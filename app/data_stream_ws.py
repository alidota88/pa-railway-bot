# app/data_stream_ws.py
import asyncio
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict

from binance import AsyncClient, BinanceSocketManager

from . import config
from .strategy_pa import PAStrategyParams, PAStrategyState, generate_signal
from .trading_engine import run_signal_once  # 我们稍后在 trading_engine.py 中新增这个函数

# === 配置 ===
# 使用 env 中的 SYMBOLS（形如 "BTC/USDT,ETH/USDT,..."）
SYMBOLS = config.SYMBOLS
# 时间粒度（和你策略保持一致；如果策略按 1h 做，可改 "1h"）
INTERVAL = (getattr(config, "TIMEFRAME", "1m")).lower()

# Binance stream 需要小写不带斜杠
def to_binance_stream_symbol(sym: str) -> str:
    return sym.replace("/", "").lower()

def to_standard_symbol(sym_stream: str) -> str:
    # 反向映射：btcusdt -> BTC/USDT
    base = sym_stream[:-4].upper()
    quote = sym_stream[-4:].upper()
    return f"{base}/{quote}"

# 内存K线缓存与策略状态
kline_cache: dict[str, pd.DataFrame] = defaultdict(lambda: pd.DataFrame())
state_cache: dict[str, PAStrategyState] = {s: PAStrategyState() for s in SYMBOLS}
params = PAStrategyParams()

def upsert_kline(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    # 追加/更新一行K线；只保留最近500根
    new_row = pd.DataFrame([row])
    if df.empty:
        out = new_row
    else:
        out = pd.concat([df, new_row]).drop_duplicates(subset=["ts"], keep="last")
    return out.tail(500).reset_index(drop=True)

async def stream_binance():
    client = await AsyncClient.create()
    try:
        bsm = BinanceSocketManager(client)

        streams = [f"{to_binance_stream_symbol(s)}@kline_{INTERVAL}" for s in SYMBOLS]
        async with bsm.multiplex_socket(streams) as ms:
            print(f"✅ WS connected. Subscribed: {streams}")
            while True:
                msg = await ms.recv()

                # 兼容单/多路复用
                data = msg.get("data") or msg
                k = (data or {}).get("k")
                s = (data or {}).get("s")  # 例如 "BTCUSDT"
                if not k or not s:
                    continue

                # 只在K线收盘时触发（k["x"] == True）
                if not k.get("x", False):
                    continue

                symbol_std = to_standard_symbol(s.lower())
                ts_open = int(k["t"])  # open time ms
                row = {
                    "ts": datetime.utcfromtimestamp(ts_open / 1000.0),
                    "open": float(k["o"]),
                    "high": float(k["h"]),
                    "low": float(k["l"]),
                    "close": float(k["c"]),
                    "volume": float(k["v"]),
                }

                df = kline_cache[symbol_std]
                df = upsert_kline(df, row)
                kline_cache[symbol_std] = df

                # 至少有一定长度再触发策略
                if len(df) < 30:
                    continue

                # 生成信号（使用你现有的策略）
                sig = generate_signal(df, state_cache[symbol_std], params)

                # 直接交给交易引擎处理这个 symbol（单币执行一次）
                try:
                    await run_signal_once(symbol_std, df, sig)
                except Exception as e:
                    print(f"run_signal_once error for {symbol_std}:", repr(e))

    finally:
        await client.close_connection()

async def main():
    while True:
        try:
            await stream_binance()
        except Exception as e:
            print("⚠️ WS reconnect in 5s:", repr(e))
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
