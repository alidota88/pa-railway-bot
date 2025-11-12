from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from . import config

# ========= 策略参数 =========

@dataclass
class PAStrategyParams:
    # 核心模块
    enable_trend_follow: bool = True
    enable_breakout: bool = True

    # RSI + EMA 趋势过滤
    use_rsi_ema_filter: bool = True
    RSI_period: int = 14
    EMA_fast: int = 20
    EMA_slow: int = 60
    RSI_long_threshold: float = 55
    RSI_short_threshold: float = 45

    # ATR止损
    ATR_period: int = 14

    # 不同周期适配参数
    timeframe: str = getattr(config, "TIMEFRAME", "4h")  # 4h / 15m / 1d


@dataclass
class PAStrategyState:
    trend_dir: int = 0
    leg_count: int = 0


# ========= 基础指标 =========

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum(abs(df["high"] - df["close"].shift(1)),
                               abs(df["low"] - df["close"].shift(1))))
    return tr.rolling(period).mean()


# ========= 核心信号函数 =========

def generate_signal(
    df: pd.DataFrame,
    state: PAStrategyState,
    params: PAStrategyParams | None = None,
) -> dict:
    """
    4h / 15m 周期 swing 策略：EMA趋势 + RSI 动量 + ATR止损
    保持接口兼容 trading_engine
    """
    if params is None:
        params = PAStrategyParams()

    if df is None or len(df) < 60:
        return {"side": None, "reason": "not_enough_bars", "atr": None, "detail": {}}

    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=params.EMA_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=params.EMA_slow, adjust=False).mean()
    df["rsi"] = compute_rsi(df["close"], params.RSI_period)
    df["atr"] = compute_atr(df, params.ATR_period)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    ema_fast = float(last["ema_fast"])
    ema_slow = float(last["ema_slow"])
    rsi_val = float(last["rsi"])
    atr_val = float(last["atr"])
    price = float(last["close"])

    # 趋势过滤：EMA方向
    bullish = ema_fast > ema_slow
    bearish = ema_fast < ema_slow

    # RSI过滤
    long_filter = bullish and rsi_val > params.RSI_long_threshold
    short_filter = bearish and rsi_val < params.RSI_short_threshold

    # 多周期微调参数
    tf = params.timeframe.lower()
    if tf == "15m":
        rsi_long = 52
        rsi_short = 48
    elif tf == "4h":
        rsi_long = 55
        rsi_short = 45
    else:
        rsi_long = 60
        rsi_short = 40

    # 趋势信号
    long_cond = long_filter and rsi_val > rsi_long
    short_cond = short_filter and rsi_val < rsi_short

    # 突破辅助（breakout）
    if params.enable_breakout:
        breakout_up = price > prev["high"] and bullish
        breakout_down = price < prev["low"] and bearish
        long_cond = long_cond or breakout_up
        short_cond = short_cond or breakout_down

    # 汇总信号
    side = None
    reason = "no_signal"
    if long_cond and not short_cond:
        side = "long"
        reason = f"{tf}_trend_long"
    elif short_cond and not long_cond:
        side = "short"
        reason = f"{tf}_trend_short"

    # 更新状态
    state.trend_dir = 1 if bullish else -1 if bearish else 0

    detail = {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "rsi": rsi_val,
        "atr": atr_val,
        "trend": "up" if bullish else "down" if bearish else "flat",
        "tf": tf,
    }

    return {"side": side, "reason": reason, "atr": atr_val, "detail": detail}
