# app/trading_engine.py
from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

# ------------ 配置读取（与 config.py 对齐）------------
def _getenv_float(k: str, default: float) -> float:
    try:
        return float(os.getenv(k, str(default)))
    except Exception:
        return default

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
USE_BINANCE_FUTURES = os.getenv("USE_BINANCE_FUTURES", "0") == "1"

SLIPPAGE_MODEL = os.getenv("SLIPPAGE_MODEL", "impact")  # fixed / spread / impact
SLIPPAGE_RATE = _getenv_float("SLIPPAGE_RATE", 0.0005)
MIN_SPREAD_SLIPPAGE = _getenv_float("MIN_SPREAD_SLIPPAGE", 0.0001)
IMPACT_TOP_LEVELS = int(os.getenv("IMPACT_TOP_LEVELS", "10"))
IMPACT_COEF = _getenv_float("IMPACT_COEF", 0.7)

TAKER_FEE_RATE = _getenv_float("TAKER_FEE_RATE", 0.0005)
MAX_LEVERAGE = _getenv_float("MAX_LEVERAGE", 1.0)
RISK_PER_TRADE_PCT = _getenv_float("RISK_PER_TRADE_PCT", 0.01)
START_EQUITY = _getenv_float("START_EQUITY", 10000.0)

SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT").split(",") if s.strip()]
TIMEFRAME = os.getenv("TIMEFRAME", "1h")

# ------------ 可选依赖（都失败不影响主流程）------------
_ccxt = None
try:
    import ccxt  # type: ignore
    _ccxt = getattr(ccxt, "binance")()
except Exception:
    _ccxt = None

_binance_client = None
try:
    from binance import Client  # type: ignore
    _binance_client = Client(api_key=BINANCE_API_KEY or None,
                             api_secret=BINANCE_API_SECRET or None)
except Exception:
    _binance_client = None

# 如果用户已经按之前建议创建了 exchange_client / slippage，这里优先使用
_use_external_exchange = False
_use_external_slippage = False
try:
    from .exchange_client import get_best_bid_ask as ext_get_best_bid_ask, \
        get_orderbook_levels as ext_get_orderbook_levels, get_mid_price as ext_get_mid_price, ccxt_exchange as ext_ccxt_exchange
    _use_external_exchange = True
except Exception:
    pass

try:
    from .slippage import compute_exec_price as ext_compute_exec_price
    _use_external_slippage = True
except Exception:
    pass

# ------------ 工具函数：符号转换 ------------
def _to_binance_symbol(sym: str) -> str:
    return sym.replace("/", "")

# ------------ 盘口/价格读取 ------------
def get_best_bid_ask(symbol: str) -> Tuple[float, float]:
    """优先用外部 exchange_client，其次 python-binance，失败抛异常。"""
    if _use_external_exchange:
        return ext_get_best_bid_ask(symbol)
    if _binance_client is None:
        raise RuntimeError("binance client unavailable")
    data = _binance_client.get_orderbook_ticker(symbol=_to_binance_symbol(symbol))
    return float(data["bidPrice"]), float(data["askPrice"])

def get_orderbook_levels(symbol: str, limit: int = IMPACT_TOP_LEVELS) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    if _use_external_exchange:
        return ext_get_orderbook_levels(symbol, limit=limit)
    if _binance_client is None:
        raise RuntimeError("binance client unavailable")
    depth = _binance_client.get_order_book(symbol=_to_binance_symbol(symbol), limit=limit)
    bids = [(float(p), float(q)) for p, q in depth["bids"]]
    asks = [(float(p), float(q)) for p, q in depth["asks"]]
    return bids, asks

def get_mid_price(symbol: str, fallback_last: Optional[float] = None) -> float:
    if _use_external_exchange:
        try:
            return ext_get_mid_price(symbol)
        except Exception:
            pass
    # 先 bookTicker
    if _binance_client:
        try:
            bid, ask = get_best_bid_ask(symbol)
            return (bid + ask) / 2.0
        except Exception:
            pass
    # 再 ccxt
    if _ccxt:
        try:
            t = _ccxt.fetch_ticker(symbol)
            return float(t["last"])
        except Exception:
            pass
    # 兜底
    if fallback_last is not None:
        return float(fallback_last)
    raise RuntimeError(f"cannot fetch mid price for {symbol}")

# ------------ 滑点计算 ------------
def compute_exec_price(symbol: str, side: str, qty: float, last_price: float) -> float:
    """side: 'long'/'short'"""
    if _use_external_slippage:
        # 如果用户已经有 slippage.py，就直接调用
        try:
            return ext_compute_exec_price(symbol, side, qty, last_price)
        except Exception:
            pass

    # 内置逻辑
    model = SLIPPAGE_MODEL.lower()
    if model == "fixed":
        rate = SLIPPAGE_RATE
        return last_price * (1 + rate) if side == "long" else last_price * (1 - rate)

    # spread/impact 需要盘口
    try:
        bid, ask = get_best_bid_ask(symbol)
        mid = (bid + ask) / 2.0
        half_spread = max(((ask - bid) / 2.0) / mid, MIN_SPREAD_SLIPPAGE)

        if model == "spread":
            return mid * (1 + half_spread) if side == "long" else mid * (1 - half_spread)

        if model == "impact":
            bids, asks = get_orderbook_levels(symbol, limit=IMPACT_TOP_LEVELS)
            depth_qty = sum(q for _, q in (asks if side == "long" else bids))
            if depth_qty <= 0:
                impact = SLIPPAGE_RATE
            else:
                impact = min(0.01, (qty / depth_qty) * IMPACT_COEF)
            total_shift = half_spread + impact
            return mid * (1 + total_shift) if side == "long" else mid * (1 - total_shift)

        # 未知模型 -> fixed
        return last_price * (1 + SLIPPAGE_RATE) if side == "long" else last_price * (1 - SLIPPAGE_RATE)
    except Exception:
        # 盘口失败 -> fixed
        return last_price * (1 + SLIPPAGE_RATE) if side == "long" else last_price * (1 - SLIPPAGE_RATE)

# ------------ 纸质账户与仓位（示例实现，可换成你的持久化）------------
@dataclass
class Position:
    symbol: str
    side: str             # 'long' or 'short'
    size: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    updated_ts: float = field(default_factory=time.time)

@dataclass
class Account:
    equity: float = START_EQUITY
    positions: Dict[str, Position] = field(default_factory=dict)

ACCOUNT = Account()

def _position_value(pos: Position, mark: float) -> float:
    if pos.side == "long":
        return pos.size * mark
    else:
        # 简化：做空名义价值
        return pos.size * (2 * pos.avg_price - mark)

# ------------ 风险控制 ------------
def _check_leverage_ok(symbol: str, new_notional: float) -> bool:
    # 简化：全仓权益 / 总名义 <= MAX_LEVERAGE
    total_notional = sum(abs(p.size * p.avg_price) for p in ACCOUNT.positions.values())
    after = total_notional + abs(new_notional)
    if MAX_LEVERAGE <= 0:
        return True
    return (after / max(ACCOUNT.equity, 1e-9)) <= MAX_LEVERAGE

# ------------ 纸质撮合 ------------
def paper_execute_order(symbol: str, side: str, qty: float, last_price: Optional[float] = None) -> Dict:
    """
    返回执行细节：{symbol, side, qty, exec_price, fee, pnl_delta, equity}
    side: 'buy'/'sell' 或 'long'/'short'
    """
    side = "long" if side in ("buy", "long") else "short"
    mark = get_mid_price(symbol, fallback_last=last_price or 0.0)
    exec_price = compute_exec_price(symbol, side, qty, mark)

    notional = qty * exec_price
    if not _check_leverage_ok(symbol, notional):
        return {"status": "rejected/leverage", "symbol": symbol, "side": side, "qty": qty}

    fee = notional * TAKER_FEE_RATE

    pos = ACCOUNT.positions.get(symbol)
    if pos is None:
        pos = Position(symbol=symbol, side=side, size=0.0, avg_price=0.0)
        ACCOUNT.positions[symbol] = pos

    pnl_delta = 0.0
    if pos.size == 0:
        # 开仓
        pos.side = side
        pos.size = qty
        pos.avg_price = exec_price
    elif pos.side == side:
        # 加仓 -> 加权均价
        total_cost = pos.avg_price * pos.size + exec_price * qty
        pos.size += qty
        pos.avg_price = total_cost / max(pos.size, 1e-9)
    else:
        # 反向单 -> 平仓/反手
        closed = min(pos.size, qty)
        # 平仓盈亏
        if pos.side == "long":
            pnl_delta += (exec_price - pos.avg_price) * closed
        else:
            pnl_delta += (pos.avg_price - exec_price) * closed
        pos.size -= closed
        if pos.size <= 1e-12:
            pos.size = 0.0
            pos.avg_price = 0.0
            # 反手剩余
            remain = qty - closed
            if remain > 1e-12:
                pos.side = side
                pos.size = remain
                pos.avg_price = exec_price
        else:
            # 部分平后仍保留原方向
            pass

    pos.updated_ts = time.time()
    ACCOUNT.equity += pnl_delta - fee

    return {
        "status": "filled",
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "exec_price": exec_price,
        "fee": fee,
        "pnl_delta": pnl_delta,
        "equity": ACCOUNT.equity,
        "ts": pos.updated_ts,
    }

# ------------ 策略入口（示例）------------
def run_strategy_and_update_positions(signals: List[Dict]) -> List[Dict]:
    """
    signals: 每个信号格式建议：
      {
        "symbol": "BTC/USDT",
        "action": "buy" | "sell" | "close",
        "strength": 1.0,         # 可选，用于按强度决定下单量
        "qty": 0.01,             # 可选，缺省将按风险百分比估算
        "last_price": 68000.0    # 可选
      }
    返回：每笔执行结果列表（便于推送到 Telegram）
    """
    results = []
    for s in signals:
        symbol = s.get("symbol", SYMBOLS[0])
        action = s.get("action", "hold").lower()
        last_price = s.get("last_price", None)

        if action not in ("buy", "sell", "close"):
            continue

        # 估算下单量：按账户权益 * 风险百分比 / 当前价格（极简演示）
        mark = None
        if last_price is not None:
            mark = last_price
        try:
            mark = mark or get_mid_price(symbol)
        except Exception:
            mark = last_price or 0.0

        if mark <= 0:
            # 无法取价，跳过
            continue

        if action == "close":
            # 平掉同向仓位
            pos = ACCOUNT.positions.get(symbol)
            if pos and pos.size > 0:
                side = "sell" if pos.side == "long" else "buy"
                res = paper_execute_order(symbol, side, pos.size, last_price=mark)
                res["note"] = "close-all"
                results.append(res)
            continue

        # buy / sell
        req_qty = s.get("qty")
        if not req_qty:
            dollar_risk = max(ACCOUNT.equity * RISK_PER_TRADE_PCT, 1.0)
            req_qty = round(dollar_risk / mark, 6)

        res = paper_execute_order(symbol, action, req_qty, last_price=mark)
        res["note"] = "from-signal"
        results.append(res)

    # 这里你可以把 ACCOUNT/positions 持久化到数据库
    # save_account_to_db(ACCOUNT)  # <- 接你自己的实现
    return results

# ------------ 便捷函数：查询账户与仓位 ------------
def get_account_snapshot() -> Dict:
    # 也可以改成数据库读
    snap_pos = []
    for p in ACCOUNT.positions.values():
        if p.size <= 0:
            continue
        mark = 0.0
        try:
            mark = get_mid_price(p.symbol, fallback_last=p.avg_price)
        except Exception:
            mark = p.avg_price
        unreal = (mark - p.avg_price) * p.size if p.side == "long" else (p.avg_price - mark) * p.size
        snap_pos.append({
            "symbol": p.symbol,
            "side": p.side,
            "size": p.size,
            "avg_price": p.avg_price,
            "mark": mark,
            "unrealized_pnl": unreal,
            "updated_ts": p.updated_ts
        }
