from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List

from . import config
from .exchange_client import get_mid_price
from .slippage import compute_exec_price

# ============== 账户/仓位模型 ==============
@dataclass
class Position:
    symbol: str
    side: str             # 'long' or 'short'
    size: float = 0.0
    avg_price: float = 0.0
    updated_ts: float = field(default_factory=time.time)

@dataclass
class Account:
    equity: float = config.START_EQUITY
    positions: Dict[str, Position] = field(default_factory=dict)

ACCOUNT = Account()

# ============== 杠杆校验（极简全仓） ==============
def _check_leverage_ok(new_notional: float) -> bool:
    total_notional = sum(abs(p.size * p.avg_price) for p in ACCOUNT.positions.values())
    after = total_notional + abs(new_notional)
    if config.MAX_LEVERAGE <= 0:
        return True
    return (after / max(ACCOUNT.equity, 1e-9)) <= config.MAX_LEVERAGE

# ============== 纸质撮合 ==============
def paper_execute_order(symbol: str, side: str, qty: float, last_price: Optional[float] = None) -> Dict:
    """
    side: 'buy'/'sell' 或 'long'/'short'
    """
    side = "long" if side in ("buy", "long") else "short"
    mark = get_mid_price(symbol) if last_price is None else last_price
    exec_price = compute_exec_price(symbol, side, qty, mark)

    notional = qty * exec_price
    if not _check_leverage_ok(notional):
        return {"status": "rejected/leverage", "symbol": symbol, "side": side, "qty": qty}

    fee = notional * config.TAKER_FEE_RATE

    pos = ACCOUNT.positions.get(symbol)
    if pos is None:
        pos = Position(symbol=symbol, side=side, size=0.0, avg_price=0.0)
        ACCOUNT.positions[symbol] = pos

    pnl_delta = 0.0
    if pos.size == 0:
        pos.side = side
        pos.size = qty
        pos.avg_price = exec_price
    elif pos.side == side:
        # 加仓加权
        total_cost = pos.avg_price * pos.size + exec_price * qty
        pos.size += qty
        pos.avg_price = total_cost / max(pos.size, 1e-9)
    else:
        # 反向单 -> 平仓/反手
        closed = min(pos.size, qty)
        if pos.side == "long":
            pnl_delta += (exec_price - pos.avg_price) * closed
        else:
            pnl_delta += (pos.avg_price - exec_price) * closed
        pos.size -= closed
        if pos.size <= 1e-12:
            pos.size = 0.0
            pos.avg_price = 0.0
            remain = qty - closed
            if remain > 1e-12:
                pos.side = side
                pos.size = remain
                pos.avg_price = exec_price

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

# ============== 批量信号入口 ==============
def run_strategy_and_update_positions(signals: List[Dict]) -> List[Dict]:
    """
    signals: [{"symbol":"BTC/USDT","action":"buy|sell|close","qty":0.01,"last_price":68000}, ...]
    """
    results = []
    for s in signals:
        symbol = s.get("symbol", config.SYMBOLS[0])
        action = s.get("action", "hold").lower()
        last_price = s.get("last_price")
        if action not in ("buy", "sell", "close"):
            continue

        if action == "close":
            pos = ACCOUNT.positions.get(symbol)
            if pos and pos.size > 0:
                side = "sell" if pos.side == "long" else "buy"
                res = paper_execute_order(symbol, side, pos.size, last_price=last_price)
                res["note"] = "close-all"
                results.append(res)
            continue

        qty = s.get("qty")
        if not qty:
            # 按风险估算数量
            mark = last_price or get_mid_price(symbol)
            dollar_risk = max(ACCOUNT.equity * config.RISK_PER_TRADE_PCT, 1.0)
            qty = round(dollar_risk / mark, 6)

        res = paper_execute_order(symbol, action, qty, last_price=last_price)
        res["note"] = "from-signal"
        results.append(res)

    return results

# ============== 账户快照 ==============
def get_account_snapshot() -> Dict:
    snap_pos = []
    for p in ACCOUNT.positions.values():
        if p.size <= 0:
            continue
        try:
            mark = get_mid_price(p.symbol)
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
        })
    return {
        "equity": ACCOUNT.equity,
        "positions": snap_pos
    }

# ============== 兼容 main.py 的初始化入口 ==============
def init_db_and_account():
    """
    若启用数据库，可在此处加载历史账户；当前返回内存账户，保证 main.py 正常导入。
    """
    global ACCOUNT
    return ACCOUNT