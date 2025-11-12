from __future__ import annotations
from . import config
from .exchange_client import get_best_bid_ask, get_orderbook_levels

def compute_exec_price(symbol: str, side: str, qty: float, last_price: float) -> float:
    """
    返回模拟成交价；side: 'long'/'short'
    """
    model = (config.SLIPPAGE_MODEL or "fixed").lower()

    if model == "fixed":
        r = config.SLIPPAGE_RATE
        return last_price * (1 + r) if side == "long" else last_price * (1 - r)

    try:
        bid, ask = get_best_bid_ask(symbol)
        mid = (bid + ask) / 2.0
        half_spread = max(((ask - bid) / 2.0) / mid, config.MIN_SPREAD_SLIPPAGE)

        if model == "spread":
            return mid * (1 + half_spread) if side == "long" else mid * (1 - half_spread)

        if model == "impact":
            bids, asks = get_orderbook_levels(symbol, limit=config.IMPACT_TOP_LEVELS)
            depth_qty = sum(q for _, q in (asks if side == "long" else bids))
            if depth_qty <= 0:
                impact = config.SLIPPAGE_RATE
            else:
                impact = min(0.01, (qty / depth_qty) * config.IMPACT_COEF)
            total = half_spread + impact
            return mid * (1 + total) if side == "long" else mid * (1 - total)

        # 未知模型 -> fixed
        r = config.SLIPPAGE_RATE
        return last_price * (1 + r) if side == "long" else last_price * (1 - r)

    except Exception:
        # 盘口失败 -> fixed
        r = config.SLIPPAGE_RATE
        return last_price * (1 + r) if side == "long" else last_price * (1 - r)