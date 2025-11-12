# app/slippage.py
from __future__ import annotations
from typing import Literal
from . import config
from .exchange_client import get_best_bid_ask, get_orderbook_levels

SlippageModel = Literal["fixed", "spread", "impact"]

def compute_exec_price(symbol: str, side: str, qty: float, last_price: float) -> float:
    """
    返回模拟成交价（含滑点）；side: "long"/"short"
    优先使用你配置的模型；last_price 为你策略/行情的最近价兜底。
    """
    model: SlippageModel = config.SLIPPAGE_MODEL  # fixed/spread/impact

    if model == "fixed":
        rate = config.SLIPPAGE_RATE
        if side == "long":
            return last_price * (1 + rate)
        else:
            return last_price * (1 - rate)

    # 基于买卖一的点差
    try:
        bid, ask = get_best_bid_ask(symbol)
        mid = (bid + ask) / 2.0
        half_spread = max((ask - bid) / 2.0 / mid, config.MIN_SPREAD_SLIPPAGE)

        if model == "spread":
            # 买入站在卖一，卖出站在买一；最少给个 half_spread 的偏移
            if side == "long":
                return mid * (1 + half_spread)
            else:
                return mid * (1 - half_spread)

        # 成交量冲击模型：点差 + 按下单量与盘口流动性给冲击
        if model == "impact":
            bids, asks = get_orderbook_levels(symbol, limit=config.IMPACT_TOP_LEVELS)
            # 估算对手盘可承接的量（只看一侧前 N 档的总量）
            depth_qty = sum(q for _, q in (asks if side == "long" else bids))
            if depth_qty <= 0:
                impact = config.SLIPPAGE_RATE  # 兜底
            else:
                # 量化冲击：下单量 / 盘口量 * 系数
                impact = min(0.01, (qty / depth_qty) * config.IMPACT_COEF)
            total_shift = half_spread + impact
            if side == "long":
                return mid * (1 + total_shift)
            else:
                return mid * (1 - total_shift)

    except Exception:
        # 读盘口失败则回退固定比例
        rate = config.SLIPPAGE_RATE
        return last_price * (1 + rate) if side == "long" else last_price * (1 - rate)
