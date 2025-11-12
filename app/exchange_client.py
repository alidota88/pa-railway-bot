from __future__ import annotations
from typing import Tuple, List
import ccxt
from binance import Client
from . import config

# 标准符号转换
def to_binance_symbol(sym: str) -> str:
    return sym.replace("/", "")

def to_standard_symbol(sym: str) -> str:
    if sym.endswith("USDT"):
        return f"{sym[:-4]}/USDT"
    return sym

# 只读 REST 客户端
client = Client(api_key=config.BINANCE_API_KEY or None,
                api_secret=config.BINANCE_API_SECRET or None)

# ccxt 客户端（ticker/ohlcv 回退）
ccxt_ex = getattr(ccxt, "binance")()

def get_best_bid_ask(symbol: str) -> Tuple[float, float]:
    """
    返回 (bid, ask)；symbol 使用标准 "BTC/USDT"
    """
    data = client.get_orderbook_ticker(symbol=to_binance_symbol(symbol))
    return float(data["bidPrice"]), float(data["askPrice"])

def get_orderbook_levels(symbol: str, limit: int = 10) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    返回 (bids, asks)，各为 [(price, qty), ...]
    """
    depth = client.get_order_book(symbol=to_binance_symbol(symbol), limit=limit)
    bids = [(float(p), float(q)) for p, q in depth["bids"]]
    asks = [(float(p), float(q)) for p, q in depth["asks"]]
    return bids, asks

def get_mid_price(symbol: str) -> float:
    bid, ask = get_best_bid_ask(symbol)
    return (bid + ask) / 2.0