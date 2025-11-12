# app/exchange_client.py
from __future__ import annotations
import os
import ccxt
from binance import Client  # 同步REST
from typing import Tuple, List

from . import config

# 统一的 symbol 转换： "BTC/USDT" <-> "BTCUSDT"
def to_binance_symbol(sym: str) -> str:
    return sym.replace("/", "")

def to_standard_symbol(sym: str) -> str:
    # "BTCUSDT" -> "BTC/USDT"
    if sym.endswith("USDT"):
        base = sym[:-4]
        return f"{base}/{'USDT'}"
    return sym  # 简化处理

# 只读 REST 客户端（python-binance）
_client = Client(api_key=config.BINANCE_API_KEY or None,
                 api_secret=config.BINANCE_API_SECRET or None,
                 # 若走Futures可加 tld/sandbox/requests_params 等
                 )

# ccxt 客户端（现有 fetch_ohlcv 用）
_ccxt_ex = getattr(ccxt, config.EXCHANGE_ID)()

def get_best_bid_ask(symbol: str) -> Tuple[float, float]:
    """
    返回 (best_bid, best_ask)；symbol 用标准 "BTC/USDT"
    """
    s = to_binance_symbol(symbol)
    data = _client.get_orderbook_ticker(symbol=s)  # 更快：bookTicker
    bid = float(data["bidPrice"])
    ask = float(data["askPrice"])
    return bid, ask

def get_orderbook_levels(symbol: str, limit: int = 10) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    返回 (bids, asks)，各为 [(price, qty), ...]，按价格优先排序
    """
    s = to_binance_symbol(symbol)
    depth = _client.get_order_book(symbol=s, limit=limit)
    bids = [(float(p), float(q)) for p, q in depth["bids"]]
    asks = [(float(p), float(q)) for p, q in depth["asks"]]
    return bids, asks

def get_mid_price(symbol: str) -> float:
    bid, ask = get_best_bid_ask(symbol)
    return (bid + ask) / 2.0

def ccxt_exchange():
    return _ccxt_ex
