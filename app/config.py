from __future__ import annotations
import os

def _getenv_float(k: str, default: float) -> float:
    try:
        return float(os.getenv(k, str(default)))
    except Exception:
        return default

# ===== Binance 只读 =====
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
USE_BINANCE_FUTURES = os.getenv("USE_BINANCE_FUTURES", "0") == "1"

# ===== 滑点模型 =====
# fixed: 固定比例滑点
# spread: 基于买卖一点差
# impact: 点差 + 成交量冲击（推荐）
SLIPPAGE_MODEL = os.getenv("SLIPPAGE_MODEL", "impact")
SLIPPAGE_RATE = _getenv_float("SLIPPAGE_RATE", 0.0005)
MIN_SPREAD_SLIPPAGE = _getenv_float("MIN_SPREAD_SLIPPAGE", 0.0001)
IMPACT_TOP_LEVELS = int(os.getenv("IMPACT_TOP_LEVELS", "10"))
IMPACT_COEF = _getenv_float("IMPACT_COEF", 0.7)

# ===== 交易参数 =====
TAKER_FEE_RATE = _getenv_float("TAKER_FEE_RATE", 0.0005)
MAX_LEVERAGE = _getenv_float("MAX_LEVERAGE", 1.0)
RISK_PER_TRADE_PCT = _getenv_float("RISK_PER_TRADE_PCT", 0.01)
START_EQUITY = _getenv_float("START_EQUITY", 10000.0)

# ===== 运行范围 =====
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT").split(",") if s.strip()]
TIMEFRAME = os.getenv("TIMEFRAME", "1h")

# ===== Telegram（可选）=====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ===== 数据库（可选）=====
DATABASE_URL = os.getenv("DATABASE_URL", "")