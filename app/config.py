import os

# 使用环境变量来配置，方便 Railway 上修改

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ccxt 的交易所 & 币种列表，直接用现货对 USDT
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")

SYMBOLS_STR = os.getenv(
    "SYMBOLS",
    "BTC/USDT,ETH/USDT,BNB/USDT,XRP/USDT,SOL/USDT"
)
SYMBOLS = [s.strip() for s in SYMBOLS_STR.split(",") if s.strip()]

TIMEFRAME = os.getenv("TIMEFRAME", "5m")

# 风控参数（虚拟盘）
START_EQUITY = float(os.getenv("START_EQUITY", "10000"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "5.0"))

# 数据库（Railway Postgres 的连接串）
DATABASE_URL = os.getenv("DATABASE_URL", "")
