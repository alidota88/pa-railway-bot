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

# 合约相关配置
# 最大杠杆（名义仓位上限 ≈ equity * MAX_LEVERAGE）
MAX_LEVERAGE = float(os.getenv("MAX_LEVERAGE", "5"))  # 比如 5 倍

# 手续费（按名义价值的比例），0.0006 = 0.06%
TAKER_FEE_RATE = float(os.getenv("TAKER_FEE_RATE", "0.0006"))

# 滑点（相对价格比例），0.0005 = 0.05%
SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0005"))


# 数据库（Railway Postgres 的连接串）
DATABASE_URL = os.getenv("DATABASE_URL", "")

PROCESS_ROLE = os.getenv("PROCESS_ROLE", "web")  # web / worker
TELEGRAM_LOOP_ENABLED = os.getenv("TELEGRAM_LOOP_ENABLED", "1") == "1"

# ===== Binance Read-Only API（不下单也可以为空；若填入仅用于更高频限速/私有查询）=====
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# 标准化品种：你的 SYMBOLS 使用 "BTC/USDT" 这种，下面会自动转换成 "BTCUSDT"
# 期现选择（目前你代码偏现货；若后续做U本位合约，可加 futures=true 的客户端初始化）
USE_BINANCE_FUTURES = os.getenv("USE_BINANCE_FUTURES", "0") == "1"

# ===== 滑点模型选择 =====
# fixed: 固定比例（兼容你现有做法）
# spread: 基于买卖一点差（mid +/- half-spread）
# impact: 成交量冲击（点差 + 按下单量/盘口量估算冲击）
SLIPPAGE_MODEL = os.getenv("SLIPPAGE_MODEL", "impact")  # fixed / spread / impact

# 固定比例（回退用）
SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0005"))  # 0.05%

# 点差模型的最小滑点下限（避免极端紧致时为 0）
MIN_SPREAD_SLIPPAGE = float(os.getenv("MIN_SPREAD_SLIPPAGE", "0.0001"))  # 0.01%

# 成交量冲击模型参数
IMPACT_TOP_LEVELS = int(os.getenv("IMPACT_TOP_LEVELS", "10"))  # 统计前N档的挂单量
IMPACT_COEF = float(os.getenv("IMPACT_COEF", "0.7"))  # 冲击强度系数(0~1，一般0.3~1之间)
