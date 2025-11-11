import pandas as pd
from dataclasses import dataclass


# ========= 策略参数 =========

@dataclass
class PAStrategyParams:
    enable_breakout: bool = True
    enable_reversal: bool = True
    enable_climax: bool = True
    enable_failedBO: bool = True

    use_IBS_filter: bool = True
    IBS_threshold: float = 0.69          # Bullish IBS
    IBS_threshold_bear: float = 0.31     # Bearish IBS

    use_MIG_filter: bool = True          # 是否使用 Micro Gap 过滤
    skip_late_wave: bool = True          # 是否跳过晚段（第3、4腿）
    skip_early_time: bool = False        # 是否跳过开盘前 X 分钟
    session_start_hour: int = 9          # 交易时段开始（小时）
    session_start_min: int = 30          # 交易时段开始（分钟）
    early_session_cutoff_min: int = 40   # 跳过前多少分钟


@dataclass
class PAStrategyState:
    """
    用来保存趋势阶段相关状态。
    这里给一个简单实现：
      - trend_dir:  1 = 上升趋势, -1 = 下降趋势, 0 = 无明显趋势
      - leg_count:  当前趋势方向上的“推进腿数”粗略计数
    你后面如果有更精细的腿计数逻辑，可以在 compute_trend_state 里替换。
    """
    trend_dir: int = 0
    leg_count: int = 0


# ========= 指标计算 =========

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算：
      - range
      - ATR10 (简单用10期均值)
      - IBS
      - Micro Gap (bull_mig / bear_mig)
      - session_minute (当天第几分钟)
    """
    df = df.copy()
    df["range"] = df["high"] - df["low"]
    df["atr10"] = df["range"].rolling(10).mean()

    # IBS: (close - low)/(high - low)，高低价相等时给一个中性值 0.5
    rng = df["high"] - df["low"]
    ibs = (df["close"] - df["low"]) / rng.replace(0, pd.NA)
    ibs = ibs.clip(0, 1).fillna(0.5)
    df["ibs"] = ibs

    # Micro Gap：当前 low 在前两根 high 之上 / 当前 high 在前两根 low 之下
    df["bull_mig"] = (df["low"] > df["high"].shift(1)) & (df["low"] > df["high"].shift(2))
    df["bear_mig"] = (df["high"] < df["low"].shift(1)_]()
