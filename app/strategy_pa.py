# app/strategy_pa.py
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass

# ========= 策略参数 =========

@dataclass
class PAStrategyParams:
    # 四大模块
    enable_breakout: bool = True
    enable_reversal: bool = True
    enable_climax: bool = True
    enable_failedBO: bool = True

    # 过滤项
    use_IBS_filter: bool = True
    IBS_threshold: float = 0.69          # Bullish IBS
    IBS_threshold_bear: float = 0.31     # Bearish IBS

    use_MIG_filter: bool = True          # 是否使用 Micro Gap 过滤
    skip_late_wave: bool = True          # 是否跳过晚段（第3、4腿）
    skip_early_time: bool = False        # 是否跳过开盘前 X 分钟（主要给日内用）
    session_start_hour: int = 9          # 交易时段开始（小时）
    session_start_min: int = 30          # 交易时段开始（分钟）
    early_session_cutoff_min: int = 40   # 跳过前多少分钟

    # === 新增：RSI + EMA 趋势过滤（第一阶段优化，可开关） ===
    use_rsi_ema_filter: bool = True
    RSI_period: int = 14
    EMA_period: int = 20
    RSI_long_threshold: float = 55
    RSI_short_threshold: float = 45


@dataclass
class PAStrategyState:
    """
    用来保存趋势阶段相关状态。
      - trend_dir:  1 = 上升趋势, -1 = 下降趋势, 0 = 无明显趋势
      - leg_count:  当前趋势方向上的“推进腿数”粗略计数
    """
    trend_dir: int = 0
    leg_count: int = 0


# ========= 指标计算 =========

def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    向量化 RSI（Welles Wilder 平滑）
    不依赖 TA-Lib，返回与 close 等长的 Series。
    """
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # EWM 等效于 Wilder 平滑（alpha=1/period）
    roll_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_indicators(df: pd.DataFrame, params: PAStrategyParams) -> pd.DataFrame:
    """
    计算：
      - range
      - ATR10 (简单用10期均值)
      - IBS
      - Micro Gap (bull_mig / bear_mig)
      - session_minute (当天第几分钟)
      - RSI / EMA20（用于趋势过滤，可开关）
    """
    df = df.copy()

    # 基础
    df["range"] = df["high"] - df["low"]
    df["atr10"] = df["range"].rolling(10).mean()

    # IBS: (close - low)/(high - low)，高低价相等时给一个中性值 0.5
    rng = df["high"] - df["low"]
    ibs = (df["close"] - df["low"]) / rng.replace(0, pd.NA)
    ibs = ibs.clip(0, 1).fillna(0.5)
    df["ibs"] = ibs

    # Micro Gap：当前 low 在前两根 high 之上 / 当前 high 在前两根 low 之下
    df["bull_mig"] = (df["low"] > df["high"].shift(1)) & (df["low"] > df["high"].shift(2))
    df["bear_mig"] = (df["high"] < df["low"].shift(1)) & (df["high"] < df["low"].shift(2))

    # 会话内分钟数（用 UTC 日内时间简单近似）
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"])
    else:
        ts = pd.to_datetime(df.index)
    df["session_minute"] = ts.dt.hour * 60 + ts.dt.minute

    # RSI + EMA（用于趋势过滤）
    if params.use_rsi_ema_filter:
        df["rsi"] = compute_rsi(df["close"], params.RSI_period)
        df["ema20"] = df["close"].ewm(span=params.EMA_period, adjust=False).mean()

    return df


def compute_trend_state(df: pd.DataFrame, state: PAStrategyState) -> None:
    """
    简化版趋势 & 腿数估计：
      - 使用 close 相对 EMA20 的位置确定 trend_dir
      - 在最近 20 根里统计“创新高次数”(多头)或“创新低次数”(空头)作为 leg_count
    """
    if len(df) < 20:
        state.trend_dir = 0
        state.leg_count = 0
        return

    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean()

    last_close = float(close.iloc[-1])
    last_ema = float(ema20.iloc[-1])

    if last_close > last_ema:
        trend_dir = 1
    elif last_close < last_ema:
        trend_dir = -1
    else:
        trend_dir = 0

    # 统计腿数：最近 20 根中创新高/创新低的次数
    leg_count = 0
    if trend_dir == 1:
        highs = df["high"].tail(20).to_list()
        max_h = highs[0]
        for h in highs[1:]:
            if h > max_h:
                leg_count += 1
                max_h = h
    elif trend_dir == -1:
        lows = df["low"].tail(20).to_list()
        min_l = lows[0]
        for l in lows[1:]:
            if l < min_l:
                leg_count += 1
                min_l = l
    else:
        leg_count = 0

    state.trend_dir = trend_dir
    state.leg_count = leg_count


# ========= 核心：生成信号 =========

def generate_signal(
    df: pd.DataFrame,
    state: PAStrategyState,
    params: PAStrategyParams | None = None,
) -> dict:
    """
    输入：最近一段 K 线（按时间升序）
    输出：
      {
        "side": "long"/"short"/None,
        "reason": str,
        "atr": float,
        "detail": {各模块布尔条件/指标快照}
      }
    """
    if params is None:
        params = PAStrategyParams()

    if df is None or len(df) < 20:
        return {"side": None, "reason": "not_enough_bars", "atr": None, "detail": {}}

    df = compute_indicators(df, params)
    compute_trend_state(df, state)

    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    # 指标取值 + 兜底，避免类型/NaN错误
    def _num(x):
        return float(pd.to_numeric(x, errors="coerce"))

    atr10 = _num(last.get("atr10"))
    if not np.isfinite(atr10) or atr10 <= 0:
        return {"side": None, "reason": "no_atr", "atr": None, "detail": {}}

    ibs = _num(last.get("ibs"))
    ibs_prev = _num(prev.get("ibs"))

    bull_MIG = bool(last.get("bull_mig", False))
    bear_MIG = bool(last.get("bear_mig", False))

    trendDir = state.trend_dir
    legCount = state.leg_count

    # Session 过滤（早盘跳过）
    session_minute = int(_num(last.get("session_minute")))
    session_start_total = params.session_start_hour * 60 + params.session_start_min
    early_cutoff = session_start_total + params.early_session_cutoff_min
    early_ok = (not params.skip_early_time) or (session_minute >= early_cutoff)

    # === 新增：RSI + EMA 趋势过滤（可关闭） ===
    rsi_val = np.nan
    ema_val = np.nan
    trend_long = True
    trend_short = True
    if params.use_rsi_ema_filter:
        rsi_val = _num(last.get("rsi"))
        ema_val = _num(last.get("ema20"))
        price = _num(last.get("close"))

        # 样本不足时直接不出信号
        if not np.isfinite(rsi_val) or not np.isfinite(ema_val):
            return {"side": None, "reason": "nan_indicators", "atr": None, "detail": {}}

        trend_long = (rsi_val > params.RSI_long_threshold) and (price > ema_val)
        trend_short = (rsi_val < params.RSI_short_threshold) and (price < ema_val)

        # RSI 落在 45~55 的震荡区 -> 不交易
        if not (trend_long or trend_short):
            return {"side": None, "reason": "rsi_neutral_zone", "atr": atr10, "detail": {}}

    # ========== Breakout Strategy ==========
    brk_long_cond = False
    brk_short_cond = False

    if params.enable_breakout:
        # Bullish breakout: 两根 K 线的延续突破
        prev_up_break = _num(prev["high"]) > _num(prev2["high"])
        prev_bull = _num(prev["close"]) > _num(prev["open"])
        curr_bull = _num(last["close"]) > _num(last["open"])
        size_ok = max(_num(prev["range"]), _num(prev2["range"])) >= atr10
        ibs_ok_prev = ibs_prev >= params.IBS_threshold
        strength_ok = (ibs_ok_prev if params.use_IBS_filter else True)

        if prev_up_break and prev_bull and curr_bull and size_ok and strength_ok and trend_long:
            if (not params.skip_late_wave) or (legCount < 2):
                if early_ok:
                    if (not params.use_IBS_filter) or (ibs >= 0.5):
                        brk_long_cond = True

        # Bearish breakout
        prev_down_break = _num(prev["low"]) < _num(prev2["low"])
        prev_bear = _num(prev["close"]) < _num(prev["open"])
        curr_bear = _num(last["close"]) < _num(last["open"])
        size_ok_down = max(_num(prev["range"]), _num(prev2["range"])) >= atr10
        ibs_ok_prev_down = ibs_prev <= params.IBS_threshold_bear
        strength_ok_down = (ibs_ok_prev_down if params.use_IBS_filter else True)

        if prev_down_break and prev_bear and curr_bear and size_ok_down and strength_ok_down and trend_short:
            if (not params.skip_late_wave) or (legCount < 2):
                if early_ok:
                    if (not params.use_IBS_filter) or (ibs <= 0.5):
                        brk_short_cond = True

    # ========== Climax Follow-through Strategy ==========
    climax_long_cond = False
    climax_short_cond = False

    if params.enable_climax:
        # Bullish climax
        huge_bull = (_num(last["close"]) > _num(last["open"])) and (_num(last["range"]) >= 2 * atr10)
        prior_bull = _num(prev["close"]) > _num(prev["open"])
        not_reversal_bar = prior_bull

        if huge_bull and not_reversal_bar and trend_long:
            momentum_gap = bull_MIG
            if ((not params.skip_late_wave) or (legCount < 2)) and \
               ((not params.use_MIG_filter) or momentum_gap or legCount < 2):
                climax_long_cond = True

        # Bearish climax
        huge_bear = (_num(last["close"]) < _num(last["open"])) and (_num(last["range"]) >= 2 * atr10)
        prior_bear = _num(prev["close"]) < _num(prev["open"])
        not_reversal_bar_down = prior_bear

        if huge_bear and not_reversal_bar_down and trend_short:
            momentum_gap_down = bear_MIG
            if ((not params.skip_late_wave) or (legCount < 2)) and \
               ((not params.use_MIG_filter) or momentum_gap_down or legCount < 2):
                climax_short_cond = True

    # ========== Failed Breakout Reversal ==========
    fail_rev_long_cond = False
    fail_rev_short_cond = False

    if params.enable_failedBO:
        # Failed bullish breakout -> 做空
        prev_breakout_up = (_num(prev["high"]) > _num(prev2["high"])) and (_num(prev["close"]) > _num(prev["open"]))
        curr_strong_down = (_num(last["close"]) < _num(last["open"])) and (_num(last["close"]) < _num(prev["low"]))
        curr_outside_down = (_num(last["low"]) < _num(prev["low"])) and (_num(last["high"]) > _num(prev["high"]))

        if prev_breakout_up and (curr_strong_down or curr_outside_down) and trend_short:
            if (not params.use_IBS_filter) or (ibs <= params.IBS_threshold_bear):
                fail_rev_short_cond = True

        # Failed bearish breakout -> 做多
        prev_breakout_down = (_num(prev["low"]) < _num(prev2["low"])) and (_num(prev["close"]) < _num(prev["open"]))
        curr_strong_up = (_num(last["close"]) > _num(last["open"])) and (_num(last["close"]) > _num(prev["high"]))
        curr_outside_up = (_num(last["high"]) > _num(prev["high"])) and (_num(last["low"]) < _num(prev["low"]))

        if prev_breakout_down and (curr_strong_up or curr_outside_up) and trend_long:
            if (not params.use_IBS_filter) or (ibs >= params.IBS_threshold):
                fail_rev_long_cond = True

    # ========== General Two-Bar Reversal ==========
    rev_long_cond = False
    rev_short_cond = False

    if params.enable_reversal:
        # Bullish reversal
        bar1_bear = _num(prev["close"]) < _num(prev["open"])
        bull_reversal = (_num(last["close"]) > _num(last["open"])) and (_num(last["close"]) > _num(prev["high"]))
        bull_rev_bar_strength = (not params.use_IBS_filter) or (ibs >= params.IBS_threshold)

        if bar1_bear and bull_reversal and bull_rev_bar_strength and trend_long:
            if (not params.skip_late_wave) or (trendDir == -1):
                rev_long_cond = True

        # Bearish reversal
        bar1_bull = _num(prev["close"]) > _num(prev["open"])
        bear_reversal = (_num(last["close"]) < _num(last["open"])) and (_num(last["close"]) < _num(prev["low"]))
        bear_rev_bar_strength = (not params.use_IBS_filter) or (ibs <= params.IBS_threshold_bear)

        if bar1_bull and bear_reversal and bear_rev_bar_strength and trend_short:
            if (not params.skip_late_wave) or (trendDir == 1):
                rev_short_cond = True

    # ========== 汇总多空信号 ==========
    long_modules: list[str] = []
    short_modules: list[str] = []

    if brk_long_cond:
        long_modules.append("breakout")
    if climax_long_cond:
        long_modules.append("climax")
    if fail_rev_long_cond:
        long_modules.append("failed_breakout")
    if rev_long_cond:
        long_modules.append("reversal")

    if brk_short_cond:
        short_modules.append("breakout")
    if climax_short_cond:
        short_modules.append("climax")
    if fail_rev_short_cond:
        short_modules.append("failed_breakout")
    if rev_short_cond:
        short_modules.append("reversal")

    side = None
    reason = "no_signal"

    # 如果多空同时出现，按“失败突破 > 突破 > 反转 > climax”优先级
    if long_modules and not short_modules:
        side = "long"
        priority = ["failed_breakout", "breakout", "reversal", "climax"]
        for r in priority:
            if r in long_modules:
                reason = f"{r}_long"
                break
    elif short_modules and not long_modules:
        side = "short"
        priority = ["failed_breakout", "breakout", "reversal", "climax"]
        for r in priority:
            if r in short_modules:
                reason = f"{r}_short"
                break
    elif long_modules and short_modules:
        side = None
        reason = "conflict_long_short"

    detail = {
        "brk_long": brk_long_cond,
        "brk_short": brk_short_cond,
        "climax_long": climax_long_cond,
        "climax_short": climax_short_cond,
        "fail_rev_long": fail_rev_long_cond,
        "fail_rev_short": fail_rev_short_cond,
        "rev_long": rev_long_cond,
        "rev_short": rev_short_cond,
        "trendDir": trendDir,
        "legCount": legCount,
        "ibs": ibs,
        "atr10": atr10,
        # 指标快照（便于 Telegram 推送观测）
        "rsi": float(rsi_val) if np.isfinite(rsi_val) else None,
        "ema": float(ema_val) if np.isfinite(ema_val) else None,
    }

    return {
        "side": side,
        "reason": reason,
        "atr": atr10,
        "detail": detail,
    }
