import pandas as pd
from dataclasses import dataclass
import numpy as np

# ========= 策略参数 =========

@dataclass
class PAStrategyParams:
    enable_breakout: bool = True
    enable_reversal: bool = True
    enable_climax: bool = True
    enable_failedBO: bool = True

    use_IBS_filter: bool = True
    IBS_threshold: float = 0.69
    IBS_threshold_bear: float = 0.31

    use_MIG_filter: bool = True
    skip_late_wave: bool = True
    skip_early_time: bool = False
    session_start_hour: int = 9
    session_start_min: int = 30
    early_session_cutoff_min: int = 40

    # === 新增 RSI + EMA 参数 ===
    use_rsi_ema_filter: bool = True
    RSI_period: int = 14
    EMA_period: int = 20
    RSI_long_threshold: float = 55
    RSI_short_threshold: float = 45


@dataclass
class PAStrategyState:
    trend_dir: int = 0
    leg_count: int = 0


# ========= 指标计算 =========

def compute_indicators(df: pd.DataFrame, params: PAStrategyParams) -> pd.DataFrame:
    df = df.copy()
    df["range"] = df["high"] - df["low"]
    df["atr10"] = df["range"].rolling(10).mean()

    rng = df["high"] - df["low"]
    ibs = (df["close"] - df["low"]) / rng.replace(0, pd.NA)
    ibs = ibs.clip(0, 1).fillna(0.5)
    df["ibs"] = ibs

    df["bull_mig"] = (df["low"] > df["high"].shift(1)) & (df["low"] > df["high"].shift(2))
    df["bear_mig"] = (df["high"] < df["low"].shift(1)) & (df["high"] < df["low"].shift(2))

    # RSI + EMA
    df["rsi"] = df["close"].rolling(params.RSI_period).apply(lambda x: _rsi_simple(x))
    df["ema20"] = df["close"].ewm(span=params.EMA_period, adjust=False).mean()

    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"])
    else:
        ts = pd.to_datetime(df.index)
    df["session_minute"] = ts.dt.hour * 60 + ts.dt.minute

    return df


def _rsi_simple(series):
    """简化RSI计算"""
    delta = series.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_trend_state(df: pd.DataFrame, state: PAStrategyState):
    if len(df) < 20:
        state.trend_dir = 0
        state.leg_count = 0
        return

    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean()

    last_close = close.iloc[-1]
    last_ema = ema20.iloc[-1]

    if last_close > last_ema:
        trend_dir = 1
    elif last_close < last_ema:
        trend_dir = -1
    else:
        trend_dir = 0

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

    state.trend_dir = trend_dir
    state.leg_count = leg_count


# ========= 核心：生成信号 =========

def generate_signal(df: pd.DataFrame, state: PAStrategyState, params: PAStrategyParams | None = None) -> dict:
    if params is None:
        params = PAStrategyParams()

    if df is None or len(df) < 20:
        return {"side": None, "reason": "not_enough_bars", "atr": None, "detail": {}}

    df = compute_indicators(df, params)
    compute_trend_state(df, state)

    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    atr10 = float(last["atr10"])
    if pd.isna(atr10) or atr10 <= 0:
        return {"side": None, "reason": "no_atr", "atr": None, "detail": {}}

    ibs = float(last["ibs"])
    ibs_prev = float(prev["ibs"])
    bull_MIG = bool(last["bull_mig"])
    bear_MIG = bool(last["bear_mig"])
    trendDir = state.trend_dir
    legCount = state.leg_count

    # === RSI + EMA 趋势过滤 ===
    if params.use_rsi_ema_filter:
        rsi = float(last["rsi"])
        ema = float(last["ema20"])
        price = float(last["close"])

        trend_long = (rsi > params.RSI_long_threshold) and (price > ema)
        trend_short = (rsi < params.RSI_short_threshold) and (price < ema)
        trend_neutral = not (trend_long or trend_short)

        if trend_neutral:
            return {"side": None, "reason": "rsi_neutral_zone", "atr": atr10, "detail": {}}
    else:
        trend_long = trend_short = True  # 不启用过滤则全通过

    # === Session 过滤 ===
    session_minute = int(last["session_minute"])
    session_start_total = params.session_start_hour * 60 + params.session_start_min
    early_cutoff = session_start_total + params.early_session_cutoff_min
    early_ok = (not params.skip_early_time) or (session_minute >= early_cutoff)

    # ========== Breakout Strategy ==========
    brk_long_cond = False
    brk_short_cond = False
    if params.enable_breakout:
        prev_up_break = prev["high"] > prev2["high"]
        prev_bull = prev["close"] > prev["open"]
        curr_bull = last["close"] > last["open"]
        size_ok = max(prev["range"], prev2["range"]) >= atr10
        ibs_ok_prev = ibs_prev >= params.IBS_threshold
        if prev_up_break and prev_bull and curr_bull and size_ok and ibs_ok_prev and trend_long:
            if (not params.skip_late_wave or legCount < 2) and early_ok:
                brk_long_cond = True

        prev_down_break = prev["low"] < prev2["low"]
        prev_bear = prev["close"] < prev["open"]
        curr_bear = last["close"] < last["open"]
        size_ok_down = max(prev["range"], prev2["range"]) >= atr10
        ibs_ok_prev_down = ibs_prev <= params.IBS_threshold_bear
        if prev_down_break and prev_bear and curr_bear and size_ok_down and ibs_ok_prev_down and trend_short:
            if (not params.skip_late_wave or legCount < 2) and early_ok:
                brk_short_cond = True

    # === 其他模块（保留原逻辑） ===
    climax_long_cond = params.enable_climax and (last["close"] > last["open"]) and (last["range"] >= 2 * atr10) and trend_long
    climax_short_cond = params.enable_climax and (last["close"] < last["open"]) and (last["range"] >= 2 * atr10) and trend_short

    fail_rev_long_cond = params.enable_failedBO and (prev["low"] < prev2["low"]) and (prev["close"] < prev["open"]) and \
        ((last["close"] > last["open"] and last["close"] > prev["high"]) or (last["high"] > prev["high"] and last["low"] < prev["low"])) and trend_long

    fail_rev_short_cond = params.enable_failedBO and (prev["high"] > prev2["high"]) and (prev["close"] > prev["open"]) and \
        ((last["close"] < last["open"] and last["close"] < prev["low"]) or (last["low"] < prev["low"] and last["high"] > prev["high"])) and trend_short

    rev_long_cond = params.enable_reversal and (prev["close"] < prev["open"]) and (last["close"] > last["open"]) and (last["close"] > prev["high"]) and trend_long
    rev_short_cond = params.enable_reversal and (prev["close"] > prev["open"]) and (last["close"] < last["open"]) and (last["close"] < prev["low"]) and trend_short

    # === 汇总 ===
    long_modules = [m for m, c in zip(["breakout", "climax", "failed_breakout", "reversal"],
                                      [brk_long_cond, climax_long_cond, fail_rev_long_cond, rev_long_cond]) if c]
    short_modules = [m for m, c in zip(["breakout", "climax", "failed_breakout", "reversal"],
                                       [brk_short_cond, climax_short_cond, fail_rev_short_cond, rev_short_cond]) if c]

    side, reason = None, "no_signal"
    if long_modules and not short_modules:
        side = "long"
        reason = f"{long_modules[0]}_long"
    elif short_modules and not long_modules:
        side = "short"
        reason = f"{short_modules[0]}_short"
    elif long_modules and short_modules:
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
        "rsi": float(last["rsi"]),
        "ema": float(last["ema20"]),
        "ibs": ibs,
        "atr10": atr10,
    }

    return {"side": side, "reason": reason, "atr": atr10, "detail": detail}
