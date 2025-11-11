import pandas as pd

class PAStrategyState:
    # 暂时简单化，后面你需要可以加 trendDir / legCount 等状态
    pass

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["range"] = df["high"] - df["low"]
    df["atr10"] = df["range"].rolling(10).mean()

    # IBS：内部强度：(close - low)/(high - low)
    ibs = (df["close"] - df["low"]) / (df["high"] - df["low"])
    ibs = ibs.clip(0, 1)
    df["ibs"] = ibs

    # micro gap 示例（只做和 Pine 类似的大致逻辑）
    df["bull_mig"] = (df["low"] > df["high"].shift(1)) & (df["low"] > df["high"].shift(2))
    df["bear_mig"] = (df["high"] < df["low"].shift(1)) & (df["high"] < df["low"].shift(2))

    return df

def generate_signal(df: pd.DataFrame, state: PAStrategyState) -> dict:
    """
    输入：最近一段K线 df（按时间升序）
    输出：{"side": "long"/"short"/None, "reason": "...", "atr": float}
    """
    if df is None or len(df) < 20:
        return {"side": None, "reason": "not_enough_bars", "atr": None}

    df = compute_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    atr10 = float(last["atr10"])
    if pd.isna(atr10) or atr10 <= 0:
        return {"side": None, "reason": "no_atr", "atr": None}

    ibs_threshold_bull = 0.69
    ibs_threshold_bear = 0.31

    # === Breakout Long 示例（你可以后面逐步加完整逻辑）===
    prev_up_break = prev["high"] > prev2["high"]
    prev_bull = prev["close"] > prev["open"]
    curr_bull = last["close"] > last["open"]
    size_ok = max(prev["range"], prev2["range"]) >= atr10
    ibs_ok_prev = prev["ibs"] >= ibs_threshold_bull

    brk_long = bool(prev_up_break and prev_bull and curr_bull and size_ok and ibs_ok_prev)

    # === Breakout Short 镜像 ===
    prev_down_break = prev["low"] < prev2["low"]
    prev_bear = prev["close"] < prev["open"]
    curr_bear = last["close"] < last["open"]
    size_ok_down = max(prev["range"], prev2["range"]) >= atr10
    ibs_ok_prev_down = prev["ibs"] <= ibs_threshold_bear

    brk_short = bool(prev_down_break and prev_bear and curr_bear and size_ok_down and ibs_ok_prev_down)

    # 这里先只用 breakouts，你之后可以把 Climax / FailedBO / Reversal 都搬进来
    if brk_long and not brk_short:
        return {"side": "long", "reason": "breakout_long", "atr": atr10}
    if brk_short and not brk_long:
        return {"side": "short", "reason": "breakout_short", "atr": atr10}

    return {"side": None, "reason": "no_signal", "atr": atr10}
