from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Tuple, List

import ccxt
import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
)
from sqlalchemy.orm import sessionmaker, declarative_base

from . import config
from .telegram_client import send_telegram
from .strategy_pa import PAStrategyState, generate_signal, PAStrategyParams

# ========== 交易开关（由 Telegram 控制） ==========

TRADING_ENABLED: bool = True  # 默认开启自动交易

def set_trading_enabled(value: bool):
    global TRADING_ENABLED
    TRADING_ENABLED = bool(value)

def is_trading_enabled() -> bool:
    return TRADING_ENABLED



# ========== 风险 & 保证金参数 ==========

# 维持保证金率（比如 0.005 = 0.5%）
MAINTENANCE_MARGIN_RATE: float = getattr(config, "MAINTENANCE_MARGIN_RATE", 0.005)


# ========== 数据库初始化 ==========

if not config.DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Please configure it in Railway env vars.")

engine = create_engine(config.DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


# ========== ORM 模型 ==========

class Account(Base):
    __tablename__ = "accounts"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    equity = Column(Float, default=0)  # 当前总权益（已实现盈亏 + 手续费）
    cash = Column(Float, default=0)    # 简化模型下的“现金”
    created_at = Column(DateTime, default=datetime.utcnow)


class Position(Base):
    __tablename__ = "positions"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String)             # "long" / "short"
    size = Column(Float)              # 仓位数量（正数）
    entry_price = Column(Float)
    atr = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed = Column(Integer, default=0)  # 0=持仓中, 1=已平仓
    account_id = Column(Integer)


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String)              # "long" / "short"
    size = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl = Column(Float)                # 净利润（含手续费）
    opened_at = Column(DateTime)
    closed_at = Column(DateTime)
    reason = Column(String)            # "tp_sl_or_reverse" / "liquidation" 等
    account_id = Column(Integer)


def init_db_and_account():
    """建表 + 初始化虚拟账户"""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        acc = db.query(Account).filter_by(name="paper").first()
        if not acc:
            acc = Account(
                name="paper",
                equity=config.START_EQUITY,
                cash=config.START_EQUITY,
            )
            db.add(acc)
            db.commit()
    finally:
        db.close()


# ========== 行情交易所 ==========

exchange = getattr(ccxt, config.EXCHANGE_ID)()


def fetch_ohlcv_df(symbol: str, limit: int = 200) -> pd.DataFrame:
    """
    从交易所拉 OHLCV，并转为 DataFrame：
      ts, open, high, low, close, volume
    """
    ohlcv = exchange.fetch_ohlcv(symbol, config.TIMEFRAME, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df


# ========== 帐户 & 持仓工具函数 ==========

def get_account(db) -> Account:
    return db.query(Account).filter_by(name="paper").first()


def get_open_positions(db, acc_id: int) -> List[Position]:
    return db.query(Position).filter_by(account_id=acc_id, closed=0).all()


# ========== 单个持仓的保证金 & 爆仓价计算 ==========

def compute_position_margin_and_liq(pos: Position) -> Tuple[float, float, float, float]:
    """
    返回：
      notional        名义价值 = entry_price * size
      initial_margin  初始保证金 = notional / MAX_LEVERAGE
      maint_margin    维持保证金 = notional * MAINTENANCE_MARGIN_RATE
      liq_price       简化版爆仓价（按 isolated 模型推导）
    """
    notional = pos.entry_price * pos.size
    lev = config.MAX_LEVERAGE

    initial_margin = notional / lev
    maint_margin = notional * MAINTENANCE_MARGIN_RATE

    # 简化公式：
    # 对多头：IM + (P_liq - entry) * size = MM
    # 对空头：IM + (entry - P_liq) * size = MM
    if pos.side == "long":
        liq_price = pos.entry_price + (maint_margin - initial_margin) / pos.size
    else:
        liq_price = pos.entry_price - (maint_margin - initial_margin) / pos.size

    return notional, initial_margin, maint_margin, liq_price


# ========== 全账户：保证金 / 杠杆 / 未实现盈亏 统计 ==========

def compute_account_margin_and_unrealized(
    db,
    acc: Account,
):
    """
    返回：
      stats: {
        total_notional,
        used_margin,
        maint_margin_total,
        total_unrealized,
        equity_mtm,
        free_margin,
        account_leverage
      }
      price_map: { symbol: last_price }
      positions: 当前所有未平仓持仓列表
    """
    positions = get_open_positions(db, acc.id)
    total_notional = 0.0
    used_margin = 0.0
    maint_margin_total = 0.0
    total_unrealized = 0.0
    price_map: Dict[str, float] = {}

    for pos in positions:
        # 为了更真实，单独拉该 symbol 最新一根 K
        try:
            df = fetch_ohlcv_df(pos.symbol, limit=1)
            last_price = float(df.iloc[-1]["close"])
        except Exception:
            # 拉失败就退化为用 entry_price
            last_price = float(pos.entry_price)

        price_map[pos.symbol] = last_price

        notional, im, mm, _ = compute_position_margin_and_liq(pos)
        total_notional += notional
        used_margin += im
        maint_margin_total += mm

        # 未实现盈亏（mark-to-market）
        if pos.side == "long":
            total_unrealized += (last_price - pos.entry_price) * pos.size
        else:
            total_unrealized += (pos.entry_price - last_price) * pos.size

    # 按市值计的账户权益
    equity_mtm = acc.equity + total_unrealized
    free_margin = equity_mtm - used_margin
    account_leverage = total_notional / equity_mtm if equity_mtm > 0 else float("inf")

    stats = dict(
        total_notional=total_notional,
        used_margin=used_margin,
        maint_margin_total=maint_margin_total,
        total_unrealized=total_unrealized,
        equity_mtm=equity_mtm,
        free_margin=free_margin,
        account_leverage=account_leverage,
    )
    return stats, price_map, positions


# ========== 账户快照推送到 Telegram ==========

def send_account_snapshot(db, acc: Account, prefix: str = "[账户快照]"):
    """
    将当前账户整体情况推送到 Telegram：
      - 实现权益 equity
      - 按市值计的权益 equity_mtm
      - 已用保证金 / 可用保证金
      - 总名义仓位 / 杠杆
      - 当前持仓数量
    """
    try:
        stats, _, positions = compute_account_margin_and_unrealized(db, acc)
    except Exception as e:
        print("send_account_snapshot error:", repr(e))
        return

    text = (
        f"{prefix}\n"
        f"💰 Equity(已实现)：{acc.equity:.2f}\n"
        f"📈 Equity(MtM)：{stats['equity_mtm']:.2f}\n"
        f"💼 名义仓位总额：{stats['total_notional']:.2f}\n"
        f"🔒 已用保证金(IM)：{stats['used_margin']:.2f}\n"
        f"⚙️ 维持保证金(MM)：{stats['maint_margin_total']:.2f}\n"
        f"💵 可用保证金：{stats['free_margin']:.2f}\n"
        f"📊 当前杠杆：{stats['account_leverage']:.2f}x\n"
        f"📌 持仓数：{len(positions)}"
    )

    send_telegram(text)


# ========== 平仓逻辑（含滑点 & 手续费 & 原因） ==========

def close_position(
    db,
    acc: Account,
    pos: Position,
    last_price: float,
    reason: str = "tp_sl_or_reverse",
):
    """
    平仓：
      - 根据方向加滑点得到 exec_price_close
      - 计算毛利润 pnl_gross
      - 计算平仓手续费 fee_close
      - 净利润 pnl_net = pnl_gross - fee_close
      - 更新账户权益 & 记录 Trade
      - 推送平仓信息 + 平仓后账户快照
    """
    slippage = config.SLIPPAGE_RATE

    if pos.side == "long":
        # 多单平仓：卖出，价格略低
        exec_price_close = last_price * (1 - slippage)
        pnl_gross = (exec_price_close - pos.entry_price) * pos.size
    else:
        # 空单平仓：买入，价格略高
        exec_price_close = last_price * (1 + slippage)
        pnl_gross = (pos.entry_price - exec_price_close) * pos.size

    notional_close = exec_price_close * pos.size
    fee_close = notional_close * config.TAKER_FEE_RATE

    pnl_net = pnl_gross - fee_close

    acc.equity += pnl_net
    acc.cash += pnl_net

    trade = Trade(
        symbol=pos.symbol,
        side=pos.side,
        size=pos.size,
        entry_price=pos.entry_price,
        exit_price=exec_price_close,
        pnl=pnl_net,
        opened_at=pos.opened_at,
        closed_at=datetime.utcnow(),
        account_id=acc.id,
        reason=reason,
    )
    pos.closed = 1
    db.add(trade)
    db.add(acc)
    db.add(pos)

    pnl_symbol = "🟢" if pnl_net > 0 else "🔴"
    msg = (
        f"{pnl_symbol} 平仓：{pos.symbol} {pos.side.upper()}\n"
        f"数量：{pos.size:.4f}\n"
        f"入场：{pos.entry_price:.2f}  平仓：{exec_price_close:.2f}\n"
        f"净收益：{pnl_net:+.2f} USDT  (fee={fee_close:.2f})\n"
        f"原因：{reason}"
    )
    send_telegram(msg)


    # 平仓后发一条账户快照
    send_account_snapshot(db, acc, prefix=f"[平仓后账户] {pos.symbol}")


# ========== 强平逻辑：权益跌到维持保证金水平自动全平 ==========

def check_and_liquidate(db, acc: Account):
    """
    计算市值权益 equity_mtm 和总维持保证金 maint_margin_total：
      - 如果 equity_mtm <= maint_margin_total，则触发强平：
        以当前市价（含滑点）一次性平掉所有持仓。
    """
    stats, price_map, positions = compute_account_margin_and_unrealized(db, acc)

    equity_mtm = stats["equity_mtm"]
    maint_margin_total = stats["maint_margin_total"]

    if not positions:
        return

    if equity_mtm <= maint_margin_total:
        send_telegram(
            f"[强平触发] equity_mtm={equity_mtm:.2f}, "
            f"maint_margin_total={maint_margin_total:.2f}, "
            f"positions={len(positions)}"
        )

        for pos in positions:
            last_price = price_map.get(pos.symbol, pos.entry_price)
            close_position(db, acc, pos, last_price, reason="liquidation")

        # 强平后再发一次整体账户快照
        send_account_snapshot(db, acc, prefix="[强平完成后账户]")


# ========== 仓位大小：ATR + 单笔风险 + 杠杆限制 ==========

def calc_position_size(equity: float, atr: float, price: float) -> float:
    """
    合约版仓位计算：
      - 单笔风险单元：equity * RISK_PER_TRADE_PCT
      - ATR 作为止损距离，raw_qty = risk_amount / atr
      - 杠杆限制：notional <= equity * MAX_LEVERAGE
    """
    if atr <= 0 or price <= 0 or equity <= 0:
        return 0.0

    risk_amount = equity * (config.RISK_PER_TRADE_PCT / 100.0)
    raw_qty = risk_amount / atr

    # 单笔仓位名义价值的最大上限（不超过账户可用杠杆）
    max_notional_by_lev = equity * config.MAX_LEVERAGE
    cap_qty_by_lev = max_notional_by_lev / price

    qty = min(raw_qty, cap_qty_by_lev)
    return max(qty, 0.0)


def get_total_notional(db, acc: Account) -> float:
    """当前所有持仓的总名义价值（按开仓价）"""
    positions = get_open_positions(db, acc.id)
    return sum(pos.entry_price * pos.size for pos in positions)


# ========== 主循环：每轮跑所有币种 + 杠杆检查 + 强平检查 ==========

def run_cycle_once():
    """
    每轮执行：
      - 先获取当前总名义仓位（用于账户杠杆限制）
      - 对每个 symbol：
          1) 拉历史K线
          2) 用策略生成信号
          3) 如果有持仓 -> 检查 TP/SL
          4) 如果空仓 -> 检查账户杠杆 -> 开新仓（含滑点 & 手续费）
             → 每次开仓后推送一条账户快照
      - 最后跑一次强平检查（equity_mtm vs 维持保证金）
    """
    db = SessionLocal()
    try:
        # 如果被 Telegram 指令暂停，则本轮什么都不做
        if not is_trading_enabled():
            return

        acc = get_account(db)
        if not acc:
            return

        params = PAStrategyParams()
        total_notional_existing = get_total_notional(db, acc)

        for symbol in config.SYMBOLS:
            try:
                df = fetch_ohlcv_df(symbol)
            except Exception as e:
                print(f"fetch_ohlcv error for {symbol}:", repr(e))
                continue

            if df is None or df.empty:
                continue

            state = PAStrategyState()
            sig = generate_signal(df, state, params)

            side = sig.get("side")
            atr = sig.get("atr")
            reason = sig.get("reason", "unknown")

            last = df.iloc[-1]
            last_price = float(last["close"])

            # 当前 symbol 是否已有持仓
            pos = (
                db.query(Position)
                .filter_by(symbol=symbol, account_id=acc.id, closed=0)
                .first()
            )

            # --- 1) 有持仓：检查 TP / SL ---
            if pos and pos.closed == 0:
                if pos.side == "long":
                    hit_sl = last_price <= pos.stop_loss
                    hit_tp = last_price >= pos.take_profit
                else:
                    hit_sl = last_price >= pos.stop_loss
                    hit_tp = last_price <= pos.take_profit

                if hit_sl or hit_tp:
                    close_position(db, acc, pos, last_price, reason="tp_sl_or_reverse")

                # 有持仓时暂时不反向开仓，避免过度复杂
                continue

            # --- 2) 无持仓：看是否开新仓 ---
            if side not in ("long", "short") or atr is None:
                continue

            # 基于当前账户权益按 ATR 计算目标仓位
            qty = calc_position_size(acc.equity, atr, last_price)
            if qty <= 0:
                continue

            # 用滑点计算预期开仓价格 + 名义价值
            slippage = config.SLIPPAGE_RATE
            if side == "long":
                exec_price = last_price * (1 + slippage)
            else:
                exec_price = last_price * (1 - slippage)

            new_notional = exec_price * qty

            # 帐户层面的总杠杆限制：
            # (已有总名义 + 新仓位名义) / 实现权益 <= MAX_LEVERAGE
            if acc.equity <= 0:
                continue

            projected_total_notional = total_notional_existing + new_notional
            projected_leverage = projected_total_notional / acc.equity

            if projected_leverage > config.MAX_LEVERAGE:
                # 杠杆上限超标，不开新仓
                send_telegram(
                    f"[拒绝开仓][杠杆过高] {symbol} 预期杠杆={projected_leverage:.2f} "
                    f"上限={config.MAX_LEVERAGE:.2f}"
                )
                continue

            # --- 3) 名义合理，正式建仓 ---
            notional_open = new_notional
            fee_open = notional_open * config.TAKER_FEE_RATE

            # 立即扣除开仓手续费
            acc.equity -= fee_open
            acc.cash -= fee_open

            # 止损止盈以开仓成交价为中心
            if side == "long":
                sl = exec_price - atr
                tp = exec_price + 2 * atr
            else:
                sl = exec_price + atr
                tp = exec_price - 2 * atr

            pos = Position(
                symbol=symbol,
                side=side,
                size=qty,
                entry_price=exec_price,
                atr=atr,
                stop_loss=sl,
                take_profit=tp,
                account_id=acc.id,
            )
            db.add(pos)

            total_notional_existing += notional_open  # 更新账户总名义

            emoji = "📈" if side == "long" else "📉"
            msg = (
                f"{emoji} 开仓：{symbol} {side.upper()}\n"
                f"数量：{qty:.4f}\n"
                f"价格：{exec_price:.2f} USDT\n"
                f"止损：{(exec_price - atr) if side == 'long' else (exec_price + atr):.2f}\n"
                f"止盈：{(exec_price + 2*atr) if side == 'long' else (exec_price - 2*atr):.2f}\n"
                f"ATR：{atr:.2f}  手续费：{fee_open:.2f}\n"
                f"信号来源：{reason}"
            )
            send_telegram(msg)


            # 开仓后发一条账户快照
            send_account_snapshot(db, acc, prefix=f"[开仓后账户] {symbol}")

        # --- 3) 本轮结束后做一次强平检查 ---
        check_and_liquidate(db, acc)

        db.commit()
    finally:
        db.close()
