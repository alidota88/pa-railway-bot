from __future__ import annotations

from datetime import datetime
from typing import Optional

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


# ================== 数据库初始化 ==================

if not config.DATABASE_URL:
    # 如果没配置 DATABASE_URL，直接报错，避免悄悄用空连接
    raise RuntimeError("DATABASE_URL is not set. Please configure it in Railway env vars.")

engine = create_engine(config.DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


# ================== ORM 模型 ==================

class Account(Base):
    __tablename__ = "accounts"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    equity = Column(Float, default=0)  # 当前总权益
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
    pnl = Column(Float)                # 净利润（已扣除手续费）
    opened_at = Column(DateTime)
    closed_at = Column(DateTime)
    reason = Column(String)            # 平仓原因（tp/sl/反向/手动等）
    account_id = Column(Integer)


def init_db_and_account():
    """建表 + 初始化一个虚拟账户（paper）"""
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


# ================== 交易所（行情） ==================

exchange = getattr(ccxt, config.EXCHANGE_ID)()


def fetch_ohlcv_df(symbol: str, limit: int = 200) -> pd.DataFrame:
    """
    从交易所拉 OHLCV，转为 DataFrame：
      ts, open, high, low, close, volume
    """
    ohlcv = exchange.fetch_ohlcv(symbol, config.TIMEFRAME, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df


# ================== 帐户 / 持仓工具函数 ==================

def get_account(db) -> Account:
    return db.query(Account).filter_by(name="paper").first()


def get_open_position(db, symbol: str, acc_id: int) -> Optional[Position]:
    return (
        db.query(Position)
        .filter_by(symbol=symbol, account_id=acc_id, closed=0)
        .first()
    )


# ================== 平仓（包含滑点 & 手续费） ==================

def close_position(db, acc: Account, pos: Position, last_price: float):
    """
    平仓时：
      - 使用滑点计算实际成交价格 exec_price_close
      - 计算毛利润 pnl_gross
      - 计算平仓手续费 fee_close
      - 净利润 pnl_net = pnl_gross - fee_close，计入 equity / cash
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
        reason="tp/sl_or_reverse",
    )
    pos.closed = 1
    db.add(trade)
    db.add(acc)
    db.add(pos)

    send_telegram(
        f"[平仓][虚拟盘] {pos.symbol} {pos.side.upper()} "
        f"size={pos.size:.4f} 入场={pos.entry_price:.2f} "
        f"平仓={exec_price_close:.2f} 净PnL={pnl_net:.2f} "
        f"(含手续费 {fee_close:.4f})"
    )


# ================== 仓位计算（合约 + 杠杆 + ATR 风险） ==================

def calc_position_size(equity: float, atr: float, price: float) -> float:
    """
    合约版仓位计算：
      - 单笔风险：equity * RISK_PER_TRADE_PCT
      - ATR 作为止损距离，raw_qty = risk_amount / atr
      - 杠杆限制：notional = price * qty <= equity * MAX_LEVERAGE
    """
    if atr <= 0 or price <= 0 or equity <= 0:
        return 0.0

    # 单笔风险金额（1R）
    risk_amount = equity * (config.RISK_PER_TRADE_PCT / 100.0)
    raw_qty = risk_amount / atr

    # 杠杆限制：单笔名义价值不能超过 equity * MAX_LEVERAGE
    max_notional_by_lev = equity * config.MAX_LEVERAGE
    cap_qty_by_lev = max_notional_by_lev / price

    qty = min(raw_qty, cap_qty_by_lev)
    return max(qty, 0.0)


# ================== 核心循环：每个周期跑一遍所有币种 ==================

def run_cycle_once():
    """
    核心执行函数：
      - 对每个 symbol：
          1) 拉最近一段 K 线
          2) 用策略生成信号（不改策略逻辑）
          3) 检查是否已有持仓 -> 判断 SL/TP 是否触发 -> 平仓
          4) 如空仓且有新信号 -> 计算仓位（含杠杆）-> 加滑点 / 手续费 -> 开仓
      - 所有 Account / Position / Trade 变化写入数据库
    """
    db = SessionLocal()
    try:
        acc = get_account(db)
        if not acc:
            return

        # 策略参数（目前用默认，你后续可以在这里改 IBS/MIG 等）
        params = PAStrategyParams()

        for symbol in config.SYMBOLS:
            try:
                df = fetch_ohlcv_df(symbol)
            except Exception as e:
                print(f"fetch_ohlcv error for {symbol}:", repr(e))
                continue

            if df is None or df.empty:
                continue

            # 策略状态（你后续如果需要在 symbol 之间共享趋势状态，可以提升到循环外）
            state = PAStrategyState()
            sig = generate_signal(df, state, params)

            side = sig.get("side")
            atr = sig.get("atr")
            reason = sig.get("reason", "unknown")
            last = df.iloc[-1]
            last_price = float(last["close"])

            pos = get_open_position(db, symbol, acc.id)

            # 1) 有持仓 -> 判断是否触发 SL / TP
            if pos and pos.closed == 0:
                if pos.side == "long":
                    hit_sl = last_price <= pos.stop_loss
                    hit_tp = last_price >= pos.take_profit
                else:
                    hit_sl = last_price >= pos.stop_loss
                    hit_tp = last_price <= pos.take_profit

                if hit_sl or hit_tp:
                    close_position(db, acc, pos, last_price)

                # 有持仓时暂时不考虑反向开新仓，保持模型简单
                continue

            # 2) 无持仓 -> 看是否开新仓
            if side not in ("long", "short") or atr is None:
                continue

            qty = calc_position_size(acc.equity, atr, last_price)
            if qty <= 0:
                continue

            # === 1. 成交价加入滑点 ===
            slippage = config.SLIPPAGE_RATE
            if side == "long":
                # 做多开仓：买得稍微贵一点
                exec_price = last_price * (1 + slippage)
            else:
                # 做空开仓：先卖出，价格略高一点
                exec_price = last_price * (1 - slippage)

            # === 2. 名义价值 & 开仓手续费 ===
            notional_open = exec_price * qty
            fee_open = notional_open * config.TAKER_FEE_RATE

            # 手续费立刻从权益 / 现金中扣除
            acc.equity -= fee_open
            acc.cash -= fee_open

            # === 3. 止损止盈（围绕 exec_price） ===
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

            send_telegram(
                f"[开仓][虚拟盘] {symbol} {side.upper()} size={qty:.4f} "
                f"价格={exec_price:.2f} ATR={atr:.2f} "
                f"原因={reason} 手续费={fee_open:.4f}"
            )

        db.commit()
    finally:
        db.close()
