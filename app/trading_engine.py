from datetime import datetime
from typing import Optional

import ccxt
import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, DateTime
)
from sqlalchemy.orm import sessionmaker, declarative_base

from . import config
from .telegram_client import send_telegram
from .strategy_pa import PAStrategyState, generate_signal

# --- DB 初始化 ---
if not config.DATABASE_URL:
    # 如果没配 DATABASE_URL，会报错，提醒你在 Railway 配置
    raise RuntimeError("DATABASE_URL is not set. Please configure it in Railway env vars.")

engine = create_engine(config.DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

# --- 模型定义 ---
class Account(Base):
    __tablename__ = "accounts"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    equity = Column(Float, default=0)  # 总权益
    cash = Column(Float, default=0)    # 现金（简单模型）
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
    closed = Column(Integer, default=0)
    account_id = Column(Integer)

class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String)
    size = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl = Column(Float)
    opened_at = Column(DateTime)
    closed_at = Column(DateTime)
    reason = Column(String)
    account_id = Column(Integer)

def init_db_and_account():
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

# --- ccxt 交易所 ---
exchange = getattr(ccxt, config.EXCHANGE_ID)()

def fetch_ohlcv_df(symbol: str, limit: int = 200) -> pd.DataFrame:
    """从交易所拉 OHLCV，并转成 DataFrame"""
    ohlcv = exchange.fetch_ohlcv(symbol, config.TIMEFRAME, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df

def get_account(db) -> Account:
    return db.query(Account).filter_by(name="paper").first()

def get_open_position(db, symbol: str, acc_id: int) -> Optional[Position]:
    return (
        db.query(Position)
        .filter_by(symbol=symbol, account_id=acc_id, closed=0)
        .first()
    )

def close_position(db, acc: Account, pos: Position, last_price: float):
    """按当前价格平仓，记录 Trade，更新权益"""
    if pos.side == "long":
        pnl = (last_price - pos.entry_price) * pos.size
    else:
        pnl = (pos.entry_price - last_price) * pos.size

    acc.equity += pnl
    acc.cash += pnl  # 简化：假设保证金模式

    trade = Trade(
        symbol=pos.symbol,
        side=pos.side,
        size=pos.size,
        entry_price=pos.entry_price,
        exit_price=last_price,
        pnl=pnl,
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
        f"[平仓][虚拟盘] {pos.symbol} {pos.side.upper()} size={pos.size:.4f} "
        f"入场={pos.entry_price:.2f} 平仓={last_price:.2f} PnL={pnl:.2f}"
    )

def calc_position_size(equity: float, atr: float, price: float) -> float:
    """
    合约版仓位计算：
      - 单笔风险：equity * RISK_PER_TRADE_PCT
      - ATR 作为止损距离，raw_qty = risk_amount / atr
      - 杠杆限制：名义价值 notional = price * qty <= equity * MAX_LEVERAGE
    """
    if atr <= 0 or price <= 0 or equity <= 0:
        return 0.0

    # 单笔风险金额
    risk_amount = equity * (config.RISK_PER_TRADE_PCT / 100.0)
    raw_qty = risk_amount / atr

    # 杠杆限制：名义仓位上限
    max_notional_by_lev = equity * config.MAX_LEVERAGE
    cap_qty_by_lev = max_notional_by_lev / price

    qty = min(raw_qty, cap_qty_by_lev)
    return max(qty, 0.0)


def run_cycle_once():
    """核心循环：对每个币种 拉数据 → 算信号 → 管理仓位"""
    db = SessionLocal()
    try:
        acc = get_account(db)
        if not acc:
            return

        state = PAStrategyState()

        for symbol in config.SYMBOLS:
            try:
                df = fetch_ohlcv_df(symbol)
            except Exception as e:
                print(f"fetch_ohlcv error for {symbol}:", e)
                continue

            if df is None or df.empty:
                continue

            sig = generate_signal(df, state)
            side = sig["side"]
            atr = sig["atr"]
            last = df.iloc[-1]
            last_price = float(last["close"])

            pos = get_open_position(db, symbol, acc.id)

            # 1) 有持仓先管理止盈/止损
            if pos and pos.closed == 0:
                # 简单止盈止损：用入场时的 ATR
                if pos.side == "long":
                    hit_sl = last_price <= pos.stop_loss
                    hit_tp = last_price >= pos.take_profit
                else:
                    hit_sl = last_price >= pos.stop_loss
                    hit_tp = last_price <= pos.take_profit

                if hit_sl or hit_tp:
                    close_position(db, acc, pos, last_price)

                # 如果当前有仓位，就不反向开新仓（保持简单）
                continue

            # 2) 没有持仓，看是否开新仓
            if side in ("long", "short") and atr is not None:
                qty = calc_position_size(acc.equity, atr, last_price)
                if qty <= 0:
                    continue

                # === 1. 计算有滑点的成交价 ===
                slippage = config.SLIPPAGE_RATE
                if side == "long":
                    # 做多开仓：买入，价格稍微高一点
                    exec_price = last_price * (1 + slippage)
                else:
                    # 做空开仓：先卖出，价格略高（对我们更有利）
                    exec_price = last_price * (1 - slippage)

                # === 2. 名义价值 & 开仓手续费 ===
                notional_open = exec_price * qty
                fee_open = notional_open * config.TAKER_FEE_RATE

                # 手续费立刻从权益 / 现金中扣除（更接近真实）
                acc.equity -= fee_open
                acc.cash -= fee_open

                # === 3. 设置止损止盈（用成交价 exec_price 为中心更合理） ===
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
                    f"价格={exec_price:.2f} ATR={atr:.2f} 手续费={fee_open:.4f}"
                )

        db.commit()
    finally:
        db.close()
