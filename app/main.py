from fastapi import FastAPI
import asyncio
import datetime
import os

from trading_engine import (
    init_db_and_account,
    run_cycle_once,
    SessionLocal,
    get_account,
    compute_account_margin_and_unrealized,
    compute_position_margin_and_liq,
)
from . import config
from .telegram_bot import telegram_command_loop


app = FastAPI()


# 从环境变量读取（避免多个实例重复启动 Telegram 循环）
PROCESS_ROLE = os.getenv("PROCESS_ROLE", "web")       # web / worker
TELEGRAM_LOOP_ENABLED = os.getenv("TELEGRAM_LOOP_ENABLED", "1") == "1"


@app.on_event("startup")
async def startup_event():
    """系统启动时初始化"""
    init_db_and_account()

    # ✅ 只在 web 进程 并且允许时启动 Telegram 命令循环
    if PROCESS_ROLE == "web" and TELEGRAM_LOOP_ENABLED:
        asyncio.create_task(telegram_command_loop())
        print("✅ Telegram 命令循环已启动（web 实例）")
    else:
        print("🚫 当前实例未启用 Telegram 命令循环")

    # ✅ 启动策略循环（只在 web 启动，worker 专注行情）
    if PROCESS_ROLE == "web":
        asyncio.create_task(worker_loop())


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/account_stats")
async def account_stats():
    """
    返回当前账户的整体保证金情况 + 持仓列表。
    """
    db = SessionLocal()
    try:
        acc = get_account(db)
        if not acc:
            return {"error": "account_not_found"}

        stats, price_map, positions = compute_account_margin_and_unrealized(db, acc)

        pos_list = []
        for pos in positions:
            notional, im, mm, liq_price = compute_position_margin_and_liq(pos)
            last_price = price_map.get(pos.symbol, pos.entry_price)
            pos_list.append(
                {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "last_price": last_price,
                    "notional": notional,
                    "initial_margin": im,
                    "maintenance_margin": mm,
                    "liq_price": liq_price,
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
                    "opened_at": pos.opened_at,
                }
            )

        return {
            "equity": acc.equity,
            "cash": acc.cash,
            "equity_mtm": stats["equity_mtm"],
            "used_margin": stats["used_margin"],
            "maint_margin_total": stats["maint_margin_total"],
            "free_margin": stats["free_margin"],
            "total_notional": stats["total_notional"],
            "account_leverage": stats["account_leverage"],
            "unrealized_pnl": stats["total_unrealized"],
            "positions": pos_list,
        }
    finally:
        db.close()


async def worker_loop():
    """
    后台循环：
      - 每隔一段时间打印一次心跳日志（方便看 Railway 日志）
      - 调用 run_cycle_once() 跑一轮策略（所有币种）
    """
    while True:
        now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now} UTC] Worker loop running...")

        try:
            run_cycle_once()
        except Exception as e:
            print("Worker error:", repr(e))

        # 运行频率：目前每 60 秒一轮
        await asyncio.sleep(60)
