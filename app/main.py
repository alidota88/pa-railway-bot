from fastapi import FastAPI
import asyncio
import datetime

from .trading_engine import (
    init_db_and_account,
    run_cycle_once,
    SessionLocal,
    get_account,
    compute_account_margin_and_unrealized,
    compute_position_margin_and_liq,
)

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/account_stats")
async def account_stats():
    """
    返回当前账户的整体保证金情况 + 持仓列表：
      - equity: 账面权益（已实现盈亏）
      - equity_mtm: 按最新价计的市值权益
      - used_margin: 已用初始保证金
      - maint_margin_total: 总维持保证金
      - free_margin: 可用保证金
      - account_leverage: 当前总杠杆
      - positions: 每个持仓的详细信息（方向/大小/爆仓价等）
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


@app.on_event("startup")
async def startup_event():
    # 初始化数据库和虚拟账户
    init_db_and_account()
    # 启动后台策略循环
    asyncio.create_task(worker_loop())


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
