from fastapi import FastAPI
import asyncio
import datetime

from .trading_engine import init_db_and_account, run_cycle_once

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.on_event("startup")
async def startup_event():
    # 初始化数据库和虚拟账户
    init_db_and_account()
    # 启动后台策略循环
    asyncio.create_task(worker_loop())


async def worker_loop():
    """
    后台循环：
    - 每隔一段时间打印一次心跳日志
    - 调用 run_cycle_once() 跑一轮策略（所有币种）
    """
    while True:
        now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now} UTC] Worker loop running...")

        try:
            run_cycle_once()
        except Exception as e:
            # 把异常打印到 Railway 日志里，方便排查
            print("Worker error:", repr(e))

        # 调整运行频率：调试时可以改成 30，稳定后可以改为 60 或更长
        await asyncio.sleep(60)
