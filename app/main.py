from fastapi import FastAPI
import asyncio

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
    # 每 60 秒运行一次策略
    while True:
        try:
            run_cycle_once()
        except Exception as e:
            print("Worker error:", e)
        await asyncio.sleep(60)
