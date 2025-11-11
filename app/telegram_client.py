import httpx
import asyncio
from . import config

async def send_telegram_async(text: str):
    """真正发消息的异步函数"""
    if not config.TELEGRAM_TOKEN or not config.TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(
            url,
            json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text},
            timeout=10.0,
        )

def send_telegram(text: str):
    """
    同步包装：
    - 如果当前没有事件循环（本地脚本模式），就直接 asyncio.run(...)
    - 如果当前已经有事件循环（FastAPI / Uvicorn 里），就用 loop.create_task(...) 挂后台任务
    """
    try:
        # 如果当前线程已经有 running loop，会返回，不会抛异常
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 没有 running loop：例如在普通脚本里直接调用
        asyncio.run(send_telegram_async(text))
    else:
        # 在 FastAPI / Uvicorn 的事件循环里：只能用 create_task 挂一个协程
        loop.create_task(send_telegram_async(text))
