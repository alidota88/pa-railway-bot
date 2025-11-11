import httpx
from . import config

async def send_telegram_async(text: str):
    if not config.TELEGRAM_TOKEN or not config.TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text})

def send_telegram(text: str):
    # 同步封装，方便在非 async 环境使用
    import anyio
    anyio.run(send_telegram_async, text)
