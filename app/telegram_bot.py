import asyncio
import httpx

from . import config
from .telegram_client import send_telegram_async
from .trading_engine import (
    SessionLocal,
    get_account,
    compute_account_margin_and_unrealized,
    compute_position_margin_and_liq,
    get_open_positions,
    set_trading_enabled,
    is_trading_enabled,
)


async def handle_command(text: str):
    """
    根据收到的文本命令，执行相应操作：
      /help           查看帮助
      /start_trading  开启自动交易
      /stop_trading   暂停自动交易
      /stats          查看账户整体情况
      /positions      查看当前持仓列表
    """
    cmd = (text or "").strip().lower()

    if cmd in ("/start", "/help"):
        msg = (
            "命令列表：\n"
            "/start_trading - 开启自动交易\n"
            "/stop_trading  - 暂停自动交易\n"
            "/stats         - 查看账户资金 & 杠杆情况\n"
            "/positions     - 查看当前持仓\n"
        )
        await send_telegram_async(msg)
        return

    if cmd == "/start_trading":
        set_trading_enabled(True)
        await send_telegram_async("✅ 已开启自动交易。")
        return

    if cmd == "/stop_trading":
        set_trading_enabled(False)
        await send_telegram_async("⏸ 已暂停自动交易（不再开新仓，已有持仓仍会走到平仓逻辑）。")
        return

    if cmd == "/stats":
        db = SessionLocal()
        try:
            acc = get_account(db)
            if not acc:
                await send_telegram_async("账户不存在。")
                return

            stats, _, positions = compute_account_margin_and_unrealized(db, acc)
            msg = (
                "📊 账户状态\n"
                f"交易开关: {'ON' if is_trading_enabled() else 'OFF'}\n"
                f"Equity(已实现): {acc.equity:.2f}\n"
                f"Equity(MtM): {stats['equity_mtm']:.2f}\n"
                f"总名义仓位: {stats['total_notional']:.2f}\n"
                f"已用保证金(IM): {stats['used_margin']:.2f}\n"
                f"维持保证金(MM): {stats['maint_margin_total']:.2f}\n"
                f"可用保证金: {stats['free_margin']:.2f}\n"
                f"当前杠杆: {stats['account_leverage']:.2f}x\n"
                f"未实现PnL: {stats['total_unrealized']:.2f}\n"
                f"持仓数: {len(positions)}"
            )
            await send_telegram_async(msg)
        finally:
            db.close()
        return

    if cmd == "/positions":
        db = SessionLocal()
        try:
            acc = get_account(db)
            if not acc:
                await send_telegram_async("账户不存在。")
                return

            positions = get_open_positions(db, acc.id)
            if not positions:
                await send_telegram_async("当前无持仓。")
                return

            lines = ["📌 当前持仓："]
            for pos in positions:
                notional, im, mm, liq = compute_position_margin_and_liq(pos)
                lines.append(
                    f"{pos.symbol} {pos.side.upper()} size={pos.size:.4f}\n"
                    f"  入场={pos.entry_price:.2f} 名义={notional:.2f}\n"
                    f"  IM={im:.2f} MM={mm:.2f} 爆仓价≈{liq:.2f}\n"
                    f"  SL={pos.stop_loss:.2f} TP={pos.take_profit:.2f}"
                )
            await send_telegram_async("\n".join(lines))
        finally:
            db.close()
        return

    # 未知命令
    await send_telegram_async("未知命令，发送 /help 查看支持的命令。")


async def telegram_command_loop():
    """
    长轮询 Telegram getUpdates，监听指令。
    """
    token = config.TELEGRAM_TOKEN
    if not token:
        print("TELEGRAM_TOKEN 未设置，命令循环不启动。")
        return

    base_url = f"https://api.telegram.org/bot{token}"
    target_chat_id = str(config.TELEGRAM_CHAT_ID) if config.TELEGRAM_CHAT_ID else None
    offset = None

    print("Telegram 命令循环启动中...")

    while True:
        try:
            params = {"timeout": 30}
            if offset is not None:
                params["offset"] = offset

            async with httpx.AsyncClient(timeout=40.0) as client:
                resp = await client.get(f"{base_url}/getUpdates", params=params)
                data = resp.json()

            for update in data.get("result", []):
                offset = update["update_id"] + 1

                message = update.get("message") or update.get("edited_message")
                if not message:
                    continue

                chat_id = str(message["chat"]["id"])
                text = message.get("text") or ""

                # 如果设置了 TELEGRAM_CHAT_ID，就只响应这个ID的消息
                if target_chat_id and chat_id != target_chat_id:
                    continue

                # 处理命令
                await handle_command(text)

        except Exception as e:
            print("telegram_command_loop error:", repr(e))

        # 避免狂刷接口，稍微休息一下
        await asyncio.sleep(2)
