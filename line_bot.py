"""
LINE Bot

LINE 後台的 Webhook URL 設成 https://你的網址/callback

收到訊息先回 200 給 LINE，再到背景跑 agent，跑完用 reply token 回覆。
agent 一輪要好幾秒，等跑完才回 200 的話，LINE 會當成逾時。
"""
import base64
import hashlib
import hmac
import json
import logging

import httpx
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request

import agent
import sessions
import settings

logger = logging.getLogger("order-agent.line")
router = APIRouter()

_API = "https://api.line.me/v2/bot"
RESET_WORDS = {"重新開始", "重來", "reset"}


def _verify_signature(body, signature):
    digest = hmac.new(settings.LINE_CHANNEL_SECRET.encode(), body, hashlib.sha256).digest()
    return hmac.compare_digest(base64.b64encode(digest).decode(), signature)


def _post(path, payload):
    response = httpx.post(
        f"{_API}{path}",
        headers={"Authorization": f"Bearer {settings.LINE_CHANNEL_ACCESS_TOKEN}"},
        json=payload,
        timeout=10,
    )
    if response.status_code >= 400:
        logger.warning("LINE API %s 失敗：%s %s", path, response.status_code, response.text)
    return response


def _reply(reply_token, text):
    # LINE 單則訊息上限 5000 字
    _post("/message/reply", {"replyToken": reply_token, "messages": [{"type": "text", "text": text[:5000]}]})


def _show_loading(user_id):
    """在聊天室顯示「輸入中」動畫，只有一對一聊天有效"""
    _post("/chat/loading/start", {"chatId": user_id, "loadingSeconds": 60})


def _format_order(order_id, order):
    lines = [f"📋 目前訂單 {order_id}"]
    for label, key in [("客戶", "CustomerName"), ("收貨人", "ReceiverName"),
                       ("地址", "DeliveryAddress"), ("到貨日", "DeliveryDate"), ("備註", "Remarks")]:
        if order.get(key):
            lines.append(f"{label}：{order[key]}")
    for item in order["Items"]:
        lines.append(f"・{item['ProductName']}（{item['ProductNo']}）× {item['Quantity']} {item['Unit']}".rstrip())
    return "\n".join(lines)


def _handle_text(event):
    source = event["source"]
    chat_id = source.get("groupId") or source.get("roomId") or source.get("userId")
    session_id = f"line-{chat_id}"
    text = event["message"]["text"].strip()

    if text in RESET_WORDS:
        sessions.reset(session_id)
        _reply(event["replyToken"], "好的，已經重新開始，請告訴我要訂什麼。")
        return

    if source.get("type") == "user":
        _show_loading(source["userId"])

    session, lock = sessions.get(session_id)
    # 同一個聊天室的訊息排隊處理，避免兩則同時改到對話歷史
    with lock:
        try:
            result = agent.continue_turn(session, text, log=logger.info)
            reply = result.reply
            if "create_order" in result.tools_used and result.order:
                reply += "\n\n" + _format_order(result.order_id, result.order)
        except Exception:
            logger.exception("處理 LINE 訊息失敗 session=%s", session_id)
            reply = "（系統忙碌中，請稍後再試）"

    _reply(event["replyToken"], reply)


@router.post("/callback")
async def callback(request: Request, background: BackgroundTasks, x_line_signature: str = Header(default="")):
    if not settings.LINE_CHANNEL_SECRET or not settings.LINE_CHANNEL_ACCESS_TOKEN:
        raise HTTPException(status_code=503, detail="LINE 尚未設定")

    body = await request.body()
    if not _verify_signature(body, x_line_signature):
        raise HTTPException(status_code=400, detail="簽章錯誤")

    for event in json.loads(body).get("events", []):
        if event.get("type") != "message":
            continue
        if event["message"].get("type") == "text":
            background.add_task(_handle_text, event)
        else:
            background.add_task(_reply, event["replyToken"], "目前只看得懂文字訊息喔")

    return {"ok": True}
