"""
訂單 Agent - 網頁版 + LINE Bot

本機執行：uvicorn app:app --reload
網頁版打開 http://localhost:8000，LINE 的 Webhook 是 /callback
"""
import logging
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import agent
import line_bot
import sessions
import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("order-agent")

app = FastAPI(title="AI Order Agent")
app.include_router(line_bot.router)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=64)
    message: str = Field(min_length=1, max_length=2000)


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=64)


def _check_token(token):
    if settings.APP_ACCESS_TOKEN and token != settings.APP_ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="通關碼錯誤")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/api/chat")
def chat(request: ChatRequest, x_access_token: str = Header(default="")):
    _check_token(x_access_token)
    session, lock = sessions.get(request.session_id)

    # 同一段對話一次只處理一則，避免使用者連按兩次把歷史弄亂
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="上一則訊息還在處理中")
    try:
        result = agent.continue_turn(session, request.message.strip(), log=logger.info)
    except Exception:
        logger.exception("處理訊息失敗 session=%s", request.session_id)
        raise HTTPException(status_code=500, detail="系統忙碌中，請稍後再試")
    finally:
        lock.release()

    return {
        "reply": result.reply,
        "order_id": result.order_id,
        "order": result.order,
        "tools_used": result.tools_used,
    }


@app.post("/api/reset")
def reset(request: ResetRequest, x_access_token: str = Header(default="")):
    _check_token(x_access_token)
    sessions.reset(request.session_id)
    return {"ok": True}
