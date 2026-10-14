"""
訂單 Agent - 網頁版

本機執行：uvicorn app:app --reload
然後打開 http://localhost:8000
"""
import logging
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import agent
import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("order-agent")

app = FastAPI(title="AI Order Agent")

# 對話狀態與訂單都放記憶體，程式重開就沒了
_sessions = {}
_locks = {}
_sessions_lock = threading.Lock()


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=64)
    message: str = Field(min_length=1, max_length=2000)


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=64)


def _check_token(token):
    if settings.APP_ACCESS_TOKEN and token != settings.APP_ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="通關碼錯誤")


def _get_session(session_id):
    with _sessions_lock:
        _evict_expired()
        if session_id not in _sessions:
            _sessions[session_id] = agent.OrderSession(session_id=session_id)
            _locks[session_id] = threading.Lock()
        return _sessions[session_id], _locks[session_id]


def _evict_expired():
    expire_before = time.time() - settings.SESSION_TTL_MINUTES * 60
    for sid in [sid for sid, s in _sessions.items() if s.last_active < expire_before]:
        _sessions.pop(sid, None)
        _locks.pop(sid, None)


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/api/chat")
def chat(request: ChatRequest, x_access_token: str = Header(default="")):
    _check_token(x_access_token)
    session, lock = _get_session(request.session_id)

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
    with _sessions_lock:
        _sessions.pop(request.session_id, None)
        _locks.pop(request.session_id, None)
    return {"ok": True}
