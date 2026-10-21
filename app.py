"""
訂單 Agent - 網頁版 + LINE Bot

本機執行：uvicorn app:app --reload
網頁版打開 http://localhost:8000，商品總覽在 /products，LINE 的 Webhook 是 /callback
"""
import logging
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import agent
import catalog
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


@app.get("/products")
def products_page():
    return FileResponse(Path(__file__).parent / "static" / "products.html")


# 不能用 /healthz：Cloud Run 保留了結尾是 z 的網址，會直接回 404
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/products")
def list_products(x_access_token: str = Header(default="")):
    _check_token(x_access_token)
    return {"products": catalog.all_products()}


@app.get("/api/search")
def search_products(q: str = Query(min_length=1, max_length=100), x_access_token: str = Header(default="")):
    """
    跟 agent 的 search_products 用同一套規則，但把門檻以下的結果也回傳，
    demo 時看得到「為什麼這筆被擋掉」。
    """
    _check_token(x_access_token)
    q = q.strip()
    exact = catalog.get_by_code(q)
    if exact or catalog.looks_like_code(q):
        return {"mode": "code", "threshold": settings.MIN_SIMILARITY,
                "results": [{**p, "similarity": 1.0, "passed": True} for p in exact]}
    return {
        "mode": "vector",
        "threshold": settings.MIN_SIMILARITY,
        "results": [{**p, "passed": p["similarity"] >= settings.MIN_SIMILARITY}
                    for p in catalog.vector_search(q, top_k=10)],
    }


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
