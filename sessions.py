"""
對話狀態

網頁版與 LINE 共用。全部放記憶體，程式重開就沒了。
"""
import threading
import time

import agent
import settings

_sessions = {}
_locks = {}
_sessions_lock = threading.Lock()


def get(session_id):
    """取得對話與它的鎖，沒有就開一段新的"""
    with _sessions_lock:
        _evict_expired()
        if session_id not in _sessions:
            _sessions[session_id] = agent.OrderSession(session_id=session_id)
            _locks[session_id] = threading.Lock()
        return _sessions[session_id], _locks[session_id]


def reset(session_id):
    with _sessions_lock:
        _sessions.pop(session_id, None)
        _locks.pop(session_id, None)


def _evict_expired():
    expire_before = time.time() - settings.SESSION_TTL_MINUTES * 60
    for sid in [sid for sid, s in _sessions.items() if s.last_active < expire_before]:
        _sessions.pop(sid, None)
        _locks.pop(sid, None)
