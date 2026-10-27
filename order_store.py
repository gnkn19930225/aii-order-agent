"""
訂單儲存（MongoDB）

一段對話只有一張訂單草稿。create_order 每次都整張取代同一筆，取消只改狀態不刪除。
config.py 沒設 ORDER_MONGO_URI 時不寫資料庫，訂單只留在對話狀態裡。
"""
from datetime import datetime, timezone

from bson import ObjectId
from pymongo import MongoClient

import settings

_collection = None


def enabled():
    return bool(settings.ORDER_MONGO_URI)


def _orders():
    global _collection
    if _collection is None:
        client = MongoClient(settings.ORDER_MONGO_URI, serverSelectionTimeoutMS=5000)
        _collection = client[settings.ORDER_DB][settings.ORDER_COLLECTION]
    return _collection


def save_draft(session_id, order_id, order):
    """有草稿就整張取代，沒有就新增。回傳訂單編號"""
    now = datetime.now(timezone.utc)
    document = {**order, "SessionId": session_id, "Status": "Pending", "LastModifiedAt": now}

    if order_id:
        result = _orders().update_one({"_id": ObjectId(order_id), "Status": "Pending"}, {"$set": document})
        if result.matched_count:
            return order_id

    document["CreatedAt"] = now
    return str(_orders().insert_one(document).inserted_id)


def cancel(order_id):
    """取消訂單。只改狀態，事後還查得到"""
    result = _orders().update_one(
        {"_id": ObjectId(order_id), "Status": "Pending"},
        {"$set": {"Status": "Cancelled", "LastModifiedAt": datetime.now(timezone.utc)}},
    )
    return result.matched_count > 0
