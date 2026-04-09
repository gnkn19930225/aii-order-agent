"""
訂單 Agent - 使用 Gemini Function Calling 進行智能打單
"""
import json
import time
import chromadb
import ollama
from google import genai
from google.genai import types, errors
from pymongo import MongoClient
from datetime import datetime, timedelta
from config import (
    GEMINI_API_KEY, GEMINI_MODEL, EMBEDDING_HOST, EMBEDDING_MODEL,
    TEST_MONGO_URI, TEST_DB, TEST_COLLECTION
)

# ============================================================
# 初始化
# ============================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
embedding_client = ollama.Client(host=EMBEDDING_HOST)
chroma = chromadb.PersistentClient(path="./vectordb")
collection = chroma.get_collection("products")

# ============================================================
# Gemini 工具定義
# ============================================================

def _s(type_str, description="", properties=None, required=None, items=None):
    """快速建立 Gemini Schema"""
    kw = {"type": type_str.upper(), "description": description}
    if properties:
        kw["properties"] = properties
    if required:
        kw["required"] = required
    if items:
        kw["items"] = items
    return types.Schema(**kw)


_item_schema = _s("object", properties={
    "product_ref":  _s("string", "客戶說的產品名稱（原始輸入）"),
    "product_no":   _s("string", "產品編號（從 search_products 結果的 ProductNo 取得，必填）"),
    "product_name": _s("string", "正式產品名稱（從 search_products 結果的 ProductName 取得，必填）"),
    "quantity":     _s("number", "數量"),
    "unit":         _s("string", "單位（包/箱/瓶/桶等）"),
}, required=["product_no", "product_name", "quantity"])

GEMINI_TOOLS = [types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="search_products",
        description="搜尋產品。當需要查詢產品資訊或確認產品是否存在時呼叫。",
        parameters=_s("object", properties={"query": _s("string", "搜尋關鍵字")}, required=["query"])
    ),
    types.FunctionDeclaration(
        name="create_order",
        description="建立訂單。當客戶提供了完整的訂單資訊（至少包含產品和數量）時呼叫此函數。",
        parameters=_s("object", properties={
            "customer_name":    _s("string", "客戶名稱"),
            "receiver_name":    _s("string", "收貨人姓名"),
            "delivery_address": _s("string", "送貨地址"),
            "delivery_date":    _s("string", "送貨日期，格式 YYYY/MM/DD，如果沒指定就填明天"),
            "items":            _s("array",  "訂單項目清單", items=_item_schema),
            "remarks":          _s("string", "備註"),
        }, required=["items"])
    ),
    types.FunctionDeclaration(
        name="ask_clarification",
        description="向客戶詢問更多資訊。當訂單資訊不完整或有多個選項需要確認時呼叫。",
        parameters=_s("object", properties={
            "question": _s("string", "要問客戶的問題"),
            "options":  _s("array",  "提供給客戶的選項（如果有的話）", items=_s("string")),
        }, required=["question"])
    ),
    types.FunctionDeclaration(
        name="update_order",
        description="修改現有訂單。當客戶要更換產品、修改數量、更改地址或日期時呼叫。",
        parameters=_s("object", properties={
            "order_id": _s("string", "要修改的訂單編號"),
            "updates":  _s("object", "要更新的欄位", properties={
                "customer_name":    _s("string"),
                "receiver_name":    _s("string"),
                "delivery_address": _s("string"),
                "delivery_date":    _s("string"),
                "remarks":          _s("string"),
                "items":            _s("array", "完整的訂單項目清單（會整個取代）", items=_item_schema),
            }),
        }, required=["order_id", "updates"])
    ),
    types.FunctionDeclaration(
        name="delete_order",
        description="刪除訂單。當客戶要取消訂單時呼叫。",
        parameters=_s("object", properties={"order_id": _s("string", "要刪除的訂單編號")}, required=["order_id"])
    ),
])]

# ============================================================
# 工具實作
# ============================================================

def get_embedding(text):
    """將文字轉成向量"""
    response = embedding_client.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text
    )
    return response['embedding']


def tool_search_products(query, top_k=5):
    """搜尋產品"""
    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )

    products = []
    for i in range(len(results['ids'][0])):
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        products.append({
            "ProductNo": metadata.get("ProductNo", ""),
            "ProductName": metadata.get("ProductName", ""),
            "ProductCategory": metadata.get("ProductCategory", ""),
            "ProductBrand": metadata.get("ProductBrand", ""),
            "similarity": round(1 - distance, 4)
        })

    return products


def lookup_product_name(product_no):
    """根據產品編號查詢產品名稱"""
    from config import MONGO_URI, MONGO_DB, MONGO_COLLECTION
    mongo = MongoClient(MONGO_URI)
    db = mongo[MONGO_DB]
    products = db[MONGO_COLLECTION]

    product = products.find_one({"ProductNo": product_no, "IsDeleted": False})
    mongo.close()

    if product:
        return product.get("ProductName", "")
    return ""


def tool_create_order(order_data):
    """建立訂單到測試 DB"""
    mongo = MongoClient(TEST_MONGO_URI)
    db = mongo[TEST_DB]
    orders = db[TEST_COLLECTION]

    # 處理 Items，確保有完整資訊
    items = []
    for item in order_data.get("items", []):
        product_no = item.get("product_no", "")
        product_name = item.get("product_name", "")

        # 如果沒有 product_name，嘗試用 product_no 查詢
        if not product_name and product_no:
            product_name = lookup_product_name(product_no)

        items.append({
            "ProductRef": item.get("product_ref", ""),      # 客戶說的
            "ProductNo": product_no,                         # 產品編號
            "ProductName": product_name,                     # 正式品名
            "Quantity": item.get("quantity", 0),
            "Unit": item.get("unit", "")
        })

    # 組裝訂單
    order = {
        "CustomerName": order_data.get("customer_name", ""),
        "ReceiverName": order_data.get("receiver_name", ""),
        "DeliveryAddress": order_data.get("delivery_address", ""),
        "DeliveryDate": order_data.get("delivery_date", get_tomorrow()),
        "Items": items,
        "Remarks": order_data.get("remarks", ""),
        "Status": "Pending",
        "CreatedAt": datetime.utcnow(),
        "CreatedBy": "OrderAgent"
    }

    result = orders.insert_one(order)
    mongo.close()

    return {
        "success": True,
        "order_id": str(result.inserted_id),
        "message": "訂單建立成功",
        "order_summary": {
            "items_count": len(items),
            "items": [{"ProductNo": i["ProductNo"], "ProductName": i["ProductName"], "Quantity": i["Quantity"]} for i in items]
        }
    }


def tool_ask_clarification(question, options=None):
    """產生詢問回應"""
    return {
        "question": question,
        "options": options or []
    }


def tool_update_order(order_id, updates):
    """修改訂單"""
    from bson import ObjectId

    mongo = MongoClient(TEST_MONGO_URI)
    db = mongo[TEST_DB]
    orders = db[TEST_COLLECTION]

    # 檢查訂單是否存在
    existing = orders.find_one({"_id": ObjectId(order_id)})
    if not existing:
        mongo.close()
        return {"success": False, "message": f"訂單 {order_id} 不存在"}

    # 處理 items（如果有更新的話）
    update_fields = {}

    if "customer_name" in updates:
        update_fields["CustomerName"] = updates["customer_name"]
    if "receiver_name" in updates:
        update_fields["ReceiverName"] = updates["receiver_name"]
    if "delivery_address" in updates:
        update_fields["DeliveryAddress"] = updates["delivery_address"]
    if "delivery_date" in updates:
        update_fields["DeliveryDate"] = updates["delivery_date"]
    if "remarks" in updates:
        update_fields["Remarks"] = updates["remarks"]

    if "items" in updates:
        items = []
        for item in updates["items"]:
            product_no = item.get("product_no", "")
            product_name = item.get("product_name", "")
            if not product_name and product_no:
                product_name = lookup_product_name(product_no)

            items.append({
                "ProductRef": item.get("product_ref", ""),
                "ProductNo": product_no,
                "ProductName": product_name,
                "Quantity": item.get("quantity", 0),
                "Unit": item.get("unit", "")
            })
        update_fields["Items"] = items

    update_fields["LastModifiedAt"] = datetime.utcnow()

    orders.update_one({"_id": ObjectId(order_id)}, {"$set": update_fields})
    mongo.close()

    return {
        "success": True,
        "order_id": order_id,
        "message": "訂單修改成功",
        "updated_fields": list(update_fields.keys())
    }


def tool_delete_order(order_id):
    """刪除訂單"""
    from bson import ObjectId

    mongo = MongoClient(TEST_MONGO_URI)
    db = mongo[TEST_DB]
    orders = db[TEST_COLLECTION]

    # 檢查訂單是否存在
    existing = orders.find_one({"_id": ObjectId(order_id)})
    if not existing:
        mongo.close()
        return {"success": False, "message": f"訂單 {order_id} 不存在"}

    orders.delete_one({"_id": ObjectId(order_id)})
    mongo.close()

    return {
        "success": True,
        "order_id": order_id,
        "message": "訂單已刪除"
    }


def get_tomorrow():
    """取得明天日期"""
    tomorrow = datetime.now() + timedelta(days=1)
    return tomorrow.strftime("%Y/%m/%d")


# ============================================================
# Agent 主邏輯
# ============================================================

def _build_system_prompt():
    today = datetime.now().strftime("%Y/%m/%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y/%m/%d")
    return f"""你是一個訂單助手。你的任務是協助客戶建立訂單。
今天日期：{today}，明天日期：{tomorrow}。日期請以此為準，不可自行推算。

工作流程：
1. 當客戶說要訂購產品時，先使用 search_products 搜尋每個產品
2. 根據搜尋結果確認產品，如果有多個相似產品，用 ask_clarification 詢問客戶要哪一個
3. 收集完整的訂單資訊：產品、數量、單位、收貨人、地址、日期
4. 資訊完整後，使用 create_order 建立訂單

create_order 的 items 欄位格式（非常重要）：
- product_ref: 客戶原本說的產品名稱
- product_no: 從 search_products 結果取得的 ProductNo（必填）
- product_name: 從 search_products 結果取得的 ProductName（必填）
- quantity: 數量
- unit: 單位

範例：客戶說「雞腿排3包」，搜尋後找到 ProductNo="A001", ProductName="去骨雞腿排"
則 items 應為：
{{"product_ref": "雞腿排", "product_no": "A001", "product_name": "去骨雞腿排", "quantity": 3, "unit": "包"}}

注意事項：
- 如果客戶沒說日期，預設是明天（{tomorrow}）
- 如果客戶沒說地址或收貨人，可以先建立訂單，這些欄位可以是空的
- 數量和產品是必要的
- product_no 和 product_name 一定要從 search_products 的結果取得，不能自己編造
- 用繁體中文回應

修改與刪除訂單：
- 當客戶說要「換」、「改」、「更換」產品或資訊時，使用 update_order 修改現有訂單，不要建立新訂單
- 當客戶說要「取消」訂單時，使用 delete_order 刪除訂單
- 修改訂單時，記住最近建立的訂單編號（order_id），用於後續修改
"""

SYSTEM_PROMPT = _build_system_prompt()

_chat_session = None


def _send_with_retry(content, max_retries=5):
    """送訊息給 Gemini，遇到 503 自動重試"""
    for attempt in range(max_retries):
        try:
            return _chat_session.send_message(content)
        except errors.ServerError as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1, 2, 4, 8 秒
                print(f"[重試 {attempt + 1}/{max_retries}] 伺服器忙碌，等待 {wait} 秒...")
                time.sleep(wait)
            else:
                raise


def reset_chat():
    global _chat_session
    _chat_session = gemini_client.chats.create(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=GEMINI_TOOLS
        )
    )


def execute_tool(tool_name, arguments):
    """執行工具"""
    if tool_name == "search_products":
        return tool_search_products(arguments.get("query", ""))
    elif tool_name == "create_order":
        return tool_create_order(dict(arguments))
    elif tool_name == "update_order":
        return tool_update_order(
            arguments.get("order_id", ""),
            dict(arguments.get("updates", {}))
        )
    elif tool_name == "delete_order":
        return tool_delete_order(arguments.get("order_id", ""))
    elif tool_name == "ask_clarification":
        return tool_ask_clarification(
            arguments.get("question", ""),
            list(arguments.get("options", []))
        )
    else:
        return {"error": f"Unknown tool: {tool_name}"}


def process_response(response):
    """處理 Gemini 回應，執行工具呼叫並回傳下一個回應"""
    function_calls = list(response.function_calls or [])

    # 備用：從 candidates 直接取（新版 SDK 有時 function_calls 屬性為空）
    if not function_calls:
        try:
            for part in (response.candidates[0].content.parts or []):
                if part.function_call and part.function_call.name:
                    function_calls.append(part.function_call)
        except (AttributeError, IndexError, TypeError):
            pass

    if not function_calls:
        return response, False

    # 執行所有工具並收集結果
    result_parts = []
    for fc in function_calls:
        tool_name = fc.name
        arguments = dict(fc.args)

        print(f"\n[執行工具] {tool_name}")
        print(f"[參數] {json.dumps(arguments, ensure_ascii=False, indent=2)}")

        result = execute_tool(tool_name, arguments)
        print(f"[結果] {json.dumps(result, ensure_ascii=False, indent=2, default=str)}")

        result_parts.append(types.Part.from_function_response(
            name=tool_name,
            response={"result": json.loads(json.dumps(result, default=str))}
        ))

    # 把所有工具結果送回 Gemini（自動重試 503）
    next_response = _send_with_retry(result_parts)
    return next_response, True


def get_multiline_input():
    """取得多行輸入（輸入空行或 END 結束）"""
    print("\n客戶（輸入完按 Enter 兩次）：")
    lines = []
    empty_count = 0

    while True:
        try:
            line = input()
            if line.strip() == "":
                empty_count += 1
                if empty_count >= 1 and lines:  # 有內容後遇到空行就結束
                    break
            else:
                empty_count = 0
                lines.append(line)
        except EOFError:
            break

    return "\n".join(lines)


def main():
    """互動式對話"""
    reset_chat()

    print("=" * 60)
    print("訂單 Agent（使用 Gemini Function Calling）")
    print("輸入 'quit' 離開, 'reset' 重置對話")
    print("可以貼上多行文字，輸入完按 Enter 結束")
    print("=" * 60)

    while True:
        user_input = get_multiline_input().strip()

        if user_input.lower() == 'quit':
            print("掰掰！")
            break

        if user_input.lower() == 'reset':
            reset_chat()
            print("對話已重置")
            continue

        if not user_input:
            continue

        print(f"\n[收到訊息]\n{user_input}\n")

        response = _send_with_retry(user_input)

        # 多輪工具呼叫直到得到文字回應
        while True:
            response, has_tool_calls = process_response(response)
            if not has_tool_calls:
                break

        print(f"\n助手：{response.text or '(no response)'}")


if __name__ == "__main__":
    main()
