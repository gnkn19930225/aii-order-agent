"""
訂單 Agent - Gemini Function Calling

做法移植自 dachan.oms 的 OllamaOrderAgent：
模型自己決定要查商品、要追問什麼、什麼時候建單，回覆文字也由模型產生。
系統只負責三件事：提供工具、驗證模型給的資料、限制迴圈不要跑掉。

跟 OMS 不同的地方：不看客戶主檔的固定叫貨清單，直接查整份商品目錄（向量搜尋，貨號精確比對）。
"""
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from google import genai
from google.genai import types, errors

import catalog
import settings

TAIPEI = ZoneInfo("Asia/Taipei")

# 這裡只留語氣與對話取向。工具怎麼用、欄位是什麼意思、什麼情況不能下單，
# 一律寫在工具說明與工具回傳裡，由模型當場讀。
# 提示詞是一次性的叮嚀，長對話走幾輪就被稀釋；工具回傳出現在決策當下，模型繞不過去。
SYSTEM_PROMPT = """你是食品批發商的訂單助理，在線上與客戶對話。用繁體中文，語氣自然簡短，像真人客服。

工具回傳的內容就是事實，照它說的做，不要自己推測或替它補完。

- 客戶提到任何商品時一律先用 search_products 查，不要憑印象回答。
- 搜尋結果有好幾個相近的商品，或同一個品名有多種規格時，問客戶要哪一種，不要自己挑。
- 客戶只是在詢問時就好好回答，不要硬把話題轉到下單。
- 訂單成立後簡短告訴客戶你記了什麼，不要重複列一次完整清單。"""

# 模型跑到輪數上限卻一句話都沒講時的收尾
EXHAUSTED_TEXT = "（AI 無法完成這次對話，請換個說法再試一次）"
# 模型一個字都沒產出時，代替它告訴客戶這一輪沒有回應。客戶再問一次通常就會自己救回來
NO_RESPONSE_TEXT = "（AI 沒有回應）"

_gemini_client = None


def _client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=60_000),
        )
    return _gemini_client


# ============================================================
# 工具定義
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


_ITEM_SCHEMA = _s("object", properties={
    "item_code":   _s("string", "必須是 search_products 回傳過的 item_code。自己編的代號會被拒絕"),
    "quantity":    _s("number", "數量，必須大於零"),
    "unit":        _s("string", "單位。search_products 回傳的 unit 有值就照抄（同一個 item_code 有多個單位時必填）；unit 是空的就填客戶說的單位，例如 包、箱、瓶"),
    "product_ref": _s("string", "客戶原本怎麼稱呼這個商品"),
}, required=["item_code", "quantity"])

TOOLS = [types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="search_products",
        description=(
            "在商品目錄裡查商品。客戶提到任何商品時一律先查，不要憑印象回答。"
            "keyword 可以是品名、俗名或貨號；是貨號時只會回傳完全相符的那一項。"
            "回傳 found=false 表示查不到，照 message 說的告訴客戶，不要自己編貨號。"
            "products 依 similarity 由高到低排列，好幾項都可能符合時要問客戶要哪一個。"
            "同一個 item_code 出現多筆時是同商品的不同單位，要讓客戶自己選，"
            "選好後把對應的 unit 一起放進 create_order。"
        ),
        parameters=_s("object", properties={
            "keyword": _s("string", "商品關鍵字，例如 沙拉油、雞胸肉、ASA022-2，一次查一個商品"),
        }, required=["keyword"]),
    ),
    types.FunctionDeclaration(
        name="create_order",
        description=(
            "建立訂單草稿。每一項都要有明確商品與數量才可呼叫。"
            "回傳 ok=true 才代表訂單成立；ok=false 時訂單沒有建立，"
            "要依照 error 說的處理，補齊後再呼叫一次，不可以告訴客戶訂單已完成。"
            "每次呼叫都會整張取代這段對話裡還沒取消的那張草稿，"
            "所以客戶只是道謝、打招呼、問別的問題時不要呼叫，直接回覆即可；"
            "只有客戶要修改或新增訂單內容時才呼叫，而且要帶完整的品項與欄位（沒改的也要帶）。"
        ),
        parameters=_s("object", properties={
            "items":          _s("array", "訂購品項", items=_ITEM_SCHEMA),
            "delivery_date":  _s("string", "到貨日期 yyyy-MM-dd，客戶沒說就不要填"),
            "address":        _s("string", "收貨地址，客戶沒說就不要填"),
            "receiver_name":  _s("string", "收貨人，客戶沒說就不要填"),
            "customer_name":  _s("string", "客戶（店家）名稱，客戶沒說就不要填"),
            "remark":         _s("string", "備註"),
        }, required=["items"]),
    ),
    types.FunctionDeclaration(
        name="cancel_order",
        description="取消這段對話目前的訂單草稿。只有客戶明確說要取消時才呼叫。",
    ),
])]


# ============================================================
# 對話狀態
# ============================================================

@dataclass
class OrderSession:
    session_id: str
    contents: list = field(default_factory=list)
    # 查過的貨號白名單。create_order 只收這裡面的，這是不讓幻覺變成訂單的最後一道防線
    searched_codes: set = field(default_factory=set)
    order_id: str = ""
    order: dict = None
    last_active: float = field(default_factory=time.time)


@dataclass
class TurnResult:
    reply: str
    order_id: str
    order: dict
    tools_used: list
    rounds: int


# ============================================================
# 主流程
# ============================================================

def _build_system_prompt():
    today = datetime.now(TAIPEI).date()
    weekdays = "一二三四五六日"
    return (
        SYSTEM_PROMPT
        + f"\n\n今天是 {today:%Y-%m-%d}（星期{weekdays[today.weekday()]}）。"
        + "客戶說「明天」「下禮拜一」「月底」時請自己換算成實際日期，不要反問客戶幾月幾號。"
    )


def _send(session, log):
    """送給 Gemini，遇到 429/5xx 自動重試"""
    config = types.GenerateContentConfig(
        system_instruction=_build_system_prompt(),
        tools=TOOLS,
        temperature=settings.GEMINI_TEMPERATURE,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=types.ThinkingConfig(thinking_level=settings.GEMINI_THINKING_LEVEL)
        if settings.GEMINI_THINKING_LEVEL else None,
    )
    for attempt in range(5):
        try:
            return _client().models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=session.contents,
                config=config,
            )
        except errors.APIError as e:
            if e.code not in (429, 500, 503, 504) or attempt == 4:
                raise
            wait = 2 ** attempt
            log(f"[重試 {attempt + 1}/5] Gemini 回 {e.code}，等待 {wait} 秒...")
            time.sleep(wait)


def _visible_text(content):
    """取出要給客戶看的文字，略過 thinking"""
    return "".join(
        part.text for part in (content.parts or [])
        if part.text and not getattr(part, "thought", False)
    ).strip()


def continue_turn(session, user_message, log=lambda _: None):
    """處理客戶的一則訊息，回傳這一輪的結果"""
    session.last_active = time.time()
    session.contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

    deadline = time.monotonic() + settings.AGENT_TOTAL_TIMEOUT_SECONDS
    tools_used = []
    rounds = 0

    # 模型常把話寫在有 function call 的那一輪。只認「沒有工具那一輪」的文字的話，
    # 下一輪剛好回空字串時客戶就什麼都收不到。所以每一輪講過的話都先留著，有新的再蓋掉。
    pending_reply = ""

    while rounds < settings.AGENT_MAX_ROUNDS and time.monotonic() < deadline:
        rounds += 1
        response = _send(session, log)

        candidate = (response.candidates or [None])[0]
        content = candidate.content if candidate else None
        if content is None or not content.parts:
            log(f"[警告] Gemini 回傳空內容，finish_reason={getattr(candidate, 'finish_reason', None)}")
            break

        # 原封不動放回歷史：Gemini 2.5 的 function call 帶有 thought_signature，少了會出錯
        session.contents.append(content)

        text = _visible_text(content)
        if text:
            pending_reply = text

        calls = [part.function_call for part in content.parts if part.function_call]
        if not calls:
            return _result(session, pending_reply or NO_RESPONSE_TEXT, tools_used, rounds)

        result_parts = []
        for call in calls:
            args = dict(call.args or {})
            tools_used.append(call.name)
            log(f"\n[執行工具] {call.name}\n[參數] {json.dumps(args, ensure_ascii=False, default=str)}")

            try:
                result = _execute_tool(session, call.name, args)
            except Exception as e:
                # 一定要回一個結果給模型：歷史裡有 function call 卻沒有對應的回應，
                # 之後每一次呼叫 Gemini 都會被拒絕，這段對話就整個壞掉
                log(f"[錯誤] 工具 {call.name} 執行失敗：{e!r}")
                result = {"ok": False, "error": "系統暫時無法處理，請告訴客戶稍後再試，不要說訂單已完成"}
            log(f"[結果] {json.dumps(result, ensure_ascii=False, default=str)}")

            result_parts.append(types.Part.from_function_response(name=call.name, response={"result": result}))

        session.contents.append(types.Content(role="user", parts=result_parts))

    log(f"[警告] 達到輪數上限或逾時（{rounds} 輪），回兜底訊息")
    return _result(session, pending_reply or EXHAUSTED_TEXT, tools_used, rounds)


def _result(session, reply, tools_used, rounds):
    return TurnResult(
        reply=reply,
        order_id=session.order_id,
        order=session.order,
        tools_used=tools_used,
        rounds=rounds,
    )


def _execute_tool(session, name, args):
    if name == "search_products":
        return _tool_search(session, str(args.get("keyword", "")))
    if name == "create_order":
        return _tool_create_order(session, args)
    if name == "cancel_order":
        return _tool_cancel_order(session)
    return {"error": f"unknown tool: {name}"}


# ============================================================
# 工具實作
# ============================================================

def _tool_search(session, keyword):
    found = catalog.search(keyword)

    if not found:
        # 這個回傳就是模型當場看到的事實，所以連「該怎麼跟客戶講」都寫在這裡
        return {
            "keyword": keyword,
            "found": False,
            "message": "商品目錄裡查不到這個商品。請告訴客戶查無此品項，"
                       "請他換個說法或提供貨號。不要自己編貨號，也不要承諾會幫他下單。",
        }

    for product in found:
        session.searched_codes.add(catalog.normalize_code(product["ProductNo"]))

    return {
        "keyword": keyword,
        "found": True,
        "products": [
            {
                "item_code": p["ProductNo"],
                "item_name": p["ProductName"],
                "brand": p["ProductBrand"],
                "category": p["ProductCategory"],
                "unit": p["Unit"],
                "similarity": p.get("similarity", 1.0),
            }
            for p in found
        ],
    }


def _resolve(code, unit):
    """
    用貨號回目錄取出權威的品名與單位。只認貨號完全相符——比錯就是下錯貨。

    同一個貨號可能有多個單位：模型有指定就照指定；沒指定而剛好只有一種就用那一種；
    沒指定又有多種就回 None，讓模型回去問客戶，不要替客戶猜。
    """
    matches = catalog.get_by_code(code)
    if not matches:
        return None

    with_unit = [m for m in matches if m["Unit"]]
    if not with_unit:
        # 目錄沒維護單位，就用客戶說的單位
        return {**matches[0], "Unit": unit or ""}

    if unit:
        return next((m for m in with_unit if m["Unit"].strip().lower() == unit.strip().lower()), None)

    return with_unit[0] if len(with_unit) == 1 else None


def _parse_date(text):
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime((text or "").strip(), fmt).date()
        except ValueError:
            pass
    return None


def _tool_create_order(session, args):
    raw_items = args.get("items") or []
    if not raw_items:
        return {"ok": False, "error": "items 是必填的"}

    items = []
    rejected = []
    for raw in raw_items:
        raw = dict(raw)
        code = str(raw.get("item_code", "")).strip()
        unit = str(raw.get("unit", "") or "").strip()
        try:
            quantity = float(raw.get("quantity", 0))
        except (TypeError, ValueError):
            quantity = 0

        if catalog.normalize_code(code) not in session.searched_codes or quantity <= 0:
            rejected.append(code)
            continue

        master = _resolve(code, unit)
        if master is None:
            # 查不到，或同一個貨號有多個單位而模型沒指定
            rejected.append(code)
            continue

        items.append({
            "ProductRef": str(raw.get("product_ref", "") or ""),
            "ProductNo": master["ProductNo"],
            "ProductName": master["ProductName"],
            "Quantity": int(quantity) if quantity.is_integer() else quantity,
            "Unit": master["Unit"],
        })

    if rejected:
        return {
            "ok": False,
            "error": "以下貨號沒有先用 search_products 查過、數量無效，"
                     "或同一個貨號有多個單位卻沒指定 unit，請重新確認",
            "rejected": rejected,
        }

    delivery_date = _parse_date(args.get("delivery_date"))
    fields = {
        "delivery_date": delivery_date,
        "address": str(args.get("address", "") or "").strip(),
        "receiver_name": str(args.get("receiver_name", "") or "").strip(),
        "customer_name": str(args.get("customer_name", "") or "").strip(),
    }

    # 必填欄位由工具把關。只靠 prompt 叮嚀的話，模型會湊到品項數量就宣稱完成
    missing = [f for f in settings.REQUIRED_ORDER_FIELDS if not fields.get(f)]
    if missing:
        return {
            "ok": False,
            "error": "訂單還不能成立，請先問客戶這些資訊，拿到之後再呼叫一次 create_order",
            "missing_fields": missing,
        }

    if delivery_date and delivery_date < datetime.now(TAIPEI).date():
        return {"ok": False, "error": f"到貨日期 {delivery_date} 已經過了，請跟客戶確認日期"}

    order = {
        "CustomerName": fields["customer_name"],
        "ReceiverName": fields["receiver_name"],
        "DeliveryAddress": fields["address"],
        "DeliveryDate": delivery_date.strftime("%Y/%m/%d") if delivery_date else "",
        "Items": items,
        "Remarks": str(args.get("remark", "") or ""),
    }
    # 訂單先只放在對話狀態裡，程式重開就沒了
    session.order_id = session.order_id or uuid.uuid4().hex[:12]
    session.order = order

    return {"ok": True, "order_id": session.order_id, "count": len(items)}


def _tool_cancel_order(session):
    if not session.order_id:
        return {"ok": False, "error": "目前沒有訂單可以取消"}

    cancelled = session.order_id
    session.order_id = ""
    session.order = None
    return {"ok": True, "cancelled_order_id": cancelled}
