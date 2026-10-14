"""
設定讀取

優先順序：環境變數 > config.py > 預設值。
平常用 config.py 就好；環境變數可以臨時蓋掉某個值，不用改檔案。
"""
import os

try:
    import config as _config
except ImportError:
    _config = None


def _get(name, default=""):
    value = os.environ.get(name)
    if value is not None and value != "":
        return value
    if _config is not None and hasattr(_config, name):
        return getattr(_config, name)
    return default


def _get_int(name, default):
    return int(_get(name, default))


def _get_float(name, default):
    return float(_get(name, default))


def _get_list(name, default):
    value = _get(name, default)
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


# MongoDB - 商品主檔（只有 build_index.py / fuzzy_search.py 會用到）
MONGO_URI = _get("MONGO_URI")
MONGO_DB = _get("MONGO_DB")
MONGO_COLLECTION = _get("MONGO_COLLECTION")
PRODUCT_UNIT_FIELD = _get("PRODUCT_UNIT_FIELD", "Unit")

# Gemini
GEMINI_API_KEY = _get("GEMINI_API_KEY")
GEMINI_MODEL = _get("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Gemini 3 官方強烈建議維持預設 1.0，調低可能造成重複呼叫工具、回覆變差。沒設定就不送
GEMINI_TEMPERATURE = _get("GEMINI_TEMPERATURE", None)
# 思考深度 minimal / low / medium / high。打單不需要深度推理，low 比較快也比較不會逾時
GEMINI_THINKING_LEVEL = _get("GEMINI_THINKING_LEVEL", "low")

# Ollama（Embedding 用）。換模型後一定要重建向量資料庫，新舊向量不能混用
EMBEDDING_HOST = _get("EMBEDDING_HOST", "http://localhost:11434")
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "bge-m3")

# 向量資料庫
VECTORDB_PATH = _get("VECTORDB_PATH", "./vectordb")
VECTOR_COLLECTION = _get("VECTOR_COLLECTION", "products")
# 低於這個相似度的結果視為「查無此商品」，太低會讓模型拿不相干的商品下單
MIN_SIMILARITY = _get_float("MIN_SIMILARITY", 0.5)

# Agent
AGENT_MAX_ROUNDS = _get_int("AGENT_MAX_ROUNDS", 5)
AGENT_TOTAL_TIMEOUT_SECONDS = _get_int("AGENT_TOTAL_TIMEOUT_SECONDS", 90)
# 建單前一定要有的欄位：delivery_date / address / receiver_name / customer_name
REQUIRED_ORDER_FIELDS = _get_list("REQUIRED_ORDER_FIELDS", "delivery_date")

# Web
# 有設定時，網頁要輸入這組通關碼才能用，避免被別人刷 Gemini 額度
APP_ACCESS_TOKEN = _get("APP_ACCESS_TOKEN")
SESSION_TTL_MINUTES = _get_int("SESSION_TTL_MINUTES", 120)
