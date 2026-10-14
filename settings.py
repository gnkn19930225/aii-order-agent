"""
設定讀取

一律從 config.py 讀，沒寫的用這裡的預設值。
"""
import config


def _get(name, default=""):
    return getattr(config, name, default)


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
MIN_SIMILARITY = _get("MIN_SIMILARITY", 0.5)

# Agent
AGENT_MAX_ROUNDS = _get("AGENT_MAX_ROUNDS", 5)
AGENT_TOTAL_TIMEOUT_SECONDS = _get("AGENT_TOTAL_TIMEOUT_SECONDS", 90)
# 建單前一定要有的欄位：delivery_date / address / receiver_name / customer_name
REQUIRED_ORDER_FIELDS = _get("REQUIRED_ORDER_FIELDS", ["delivery_date"])

# Web
# 有設定時，網頁要輸入這組通關碼才能用，避免被別人刷 Gemini 額度
APP_ACCESS_TOKEN = _get("APP_ACCESS_TOKEN")
SESSION_TTL_MINUTES = _get("SESSION_TTL_MINUTES", 120)

# LINE Bot（LINE Developers 後台 Messaging API 頁面取得）
LINE_CHANNEL_SECRET = _get("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = _get("LINE_CHANNEL_ACCESS_TOKEN")
