# 複製成 config.py 後填入（config.py 不會進 git）

# MongoDB - 商品主檔（build_index.py / fuzzy_search.py 用）
MONGO_URI = "mongodb://user:password@your-mongodb-host:27017/"
MONGO_DB = "YourDatabase"
MONGO_COLLECTION = "YourCollection"
PRODUCT_UNIT_FIELD = "Unit"          # 商品主檔裡單位的欄位名稱，沒有就留著不影響

# Gemini（對話用）
GEMINI_API_KEY = "your-gemini-api-key"
GEMINI_MODEL = "gemini-3.5-flash-lite"

# Gemini Embedding（向量搜尋用）。換了要重跑 build_index.py
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

# 建單前一定要有的欄位：delivery_date / address / receiver_name / customer_name
REQUIRED_ORDER_FIELDS = ["delivery_date"]

# LINE Bot（LINE Developers 後台 > Messaging API 頁面）
LINE_CHANNEL_SECRET = ""
LINE_CHANNEL_ACCESS_TOKEN = ""

# MongoDB - 訂單寫入位置（MongoDB Atlas 連線字串）。留空就只放記憶體，程式重開訂單就沒了
ORDER_MONGO_URI = ""
ORDER_DB = "aii_order_agent"
ORDER_COLLECTION = "Orders"
