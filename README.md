# RAG 產品查詢 & 訂單 Agent Demo

從 MongoDB 建立向量索引，讓客戶用自然語言查詢產品，並使用 LLM Tools 進行智能打單。

## 檔案結構

```
rag-demo/
├── config.py              ← 設定檔（MongoDB、Ollama 連線）
├── build_index.py         ← 建立向量索引
├── rag_query.py           ← RAG 產品查詢（LLM 回應）
├── vector_search.py       ← 純向量搜尋（不經過 LLM）
├── fuzzy_search.py        ← 純模糊搜尋（不用向量）
├── order_agent.py         ← 訂單 Agent（使用 LLM Tools）
├── requirements.txt       ← 套件需求
├── vectordb/              ← 向量資料庫（自動產生）
└── 向量資料庫說明.md       ← 向量資料庫運作原理說明
```

## 使用步驟

### 1. 安裝套件

```bash
pip install -r requirements.txt
pip install rapidfuzz  # fuzzy_search.py 需要
```

### 2. 修改設定檔

編輯 `config.py`：

```python
# MongoDB 設定
MONGO_URI = "mongodb://你的IP:27017"
MONGO_DB = "你的資料庫名稱"
MONGO_COLLECTION = "你的collection名稱"

# Ollama 設定 - 遠端（對話用）
OLLAMA_HOST = "http://遠端Ollama的IP:11434"
OLLAMA_MODEL = "gpt-oss:120b"  # 或其他 LLM

# Ollama 設定 - 本地（Embedding 用）
EMBEDDING_HOST = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"
```

### 3. 安裝 Ollama 及模型

```bash
# 本地安裝 embedding 模型
ollama pull bge-m3
```

### 4. 建立向量索引

```bash
python build_index.py
```

這會從 MongoDB 撈出所有 `IsDeleted: false` 的產品，建立向量索引。

### 5. 開始使用

```bash
# RAG 產品查詢（有 LLM 回應）
python rag_query.py

# 純向量搜尋（不經過 LLM）
python vector_search.py

# 純模糊搜尋（不用向量）
python fuzzy_search.py

# 訂單 Agent（使用 LLM Tools）
python order_agent.py
```

---

## 程式說明

### rag_query.py - RAG 產品查詢

用向量搜尋找相似產品，再用 LLM 生成回應。

```
客戶說：我要五穀米

助手：您好！我們有以下五穀米產品：
1. 五穀米1.8kg (編號: ASA022-2)
2. 五穀米3kg (編號: ASA022-3)
```

### vector_search.py - 純向量搜尋

不經過 LLM，直接顯示向量搜尋結果。

### fuzzy_search.py - 純模糊搜尋

用字串模糊比對（Fuse.js 風格），不需要向量資料庫。

### order_agent.py - 訂單 Agent

使用 LLM Tools 進行智能打單，支援多行輸入。

#### 可用的 Tools

| Tool | 用途 |
|------|------|
| `search_products` | 搜尋產品（向量搜尋） |
| `create_order` | 建立訂單 |
| `update_order` | 修改訂單 |
| `delete_order` | 刪除訂單 |
| `ask_clarification` | 詢問客戶補充資訊 |

#### 訂單結構

```json
{
  "CustomerName": "客戶名稱",
  "ReceiverName": "收貨人",
  "DeliveryAddress": "送貨地址",
  "DeliveryDate": "2026/02/03",
  "Items": [
    {
      "ProductRef": "客戶說的",
      "ProductNo": "ASE004",
      "ProductName": "正式品名",
      "Quantity": 10,
      "Unit": "箱"
    }
  ],
  "Remarks": "備註",
  "Status": "Pending",
  "CreatedAt": "...",
  "CreatedBy": "OrderAgent"
}
```

#### 使用範例

```
客戶（輸入完按 Enter 兩次）：
協調中心
臺北市內湖區文湖街20號7F
冰糖2包 豆腐乳5箱 紹興酒1瓶
週五前到
郭威志收

[執行工具] search_products ...
[執行工具] create_order ...

助手：訂單已建立！訂單編號：698061b01f94dc68638a472b
```

---

## 向量搜尋 vs 模糊搜尋

| | 模糊搜尋 | 向量搜尋 |
|---|---|---|
| **原理** | 字串相似度（編輯距離） | 語意相似度（向量夾角） |
| **找什麼** | 拼寫相近 | 意思相近 |
| **範例** | 「雞腿」→「雞腿排」✓「棒棒腿」✗ | 「雞腿」→「雞腿排」✓「棒棒腿」✓ |
| **速度** | 快 | 較慢（要轉向量） |
| **資源** | 輕量 | 需要 embedding 模型 |

---

## 注意事項

- 確保 Ollama 有安裝 embedding 模型（如 `bge-m3`）
- 確保遠端 Ollama 有安裝 LLM 模型（如 `gpt-oss:120b`）
- MongoDB 資料需要有 `ProductName`、`ProductNo` 欄位
- 訂單存到 `OMSDB_Test.Orders`（測試用 DB）
- 換 embedding 模型後需要重建向量資料庫：
  ```bash
  rmdir /s /q vectordb
  python build_index.py
  ```
