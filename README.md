# RAG 產品查詢 & 訂單 Agent Demo

從 MongoDB 建立向量索引，讓客戶用自然語言查詢產品，並使用 Gemini Function Calling 進行智能打單。
打單邏輯移植自 dachan.oms 的 AI 打單（OllamaOrderAgent），但不看客戶主檔，直接查整份商品目錄。

## 檔案結構

```
aii-order-agent/
├── settings.py            ← 讀設定（環境變數 > config.py > 預設值）
├── catalog.py             ← 商品查詢：貨號精確比對、品名向量搜尋
├── agent.py               ← 訂單 Agent 核心（工具、驗證、迴圈控制）
├── order_agent.py         ← 訂單 Agent 命令列版
├── app.py                 ← 訂單 Agent 網頁版（FastAPI）
├── static/index.html      ← 網頁聊天介面
├── build_index.py         ← 建立向量索引
├── rag_query.py           ← RAG 產品查詢（LLM 回應）
├── vector_search.py       ← 純向量搜尋（調門檻用）
├── fuzzy_search.py        ← 純模糊搜尋（不用向量）
├── config.example.py      ← 設定範例
└── vectordb/              ← 向量資料庫（build_index.py 產生）
```

## 使用步驟

### 1. 安裝套件

```bash
pip install -r requirements.txt
```

### 2. 設定

複製 `config.example.py` 成 `config.py` 並填入。也可以用同名的環境變數臨時蓋掉。

### 3. 安裝 Ollama 及 embedding 模型

```bash
ollama pull bge-m3
```

### 4. 建立向量索引

```bash
python build_index.py
```

從 MongoDB 撈出所有 `IsDeleted: false` 的產品建立向量索引。**每次都會整個重建**。
換了 embedding 模型一定要重跑。舊版建的 `vectordb/` 也能直接用，只是沒有單位資料。

### 5. 開始使用

```bash
python order_agent.py      # 訂單 Agent 命令列版
uvicorn app:app --reload   # 訂單 Agent 網頁版，打開 http://localhost:8000
python rag_query.py        # RAG 產品查詢
python vector_search.py    # 純向量搜尋
python fuzzy_search.py     # 純模糊搜尋
```

---

## 訂單 Agent 設計

核心原則（沿用 OMS）：**模型負責對話與判斷，系統負責提供工具、驗證資料、限制迴圈。**

### 工具

| Tool | 用途 |
|------|------|
| `search_products` | 查商品。貨號走精確比對，品名走向量搜尋 |
| `create_order` | 建立 / 整張取代這段對話的訂單草稿（目前只存在記憶體） |
| `cancel_order` | 取消這段對話的訂單草稿 |

### 防呆機制

| 機制 | 為什麼 |
|------|------|
| 規則寫在工具說明與工具回傳，system prompt 只留語氣 | prompt 會被長對話稀釋；工具回傳出現在決策當下，模型繞不過去 |
| `create_order` 只收 `search_products` 查過的貨號 | 不讓模型幻覺出來的貨號變成訂單 |
| 建單時用貨號回目錄取權威品名與單位 | 不拿模型給的文字當品名 |
| 同一個貨號有多個單位而沒指定 → 退回 | 不替客戶猜規格 |
| 必填欄位由工具檢查，回 `missing_fields` | 只靠 prompt 叮嚀，模型會湊到品項數量就宣稱完成 |
| 一段對話只有一張草稿，`create_order` 整張取代 | 模型不用記訂單編號，不會改錯單 |
| 輪數上限、總逾時、空回覆兜底訊息 | 迴圈不會跑掉，客戶不會什麼都收不到 |

### 訂單結構

```json
{
  "CustomerName": "客戶名稱",
  "ReceiverName": "收貨人",
  "DeliveryAddress": "送貨地址",
  "DeliveryDate": "2026/02/03",
  "Items": [
    { "ProductRef": "客戶說的", "ProductNo": "ASE004", "ProductName": "正式品名", "Quantity": 10, "Unit": "箱" }
  ],
  "Remarks": "備註"
}
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

### 貨號不走向量搜尋

貨號（如 `AG001`、`AG002`）是無語意的英數代碼，embedding 分不出來。所以：

- 建索引時 **不把貨號放進 embedding 文字**，只放品名、類別、品牌
- 查詢時先用貨號精確比對；看起來像貨號但查不到就回「查無」，不退回向量搜尋拿長得像的充數
- `MIN_SIMILARITY`（預設 0.5）以下的向量結果視為查無，可用 `vector_search.py` 觀察實際分數後調整

