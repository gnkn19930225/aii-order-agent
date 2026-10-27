# RAG 產品查詢 & 訂單 Agent Demo

用 Gemini Embedding 建立商品向量索引，讓客戶用自然語言查詢產品，並使用 Gemini Function Calling 進行智能打單。
打單邏輯移植自 dachan.oms 的 AI 打單（OllamaOrderAgent），但不看客戶主檔，直接查整份商品目錄。

## 檔案結構

```
aii-order-agent/
├── settings.py            ← 讀設定（config.py，沒寫的用預設值）
├── catalog.py             ← 商品查詢：貨號精確比對、品名向量搜尋
├── agent.py               ← 訂單 Agent 核心（工具、驗證、迴圈控制）
├── order_agent.py         ← 訂單 Agent 命令列版
├── app.py                 ← 訂單 Agent 網頁版 + LINE Webhook（FastAPI）
├── line_bot.py            ← LINE Bot（/callback）
├── sessions.py            ← 對話狀態（網頁版與 LINE 共用，放記憶體）
├── order_store.py         ← 訂單寫入 MongoDB Atlas
├── static/index.html      ← 網頁聊天介面（/）
├── static/products.html   ← 商品總覽與搜尋示範（/products）
├── build_index.py         ← 建立向量索引
├── rag_query.py           ← RAG 產品查詢（LLM 回應）
├── vector_search.py       ← 純向量搜尋（調門檻用）
├── fuzzy_search.py        ← 純模糊搜尋（不用向量）
├── config.example.py      ← 設定範例
├── Dockerfile             ← Cloud Run 部署用
└── vectordb/              ← 向量資料庫（build_index.py 產生，部署時打包進映像檔）
```

## 使用步驟

### 1. 安裝套件

```bash
pip install -r requirements.txt
```

### 2. 設定

複製 `config.example.py` 成 `config.py` 並填入，所有設定都寫在這裡。

### 3. 建立向量索引

```bash
python build_index.py                                   # 從 MongoDB 商品主檔建立
python build_index.py --from-vectordb vectordb_bge-m3   # 從舊的向量資料庫取商品資料重建（不用連 MongoDB）
```

**每次都會整個重建**。換了 embedding 模型或維度一定要重跑，程式啟動時會檢查兩邊是否一致。
Gemini 免費額度每分鐘能轉的筆數有限，幾千筆商品要跑幾十分鐘；中斷了加上 `--resume` 接著跑。

### 4. 開始使用

```bash
python order_agent.py      # 訂單 Agent 命令列版
uvicorn app:app --reload --port 8000   # 網頁版 http://localhost:8000，商品總覽 /products，LINE Webhook 是 /callback
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
| `create_order` | 建立 / 整張取代這段對話的訂單草稿，寫入 MongoDB Atlas（`ORDER_MONGO_URI`） |
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
- `MIN_SIMILARITY`（預設 0.68）以下的向量結果視為查無。這個值跟 embedding 模型綁在一起，換模型要用 `vector_search.py` 觀察實際分數後重調

---

## 部署到 Google Cloud Run

```
LINE ──► Cloud Run（app.py + vectordb/ 打包在映像檔，config.py 由 Secret Manager 掛載）
                 ├──► Gemini API（對話 + embedding）
                 └──► MongoDB Atlas（訂單）
```

訂單資料庫用 MongoDB Atlas 免費 M0 叢集。Network Access 要加入 `0.0.0.0/0`：Cloud Run 沒有固定 IP，
安全性靠資料庫帳號的強密碼與 TLS。連線字串填在 `config.py` 的 `ORDER_MONGO_URI`；留空則訂單只放記憶體。

### 事前準備

- 安裝 gcloud CLI：`winget install Google.CloudSDK`，裝好後開新終端機執行 `gcloud init` 登入
- 在 https://console.cloud.google.com 建一個**新的專案**並綁定帳單帳戶（Cloud Run 需要，用量小在免費額度內）
  - 不要把帳單綁在 Gemini API key 所在的 AI Studio 專案上，那個專案開了帳單就不再走免費額度
- `config.py` 填好 `APP_ACCESS_TOKEN`，避免網頁版被別人刷額度

### 第一次部署

```bash
gcloud config set project 你的專案ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com

# 把整個 config.py 存進 Secret Manager，並只允許 Cloud Run 讀取這一個密鑰
gcloud secrets create order-agent-config --data-file=config.py
gcloud secrets add-iam-policy-binding order-agent-config --member=serviceAccount:你的專案編號-compute@developer.gserviceaccount.com --role=roles/secretmanager.secretAccessor

gcloud run deploy ai-order-agent --source . --region asia-east1 --allow-unauthenticated --max-instances 1 --memory 1Gi --no-cpu-throttling --set-secrets /config/config.py=order-agent-config:latest
```

部署完會印出網址，LINE 後台的 Webhook URL 改成 `https://那個網址/callback`。

- `--max-instances 1`：對話狀態放記憶體，開多台的話同一個人的訊息可能跑到不同台
- `--no-cpu-throttling`：LINE 訊息是回 200 之後才在背景處理，沒開的話 Cloud Run 會在回應後把 CPU 降到幾乎為零
- 專案裡不能有中文檔名的檔案被打包：從 Windows 上傳時檔名編碼會壞掉，部署會出現 `Container import failed`（`.gcloudignore` 已排除 `*.md`）
- 健康檢查是 `/health`，不能用 `/healthz`：Cloud Run 保留了結尾是 z 的網址

### 之後更新

```bash
# 改了程式或重建索引
gcloud run deploy ai-order-agent --source . --region asia-east1

# 改了 config.py：新增一版 secret 再重新部署
gcloud secrets versions add order-agent-config --data-file=config.py
gcloud run deploy ai-order-agent --source . --region asia-east1
```

---

## LINE Bot 本機測試（cloudflared 通道）

1. LINE Developers 建立 Messaging API channel，把 Channel secret、Channel access token 填進 `config.py`
2. LINE Official Account Manager →「回應設定」：關閉自動回應訊息、開啟 Webhook
3. 開兩個終端機：
   ```bash
   uvicorn app:app --reload --port 8000
   cloudflared tunnel --url http://localhost:8000
   ```
4. 把 cloudflared 印出的 `https://xxx.trycloudflare.com` 加上 `/callback`，填到 LINE 後台的 Webhook URL，按 Verify
5. 傳「重新開始」可以清掉對話

注意：快速通道每次重開網址都會變，要重新填 Webhook URL。
