"""
商品目錄

- 貨號：走精確比對，不走向量。貨號是沒有語意的英數代碼，
  embedding 分不出 AG001 跟 AG002，拿去做向量搜尋只會撈到一堆長得像的商品。
- 品名、俗名：走向量搜尋，客戶說「雞腿」也找得到「去骨雞腿排」。
"""
import re
import time
import chromadb
from google import genai
from google.genai import errors, types
import settings

# 看起來像貨號：英文開頭、含數字、沒有中文，例如 ASA022-2、AG001
_CODE_PATTERN = re.compile(r"^[A-Za-z]{1,6}[-_]?\d+[A-Za-z0-9\-_]*$")

# 免費額度每分鐘 100 筆，而且一批裡的每一筆都算一次。一次送滿 100 筆的話，
# 同一分鐘只要多一筆就整批被拒，所以每批送 50 筆
_EMBED_BATCH = 50

_collection = None
_client = None


class DailyQuotaExceeded(Exception):
    """Gemini embedding 免費額度一天 1000 筆用完了"""


def embed(texts, task_type="RETRIEVAL_DOCUMENT"):
    """
    文字轉向量（Gemini Embedding）。建索引用 RETRIEVAL_DOCUMENT，查詢用 RETRIEVAL_QUERY，
    兩種是成對設計的，混用會讓相似度變差。
    """
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)

    vectors = []
    for start in range(0, len(texts), _EMBED_BATCH):
        for attempt in range(6):
            try:
                result = _client.models.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    contents=texts[start:start + _EMBED_BATCH],
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=settings.EMBEDDING_DIM,
                    ),
                )
                break
            except errors.APIError as e:
                if e.code == 429 and "PerDay" in str(e):
                    raise DailyQuotaExceeded() from e
                # 免費額度每分鐘有次數上限，建索引時很容易撞到 429，Google 建議等 30 秒再試
                if e.code not in (429, 500, 503, 504) or attempt == 5:
                    raise
                time.sleep(30 if e.code == 429 else 2 ** attempt)
        vectors.extend(e.values for e in result.embeddings)
    return vectors


def get_embedding(text):
    """查詢用：將一段文字轉成向量"""
    return embed([text], task_type="RETRIEVAL_QUERY")[0]


def normalize_code(code):
    """去空白、轉大寫，比對貨號用"""
    return "".join((code or "").split()).upper()


def looks_like_code(keyword):
    return bool(_CODE_PATTERN.match((keyword or "").strip()))


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=settings.VECTORDB_PATH)
        _collection = client.get_collection(settings.VECTOR_COLLECTION, embedding_function=None)
        built_with = (_collection.metadata or {}).get("embedding")
        if built_with != settings.EMBEDDING_MODEL:
            raise RuntimeError(
                f"向量資料庫是用 {built_with or '舊版設定'} 建的，目前設定是 {settings.EMBEDDING_MODEL}，"
                "新舊向量不能混用，請重新執行 build_index.py"
            )
    return _collection


def _to_product(metadata, similarity=None):
    product = {
        "ProductNo": metadata.get("ProductNo", ""),
        "ProductName": metadata.get("ProductName", ""),
        "ProductCategory": metadata.get("ProductCategory", ""),
        "ProductBrand": metadata.get("ProductBrand", ""),
        "Unit": metadata.get("Unit", ""),
    }
    if similarity is not None:
        product["similarity"] = round(similarity, 4)
    return product


def all_products():
    """目錄裡所有商品，依貨號排序（商品總覽頁用）"""
    result = _get_collection().get(include=["metadatas"])
    return sorted((_to_product(m) for m in result["metadatas"]), key=lambda p: p["ProductNo"])


def get_by_code(code):
    """用貨號精確查商品，查不到回空清單（同一個貨號可能有多筆不同單位）"""
    key = normalize_code(code)
    if not key:
        return []
    result = _get_collection().get(where={"ProductNoKey": key})
    return [_to_product(m) for m in result["metadatas"]]


def vector_search(query, top_k=5):
    """純向量搜尋，不做門檻過濾"""
    # 索引用 cosine 距離建立，相似度 = 1 - 距離
    result = _get_collection().query(
        query_embeddings=[get_embedding(query)],
        n_results=top_k,
    )
    return [
        _to_product(metadata, 1 - distance)
        for metadata, distance in zip(result["metadatas"][0], result["distances"][0])
    ]


def search(keyword, top_k=5):
    """
    查商品。貨號完全相符就直接回傳；看起來像貨號但查不到就回空，
    不要退回向量搜尋拿一個長得像的貨號充數。
    """
    keyword = (keyword or "").strip()
    if not keyword:
        return []

    exact = get_by_code(keyword)
    if exact:
        return exact

    if looks_like_code(keyword):
        return []

    return [p for p in vector_search(keyword, top_k) if p["similarity"] >= settings.MIN_SIMILARITY]
