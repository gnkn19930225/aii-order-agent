"""
商品目錄

- 貨號：走精確比對，不走向量。貨號是沒有語意的英數代碼，
  embedding 分不出 AG001 跟 AG002，拿去做向量搜尋只會撈到一堆長得像的商品。
- 品名、俗名：走向量搜尋，客戶說「雞腿」也找得到「去骨雞腿排」。
"""
import re
import chromadb
import numpy as np
import ollama
import settings

# 看起來像貨號：英文開頭、含數字、沒有中文，例如 ASA022-2、AG001
_CODE_PATTERN = re.compile(r"^[A-Za-z]{1,6}[-_]?\d+[A-Za-z0-9\-_]*$")

_collection = None
_embedding_client = None


def get_embedding(text):
    """將文字轉成向量"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = ollama.Client(host=settings.EMBEDDING_HOST)
    return _embedding_client.embeddings(model=settings.EMBEDDING_MODEL, prompt=text)["embedding"]


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


def get_by_code(code):
    """用貨號精確查商品，查不到回空清單（同一個貨號可能有多筆不同單位）"""
    key = normalize_code(code)
    if not key:
        return []
    # 舊版 build_index.py 建的索引沒有 ProductNoKey，退回直接比 ProductNo
    result = _get_collection().get(where={"$or": [
        {"ProductNoKey": key},
        {"ProductNo": code.strip()},
        {"ProductNo": key},
    ]})
    return [_to_product(m) for m in result["metadatas"]]


def vector_search(query, top_k=5):
    """純向量搜尋，不做門檻過濾"""
    # Ollama 的 bge-m3 向量沒有正規化，舊版索引又是用 L2 距離排序，
    # 距離換算不出可比較的相似度。所以多撈一些候選，自己算 cosine 再重新排序
    query_vector = np.asarray(get_embedding(query))
    result = _get_collection().query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_k * 4,
        include=["metadatas", "embeddings"],
    )
    vectors = np.asarray(result["embeddings"][0])
    similarities = vectors @ query_vector / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector))
    ranked = sorted(zip(result["metadatas"][0], similarities), key=lambda x: x[1], reverse=True)
    return [_to_product(metadata, float(similarity)) for metadata, similarity in ranked[:top_k]]


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
