"""
從 MongoDB 建立向量索引

每次執行都會整個重建，避免換了 embedding 模型後新舊向量混在一起。
"""
from pymongo import MongoClient
import chromadb
import settings
from catalog import get_embedding, normalize_code

BATCH_SIZE = 100


def build_document(product):
    """
    要 embedding 的文字。刻意不放貨號：
    貨號沒有語意，放進去只會讓同品名不同貨號的商品向量更難分開，
    貨號查詢改由 catalog.get_by_code 精確比對處理。
    """
    return " ".join(filter(None, [
        product.get("ProductName", ""),
        product.get("ProductCategory", ""),
        product.get("ProductBrand", ""),
    ]))


def build_metadata(product):
    return {
        "ProductNo": product.get("ProductNo", ""),
        "ProductNoKey": normalize_code(product.get("ProductNo", "")),
        "ProductName": product.get("ProductName", ""),
        "ProductCategory": product.get("ProductCategory", "") or "",
        "ProductBrand": product.get("ProductBrand", "") or "",
        "Unit": str(product.get(settings.PRODUCT_UNIT_FIELD, "") or ""),
    }


def main():
    print("連接 MongoDB...")
    mongo = MongoClient(settings.MONGO_URI)
    source = mongo[settings.MONGO_DB][settings.MONGO_COLLECTION]

    print(f"Embedding：{settings.EMBEDDING_MODEL}（{settings.EMBEDDING_HOST}）")
    chroma = chromadb.PersistentClient(path=settings.VECTORDB_PATH)
    try:
        chroma.delete_collection(settings.VECTOR_COLLECTION)
        print("已刪除舊的向量索引")
    except Exception:
        pass
    collection = chroma.create_collection(
        settings.VECTOR_COLLECTION,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )

    print("從 MongoDB 撈取產品資料...")
    products = [p for p in source.find({"IsDeleted": False}) if p.get("ProductNo") and p.get("ProductName")]
    mongo.close()

    for start in range(0, len(products), BATCH_SIZE):
        batch = products[start:start + BATCH_SIZE]
        documents = [build_document(p) for p in batch]
        collection.add(
            ids=[str(p["_id"]) for p in batch],
            embeddings=[get_embedding(d) for d in documents],
            documents=documents,
            metadatas=[build_metadata(p) for p in batch],
        )
        print(f"已處理 {start + len(batch)} / {len(products)} 筆...")

    print(f"完成！共建立 {len(products)} 筆向量索引")


if __name__ == "__main__":
    main()
