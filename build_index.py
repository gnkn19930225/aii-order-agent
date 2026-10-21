"""
建立向量索引

每次執行都會整個重建，避免換了 embedding 模型後新舊向量混在一起。

    python build_index.py                              # 從 MongoDB 商品主檔建立
    python build_index.py --from-vectordb vectordb_old # 從舊的向量資料庫取商品資料重建（不用連 MongoDB）
    python build_index.py ... --resume                 # 中斷後接著做，跳過已經建好的商品

Gemini 免費額度每分鐘能轉的筆數有限，幾千筆商品要跑幾十分鐘，中斷了就用 --resume 接著跑。
"""
import argparse
import os
import chromadb
import settings
from catalog import DailyQuotaExceeded, embed, normalize_code

BATCH_SIZE = 50


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
        "Unit": str(product.get(settings.PRODUCT_UNIT_FIELD, "") or product.get("Unit", "") or ""),
    }


def load_from_mongo():
    from pymongo import MongoClient
    print("從 MongoDB 撈取產品資料...")
    mongo = MongoClient(settings.MONGO_URI)
    source = mongo[settings.MONGO_DB][settings.MONGO_COLLECTION]
    products = [{**p, "_id": str(p["_id"])} for p in source.find({"IsDeleted": False})]
    mongo.close()
    return products


def load_from_vectordb(path):
    print(f"從舊的向量資料庫 {path} 取商品資料...")
    collection = chromadb.PersistentClient(path=path).get_collection(
        settings.VECTOR_COLLECTION, embedding_function=None)
    result = collection.get(include=["metadatas"])
    return [{**metadata, "_id": id_} for id_, metadata in zip(result["ids"], result["metadatas"])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-vectordb", help="從這個舊的向量資料庫資料夾取商品資料，而不是 MongoDB")
    parser.add_argument("--resume", action="store_true", help="不刪除既有索引，跳過已經建好的商品")
    args = parser.parse_args()

    if args.from_vectordb and os.path.abspath(args.from_vectordb) == os.path.abspath(settings.VECTORDB_PATH):
        raise SystemExit("來源跟目標是同一個資料夾，請先把舊的向量資料庫改名再執行")

    products = load_from_vectordb(args.from_vectordb) if args.from_vectordb else load_from_mongo()
    products = [p for p in products if p.get("ProductNo") and p.get("ProductName")]

    print(f"Embedding：{settings.EMBEDDING_MODEL}（{settings.EMBEDDING_DIM} 維）")
    chroma = chromadb.PersistentClient(path=settings.VECTORDB_PATH)
    if not args.resume:
        try:
            chroma.delete_collection(settings.VECTOR_COLLECTION)
            print("已刪除舊的向量索引")
        except Exception:
            pass
    collection = chroma.get_or_create_collection(
        settings.VECTOR_COLLECTION,
        embedding_function=None,
        metadata={"hnsw:space": "cosine", "embedding": settings.EMBEDDING_MODEL},
    )
    if args.resume:
        if (collection.metadata or {}).get("embedding") != settings.EMBEDDING_MODEL:
            raise SystemExit("既有索引的 embedding 模型跟目前設定不同，不能接著做，請拿掉 --resume 重建")
        done = set(collection.get(include=[])["ids"])
        products = [p for p in products if p["_id"] not in done]
        print(f"已有 {len(done)} 筆，剩下 {len(products)} 筆")

    for start in range(0, len(products), BATCH_SIZE):
        batch = products[start:start + BATCH_SIZE]
        documents = [build_document(p) for p in batch]
        try:
            vectors = embed(documents)
        except DailyQuotaExceeded:
            print(f"今天的免費額度用完了，目前索引共 {collection.count()} 筆。"
                  "明天加上 --resume 再跑一次就會接著做。")
            return
        collection.add(
            ids=[p["_id"] for p in batch],
            embeddings=vectors,
            documents=documents,
            metadatas=[build_metadata(p) for p in batch],
        )
        print(f"已處理 {start + len(batch)} / {len(products)} 筆...", flush=True)

    print(f"完成！向量索引共 {collection.count()} 筆")


if __name__ == "__main__":
    main()
