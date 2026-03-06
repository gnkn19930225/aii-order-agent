"""
從 MongoDB 建立向量索引
"""
from pymongo import MongoClient
import chromadb
import ollama
from config import (
    MONGO_URI, MONGO_DB, MONGO_COLLECTION,
    EMBEDDING_HOST, EMBEDDING_MODEL
)

def main():
    # 連接 MongoDB
    print("連接 MongoDB...")
    mongo = MongoClient(MONGO_URI)
    db = mongo[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    # 連接本地 Ollama（用於 Embedding）
    print(f"連接 Ollama (Embedding): {EMBEDDING_HOST}")
    embedding_client = ollama.Client(host=EMBEDDING_HOST)

    # 建立向量資料庫
    print("建立向量資料庫...")
    chroma = chromadb.PersistentClient(path="./vectordb")
    vector_collection = chroma.get_or_create_collection("products")

    # 從 MongoDB 撈資料
    print("從 MongoDB 撈取產品資料...")
    products = collection.find({"IsDeleted": False})

    count = 0
    for product in products:
        product_id = str(product["_id"])

        # 組合要 embedding 的文字（客戶可能會這樣講）
        text_to_embed = f"{product.get('ProductName', '')} {product.get('ProductNo', '')} {product.get('ProductCategory', '')} {product.get('ProductBrand', '')}"

        # 產生 embedding
        response = embedding_client.embeddings(
            model=EMBEDDING_MODEL,
            prompt=text_to_embed
        )
        embedding = response['embedding']

        # 存入向量資料庫
        vector_collection.upsert(
            ids=[product_id],
            embeddings=[embedding],
            documents=[text_to_embed],
            metadatas=[{
                "ProductNo": product.get("ProductNo", ""),
                "ProductName": product.get("ProductName", ""),
                "ProductCategory": product.get("ProductCategory", ""),
                "ProductBrand": product.get("ProductBrand", ""),
            }]
        )

        count += 1
        if count % 10 == 0:
            print(f"已處理 {count} 筆...")

    print(f"完成！共建立 {count} 筆向量索引")
    mongo.close()

if __name__ == "__main__":
    main()
