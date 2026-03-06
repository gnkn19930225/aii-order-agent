"""
直接向量搜尋 - 不經過 LLM
"""
import chromadb
import ollama
from config import EMBEDDING_HOST, EMBEDDING_MODEL

# 初始化
embedding_client = ollama.Client(host=EMBEDDING_HOST)
chroma = chromadb.PersistentClient(path="./vectordb")
collection = chroma.get_collection("products")


def get_embedding(text):
    """將文字轉成向量"""
    response = embedding_client.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text
    )
    return response['embedding']


def search(query, top_k=5):
    """搜尋相似產品"""
    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    return results


def main():
    print("=" * 50)
    print("向量搜尋（不經過 LLM，輸入 'quit' 離開）")
    print("=" * 50)

    while True:
        query = input("\n搜尋：").strip()

        if query.lower() == 'quit':
            print("掰掰！")
            break

        if not query:
            continue

        results = search(query, top_k=5)

        print(f"\n找到 {len(results['ids'][0])} 筆結果：")
        print("-" * 50)

        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            similarity = 1 - distance  # 轉成相似度

            print(f"{i+1}. {metadata.get('ProductName', '')}")
            print(f"   編號: {metadata.get('ProductNo', '')}")
            print(f"   類別: {metadata.get('ProductCategory', '')}")
            print(f"   品牌: {metadata.get('ProductBrand', '')}")
            print(f"   相似度: {similarity:.4f}")
            print()


if __name__ == "__main__":
    main()
