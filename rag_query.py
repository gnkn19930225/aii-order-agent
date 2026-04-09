"""
RAG 產品查詢 - 支援反問確認
"""
import chromadb
import ollama
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL, EMBEDDING_HOST, EMBEDDING_MODEL

# 初始化
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
embedding_client = ollama.Client(host=EMBEDDING_HOST)  # 本地 - Embedding 用
chroma = chromadb.PersistentClient(path="./vectordb")
collection = chroma.get_or_create_collection("products")


def get_embedding(text):
    """將文字轉成向量（使用本地 Ollama）"""
    response = embedding_client.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text
    )
    return response['embedding']


def search_products(query, top_k=5):
    """搜尋相似產品"""
    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    return results


def format_products(results):
    """格式化產品清單"""
    products = []
    for i, metadata in enumerate(results['metadatas'][0]):
        products.append({
            "index": i + 1,
            "ProductNo": metadata.get("ProductNo", ""),
            "ProductName": metadata.get("ProductName", ""),
            "ProductCategory": metadata.get("ProductCategory", ""),
            "ProductBrand": metadata.get("ProductBrand", ""),
            "similarity": 1 - results['distances'][0][i]
        })
    return products


def ask_llm(question, products):
    """讓 Gemini 決定回應方式"""
    product_list = "\n".join([
        f"{p['index']}. {p['ProductName']} (編號: {p['ProductNo']}, 品牌: {p['ProductBrand']}, 相似度: {p['similarity']:.2f})"
        for p in products
    ])

    system_prompt = f"""你是一個產品查詢助手。根據客戶的需求和以下產品清單來回應。

產品清單：
{product_list}

規則：
1. 如果只有一個產品明顯符合，直接確認該產品
2. 如果有多個類似產品，列出前幾項讓客戶選擇
3. 如果沒有符合的產品，告知客戶並建議相近的選項
4. 回應要簡潔，用繁體中文
5. 列出選項時，顯示編號、品名、規格讓客戶方便選擇"""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=question,
        config=types.GenerateContentConfig(system_instruction=system_prompt)
    )
    return response.text


def query(user_input):
    """主查詢函數"""
    results = search_products(user_input, top_k=5)
    products = format_products(results)
    response = ask_llm(user_input, products)
    return response, products


def main():
    """互動式查詢"""
    print("=" * 50)
    print("產品查詢系統（輸入 'quit' 離開）")
    print("=" * 50)

    while True:
        user_input = input("\n客戶說：").strip()

        if user_input.lower() == 'quit':
            print("掰掰！")
            break

        if not user_input:
            continue

        response, products = query(user_input)

        print(f"\n助手：{response}")
        print("-------------------------------------------------------------------------\n\n-------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
