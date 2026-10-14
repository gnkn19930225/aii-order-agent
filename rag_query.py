"""
RAG 產品查詢 - 支援反問確認
"""
from google import genai
from google.genai import types
import catalog
import settings

gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)


def ask_llm(question, products):
    """讓 Gemini 決定回應方式"""
    product_list = "\n".join([
        f"{i}. {p['ProductName']} (編號: {p['ProductNo']}, 品牌: {p['ProductBrand']}, 相似度: {p.get('similarity', 1.0):.2f})"
        for i, p in enumerate(products, start=1)
    ]) or "（查無符合的產品）"

    system_prompt = f"""你是一個產品查詢助手。根據客戶的需求和以下產品清單來回應。

產品清單：
{product_list}

規則：
1. 如果只有一個產品明顯符合，直接確認該產品
2. 如果有多個類似產品，列出前幾項讓客戶選擇
3. 如果沒有符合的產品，告知客戶並請他換個說法或提供貨號
4. 回應要簡潔，用繁體中文
5. 列出選項時，顯示編號、品名、規格讓客戶方便選擇"""

    response = gemini_client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=question,
        config=types.GenerateContentConfig(system_instruction=system_prompt)
    )
    return response.text


def query(user_input):
    """主查詢函數"""
    products = catalog.search(user_input, top_k=5)
    return ask_llm(user_input, products), products


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

        response, _ = query(user_input)
        print(f"\n助手：{response}")
        print("-" * 70)


if __name__ == "__main__":
    main()
