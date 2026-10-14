"""
直接向量搜尋 - 不經過 LLM，用來調整 MIN_SIMILARITY 門檻
"""
import catalog


def main():
    print("=" * 50)
    print("向量搜尋（不經過 LLM，輸入 'quit' 離開）")
    print("貨號會走精確比對，其他走向量搜尋")
    print("=" * 50)

    while True:
        query = input("\n搜尋：").strip()

        if query.lower() == 'quit':
            print("掰掰！")
            break

        if not query:
            continue

        exact = catalog.get_by_code(query)
        results = exact or catalog.vector_search(query, top_k=5)

        print(f"\n找到 {len(results)} 筆結果（{'貨號精確比對' if exact else '向量搜尋'}）：")
        print("-" * 50)

        for i, p in enumerate(results):
            print(f"{i+1}. {p['ProductName']}")
            print(f"   編號: {p['ProductNo']}")
            print(f"   類別: {p['ProductCategory']}")
            print(f"   品牌: {p['ProductBrand']}")
            if p['Unit']:
                print(f"   單位: {p['Unit']}")
            if 'similarity' in p:
                print(f"   相似度: {p['similarity']:.4f}")
            print()


if __name__ == "__main__":
    main()
