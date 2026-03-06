"""
純 MongoDB + 模糊搜尋（不用向量資料庫）
"""
from pymongo import MongoClient
from rapidfuzz import fuzz, process
from config import MONGO_URI, MONGO_DB, MONGO_COLLECTION


def get_all_products():
    """從 MongoDB 取得所有產品"""
    mongo = MongoClient(MONGO_URI)
    db = mongo[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    products = list(collection.find({"IsDeleted": False}))
    mongo.close()
    return products


def fuzzy_search(query, products, top_k=5):
    """模糊搜尋"""
    # 組合搜尋文字
    search_texts = []
    for p in products:
        text = f"{p.get('ProductName', '')} {p.get('ProductNo', '')} {p.get('ProductCategory', '')} {p.get('ProductBrand', '')}"
        search_texts.append(text)

    # 模糊比對
    results = process.extract(query, search_texts, scorer=fuzz.partial_ratio, limit=top_k)

    matched = []
    for text, score, idx in results:
        matched.append({
            'product': products[idx],
            'score': score
        })
    return matched


def main():
    print("=" * 50)
    print("模糊搜尋（純 MongoDB，輸入 'quit' 離開）")
    print("=" * 50)

    print("從 MongoDB 載入產品...")
    products = get_all_products()
    print(f"共 {len(products)} 筆產品")

    while True:
        query = input("\n搜尋：").strip()

        if query.lower() == 'quit':
            print("掰掰！")
            break

        if not query:
            continue

        results = fuzzy_search(query, products, top_k=5)

        print(f"\n找到 {len(results)} 筆結果：")
        print("-" * 50)

        for i, r in enumerate(results):
            p = r['product']
            print(f"{i+1}. {p.get('ProductName', '')}")
            print(f"   編號: {p.get('ProductNo', '')}")
            print(f"   類別: {p.get('ProductCategory', '')}")
            print(f"   品牌: {p.get('ProductBrand', '')}")
            print(f"   分數: {r['score']}")
            print()


if __name__ == "__main__":
    main()
