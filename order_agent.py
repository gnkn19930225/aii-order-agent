"""
訂單 Agent - 命令列版

邏輯都在 agent.py，這裡只負責讀輸入、印結果。網頁版見 app.py。
"""
import json
import uuid
import agent


def get_multiline_input():
    """取得多行輸入（有內容後輸入空行結束）"""
    print("\n客戶（輸入完按 Enter 兩次）：")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            if lines:
                break
        else:
            lines.append(line)
    return "\n".join(lines)


def new_session():
    return agent.OrderSession(session_id=f"cli-{uuid.uuid4().hex[:8]}")


def main():
    session = new_session()

    print("=" * 60)
    print("訂單 Agent（使用 Gemini Function Calling）")
    print("輸入 'quit' 離開, 'reset' 重置對話")
    print("可以貼上多行文字，輸入完按 Enter 結束")
    print("=" * 60)

    while True:
        user_input = get_multiline_input().strip()

        if user_input.lower() == "quit":
            print("掰掰！")
            break

        if user_input.lower() == "reset":
            session = new_session()
            print("對話已重置")
            continue

        if not user_input:
            continue

        result = agent.continue_turn(session, user_input, log=print)
        print(f"\n助手：{result.reply}")
        if result.order:
            print(f"\n[目前訂單 {result.order_id}]")
            print(json.dumps(result.order, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
