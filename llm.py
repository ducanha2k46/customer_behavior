import pandas as pd
from ollama import chat
from ollama import ChatResponse
import json
import re
import time

# Đọc dữ liệu gốc
df = pd.read_csv('Data/cell2cell-duke univeristy.csv')
sample = df[df['churn'] == 1].iloc[0].dropna().to_dict()

# Prompt cố định
system_prompt = """
You are a data augmentation assistant for a customer churn prediction model.

Your task is to generate 3 NEW customer data points similar to the given record.
Your response MUST be in valid JSON format as a list of dictionaries (e.g. [ { ... }, { ... }, { ... } ]).
DO NOT include any markdown, explanation, or triple backticks. Just return JSON.
"""

# Mục tiêu: 2000 bản ghi
TARGET = 2000
BATCH_SIZE = 5
generated_records = []

# Lặp đến khi đủ dữ liệu
while len(generated_records) < TARGET:
    user_prompt = f"""
    Given the following customer record:

    {json.dumps(sample, indent=2)}

    Please generate {BATCH_SIZE} realistic synthetic customer records similar in structure and logic.
    Return a JSON array of {BATCH_SIZE} dictionaries.
    """

    try:
        response: ChatResponse = chat(
            model='mistral',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        )

        content = response.message.content.strip()

        # In để theo dõi từng vòng lặp (nên tắt nếu chạy nhiều)
        print(f"🔵 Batch {len(generated_records) + 1}-{len(generated_records) + BATCH_SIZE} output:\n", content[:200], "...")

        # Làm sạch định dạng markdown nếu có
        content = re.sub(r"^```(?:json)?", "", content)
        content = re.sub(r"```$", "", content).strip()

        # Parse
        new_records = json.loads(content)
        if isinstance(new_records, list):
            generated_records.extend(new_records)
        else:
            print("⚠️ Output is not a list.")
    except Exception as e:
        print(f"❌ Error: {e}")

    time.sleep(0.5)  # nghỉ nhẹ giữa các lần gọi để tránh overload

# Cắt đúng 2000 record đầu tiên (trong trường hợp thừa)
final_df = pd.DataFrame(generated_records[:TARGET])
final_df.to_csv("augmented_churn_2000.csv", index=False)

print("✅ Done! Saved 2000 synthetic records to 'augmented_churn_2000.csv'")
