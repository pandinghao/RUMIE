import json
import os

# 输入：你已经处理好的 m2e2 event json
input_file = "data_process/processed_data/event/m2e2_test_ED.json"

# 输出：VOA 风格
output_file = "LVLM_test_datasets/m2e2/test.json"
image_base_dir = "datasets/m2e2/m2e2_rawdata/image/image"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

INSTRUCTION = (
    "Please extract the following event type: "
    "Justice:Arrest-Jail, Life:Die, Movement:Transport, Conflict:Attack, "
    "Conflict:Demonstrate, Contact:Meet, Contact:Phone-Write, "
    "Transaction:Transfer-Money."
)

def process_voa_style_to_messages(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 期望是一个 list

    results = []
    for item in data:
        text = item.get("text", "")
        label = item.get("label", "")
        image_id = item.get("image_id", "")

        # 基本校验：没有 label 或 image_id 的样本直接跳过（你也可以改成保留）
        if not label or not image_id:
            continue

        user_content = f"<image>{INSTRUCTION} text: {text}"

        sample = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": label}
            ],
            "images": [os.path.join(image_base_dir, image_id)]
        }

        results.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

process_voa_style_to_messages(input_file, output_file)