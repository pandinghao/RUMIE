import os
import json
import cv2
from tqdm import tqdm
# 输入 / 输出路径
input_file = "data_process/processed_data/relation/MNRE-V2/test.fixed.jsonl"
output_file = "LVLM_test_datasets/MNRE-V2/test.json"
image_forder = "datasets/MNRE-V2/mnre_image/img_org/test"
# 确保输出目录存在
os.makedirs(os.path.dirname(output_file), exist_ok=True)

INSTRUCTION = (
    "Please extract the following relation between [head] and [tail]: "
    "part of, contain, present in, none, held on, member of, peer, "
    "place of residence, locate at, alternate names, neighbor, subsidiary, "
    "awarded, couple parent, nationality, place of birth, charges, siblings, "
    "religion, race."
)

def process_re_data(input_file, output_file):
    results = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            data = json.loads(line.strip())

            text = data.get("text", "")
            entities = data.get("entity", [])
            relations = data.get("relation", [])
            image_id = data.get("image_id", "")
            image_path = f"{image_forder}/{image_id}"
            img = cv2.imread(image_path)
            if img is None:
                iamge_path = f"{image_forder}/{image_id}.jpg"
                img = cv2.imread(iamge_path)
                data["iamge_id"] = str(image_id)+".jpg"
                if img is None:
                    print(f"找不到图片: {image_path}")

                    continue
            head = entities[0]["text"]
            tail = entities[1]["text"]

            # relation type
            if relations:
                relation_type = relations[0]["type"]
            else:
                relation_type = "none"

            # User message
            user_content = (
                f"{INSTRUCTION} "
                f"<image> text: {text} "
                f"Head entity: {head} ; Tail entity: {tail}"
            )

            # Assistant message（严格格式）
            assistant_content = f"{head} <spot> {relation_type} <spot> {tail}"

            sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": user_content
                    },
                    {
                        "role": "assistant",
                        "content": assistant_content
                    }
                ],
                "images": [f"{image_forder}/{image_id}"] if image_id else []
            }

            results.append(sample)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

process_re_data(input_file, output_file)