import os
import json
import random

ROOT_DIR = "LVLM_test_datasets"
RELATION_SUBDIRS = ["MNRE-V1", "MNRE-V2"]
ENTITY_SUBDIRS = ["twitter15", "twitter17"]
IN_NAME = "test.json"

OUT_DIR = os.path.join(ROOT_DIR, "mner")
OUT_PATH = os.path.join(OUT_DIR, "test.json")

def load_json_list(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, but got: {type(data)}")

    return data

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    merged = []
    stats = {}

    for sd in ENTITY_SUBDIRS:
        in_path = os.path.join(ROOT_DIR, sd, IN_NAME)
        items = load_json_list(in_path)

        # 可选：给每条样本加一个来源字段，便于排查/分析（不需要就注释掉）
        # for x in items:
        #     if isinstance(x, dict) and "source" not in x:
        #         x["source"] = sd

        merged.extend(items)
        stats[sd] = len(items)

    # 打乱合并后的数据
    random.shuffle(merged)

    # 将合并并打乱的数据写入输出文件
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)

    print("Merged finished.")
    print("Output:", OUT_PATH)
    print("Counts per dataset:", stats)
    print("Total:", len(merged))

if __name__ == "__main__":
    main()