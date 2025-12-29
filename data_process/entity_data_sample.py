import json
import random
from collections import Counter, defaultdict
from typing import List, Dict, Any
import os
import cv2
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """读取 jsonl 文件，一行一个 json 对象"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def count_entity_types(data: List[Dict[str, Any]]) -> Counter:
    """
    统计所有样本中 entity.type 的数量。
    每一个 entity 都单独计数。
    """
    type_counter = Counter()
    for ex in data:
        for ent in ex.get("entity", []):
            ent_type = ent.get("type")
            if ent_type is not None:
                type_counter[ent_type] += 1
    return type_counter


def stratified_sample_keep_entity_ratio(
    data: List[Dict[str, Any]],
    sample_ratio: float = 0.2,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    在尽量保持实体类别整体比例不变的前提下，对样本进行分层采样。

    思路（实体级分布约束，样本为单位）：
      1. 先数出每个实体类别在全体数据中的数量 total_counts。
      2. 计算目标采样数量 target_counts = total_counts * sample_ratio。
      3. 随机打乱所有样本顺序，按顺序遍历；
         - 对每个样本，看它包含的实体类别，如果这些类别中还有没到 target 的，就把这个样本加入采样集；
         - 加入样本后，更新 current_counts（按实体计数）。
      4. 当所有类别的 current_counts >= target_counts 时停止。
    
    注意：由于一个样本中可能同时包含多个类别，最终每个类别的数量会有轻微的偏差，这是多标签数据下比较常见的情况。
    """
    random.seed(seed)

    # 1. 总的实体类别计数
    total_counts = count_entity_types(data)
    print("总实体类别计数：", total_counts)

    # 2. 目标计数，至少为 1（如果原本就有该类别）
    target_counts = {
        t: max(1, int(cnt * sample_ratio))
        for t, cnt in total_counts.items()
    }
    print("目标采样实体类别计数：", target_counts)

    current_counts = Counter()
    indices = list(range(len(data)))
    random.shuffle(indices)

    selected_indices = []

    for idx in indices:
        ex = data[idx]
        image_name = ex.get("image_id")
        image_path = f"datasets/twitter17_data/twitter17_images/{image_name}"
        img = cv2.imread(image_path)
        if img is None:
            print(f"找不到图片: {image_path}")
            continue
        
        entities = ex.get("entity", [])
        if not entities:
            # 没有实体的样本可以根据需要选择是否保留，这里简单跳过
            continue

        # 该样本中出现的实体类别及其数量
        ex_type_counts = Counter(ent["type"] for ent in entities if "type" in ent)
        ex_types = set(ex_type_counts.keys())

        # 判断“加入这个样本”是否还有意义（即至少有一个类别还没达到 target）
        if not any(current_counts[t] < target_counts[t] for t in ex_types):
            continue

        # 加入样本
        selected_indices.append(idx)

        # 更新当前计数（按照实体数量更新）
        for t, c in ex_type_counts.items():
            if current_counts[t] < target_counts[t]:
                # 最多累积到 target_counts[t]，避免严重超采样
                current_counts[t] = min(current_counts[t] + c, target_counts[t])

        # 如果所有类别都达到或超过 target，则可以提前停止
        if all(current_counts[t] >= target_counts[t] for t in target_counts):
            break

    sampled_data = [data[i] for i in sorted(selected_indices)]

    print("实际采样后实体类别计数：", count_entity_types(sampled_data))
    print("采样样本数量：", len(sampled_data))

    return sampled_data

def build_output_path(
    input_path: str,
    base_out_dir: str = "data_process/processed_data"
) -> str:
    """
    根据输入路径自动生成输出路径：
    例如：
      input:  UMIE/.../text2spotasoc/entity/twitter2015/train.json
      output: data_process/processed_data/entity/twitter2015/train.jsonl
    """
    norm_path = os.path.normpath(input_path)
    parts = norm_path.split(os.sep)

    # 期望末尾结构为: .../<category>/<dataset>/<split>.json
    if len(parts) < 3:
        raise ValueError(f"输入路径层级太浅，无法解析类别和数据集: {input_path}")

    category = parts[-3]   # entity
    dataset = parts[-2]    # twitter2015
    split = os.path.splitext(parts[-1])[0]  # train

    out_dir = os.path.join(base_out_dir, category, dataset)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, split + ".jsonl")
    return out_path


if __name__ == "__main__":
    # 1. 读取数据
    data_path = "UMIE/text_processing/converted_data/text2spotasoc/entity/twitter2017/test.json"
    data = load_jsonl(data_path)

    # 2. 统计整体实体类别数量
    all_type_counts = count_entity_types(data)
    print("数据集中实体类别分布：")
    for t, c in all_type_counts.items():
        print(f"  {t}: {c}")

    # 3. 做分层采样（例如采 20%）
    sampled = stratified_sample_keep_entity_ratio(data, sample_ratio=0.690, seed=2025)

    # 4. 把采样结果写到文件（可选）
    out_path = build_output_path(data_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in sampled:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"采样数据已保存到: {out_path}")
