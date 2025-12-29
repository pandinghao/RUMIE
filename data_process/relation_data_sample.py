import json
import random
from collections import Counter
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


def count_relation_types(data: List[Dict[str, Any]]) -> Counter:
    """
    统计所有样本中 relation.type 的数量。
    每一条关系单独计数，包括 type == "none" 的负例关系（如果数据里有）。
    """
    type_counter = Counter()
    for ex in data:
        for rel in ex.get("relation", []):
            rel_type = rel.get("type")
            if rel_type is not None:
                type_counter[rel_type] += 1
    return type_counter


def stratified_sample_keep_relation_ratio(
    data: List[Dict[str, Any]],
    sample_ratio: float = 0.2,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    在尽量保持“关系类别”整体比例不变的前提下，对 MNRE 样本进行分层采样。
    （包括 none 类别，只要它在 relation.type 里出现）

    思路（关系级分布约束，样本为单位）：
      1. 统计所有样本中的关系类型计数 total_counts（逐条 relation 计数）。
      2. 计算目标采样计数 target_counts = total_counts * sample_ratio，且每类至少为 1。
      3. 打乱样本顺序，依次遍历：
         - 对每个样本，看它包含的关系类型，如果其中至少有一个类型 current_counts[t] < target_counts[t]，
           就把这个样本加入采样集；
         - 加入样本后，根据该样本里每种关系类型的数量，更新 current_counts；
           更新时不超过 target_counts[t]，避免严重超采样。
      4. 当所有关系类型都达到或超过 target_counts 时提前停止。

    注意：
      - 一个样本可能含有多条、多种关系，因此最终比例会有轻微偏差，这是多标签数据常见情况。
      - 如果想把 “没有任何 relation 的样本” 也当成一个样本级别的 none 类别控制比例，
        可以参考代码中的注释位置加逻辑。
    """
    random.seed(seed)

    # 1. 统计整体关系类别分布
    total_counts = count_relation_types(data)
    print("总关系类别计数：", total_counts)

    if not total_counts:
        print("警告：数据中没有任何关系（relation 为空），无法按关系类别分层采样。")
        return []

    # 2. 目标计数，至少为 1
    target_counts = {
        t: max(1, int(cnt * sample_ratio))
        for t, cnt in total_counts.items()
    }
    print("目标采样关系类别计数：", target_counts)

    current_counts = Counter()
    indices = list(range(len(data)))
    random.shuffle(indices)

    selected_indices = []

    for idx in indices:
        ex = data[idx]
        image_name = ex.get("image_id")
        image_path = f"datasets/MNRE-V2/mnre_image/img_org/test/{image_name}"
        img = cv2.imread(image_path)
        if img is None:
            print(f"找不到图片: {image_path}")
            continue

        relations = ex.get("relation", [])

        # 👉 如果你希望“完全没有关系的样本”当作一个样本级 none 类别来控制比例，
        # 可以把下面一段注释打开，并在 total_counts 里事先把 "none" 加进去。
        #
        # if not relations:
        #     none_type = "none"
        #     if none_type in target_counts and current_counts[none_type] < target_counts[none_type]:
        #         selected_indices.append(idx)
        #         current_counts[none_type] += 1
        #     continue

        if not relations:
            # 默认：relation 为空的样本不参与基于关系分布的采样
            continue

        # 该样本中关系类别及其数量
        ex_type_counts = Counter(
            rel["type"] for rel in relations if "type" in rel
        )
        if not ex_type_counts:
            continue

        ex_types = set(ex_type_counts.keys())

        # 判断加入该样本是否有助于“补足”某些关系类别
        if not any(current_counts[t] < target_counts.get(t, 0) for t in ex_types):
            continue

        # 加入样本
        selected_indices.append(idx)

        # 更新当前计数（按关系条数更新），最多累积到 target_counts[t]
        for t, c in ex_type_counts.items():
            if t not in target_counts:
                continue
            if current_counts[t] < target_counts[t]:
                current_counts[t] = min(current_counts[t] + c, target_counts[t])

        # 如果所有关系类别都已经达到目标数量，可以提前停止
        if all(current_counts[t] >= target_counts[t] for t in target_counts):
            break

    sampled_data = [data[i] for i in sorted(selected_indices)]

    print("实际采样后关系类别计数：", count_relation_types(sampled_data))
    print("采样样本数量：", len(sampled_data))

    return sampled_data


def build_output_path(
    input_path: str,
    base_out_dir: str = "data_process/processed_data"
) -> str:
    """
    根据输入路径自动生成输出路径：
    例如：
      input:  UMIE/.../text2spotasoc/relation/mnre/train.json
      output: data_process/processed_data/relation/mnre/train.jsonl
    """
    norm_path = os.path.normpath(input_path)
    parts = norm_path.split(os.sep)

    # 期望末尾结构为: .../<category>/<dataset>/<split>.json
    if len(parts) < 3:
        raise ValueError(f"输入路径层级太浅，无法解析类别和数据集: {input_path}")

    category = parts[-3]   # relation
    dataset = parts[-2]    # mnre
    split = os.path.splitext(parts[-1])[0]  # train / dev / test

    out_dir = os.path.join(base_out_dir, category, dataset)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, split + ".jsonl")
    return out_path


if __name__ == "__main__":
    # 1. 读取 MNRE 数据（jsonl / 一行一个样本）
    data_path = "UMIE/text_processing/converted_data/text2spotasoc/relation/MNRE-V2/test.json"
    data = load_jsonl(data_path)
    
    # 2. 统计整体关系类别数量
    all_rel_counts = count_relation_types(data)
    print("数据集中关系类别分布：")
    for t, c in all_rel_counts.items():
        print(f"  {t}: {c}")

    # 3. 做分层采样（例如采 70.1%）
    sampled = stratified_sample_keep_relation_ratio(
        data,
        sample_ratio=0.250,
        seed=2025
    )

    # 4. 把采样结果写到文件
    out_path = build_output_path(data_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in sampled:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"采样数据已保存到: {out_path}")