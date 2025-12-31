#!/usr/bin/env bash
set -euo pipefail

# ====== 你可能需要改的路径 ======
V1_BASE="LVLM_test_datasets/twitter15"
V2_BASE="LVLM_test_datasets/twitter17"

OUT_BASE="LVLM_test_datasets/mner"

# 两大噪声类型
NOISE_GROUPS=("rule_vision_noise" "text_noise")

# 规则：每个 type 目录下都有 test.json
FNAME="test.json"
# =================================

mkdir -p "$OUT_BASE"

merge_two_json_arrays () {
  local f1="$1"
  local f2="$2"
  local out="$3"
  mkdir -p "$(dirname "$out")"

  # 用 python 做 JSON list 合并（安全、不会破坏格式）
  python - "$f1" "$f2" "$out" << 'PY'
import json, sys, os

f1, f2, out = sys.argv[1], sys.argv[2], sys.argv[3]

def load_list(p):
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        x = json.load(f)
    if isinstance(x, list):
        return x
    raise ValueError(f"{p} is not a JSON list")

a = load_list(f1)
b = load_list(f2)

merged = a + b

os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=4)

print(f"[OK] {out}  (v1={len(a)} + v2={len(b)} => {len(merged)})")
PY
}

echo "========== Merging MNRE-V1 and MNRE-V2 into ${OUT_BASE} =========="

for group in "${NOISE_GROUPS[@]}"; do
  v1_group_dir="${V1_BASE}/${group}"
  v2_group_dir="${V2_BASE}/${group}"

  # 收集所有 type：V1 与 V2 目录名取并集
  types=()
  if [[ -d "$v1_group_dir" ]]; then
    while IFS= read -r d; do types+=("$d"); done < <(find "$v1_group_dir" -mindepth 1 -maxdepth 1 -type d -printf "%f\n")
  fi
  if [[ -d "$v2_group_dir" ]]; then
    while IFS= read -r d; do types+=("$d"); done < <(find "$v2_group_dir" -mindepth 1 -maxdepth 1 -type d -printf "%f\n")
  fi

  if [[ ${#types[@]} -eq 0 ]]; then
    echo "[WARN] No types found for group: ${group}"
    continue
  fi

  # 去重 + 排序
  mapfile -t uniq_types < <(printf "%s\n" "${types[@]}" | sort -u)

  echo "[INFO] group=${group}, types=${#uniq_types[@]}"

  for t in "${uniq_types[@]}"; do
    f1="${v1_group_dir}/${t}/${FNAME}"
    f2="${v2_group_dir}/${t}/${FNAME}"
    out="${OUT_BASE}/${group}/${t}/${FNAME}"

    # 两边都不存在就跳过
    if [[ ! -f "$f1" && ! -f "$f2" ]]; then
      echo "[SKIP] missing both: ${group}/${t}"
      continue
    fi

    echo "[MERGE] ${group}/${t}"
    merge_two_json_arrays "$f1" "$f2" "$out"
  done
done

echo "[DONE] All merged outputs under: ${OUT_BASE}"