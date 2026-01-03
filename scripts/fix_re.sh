#!/usr/bin/env bash
set -e

ROOT="LVLM_RUMIE/results/roubust_results/Qwen3_VL_4B_UMIE_76422/mre"
GET_METRIC="LVLM_RUMIE/get_metric.py"

# 占位文件（required 参数，但不会真的用）
DUMMY=""

# 找到所有 result(s).jsonl
find "${ROOT}" -type f \( -name "result.jsonl" -o -name "results.jsonl" \) | sort | while read -r f; do
    dir="$(dirname "$f")"
    out="${dir}/metric_mre.txt"

    echo "[RUN] $f"
    python "${GET_METRIC}" \
        --mee_file "${DUMMY}" \
        --mner_file "${DUMMY}" \
        --mre_file "$f" \
        > "$out"

    echo "  -> saved to $out"
done

echo "[DONE] All MRE metrics recomputed."