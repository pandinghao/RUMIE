#!/usr/bin/env bash
set -euo pipefail

# ====== 配置区 ======
STEP=76422
CUDA=0
MERGE_FLAG=false
MNER_FLAG=true
MRE_FLAG=true
MEE_FLAG=true

BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_DIR="LVLM_RUMIE/saves/Qwen2.5_UMIE/checkpoint-${STEP}"
MERGED_DIR="LVLM_RUMIE/merge_output/Qwen2.5_UMIE_${STEP}"

TEMPLATE="qwen2_vl"
TEMP=0.5

EVAL_PY="LVLM_RUMIE/evaluate.py"
METRIC_PY="LVLM_RUMIE/get_metric.py"

# 结果根目录
RES_ROOT="LVLM_RUMIE/results/roubust_results/Qwen2.5_UMIE_${STEP}"

# ====== dataset keys（需与你的 dataset_info.json 一致） ======
# --- MNER ---
DATASETS_MNER=(
  "mner_rule_vision_color_shift"
  "mner_rule_vision_gaussian_noise"
  "mner_rule_vision_jpeg_compression"
  "mner_rule_vision_low_resolusion"
  "mner_rule_vision_Image_Side_Contradictory_Perturbation_clip"
  "mner_text_change_context"
  "mner_text_extend_sentence"
  "mner_text_replace_entity"
  "mner_text_Text_Side_Contradictory_Perturbation"
)

# --- MRE ---
DATASETS_MRE=(
  "mre_rule_vision_color_shift"
  "mre_rule_vision_gaussian_noise"
  "mre_rule_vision_jpeg_compression"
  "mre_rule_vision_low_resolusion"
  "mre_rule_vision_Image_Side_Contradictory_Perturbation_clip"
  "mre_text_extend_sentence"
  "mre_text_replace_triple"
  "mre_text_Text_Side_Contradictory_Perturbation"
)

# --- MEE (m2e2) ---
DATASETS_MEE=(
  "mee_rule_vision_color_shift"
  "mee_rule_vision_gaussian_noise"
  "mee_rule_vision_jpeg_compression"
  "mee_rule_vision_low_resolusion"
  "mee_rule_vision_Image_Side_Contradictory_Perturbation_clip"
  "mee_text_change_context"
  "mee_text_extend_sentence"
  "mee_text_Text_Side_Contradictory_Perturbation"
)
# =====================

mkdir -p "$RES_ROOT"

run_eval () {
  local task="$1"       # mner / mre / mee
  local dataset="$2"    # dataset key
  local out_file="$3"

  echo "[EVAL] task=${task} dataset=${dataset}"
  mkdir -p "$(dirname "$out_file")"

  if [[ "$task" == "mee" ]]; then
    CUDA_VISIBLE_DEVICES=${CUDA} VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 DISABLE_VERSION_CHECK=1 python "$EVAL_PY" \
      --model_name_or_path "$MERGED_DIR" \
      --dataset "$dataset" \
      --template "$TEMPLATE" \
      --save_name "$out_file" \
      --temperature "$TEMP" \
      --skip_special_tokens true
  else
    CUDA_VISIBLE_DEVICES=${CUDA} DISABLE_VERSION_CHECK=1 python "$EVAL_PY" \
      --model_name_or_path "$MERGED_DIR" \
      --dataset "$dataset" \
      --template "$TEMPLATE" \
      --save_name "$out_file" \
      --temperature "$TEMP" \
      --skip_special_tokens true
  fi
}

run_metric () {
  local mee_file="$1"
  local mner_file="$2"
  local mre_file="$3"
  local metric_out="$4"

  echo "[METRIC] => ${metric_out}"
  python "$METRIC_PY" \
    --mee_file  "$mee_file" \
    --mner_file "$mner_file" \
    --mre_file  "$mre_file" | tee "$metric_out"
}

# ====== 1) merge model ======
if [[ "$MERGE_FLAG" == true ]]; then
  echo "[MERGE] exporting merged model to: ${MERGED_DIR}"
  DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=${CUDA} llamafactory-cli export \
    --model_name_or_path "$BASE_MODEL" \
    --adapter_name_or_path "$ADAPTER_DIR" \
    --template "$TEMPLATE" \
    --trust_remote_code true \
    --export_dir "$MERGED_DIR" \
    --export_size 5 \
    --export_device "cpu" \
    --export_legacy_format false
else
  echo "[MERGE] skipped."
fi

# ====== 2) evaluate all perturbations ======
# 我们按扰动类型组织：rule_vision_noise 和 text_noise
# 并且对每个扰动，汇总一次三任务指标（若该任务存在该扰动）

# 统一扰动列表（用于汇总）
PERTS_RULE_VISION=("color_shift" "gaussian_noise" "jpeg_compression" "low_resolusion" "Image_Side_Contradictory_Perturbation_clip")
#PERTS_RULE_VISION=()
PERTS_TEXT=("change_context" "extend_sentence" "replace_entity" "replace_triple" "Text_Side_Contradictory_Perturbation")
#PERTS_TEXT=("Text_Side_Contradictory_Perturbation")
# Helper: find dataset key from arrays
find_dataset_key () {
  local target="$1"; shift
  local arr=("$@")
  for x in "${arr[@]}"; do
    if [[ "$x" == "$target" ]]; then
      echo "$x"
      return 0
    fi
  done
  return 1
}

# --- rule_vision_noise ---
for pert in "${PERTS_RULE_VISION[@]}"; do
  echo "========== [RULE_VISION] ${pert} =========="

  mner_out=""
  mre_out=""
  mee_out=""

  if [[ "$MNER_FLAG" == true ]]; then
    key="mner_rule_vision_${pert}"
    if find_dataset_key "$key" "${DATASETS_MNER[@]}" >/dev/null; then
      mner_out="${RES_ROOT}/mner/rule_vision_noise/${pert}/result.jsonl"
      run_eval "mner" "$key" "$mner_out"
    fi
  fi

  if [[ "$MRE_FLAG" == true ]]; then
    key="mre_rule_vision_${pert}"
    if find_dataset_key "$key" "${DATASETS_MRE[@]}" >/dev/null; then
      mre_out="${RES_ROOT}/mre/rule_vision_noise/${pert}/result.jsonl"
      run_eval "mre" "$key" "$mre_out"
    fi
  fi

  if [[ "$MEE_FLAG" == true ]]; then
    key="mee_rule_vision_${pert}"
    if find_dataset_key "$key" "${DATASETS_MEE[@]}" >/dev/null; then
      mee_out="${RES_ROOT}/mee/rule_vision_noise/${pert}/result.jsonl"
      run_eval "mee" "$key" "$mee_out"
    fi
  fi

  # 如果三者都存在，则汇总一次 metric
  if [[ -n "$mner_out" && -n "$mre_out" && -n "$mee_out" ]]; then
    metric_out="${RES_ROOT}/metrics/rule_vision_noise/${pert}/metric.txt"
    mkdir -p "$(dirname "$metric_out")"
    run_metric "$mee_out" "$mner_out" "$mre_out" "$metric_out"
  else
    echo "[METRIC] skip for ${pert} (some task output missing)."
  fi
done

# --- text_noise ---
for pert in "${PERTS_TEXT[@]}"; do
  echo "========== [TEXT_NOISE] ${pert} =========="

  mner_out=""
  mre_out=""
  mee_out=""

  if [[ "$MNER_FLAG" == true ]]; then
    key="mner_text_${pert}"
    if find_dataset_key "$key" "${DATASETS_MNER[@]}" >/dev/null; then
      mner_out="${RES_ROOT}/mner/text_noise/${pert}/result.jsonl"
      run_eval "mner" "$key" "$mner_out"
    fi
  fi

  if [[ "$MRE_FLAG" == true ]]; then
    key="mre_text_${pert}"
    if find_dataset_key "$key" "${DATASETS_MRE[@]}" >/dev/null; then
      mre_out="${RES_ROOT}/mre/text_noise/${pert}/result.jsonl"
      run_eval "mre" "$key" "$mre_out"
    fi
  fi

  if [[ "$MEE_FLAG" == true ]]; then
    key="mee_text_${pert}"
    if find_dataset_key "$key" "${DATASETS_MEE[@]}" >/dev/null; then
      mee_out="${RES_ROOT}/mee/text_noise/${pert}/result.jsonl"
      run_eval "mee" "$key" "$mee_out"
    fi
  fi

  # 改动：只要至少有一个任务产出了结果，就做一次 metric 汇总
  # get_metric.py 内部会对缺失文件输出 warning 并返回 0 指标占位
  if [[ -n "$mner_out" || -n "$mre_out" || -n "$mee_out" ]]; then
    metric_out="${RES_ROOT}/metrics/text_noise/${pert}/metric.txt"
    mkdir -p "$(dirname "$metric_out")"
    run_metric "$mee_out" "$mner_out" "$mre_out" "$metric_out"
  else
    echo "[METRIC] skip for ${pert} (no task output generated)."
  fi
done