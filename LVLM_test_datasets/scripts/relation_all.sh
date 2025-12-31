#!/usr/bin/env bash
set -euo pipefail

# ====== 配置区：按需改 ======
PY_SCRIPT="LVLM_test_datasets/convert_mnre.py"   # 你的Python argparse脚本路径
OUT_BASE_DIR="LVLM_test_datasets/MNRE-V2"

# 原始图片目录（给 text_noise 用）
ORIG_IMAGE_DIR="datasets/MNRE-V2/mnre_image/img_org/test"

# 视觉扰动目录（MNRE-V2）
VISION_NOISE_BASE="rumie_datasets/MNRE-V2/rule_vision_noise"
VISION_INPUT_FILE="data_process/processed_data/relation/MNRE-V2/test.fixed.jsonl"   # 你之前用的那个
VISION_PERTS=("color_shift" "gaussian_noise" "jpeg_compression" "low_resolution")

# 文本扰动目录（MNRE-V2）
TEXT_NOISE_BASE="rumie_datasets/MNRE-V2/text_noise"
TEXT_PERTS=("extend_sentence" "replace_triple")

# 假设每个 text_noise 子目录里都有这个文件名；如不一致请改这里
TEXT_INPUT_BASENAME="test.fixed.jsonl"
# ===========================


echo "========== [1/2] Vision noise conversions =========="
for pert in "${VISION_PERTS[@]}"; do
  img_dir="${VISION_NOISE_BASE}/${pert}"

  # 兼容 low_resolusion 拼写
  if [[ ! -d "$img_dir" && "$pert" == "low_resolution" ]]; then
    if [[ -d "${VISION_NOISE_BASE}/low_resolusion" ]]; then
      img_dir="${VISION_NOISE_BASE}/low_resolusion"
      pert="low_resolusion"
    fi
  fi

  if [[ ! -d "$img_dir" ]]; then
    echo "[ERROR] Vision image folder not found: $img_dir"
    exit 1
  fi

  out_dir="${OUT_BASE_DIR}/rule_vision_noise/${pert}"
  out_file="${out_dir}/test.json"
  mkdir -p "$out_dir"

  echo "[INFO] Vision pert=${pert}"
  python "$PY_SCRIPT" \
    --input_file "$VISION_INPUT_FILE" \
    --output_file "$out_file" \
    --image_folder "$img_dir"
done


echo "========== [2/2] Text noise conversions =========="
for pert in "${TEXT_PERTS[@]}"; do
  in_file="${TEXT_NOISE_BASE}/${pert}/test.jsonl"

  if [[ ! -f "$in_file" ]]; then
    echo "[ERROR] Text noise input file not found: $in_file"
    exit 1
  fi

  if [[ ! -d "$ORIG_IMAGE_DIR" ]]; then
    echo "[ERROR] Original image folder not found: $ORIG_IMAGE_DIR"
    exit 1
  fi

  out_dir="${OUT_BASE_DIR}/text_noise/${pert}"
  out_file="${out_dir}/test.json"
  mkdir -p "$out_dir"

  echo "[INFO] Text pert=${pert}"
  python "$PY_SCRIPT" \
    --input_file "$in_file" \
    --output_file "$out_file" \
    --image_folder "$ORIG_IMAGE_DIR"
done

echo "[DONE] All outputs written under: ${OUT_BASE_DIR}"