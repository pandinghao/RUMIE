#!/usr/bin/env bash
set -euo pipefail

# ================= 配置区（按需修改） =================
PY_SCRIPT="LVLM_test_datasets/convert_mner.py"   # 你的Python脚本路径（就是你贴的那份）
OUT_BASE_DIR="LVLM_test_datasets/twitter15"

# 原始图片目录（text_noise 用）
ORIG_IMAGE_DIR="datasets/twitter15_data/twitter15_images"

# text_noise 输入根目录：每个子目录一个扰动
TEXT_NOISE_BASE="rumie_datasets/twitter15/text_noise"
TEXT_INPUT_BASENAME="test.jsonl"

# vision_noise 图片根目录：每个子目录一个扰动
# 注意：这里我按你前面 MNRE 的命名写成 rule_vision_noise；
# 若你的 Twitter17 视觉扰动目录名不同（如 vision_noise），改这里即可。
VISION_NOISE_BASE="rumie_datasets/twitter15/rule_vision_noise"

# vision_noise 的输入文本：通常用干净测试集（不带 text_noise）
# 你需要把它改成你真实的干净 test.jsonl 路径
CLEAN_INPUT_FILE="data_process/processed_data/entity/twitter15/test.jsonl"
# 若不想自动扫描目录，可手动指定扰动类型数组；留空则自动扫描
TEXT_PERTS=()     # e.g., ("change_context" "swap_entity")
VISION_PERTS=()   # e.g., ("color_shift" "gaussian_noise")
# =====================================================


# ---------- 基本检查 ----------
if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] PY_SCRIPT not found: $PY_SCRIPT"
  exit 1
fi
if [[ ! -d "$ORIG_IMAGE_DIR" ]]; then
  echo "[ERROR] ORIG_IMAGE_DIR not found: $ORIG_IMAGE_DIR"
  exit 1
fi
if [[ ! -d "$TEXT_NOISE_BASE" ]]; then
  echo "[ERROR] TEXT_NOISE_BASE not found: $TEXT_NOISE_BASE"
  exit 1
fi
if [[ ! -d "$VISION_NOISE_BASE" ]]; then
  echo "[ERROR] VISION_NOISE_BASE not found: $VISION_NOISE_BASE"
  exit 1
fi
if [[ ! -f "$CLEAN_INPUT_FILE" ]]; then
  echo "[ERROR] CLEAN_INPUT_FILE not found: $CLEAN_INPUT_FILE"
  exit 1
fi


# ---------- 自动收集扰动目录 ----------
if [[ ${#TEXT_PERTS[@]} -eq 0 ]]; then
  mapfile -t TEXT_PERTS < <(find "$TEXT_NOISE_BASE" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
fi
if [[ ${#VISION_PERTS[@]} -eq 0 ]]; then
  mapfile -t VISION_PERTS < <(find "$VISION_NOISE_BASE" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
fi

echo "========== [1/2] Text-noise conversions =========="
echo "[INFO] Text perturbations: ${TEXT_PERTS[*]}"
for pert in "${TEXT_PERTS[@]}"; do
  in_file="${TEXT_NOISE_BASE}/${pert}/${TEXT_INPUT_BASENAME}"
  if [[ ! -f "$in_file" ]]; then
    echo "[WARN] Skip (input missing): $in_file"
    continue
  fi

  out_dir="${OUT_BASE_DIR}/text_noise/${pert}"
  out_file="${out_dir}/test.json"
  mkdir -p "$out_dir"

  echo "[INFO] text_noise=${pert}"
  python "$PY_SCRIPT" \
    --input_file "$in_file" \
    --output_file "$out_file" \
    --image_folder "$ORIG_IMAGE_DIR"
done


echo "========== [2/2] Vision-noise conversions =========="
echo "[INFO] Vision perturbations: ${VISION_PERTS[*]}"
for pert in "${VISION_PERTS[@]}"; do
  img_dir="${VISION_NOISE_BASE}/${pert}"
  if [[ ! -d "$img_dir" ]]; then
    echo "[WARN] Skip (image folder missing): $img_dir"
    continue
  fi

  out_dir="${OUT_BASE_DIR}/rule_vision_noise/${pert}"
  out_file="${out_dir}/test.json"
  mkdir -p "$out_dir"

  echo "[INFO] vision_noise=${pert}"
  python "$PY_SCRIPT" \
    --input_file "$CLEAN_INPUT_FILE" \
    --output_file "$out_file" \
    --image_folder "$img_dir"
done

echo "[DONE] All outputs written under: $OUT_BASE_DIR"