
#!/home/sda/pandinghao/anaconda3/envs/vlm-r1
STEP=114633
MERGE_FLAG=false
CUDA=0
MNER_FLAG=false
MRE_FLAG=false
MEE_FLAG=false

if [ "$MERGE_FLAG" = true ]; then
    DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=${CUDA} llamafactory-cli export \
        --model_name_or_path "Qwen/Qwen3-VL-4B-Instruct" \
        --adapter_name_or_path "LVLM_RUMIE/saves/Qwen3_VL_4B_UMIE/checkpoint-${STEP}" \
        --template "qwen3_vl" \
        --trust_remote_code true \
        --export_dir "LVLM_RUMIE/merge_output/Qwen3_VL_4B_UMIE_${STEP}" \
        --export_size 5 \
        --export_device "cpu" \
        --export_legacy_format false
else
    echo "跳过模型合并..."
fi

if [ "$MNER_FLAG" = true ]; then
    CUDA_VISIBLE_DEVICES=${CUDA} DISABLE_VERSION_CHECK=1 python LVLM_RUMIE/evaluate.py \
        --model_name_or_path LVLM_RUMIE/merge_output/Qwen3_VL_4B_UMIE_${STEP} \
        --dataset mner_test \
        --template qwen3_vl \
        --save_name LVLM_RUMIE/results/ori_mner_results/Qwen3_VL_4B_UMIE_${STEP}_result.jsonl \
        --temperature 0.3 \
        --skip_special_tokens true 
else
    echo "跳过MNER评估..."
fi

if [ "$MRE_FLAG" = true ]; then
    CUDA_VISIBLE_DEVICES=${CUDA} DISABLE_VERSION_CHECK=1 python LVLM_RUMIE/evaluate.py \
        --model_name_or_path LVLM_RUMIE/merge_output/Qwen3_VL_4B_UMIE_${STEP} \
        --dataset mre_test \
        --template qwen3_vl \
        --save_name LVLM_RUMIE/results/ori_mre_results/Qwen3_VL_4B_UMIE_${STEP}_result.jsonl \
        --temperature 0.5 \
        --skip_special_tokens true 
else
    echo "跳过MRE评估..."
fi

if [ "$MEE_FLAG" = true ]; then
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1  DISABLE_VERSION_CHECK=1 python LVLM_RUMIE/evaluate.py \
        --model_name_or_path LVLM_RUMIE/merge_output/Qwen3_VL_4B_UMIE_${STEP} \
        --dataset mee_test \
        --template qwen3_vl \
        --save_name LVLM_RUMIE/results/ori_mee_results/Qwen3_VL_4B_UMIE_${STEP}_result.jsonl \
        --temperature 0.5 \
        --skip_special_tokens true 
else
    echo "跳过MEE评估..."
fi
python LVLM_RUMIE/get_metric.py \
    --mee_file  LVLM_RUMIE/results/ori_mee_results/Qwen3_VL_4B_UMIE_${STEP}_result.jsonl \
    --mner_file LVLM_RUMIE/results/ori_mner_results/Qwen3_VL_4B_UMIE_${STEP}_result.jsonl \
    --mre_file  LVLM_RUMIE/results/ori_mre_results/Qwen3_VL_4B_UMIE_${STEP}_result.jsonl