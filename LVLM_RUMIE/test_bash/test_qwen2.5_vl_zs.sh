CUDA_VISIBLE_DEVICES=0 DISABLE_VERSION_CHECK=1 python MH-ZS-JMERE/evaluate.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset zs_JMERE_V2_test \
    --template qwen2_vl \
    --save_name MH-ZS-JMERE/zs_results/qwen2.5vl_result/test_result.jsonl \
    --temperature 0.8
CUDA_VISIBLE_DEVICES=1 DISABLE_VERSION_CHECK=1 python MH-ZS-JMERE/get_metric.py \
    --result_file_path MH-ZS-JMERE/zs_results/qwen2.5vl_result/test_result.jsonl \
