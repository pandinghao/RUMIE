import os
import json
import argparse
from typing import Optional, Any, Dict, List

import torch
from PIL import Image
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import check_version, get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer
from pathlib import Path
if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest


def _load_single_image(img_obj: Any) -> Image.Image:
    """
    将 dataset 里各种可能的 image 表示统一转换为 PIL.Image。
    支持：
      - 字符串路径
      - PIL.Image
      - bytes
    """
    if img_obj is None:
        return None

    if isinstance(img_obj, Image.Image):
        return img_obj.convert("RGB")

    if isinstance(img_obj, str):
        # 常见：dataset 里存的是图片路径
        return Image.open(img_obj).convert("RGB")

    if isinstance(img_obj, (bytes, bytearray)):
        import io
        return Image.open(io.BytesIO(img_obj)).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(img_obj)}")


def _extract_first_image(sample_images: Any) -> Optional[Image.Image]:
    """
    LLaMA-Factory 的 sample["images"] 可能是：
      - []
      - ["path1", "path2", ...]
      - "path"
      - [{"path": ...}, ...]（少见，取决于你的数据构造）
    这里尽量做鲁棒处理：只取第一张（你 engine_args 里也限制了 image:1）。
    """
    if not sample_images:
        return None

    # 如果直接就是一个路径/对象
    if isinstance(sample_images, (str, Image.Image, bytes, bytearray)):
        return _load_single_image(sample_images)

    # 如果是 list/tuple
    if isinstance(sample_images, (list, tuple)):
        first = sample_images[0]
        # 有些数据会包一层 dict
        if isinstance(first, dict):
            # 常见键名猜测：path / image / uri
            for k in ("path", "image", "uri", "file", "filename"):
                if k in first:
                    return _load_single_image(first[k])
            raise KeyError(f"Unsupported image dict keys: {list(first.keys())}")
        return _load_single_image(first)

    raise TypeError(f"Unsupported sample_images type: {type(sample_images)}")
def _resize_for_vlm(img: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = img.size
    scale = max(w, h) / max_side
    if scale > 1:
        img = img.resize((int(w/scale), int(h/scale)))
    return img

def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "./",
    template: str = "default",
    cutoff_len: int = 10000,
    max_samples: Optional[int] = None,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 500,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    seed: Optional[int] = None,
    pipeline_parallel_size: int = 1,
):
    """
    关键点：multi_modal_data["image"] 传 PIL.Image（或 list[PIL.Image]），不要传 dict。
    """
    if not is_vllm_available():
        raise RuntimeError("vLLM is not available in your environment.")

    # vLLM / LLaMA-Factory 版本约束
    check_version("vllm>=0.4.3,<=0.7.3")
    #if pipeline_parallel_size > get_device_count():
        #raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    # vLLM generate 时一般建议 False（你原来就是 False）
    template_obj.mm_plugin.expand_mm_tokens = False

    # 你的代码里强行 data_args.dataset_dir="./" 我保留你的行为，但更建议别乱改
    data_args.dataset_dir = dataset_dir
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)

    inputs: List[Dict[str, Any]] = []
    prompts, labels, raw_images = [], [], []

    for idx, sample in enumerate(dataset_module["train_dataset"]):
        if max_samples is not None and idx >= max_samples:
            break

        # prompt token ids
        prompt_ids = sample["input_ids"]
        prompts.append(tokenizer.decode(prompt_ids, skip_special_tokens=skip_special_tokens))

        # label text
        labels.append(
            tokenizer.decode(
                list(filter(lambda x: x != IGNORE_INDEX, sample["labels"])),
                skip_special_tokens=skip_special_tokens,
            )
        )

        # --- 关键修复：不要使用 _regularize_images，它会产出 dict({'images':...})，触发 vLLM embedding 分支 ---
        pil_img = _extract_first_image(sample.get("images", None))
        raw_images.append(sample.get("images", None))
        
        if pil_img is not None:
            pil_img = _resize_for_vlm(pil_img, max_side=1024)
            multi_modal_data = {"image": pil_img}  # 这里必须是 PIL.Image（或 list[PIL.Image]）
        else:
            multi_modal_data = None

        inputs.append({"prompt_token_ids": prompt_ids, "multi_modal_data": multi_modal_data})

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,
        top_k=generating_args.top_k or -1,
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
    )

    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    engine_args: Dict[str, Any] = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
        "gpu_memory_utilization": 0.9,
    }

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)
    save_path = Path(save_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_name, "w", encoding="utf-8") as f:
        results = LLM(**engine_args).generate(inputs, sampling_params, lora_request=lora_request)
        preds = [r.outputs[0].text for r in results]
        for text, pred, label, img in zip(prompts, preds, labels, raw_images):
            f.write(json.dumps(
                {"prompt": text, "predict": pred, "label": label, "image": img},
                ensure_ascii=False
            ) + "\n")

    print("*" * 70)
    print(f"{len(prompts)} generated results have been saved at {save_name}.")
    print("*" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Inference Script (fixed mm input)")
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--adapter_name_or_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="alpaca_en_demo")
    parser.add_argument('--dataset_dir', type=str, default="./")
    parser.add_argument('--template', type=str, default="default")
    parser.add_argument('--cutoff_len', type=int, default=5000)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--vllm_config', type=str, default="{}")
    parser.add_argument('--save_name', type=str, default="generated_predictions.jsonl")
    parser.add_argument('--temperature', type=float, default=0.95)
    parser.add_argument('--top_p', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--skip_special_tokens', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--pipeline_parallel_size', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    vllm_infer(
        model_name_or_path=args.model_name_or_path,
        adapter_name_or_path=args.adapter_name_or_path,
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        template=args.template,
        cutoff_len=args.cutoff_len,
        max_samples=args.max_samples,
        vllm_config=args.vllm_config,
        save_name=args.save_name,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        skip_special_tokens=args.skip_special_tokens,
        seed=args.seed,
        pipeline_parallel_size=args.pipeline_parallel_size,
    )
