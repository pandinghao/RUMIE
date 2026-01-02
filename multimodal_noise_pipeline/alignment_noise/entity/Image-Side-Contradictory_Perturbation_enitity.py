import os
import json
import shutil
import time
import base64
from typing import Optional
from tqdm import tqdm
from PIL import Image, ImageDraw
from openai import OpenAI
import threading
import random
import math

# =========================================================
# 1. API CLIENT
# =========================================================
client = OpenAI(
    api_key="sk-C1MM4tqRF6b3sJ26bmcUZAm8xGPTAybP1fI9vH6nnfTaCxYX",
    base_url="http://35.164.11.19:3887/v1"
)

# =========================================================
# 2. PATH CONFIG
# =========================================================
IMAGE_ROOT = "datasets/twitter17_data/twitter17_images"
OUTPUT_ROOT = "rumie_datasets/twitter17/alignment_noise/Image-Side-Contradictory_Perturbation"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =========================================================
# 3. CONTEXT-AWARE LLM PROMPT
# =========================================================
CONFUSING_ENTITY_PROMPT = """
You are generating visual ambiguity for a multimodal dataset.

Given the following CONTEXT text, generate ONE named entity that:
1. Matches the specified entity type.
2. Is highly plausible in the given context.
3. Is semantically close to other possible entities that could appear.
4. Would be visually confusing if shown in an image related to the context.

Output ONLY the entity name.
Do NOT include explanations or extra text.

Entity Type: {entity_type}

Context:
{context}
"""

def preprocess_image_to_size(image_path: str, output_path: str, target_size=(1024, 1024)):
    """Convert to RGBA and resize to target_size."""
    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        img = img.resize(target_size, Image.BICUBIC)
        img.save(output_path, format="PNG")

def generate_confusing_entity_llm(entity_type: str, context: str, max_retries: int = 10) -> Optional[str]:
    """Generate confusing entity using LLM with retry mechanism."""
    prompt = CONFUSING_ENTITY_PROMPT.format(entity_type=entity_type, context=context[:800])  # 防御：避免 prompt 过长
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-5.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )
            text = resp.choices[0].message.content.strip()
            if text and len(text.split()) <= 8:
                return text
        except Exception:
            time.sleep(0.5)
    return None

# =========================================================
# 4. MASK GENERATION
# =========================================================

def generate_strict_mask_rgba(width: int, height: int, mask_path: str, box_ratio=(0.55, 0.65, 0.95, 0.9)):
    """Generate mask for the image, keeping transparent area as editable and white as non-editable."""
    mask = Image.new("RGBA", (width, height), (255, 255, 255, 255))  # Create a fully transparent mask (editable area)
    draw = ImageDraw.Draw(mask)

    x1 = int(width * box_ratio[0])
    y1 = int(height * box_ratio[1])
    x2 = int(width * box_ratio[2])
    y2 = int(height * box_ratio[3])

    # Opaque white region = non-editable (this area won't be altered)
    draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 0))  # White part (non-editable)

    # The area outside of the white part will remain transparent (editable)

    mask.save(mask_path, format="PNG")

# Example usage:
# generate_strict_mask_rgba(1024, 1024, 'mask.png')

# =========================================================
# 5. IMAGE EDIT PROMPT
# =========================================================
def build_image_edit_prompt(confuse_entity: str, entity_type: str) -> str:
    return f"""
ONLY modify the content inside the masked region. 

The area outside the mask MUST remain completely unchanged, including all people, objects, colors, lighting, and composition.

Inside the masked region, add a realistic visual element that clearly displays the text "{confuse_entity}" of type {entity_type}, such as text on a billboard, poster, sign, screen, or clothing print.

Do not alter anything outside the masked area.
"""

# =========================================================
# 6. IMAGE → IMAGE EDIT
# =========================================================
def image_edit(image_path: str, mask_path: str, prompt: str, output_path: str, max_retries: int = 10, retry_delay: float = 2.0):
    """使用遮罩和提示符编辑图片，并增加重试机制。"""
    for attempt in range(max_retries):  # 最大重试次数
        try:
            # 调用 OpenAI API 进行图片编辑
            result = client.images.edit(
                model="gpt-image-1",
                quality="low",
                image=open(image_path, "rb"),
                mask=open(mask_path, "rb"),
                prompt=prompt,
                size="1024x1024"
            )

            # 如果编辑成功，解码返回的图片字节流并保存
            img_bytes = base64.b64decode(result.data[0].b64_json)
            with open(output_path, "wb") as f:
                f.write(img_bytes)
            return  # 成功时退出函数

        except Exception as e:
            # 捕获错误并打印，记录当前是第几次重试
            print(f"编辑图片 {image_path} 失败 (第 {attempt + 1} 次尝试/{max_retries} 次重试): {e}")
            
            if attempt < max_retries - 1:
                # 如果不是最后一次重试，等待指定时间后再次尝试
                print(f"正在重试，等待 {retry_delay} 秒...")
                time.sleep(retry_delay)
            else:
                # 如果已经达到最大重试次数，打印失败信息
                print(f"达到最大重试次数，图片编辑失败: {image_path}")
                # 可以在这里记录日志，或者根据需要处理失败情况
                # 复制原图到输出路径
                shutil.copy(image_path, output_path)
                print(f"复制原始图片到输出路径: {output_path}")
    # 如果所有重试都失败，打印失败信息
    print(f"编辑图片失败，重试了 {max_retries} 次后仍未成功: {image_path}")
# =========================================================
# 7. MAIN PIPELINE (IMAGE-ONLY, CONTEXT-AWARE)
# =========================================================
def run_image_only_context_confusion(input_jsonl: str, num_threads: int):
    with open(input_jsonl, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Split the data into chunks for parallel processing
    chunk_size = math.ceil(len(data) / num_threads)
    data_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    if not os.path.exists(OUTPUT_ROOT + "/norm"):
        os.mkdir(OUTPUT_ROOT + "/norm")

    if not os.path.exists(OUTPUT_ROOT + "/mask"):
        os.mkdir(OUTPUT_ROOT + "/mask")

    def process_chunk(chunk):

        for item in tqdm(chunk):
            image_name = item.get("image_id")
            context = item["text"]
            entities = item.get("entity", [])

            # Basic check
            if not image_name or not context or not entities:
                print("缺图片 上下文或者实体:", item)
                continue

            image_path = os.path.join(IMAGE_ROOT, image_name)
            if not os.path.exists(image_path):
                print("Image not found:", image_name)
                continue

            target_entity = random.choice(entities)
            entity_type = target_entity.get("type","person")
            entity_text = target_entity.get("text","")

            #if not entity_type or not entity_text:
                #continue

            confuse_entity = generate_confusing_entity_llm(entity_type=entity_type, context=context)

            if confuse_entity is None:
                continue

            # Output paths
            norm_image_path = os.path.join(OUTPUT_ROOT+"/norm", image_name.replace(".jpg", "_norm.png"))
            mask_path = os.path.join(OUTPUT_ROOT+"/mask", image_name.replace(".jpg", "_mask.png"))
            out_path = os.path.join(OUTPUT_ROOT, image_name)

            try:
                preprocess_image_to_size(image_path=image_path, output_path=norm_image_path)
                generate_strict_mask_rgba(width=1024, height=1024, mask_path=mask_path)
                prompt = build_image_edit_prompt(confuse_entity=confuse_entity, entity_type=entity_type)
                image_edit(image_path=norm_image_path, mask_path=mask_path, prompt=prompt, output_path=out_path)
            except Exception as e:
                print("Failed:", image_name, e)

    # Use threading to process the chunks in parallel
    threads = []
    for chunk in data_chunks:
        thread = threading.Thread(target=process_chunk, args=(chunk,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

# =========================================================
# 8. ENTRY
# =========================================================
if __name__ == "__main__":
    INPUT_JSONL = "data_process/processed_data/entity/twitter17/test.jsonl"
    NUM_THREADS = 20 # You can adjust this based on your CPU cores
    run_image_only_context_confusion(input_jsonl=INPUT_JSONL, num_threads=NUM_THREADS)
