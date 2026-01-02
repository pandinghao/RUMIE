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
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

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
IMAGE_ROOT = "datasets/twitter15_data/twitter15_images"
OUTPUT_ROOT = "rumie_datasets/twitter15/alignment_noise/Image-Side-Contradictory_Perturbation_clip"

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

# =========================================================
# 4. Load CLIP Model for Text-to-Image Similarity Calculation
# =========================================================
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def calculate_text_image_similarity(text: str, image: Image.Image) -> float:
    """Calculate similarity between text and image region using CLIP."""
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
    similarity = torch.cosine_similarity(image_features, text_features)
    return similarity.item()

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
    return ""

# =========================================================
# 4. MASK GENERATION BASED ON ENTITY SIMILARITY
# =========================================================

def generate_dynamic_mask(image_path: str, text: str, mask_path: str, box_ratio=(0.55, 0.65, 0.95, 0.9)):
    """Generate mask for the image, making transparent the area with highest similarity to the entity text."""
    image = Image.open(image_path)
    width, height = image.size
    best_similarity = -1
    best_box = None

    # Try different regions to find the best match
    regions = [
        (int(width * box_ratio[0]), int(height * box_ratio[1]), int(width * box_ratio[2]), int(height * box_ratio[3])),
        (0, 0, width // 2, height // 2),
        (width // 2, 0, width, height // 2),
        (0, height // 2, width // 2, height),
        (width // 2, height // 2, width, height)
    ]

    for region in regions:
        region_img = image.crop(region)
        similarity = calculate_text_image_similarity(text, region_img)

        if similarity > best_similarity:
            best_similarity = similarity
            best_box = region

    # Create the mask: transparent area for the selected region
    mask = Image.new("RGBA", (width, height), (255, 255, 255, 255))  # Fully transparent
    draw = ImageDraw.Draw(mask)
    
    if best_box:
        draw.rectangle(best_box, fill=(0, 0, 0, 0))  # Transparent region
    
    mask.save(mask_path, format="PNG")

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
    """Use the mask and prompt to edit the image with retries."""
    for attempt in range(max_retries):  # Max retries
        try:
            # Call OpenAI API to edit the image
            result = client.images.edit(
                model="gpt-image-1",
                quality="low",
                image=open(image_path, "rb"),
                mask=open(mask_path, "rb"),
                prompt=prompt,
                size="1024x1024"
            )

            # If successful, decode and save the image
            img_bytes = base64.b64decode(result.data[0].b64_json)
            with open(output_path, "wb") as f:
                f.write(img_bytes)
            return  # Exit if successful

        except Exception as e:
            print(f"Image edit failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                shutil.copy(image_path, output_path)  # If max retries failed, copy the original image
                print(f"Copied original image to output path: {output_path}")

# =========================================================
# 7. MAIN PIPELINE (IMAGE-ONLY, CONTEXT-AWARE)
# =========================================================
def run_image_only_context_confusion(input_jsonl: str, num_threads: int):
    with open(input_jsonl, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

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

            if not image_name or not context or not entities:
                continue

            image_path = os.path.join(IMAGE_ROOT, image_name)
            if not os.path.exists(image_path):
                continue

            target_entity = random.choice(entities)
            entity_type = target_entity.get("type", "person")
            entity_text = target_entity.get("text", "")

            confuse_entity = generate_confusing_entity_llm(entity_type=entity_type, context=context)

            if confuse_entity is None:
                continue

            norm_image_path = os.path.join(OUTPUT_ROOT + "/norm", image_name.replace(".jpg", "_norm.png"))
            mask_path = os.path.join(OUTPUT_ROOT + "/mask", image_name.replace(".jpg", "_mask.png"))
            out_path = os.path.join(OUTPUT_ROOT, image_name)

            try:
                preprocess_image_to_size(image_path=image_path, output_path=norm_image_path)
                generate_dynamic_mask(image_path=norm_image_path, text=confuse_entity, mask_path=mask_path)
                prompt = build_image_edit_prompt(confuse_entity=confuse_entity, entity_type=entity_type)
                image_edit(image_path=norm_image_path, mask_path=mask_path, prompt=prompt, output_path=out_path)
            except Exception as e:
                print("Failed:", image_name, e)

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
    INPUT_JSONL = "data_process/processed_data/entity/twitter15/test.jsonl"
    NUM_THREADS = 10
    run_image_only_context_confusion(input_jsonl=INPUT_JSONL, num_threads=NUM_THREADS)
