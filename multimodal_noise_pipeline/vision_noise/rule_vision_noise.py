import cv2
import json
import os
import numpy as np
from tqdm import tqdm
# ----------- 1. 噪声（高斯噪声） -----------
def add_gaussian_noise(img, mean=0, std=15):
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

# ----------- 2. 颜色偏移（色彩通道整体偏移） -----------
def apply_color_shift(img, shift_b=0, shift_g=0, shift_r=20):
    shift = np.array([shift_b, shift_g, shift_r], dtype=np.float32)
    shifted = img.astype(np.float32) + shift
    shifted = np.clip(shifted, 0, 255).astype(np.uint8)
    return shifted

# ----------- 3. 模糊处理（低分辨率） -----------
def apply_low_resolution(img, scale=0.25, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    small = cv2.resize(img, (new_w, new_h), interpolation=interpolation_down)
    lowres = cv2.resize(small, (w, h), interpolation=interpolation_up)
    return lowres

# ----------- 4. JPEG 压缩失真 -----------
def apply_jpeg_compression(img, quality=30):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encimg = cv2.imencode('.jpg', img, encode_param)
    if not success:
        raise RuntimeError("JPEG 编码失败")
    decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return decimg



# ===================== 批量处理 JSONL 中的全部图片 =====================
if __name__ == "__main__":
    
    jsonl_path = "data_process/processed_data/event/m2e2_test_ED.jsonl"

    # 你的输出目录
    output_dir = "rumie_datasets/m2e2/rule_vision_noise"
    image_folder = "datasets/m2e2/m2e2_rawdata/image/image"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir+"/gaussian_noise", exist_ok=True)
    os.makedirs(output_dir+"/jpeg_compression", exist_ok=True)
    os.makedirs(output_dir+"/low_resolusion", exist_ok=True)
    os.makedirs(output_dir+"/color_shift", exist_ok=True)
    # 读取 JSONL
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        data = json.loads(line.strip())

        # 根据 JSON 字段获取图片文件名（请确认字段名）
        image_name = data["image_id"]          # 如果字段名不同，你需要改这里
        image_path = image_folder + f"/{image_name}"

        img = cv2.imread(image_path)
        if img is None:
            print(f"找不到图片: {image_path}")
            continue

        # 应用失真
        img_gauss_noise = add_gaussian_noise(img, mean=0, std=100)
        img_jpeg = apply_jpeg_compression(img, quality=5)
        img_lowres = apply_low_resolution(img, scale=0.01)
        img_color_shift = apply_color_shift(img, shift_b=-100, shift_g=-100, shift_r=50)


        # 输出文件前缀
        base = image_name.replace(".jpg", "")

        # 保存到指定路径
        cv2.imwrite(f"{output_dir}/gaussian_noise/{image_name}", img_gauss_noise)
        cv2.imwrite(f"{output_dir}/jpeg_compression/{image_name}", img_jpeg)
        cv2.imwrite(f"{output_dir}/low_resolusion/{image_name}", img_lowres)
        cv2.imwrite(f"{output_dir}/color_shift/{image_name}", img_color_shift)

        #print(f"处理完成：{image_name}")

    print("全部图片处理完成。")