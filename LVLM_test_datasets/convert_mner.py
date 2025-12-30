import json
import os
import cv2
# 读取原始数据文件
input_file = 'data_process/processed_data/entity/twitter15/test.jsonl'
output_file = 'LVLM_test_datasets/twitter15/test.json'
image_forder = 'datasets/twitter15_data/twitter15_images'
# 确保输出文件夹存在
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
def process_data(input_file, output_file):
    processed_data = []  # 用于存储所有处理后的数据
    
    with open(input_file, 'r') as infile:
        for line in infile:
            data = json.loads(line.strip())

            # 提取用户的指令
            instruction = "Please extract the following entity type: person, location, miscellaneous, organization."
            
            # 从原始数据提取所需的字段
            text = data.get('text', '')
            entities = data.get('entity', [])
            image_id = data.get('image_id', '')
           
            image_path = f"{image_forder}/{image_id}"
            '''
            img = cv2.imread(image_path)
            if img is None:
                print(f"找不到图片: {image_path}")
                continue
            '''
            # 准备 messages
            messages = [
                {
                    "content": f"{instruction} <image> text: {text}",
                    "role": "user"
                }
            ]
            
            # 构建实体关系部分
            extracted_entities = []
            for entity in entities:
                entity_type = entity['type']
                entity_text = entity['text']
                if entity_type in ["person", "location", "miscellaneous", "organization"]:
                    extracted_entities.append(f"{entity_type}, {entity_text}")
            
            # 准备 assistant 的回复，实体之间用分号隔开
            response = "; ".join(extracted_entities)
            
           
            # 添加 assistant 的回复
            messages.append({
                "content": response,
                "role": "assistant"
            })
            
            # 添加图片信息
            if image_id:
                images = [f"{image_forder}/{image_id}"]
            else:
                images = []
            
            # 最终构建的数据格式
            result = {
                "messages": messages,
                "images": images
            }
            
            # 将处理后的数据添加到列表中
            processed_data.append(result)

    # 将所有处理后的数据写入一个单一的 JSON 文件
    with open(output_file, 'w') as outfile:
        json.dump(processed_data, outfile, ensure_ascii=False, indent=4)

# 调用函数处理数据
process_data(input_file, output_file)

