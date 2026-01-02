
multimodal_decouple_text_prompt = '''The above is a Multimodal Event Detection data entry, where "sentence" contains the textual description, 
"events" contains the event trigger information, and an image is implicitly associated with the sentence.
Now, you need to keep the original sentence unchanged, while adding additional textual content that is 
plausible but weakly aligned or potentially inconsistent with the visual scene at the end.
The added content should:
1. Not introduce new event triggers or event types.
2. Not contradict the original event explicitly.
3. Increase cross-modal ambiguity.
Please output in the following format, without any additional content:
{"sentence": "", "events": []}, where "sentence" should be the expanded sentence and follow the original token list format, and "events" should contain the original event trigger information.
'''

import openai
import json
from openai import OpenAI
import base64
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import math
import os
import time
import ast
from typing import Dict, List, Tuple, Any
client = OpenAI(
                api_key="sk-C1MM4tqRF6b3sJ26bmcUZAm8xGPTAybP1fI9vH6nnfTaCxYX",
                base_url="http://35.164.11.19:3887/v1"
        )


import mimetypes

IMAGE_ROOT = "datasets/m2e2/m2e2_rawdata/image/image"
from PIL import Image
import io
import math

def _file_to_data_url_low_quality(
    path: str,
    max_side: int = 768,
    max_pixels: int = 768 * 768,
    jpeg_quality: int = 45
) -> str:
    """
    Resize + JPEG compress to reduce payload/compute, then return data URL.
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size

        # 先按 max_side 缩放
        scale_side = min(1.0, max_side / max(w, h))
        # 再按 max_pixels 缩放
        scale_px = min(1.0, math.sqrt(max_pixels / (w * h)))
        scale = min(scale_side, scale_px)

        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.BICUBIC)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{b64}"

def build_multimodal_user_content(example_text: str, image_filenames: Optional[List[str]] = None, max_images: int = 4) -> List[Dict[str, Any]]:
    """
    Build OpenAI chat content list: text + up to max_images images.
    """
    content: List[Dict[str, Any]] = [{"type": "text", "text": example_text}]
    if not image_filenames:
        return content

    # 防御：确保是 list
    if not isinstance(image_filenames, list):
        image_filenames = [str(image_filenames)]

    for fn in image_filenames[:max_images]:
        img_path = os.path.join(IMAGE_ROOT, fn)
        if not os.path.exists(img_path):
            # 如果数据里可能是相对路径/带子目录，也尝试直接 join
            # 这里选择跳过不存在的图片，避免整个样本失败
            continue
        try:
            data_url = _file_to_data_url_low_quality(
                img_path,
                max_side=768,          # 可调：512/768/1024
                max_pixels=768*768,    # 可调：512*512 更省
                jpeg_quality=45        # 可调：30-60，越低越省但越糊
            )
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        except Exception:
            # 编码失败则跳过该图片
            continue

    return content

def load_crossmedia_coref(coref_path: str) -> List[Tuple[str, str, str]]:
    rows = []
    with open(coref_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            sent_id, img_file, ev_type = parts[0], parts[1], parts[2]
            rows.append((sent_id, img_file, ev_type))
    return rows

def convert_event_jsonl(
    input_path: str = "",
) -> List[str]:
   
    results: List[str] = []
    rows = load_crossmedia_coref(coref_path="datasets/m2e2/data/m2e2_annotations/crossmedia_coref.txt")
    new_obj = []
    with open(input_path, "r", encoding="utf-8") as f_in:
        #event_count = 0
        data = json.load(f_in)
        for line_no, obj in enumerate(data):
            event_label = obj['golden-event-mentions']
            #for event in event_label:
                #event_count += 1
            results.append(obj)
            #results.append(json.dumps(new_obj, ensure_ascii=False))

    return results



def get_cot_data_from_api(data):

    all_new_data = []
    
    for entry in tqdm(data):
        new_data = {}
        
        example_data = json.dumps({"sentence": entry["words"], "events": entry["golden-event-mentions"]}, ensure_ascii=False)
        new_entry = entry.copy()
        # 构建请求
        original_event = entry["golden-event-mentions"]
        max_retries = 10 
        base_sleep = 0.5 
        result_flag = True

        user_text = example_data + multimodal_decouple_text_prompt
        user_content = build_multimodal_user_content(
            example_text=user_text,
            image_filenames=entry.get("image", []),  # 关键：从 obj 里读 "image"
            max_images=4
        )

        for attempt in range(1, max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": user_content
                        }
                    ],
                    temperature=0.5,
                    logprobs=True
                )
                response = completion.choices[0].message.content
                json_response = json.loads(response)
                for index, event in enumerate(json_response['events']):
                    trigger = event['trigger']
                    start = trigger['start']
                    end = trigger['end']
                    event_type = event['event_type']
                    assert "".join(json_response['sentence'][start:end]).lower().replace(" ","") == trigger['text'].lower().replace(" ","")
                    assert event_type == original_event[index]['event_type']
                break  # 成功就退出循环
            except Exception as e:
                if attempt < max_retries:
                    print(f"尝试第 {attempt + 1} 次（上次失败原因：{e}）")
                    time.sleep(base_sleep * attempt)  # 简单退避：0.5s, 1.0s, 1.5s...
                else:
                    print("没得到结果")
                    response = ""
                    json_response = None
                    result_flag = False
        
        response = completion.choices[0].message.content
        

        #entry["think_from_gpt"] = response
        #new_data.append(entry)
        if not result_flag:
            all_new_data.append({})
        else:
            new_entry['words']  = json_response['sentence']
            new_entry['events'] = json_response['events']
            new_entry['sentence'] = " ".join(json_response['sentence'])
            all_new_data.append(new_entry)
    return all_new_data

   
import threading
class MyThread(threading.Thread):  
    def __init__(self, func, args=()):  
        super(MyThread, self).__init__()  
        self.func = func  
        self.args = args  
  
    def run(self):  
        self.result = self.func(self.args)  # 在执行函数的同时，把结果赋值给result,  
        # 然后通过get_result函数获取返回的结果  
  
    def get_result(self):  
        return self.result  


if __name__ == "__main__":

    #file_path = "data/JMERE/V2/train.json"
    #output_file_path = "MH-ZS-JMERE/gpt-zero-result_V2/train_distill.jsonl"
    input_path = "datasets/m2e2/data/m2e2_annotations/text_multimedia_event.json"
    output_file_path = "rumie_datasets/m2e2/alignment_noise/multimodal_decouple_text/text_multimedia_event.json"
    data = convert_event_jsonl(input_path=input_path)
    dir_path = os.path.dirname(output_file_path)

    fix_flag = False
    # 如果目录不存在，则递归创建
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    if fix_flag:
        fix_data =  convert_event_jsonl(input_path= "")
        new_data = []
        for index,line in enumerate(fix_data):
            if line == {}:
                data[index]['line_num'] = index + 1
                new_data.append(data[index])
        output_file_path = ""
        data = new_data
    data = data
    print(f"处理前一共有{len(data)}条数据")
    thread_count =8
    threads = []
    all_data = []
    for i in range(thread_count):
        t = MyThread(func=get_cot_data_from_api, args=data[int(i*len(data)/thread_count):int((i+1)*len(data)/thread_count)])
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
        try:
            all_data.extend(t.get_result())  
        except:
            continue


    print(f"处理前一共有{len(data)}条数据")
    print(f"处理完一共有{len(all_data)}条数据")
    with open(output_file_path, "w", encoding="utf-8") as output_f:

        json.dump(all_data,output_f,ensure_ascii=False,indent=4)
    
    