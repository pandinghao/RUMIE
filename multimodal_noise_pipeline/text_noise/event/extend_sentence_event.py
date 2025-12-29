extend_sentence_event_prompt = '''
{example data} The above is an Event Detection data entry, where the "sentence" contains the sentence information, and the "events" contains the event trigger information. \
Now, you need to ensure that the original sentence remains unchanged while adding semantically related content at the end. \
The additional content should not be too simplistic and not contain new event information. \
Please output in the following format, without any additional content: {"sentence": "", "events": []}, where "sentence" should be the expanded sentence and follow the original token list format, and "events" should contain the original event trigger information.
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
        for attempt in range(1, max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model="gpt-5.1-chat-latest",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": example_data +extend_sentence_event_prompt},
                            ]
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
    output_file_path = "rumie_datasets/m2e2/text_noise/extend_sentence/text_multimedia_event.json"
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
    
    