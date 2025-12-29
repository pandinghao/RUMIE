replace_entity_promot ='''The above is a Named Entity Recognition data entry, where "sentence" contains the sentence information, and "entities" contains the entity label information. \
Now, based on the following rules, the entities in the sentence need to be changed: 1. Change the entity while keeping the original entity type. \
2. The changed entities should be difficult and uncommon, and the number of entity words can vary. Remember to update the offsets as well. Offsets should be represented as a list of all tokens indices, not as a [start, end] span. 3. Only change the entity content, do not change other content. \
4. Changed entities should be updated in both "sentence" and "entities". Please output in the following format and do not output any extra content. {"sentence": "", "entities": []}
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

client = OpenAI(
                api_key="sk-cE0PRjWuchJZ3XfXB6Be163cA1574d91B0598c6037D5Be2a",
                base_url="https://api.sttai.cc/v1"
        )
    
def convert_twitter_merge_jsonl(
    input_path: str = "/home/sda/pandinghao/RUMIE/data_process/processed_data/entity/twitter_merge/twitter_merge.jsonl",
) -> List[str]:
    """
    读取 twitter_merge.jsonl，转换为每条样本一个字符串：
        '{"sentence": "...", "entities": [...]}'
    最终返回所有字符串组成的 list[str]。

    说明：
    - sentence: 使用原始字段 text（如果不存在则尽量回退到 tokens 拼接）
    - entities: 从 entity 字段抽取，输出为:
        [{"text": "...", "type": "...", "offset": [start, end]}, ...]
      若某条没有 entity 字段，则给空列表。
    - output_path: 若提供，则将转换结果逐行写入一个新的 jsonl 文件（每行一个字符串对应的 JSON 对象）。
    """
    results: List[str] = []

    with open(input_path, "r", encoding="utf-8") as f_in:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {line_no} 行不是合法 JSON：{e}") from e

            # sentence
            if isinstance(obj.get("text"), str):
                sentence = obj["text"]
            elif isinstance(obj.get("tokens"), list):
                sentence = " ".join(str(t) for t in obj["tokens"])
            else:
                sentence = ""

        
            #obj["sentence"] = sentence
            #obj["entities"] = entities
            new_obj = obj

            # 每条样本放到一个字符串里
            results.append(new_obj)
            #results.append(json.dumps(new_obj, ensure_ascii=False))

    return results



def get_cot_data_from_api(data):

    all_new_data = []
    
    for entry in tqdm(data):
        new_data = {}
        
        example_data = json.dumps({"sentence": entry["tokens"], "entities": entry["entity"]}, ensure_ascii=False)
        new_entry = entry.copy()
        # 构建请求
        max_retries = 10 
        base_sleep = 0.5 
        result_flag = True
        for attempt in range(1, max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model="gpt-5-chat-2025-08-07",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": example_data + replace_entity_promot },
                            ]
                        }
                    ],
                    temperature=0.5,
                    logprobs=True
                )
                response = completion.choices[0].message.content
                json_response = json.loads(response)
                for index, ent in enumerate(json_response['entities']):
                    if len(json_response['entities'][index]['offset']) > 1:
                        assert "".join(json_response['sentence'][json_response['entities'][index]['offset'][0]:json_response['entities'][index]['offset'][-1]+1]).lower().replace(" ","") == json_response['entities'][index]['text'].lower().replace(" ","")
                    else:
                        assert "".join(json_response['sentence'][json_response['entities'][index]['offset'][0]:json_response['entities'][index]['offset'][0]+1]).lower().replace(" ","") == json_response['entities'][index]['text'].lower().replace(" ","")
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
            new_entry['tokens']  = json_response['sentence']
            new_entry['entity'] = json_response['entities']
            new_entry['text'] = " ".join(json_response['sentence'])
            #new_data["predict"] = response.replace("\n", "").replace("``json","").replace("[    [","[[").replace("[   [","[[").replace("[  [","[[").strip("```")
            #new_data['prompt'] = schema_Instruction + text_content
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
    input_path = "/home/sda/pandinghao/RUMIE/data_process/processed_data/entity/twitter17/test.jsonl"
    output_file_path = "rumie_datasets/twitter17/text_noise/replace_entity/test.jsonl"
    data = convert_twitter_merge_jsonl(input_path=input_path)
    dir_path = os.path.dirname(output_file_path)

    # 如果目录不存在，则递归创建
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    data = data
    print(f"处理前一共有{len(data)}条数据")
    thread_count = 10
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
        for new_data in all_data:
            output_f.write(json.dumps(new_data,ensure_ascii=False)+ '\n')
    
    