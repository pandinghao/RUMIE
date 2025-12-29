change_context_entity_prompt = '''
{example data} The above is a Named Entity Recognition data entry, where the "sentence" contains the sentence information, and the "entities" contains the entity label information. \
The sentence containing 1-4 [MASK] tokens. These [MASK] tokens mask certain words, with each [MASK] token potentially masking one or more words. \
You need to generate some challenging words to replace these [MASK] tokens to create a difficult sentence. The generated words can not contain any entity information. \
Do not change the content of the existing entities in the original text. \
Remember to update the entity offsets as well. Offsets should be represented as a list of all tokens indices, not as a [start, end] span. \
You need to provide predictions for each [MASK] token. Please output in the following format, without any additional content: {"sentence": "", "entities": []}, \
where "sentence" should be the changed sentence and follow the original token list format, and "entities" should contain the original entity label information.
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

import random
from copy import deepcopy
from typing import Set, List, Dict, Any, Optional
client = OpenAI(
                api_key="sk-C1MM4tqRF6b3sJ26bmcUZAm8xGPTAybP1fI9vH6nnfTaCxYX",
                base_url="http://35.164.11.19:3887/v1"
        )
def _iter_entities(obj: Dict[str, Any]):
    ents = obj.get("entity")
    if isinstance(ents, list):
        return ents
    ents = obj.get("entities")
    if isinstance(ents, list):
        return ents
    return []

def _protected_token_indices(obj: Dict[str, Any], tokens_len: int) -> Set[int]:
    """
    保护实体覆盖的 token 下标。
    你的 offset 格式：offset = [idx1, idx2, idx3, ...]
    """
    protected: Set[int] = set()
    for ent in _iter_entities(obj):
        off = ent.get("offset")

        #offset 是 token indices 列表（长度可能 1~N）
        if isinstance(off, list) and off and all(isinstance(x, int) for x in off):
            for idx in off:
                if 0 <= idx < tokens_len:
                    protected.add(idx)

        # 兼容：有些数据仍然可能是 [start, end]
        elif (
            isinstance(off, (list, tuple))
            and len(off) == 2
            and all(isinstance(x, int) for x in off)
        ):
            s, e = off
            if 0 <= s <= e < tokens_len:
                protected.update(range(s, e + 1))

    return protected

def mask_sample_keep_schema(
    obj: dict,
    min_masks: int = 1,
    max_masks: int = 4,
    seed: int  = 42,
    mask_token: str = "[MASK]",
    update_text: bool = False,
) -> dict:
    """
    返回与原始 obj 相同的 schema：不新增/不删除字段，只修改 tokens（可选修改 text）。
    - 不 mask entity 覆盖的 tokens（按 obj["entity"][i]["offset"] 保护）
    - 每条样本随机 mask 数量在 [min_masks, max_masks]
    """
    if "tokens" not in obj or not isinstance(obj["tokens"], list):
        raise ValueError('obj 必须包含 list 类型的 "tokens" 字段')

    rng = random.Random(seed)
    new_obj = deepcopy(obj)

    tokens = new_obj["tokens"]
    protected = _protected_token_indices(new_obj, len(tokens))

    candidates = [i for i, t in enumerate(tokens) if i not in protected and t != mask_token]
    if candidates:
        n_to_mask = rng.randint(min_masks, max_masks)
        n_to_mask = min(n_to_mask, len(candidates))
        for i in rng.sample(candidates, k=n_to_mask):
            tokens[i] = mask_token

    # 可选：如果你希望 text 与 tokens 同步（注意：你的 text 里有引号/空格/URL 等，简单 join 可能不完全一致）
    if update_text and isinstance(new_obj.get("text"), str):
        new_obj["text"] = " ".join(tokens)

    return new_obj

def convert_twitter_merge_jsonl(
    input_path: str = "/home/sda/pandinghao/RUMIE/data_process/processed_data/entity/twitter_merge/twitter_merge.jsonl",
) -> List[str]:
    seed = 42
    results = []
    with open(input_path, "r", encoding="utf-8") as f_in:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {line_no} 行不是合法 JSON：{e}") from e

            # 为保证可复现：同一个 seed 下，不同行号会产生不同 mask
            new_obj = mask_sample_keep_schema(
                obj,
                min_masks=1,
                max_masks=4,
                seed=(seed + line_no) if seed is not None else None,
                update_text=True,
            )

            results.append(new_obj)
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
                    model="gpt-5.1",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": example_data + change_context_entity_prompt},
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
    output_file_path = "rumie_datasets/twitter17/text_noise/change_context/test.jsonl"
    data = convert_twitter_merge_jsonl(input_path=input_path)
    dir_path = os.path.dirname(output_file_path)

    fix_flag = False
    # 如果目录不存在，则递归创建
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    if fix_flag:
        fix_data = convert_twitter_merge_jsonl(input_path= "")
        new_data = []
        for index,line in enumerate(fix_data):
            if line == {}:
                data[index]['line_num'] = index + 1
                new_data.append(data[index])
        output_file_path = ""
        data = new_data
    data = data
    print(f"处理前一共有{len(data)}条数据")
    thread_count =10
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
    
    