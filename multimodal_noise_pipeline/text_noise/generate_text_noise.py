replace_entity_promot ='''
{example data} The above is a Named Entity Recognition data entry, where "sentence" contains the sentence information, and "entities" contains the entity label information. \
Now, based on the following rules, the entities in the sentence need to be changed: 1. Change the entity while keeping the original entity type. \
2. The changed entities should be difficult and uncommon, and the number of entity words can vary. 3. Only change the entity content, do not change other content. \
4. Changed entities should be updated in both "sentence" and "entities". Please output in the following format and do not output any extra content. {"sentence": "", "entities": []}
'''

replace_triple_prompt = '''{example data} The above is a Relation Extraction data entry, where "sentence" contains the sentence information, and "relations" contains the relational triple information. \
Now, based on the following rules, the head and tail entities in the relational triples need to be changed: \
1. Change the head and tail entities while keeping the original entity type. 2. The changed entity should be significantly different from the original entity and can vary in length. \
3. Only change the head and tail entities, do not change other content. 4. Changed content should be updated in both "sentence" and "relations". \
5. After replacement, ensure the semantics remain consistent: the sentence and the relation triples must still be logically correct, coherent, and readable. \
Please output in the following format and do not output any extra content. {"sentence": "", "relations": []}'''


replace_Trigger_prompt = '''{example data} The above is a Event Detection data entry, where "sentence" contains the sentence information, and "events" contains the event trigger information. \
Now, based on the following rules, the event triggers in the sentence need to be changed: \
1. Change the trigger while keeping the original event type. 2. The changed trigger should be significantly different from the original trigger and can vary in length. \
3. Only change the trigger content, do not change other content. 4. Changed triggers should be updated in both "sentence" and "events". Please output in the following format and do not output any extra content. \
{"sentence": "", "events": []}'''


change_context_entity_prompt = '''
{example data} The above is a Named Entity Recognition data entry, where the "sentence" contains the sentence information, and the "entities" contains the entity label information. \
The sentence containing 1-4 [MASK] tokens. These [MASK] tokens mask certain words, with each [MASK] token potentially masking one or more words. \
You need to generate some challenging words to replace these [MASK] tokens to create a difficult sentence. The generated words can not contain any entity information. \
You need to provide 3 predictions for each [MASK] token. Please output in the following format, without any additional content: [MASK]1: 1: " ", 2: " ", 3: " " ...... [MASK]n: 1: " ", 2: " ", 3: " "
'''
change_context_event_prompt = '''
{example data} The above is an Event Detection data entry, where the "sentence" contains the sentence information, and the "events" contains the event trigger information.\
The sentence containing 1-4 [MASK] tokens. These [MASK] tokens mask certain words, with each [MASK] token potentially masking one or more words. \
You need to generate some challenging words to replace these [MASK] tokens to create a difficult sentence. \
The generated words can not contain any event triger information. You need to provide 3 predictions for each [MASK] token. \
Please output in the following format, without any additional content: [MASK]1: 1: " ", 2: " ", 3: " " ...... [MASK]n: 1: " ", 2: " ", 3: " "
'''

extend_sentence_entity_prompt = '''
{example data} The above is a Named Entity Recognition data entry, where the "sentence" contains the sentence information, and the "entities" contains the entity label information. \
Now, you need to ensure that the original sentence remains unchanged while adding semantically related content at the end of the sentence. \
The additional content should not be too simplistic and not not introduce new entity information. \
Please output in the following format, without any additional content: {"sentence": "", "entities": []}, where "sentence" should be the expanded sentence, and "entities" should contain the original entity label information.
'''


extend_sentence_relation_prompt = '''
{example data} The above is a Relation Extraction data entry, where the "sentence" contains the sentence information, and the "relations" contains the relational triple information. \
Now, you need to ensure that the original sentence remains unchanged while adding semantically related content at the end or beginning of the sentence, with a preference for the end. \
The additional content should not be too simplistic, and should not introduce new relational triple information. \
Please output in the following format, without any additional content: {"sentence": "", "relations": []}, where "sentence" should be the expanded sentence, and "relations" should contain the original relational triple information.
'''

extend_sentence_event_prompt = '''
{example data} The above is an Event Detection data entry, where the "sentence" contains the sentence information, and the "events" contains the event trigger information. \
Now, you need to ensure that the original sentence remains unchanged while adding semantically related content at the end or beginning of the sentence, with a preference for the end. \
The additional content should not be too simplistic and not contain new event information. \
Please output in the following format, without any additional content: {"sentence": "", "events": []}, where "sentence" should be the expanded sentence, and "events" should contain the original event trigger information.
'''



import openai
import json
from openai import OpenAI
import base64
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import math

generate_instruction = replace_entity_promot


def get_cot_data_from_api(data):

    all_new_data = []
    
    for entry in tqdm(data):
        new_data = {}
        image_path = f"data/JMERE_img/train/{entry['img_id']}"
    #image_path = f"data/JMERE_img/train/{data[0]['img_id']}"
        text_content = " ".join(entry['token'])
    #text_content = " ".join(data[0]['token'])
    # 加载图像文件
        
        relations = []
        for relation_entry in entry['label_list']:
            beg_ent = relation_entry[0]['beg_ent']
            sec_ent = relation_entry[0]['sec_ent']
            relation = relation_entry[0]['relation']    
            relations.append([beg_ent['name'], beg_ent['tags'], sec_ent['name'], sec_ent['tags'], relation])
        new_data['label'] = json.dumps(relations)
        new_data['image'] = image_path
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        # 构建请求
        try:
            completion = client.chat.completions.create(
                model="gpt-5-chat-2025-08-07",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": schema_Instruction + text_content },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64," + encoded_image,
                                    "detail": "low"  # 可选值："low" 或 "high"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.8,
                logprobs=True
            )
        except:
            print("尝试第二次")
            try:
                completion = client.chat.completions.create(
                    model="gpt-5-chat-2025-08-07",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": 
                                  schema_Instruction + text_content},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/jpeg;base64," + encoded_image,
                                        "detail": "low"  # 可选值："low" 或 "high"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.8,
                )
            except:
                try:
                    print("尝试第三次")
                    completion = client.chat.completions.create(
                        model="gpt-5-chat-2025-08-07",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text":  schema_Instruction + text_content},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:image/jpeg;base64," + encoded_image,
                                            "detail": "low"  # 可选值："low" 或 "high"
                                        }
                                    }
                                ]
                            }
                        ],
                        temperature=0.8,
                    )
                except:
                    print("没得到结果")
                    response = ""



        try:
            response = completion.choices[0].message.content
        except:
            response = ""
        lp = extract_logprobs_from_openai_response(completion)

        entry["think_from_gpt"] = response
        #new_data.append(entry)
        new_data["predict"] = response.replace("\n", "").replace("``json","").replace("[    [","[[").replace("[   [","[[").replace("[  [","[[").strip("```")
        new_data['prompt'] = schema_Instruction + text_content
        all_new_data.append(new_data)
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

    client = OpenAI(
                api_key="sk-cE0PRjWuchJZ3XfXB6Be163cA1574d91B0598c6037D5Be2a",
                base_url="https://api.sttai.cc/v1"
            )
    file_paths = {
        "test": "data/JMERE/V2/test.json",
        "train": "data/JMERE/V2/train.json",
        "val": "data/JMERE/V2/val.json"
    }

#file_path = "data/JMERE/V2/train.json"
#output_file_path = "MH-ZS-JMERE/gpt-zero-result_V2/train_distill.jsonl"
file_path = "data/JMERE/V2/train.json"
output_file_path = "MH-ZS-JMERE/gpt5-zero-result_V2/train_distill.jsonl"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)
data = data
print(f"处理前一共有{len(data)}条数据")
thread_count = 1
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
        output_f.write(json.dumps(new_data)+ '\n')
    
    