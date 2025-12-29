import json
import codecs
from collections import defaultdict

# ===== consts =====
UNK_LABEL = "<unk>"
UNK_IDX = 0
PADDING_LABEL = "<pad>"
PADDING_IDX = 1
VOCAB_START_IDX = 2
CUTOFF = 50
TRIGGER_GOLDEN_COMPENSATION = 12
ARGUMENT_GOLDEN_COMPENSATION = 27
O_LABEL = None
ROLE_O_LABEL = None
O_LABEL_NAME = "O"
ROLE_O_LABEL_NAME = "OTHER"

def event_type_norm(type_str: str) -> str:
    return type_str.replace('.', '||').replace(':', '||').replace('-', '|').upper()

def role_name_norm(type_str: str) -> str:
    return type_str.upper()

def load_mapping_all(mapping_path: str):
    """
    映射文件每行4列(tab分隔): sr_verb sr_role ee_event ee_role
    返回:
      sr_verb_mapping: dict[sr_verb] = ee_event
      sr_role_mapping: dict[sr_verb][sr_role] = ee_role
    """
    sr_verb_mapping = defaultdict(str)
    sr_role_mapping = defaultdict(lambda: defaultdict(str))

    if not mapping_path:
        return sr_verb_mapping, sr_role_mapping

    with codecs.open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            tabs = line.split("\t")
            if len(tabs) < 4:
                continue
            sr_verb, sr_role, ee_event, ee_role = tabs[0], tabs[1], tabs[2], tabs[3]
            sr_verb_mapping[sr_verb] = ee_event
            sr_role_mapping[sr_verb][sr_role] = ee_role

    return sr_verb_mapping, sr_role_mapping

def load_imsitu_nouns(imsitu_ontology_file: str):
    """
    对齐 data_loader_situation.py：imsitu_info = json.load(...); self.nouns = imsitu_info["nouns"] :contentReference[oaicite:4]{index=4}
    """
    if not imsitu_ontology_file:
        return {}
    with open(imsitu_ontology_file, "r", encoding="utf-8") as f:
        imsitu_info = json.load(f)
    return imsitu_info.get("nouns", {})  # noun_id -> {"gloss": ...}

def noun_id_to_text(noun_id, nouns: dict) -> str:
    """
    对齐 data_loader_situation.py：role_value = self.nouns[role_value_id]['gloss'] :contentReference[oaicite:5]{index=5}
    gloss 可能是 str 或 list（有些版本的 imsitu gloss 会是 token list）
    """
    if noun_id is None or noun_id == "" or noun_id == -1:
        return ""

    # dev.json 里常见是字符串 id；保险起见统一转 str
    nid = str(noun_id)

    if nid not in nouns or "gloss" not in nouns[nid]:
        # 找不到就回退为原 id（至少不丢信息）
        return nid

    gloss = nouns[nid]["gloss"]
    if isinstance(gloss, list):
        text = " ".join([str(x) for x in gloss if x is not None and str(x) != ""])
    else:
        text = str(gloss)

    return text.strip()

def convert_item(img_id, info, sr_verb_mapping, sr_role_mapping, nouns):
    """
    规则：没映射到类别的数据就不要
      - verb 无事件类型映射 -> 不生成事件
      - role 无论元类型映射 -> 丢该论元
      - noun_id 无 gloss 映射 -> 丢该论元
    """
    verb = info.get("verb", None)
    frames = info.get("frames", [])

    # 没 verb 直接返回空事件
    if not verb:
        return {
            "img_id": img_id,
            "tokens": [],
            "golden-entity-mentions": [],
            "golden-event-mentions": [],
        }

    # verb 与映射表对齐（lower）
    verb_key = verb.lower() if isinstance(verb, str) else verb

    # ---------- 事件类型：没映射到就不要 ----------
    if (verb_key not in sr_verb_mapping) or (not sr_verb_mapping[verb_key]):
        # 不生成事件（保留该 img 记录，事件列表为空）
        
        return {
            "img_id": img_id,
            "tokens": [verb],
            "golden-entity-mentions": [],
            "golden-event-mentions": [],
        }

    mapped_event = sr_verb_mapping[verb_key]
    event_type =  event_type_norm(mapped_event)

    # 收集 role->noun_ids
    role2vals = defaultdict(set)
    for frame in frames:
        for role, val in frame.items():
            if val is None or val == "" or val == -1:
                continue
            role2vals[role].add(val)

    arguments = []

    # ---------- 论元：没映射到就不要 ----------
    for role in sorted(role2vals.keys()):
        role_key = role.lower() if isinstance(role, str) else role

        # role 类型映射必须存在，否则丢弃该 role 下所有论元
        if (verb_key not in sr_role_mapping) or (role_key not in sr_role_mapping[verb_key]) or (not sr_role_mapping[verb_key][role_key]):
            continue

        mapped_role = role_name_norm(sr_role_mapping[verb_key][role_key])

        for noun_id in sorted(role2vals[role], key=lambda x: str(x)):
            nid = str(noun_id)

            # noun 原文映射必须存在，否则丢弃该论元
            if (nid not in nouns) or ("gloss" not in nouns[nid]):
                continue

            gloss = nouns[nid]["gloss"]
            if isinstance(gloss, list):
                arg_text = " ".join([str(x) for x in gloss if x is not None and str(x) != ""]).strip()
            else:
                arg_text = str(gloss).strip()

            # gloss 为空也丢弃
            if not arg_text:
                continue

            arguments.append({
                "role": mapped_role,
                "text": arg_text
            })

    golden_events = [{
        "event_type": event_type,
        "trigger": {"text": verb},
        "arguments": arguments
    }]

    # 如果你希望 “arguments 为空就不要事件”，打开下面两行：
    # if len(arguments) == 0:
    #     golden_events = []

    return {
        "img_id": img_id,
        "tokens": [verb],
        "golden-entity-mentions": [],
        "golden-event-mentions": golden_events,
    }
def main(dev_path="dev.json",
         out_path="swig_event.json",
         mapping_path="verb_role_mapping.tsv",
         imsitu_ontology_file="imsitu_space.json"):
    sr_verb_mapping, sr_role_mapping = load_mapping_all(mapping_path)
    nouns = load_imsitu_nouns(imsitu_ontology_file)

    with open(dev_path, "r", encoding="utf-8") as f:
        dev = json.load(f)

    items = list(dev.items())
    print(f"loaded {len(items)} images from {dev_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        #f.write("[\n")
        for idx, (img_id, info) in enumerate(items):
            obj = convert_item(img_id, info, sr_verb_mapping, sr_role_mapping, nouns)
            if len(obj["golden-event-mentions"]) == 0:
                continue  # 没事件就不写入输出文件
            line = json.dumps(obj, ensure_ascii=False)
            f.write(line + ("\n" if idx < len(items) - 1 else "\n"))
        #f.write("]\n")

    print(f"Done. Wrote {len(items)} items to {out_path}")

if __name__ == "__main__":
    main("UMIE/text_processing/data/swig_event/dev.json", "UMIE/text_processing/data/swig_event/swig_val_event.json", "m2e2/data/ace/ace_sr_mapping.txt", "UMIE/text_processing/data/swig_event/imsitu_space.json")