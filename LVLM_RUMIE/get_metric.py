import json
import re
import argparse
from typing import Any, Dict, List, Tuple, Optional, Set
import os

def save_metrics(out_path: str, mee_res: Dict[str, Any], mner_res: Dict[str, Any], mre_res: Dict[str, Any]):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "mee": mee_res,
        "mner": mner_res,
        "mre": mre_res,
        "summary": {
            "mee_micro_f1": mee_res.get("micro_f1", 0.0),
            "mner_micro_f1": mner_res.get("micro_f1", 0.0),
            "mre_micro_f1": mre_res.get("micro_f1", 0.0),
            "avg_micro_f1": (
                mee_res.get("micro_f1", 0.0) +
                mner_res.get("micro_f1", 0.0) +
                mre_res.get("micro_f1", 0.0)
            ) / 3.0,
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
# ----------------------------
# IO
# ----------------------------
def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    # remove ```json ... ``` or ``` ... ```
    s = re.sub(r"^```(?:json)?", "", s.strip(), flags=re.IGNORECASE).strip()
    s = re.sub(r"```$", "", s.strip()).strip()
    return s


def safe_json_loads(x: Any) -> Any:
    """
    x can be:
      - already a python object (list/dict)
      - a json string
      - a string wrapped with code fences / extra newlines
    """
    if x is None:
        return None
    if isinstance(x, (list, dict, int, float, bool)):
        return x
    if not isinstance(x, str):
        # last resort: try json dump/loads
        try:
            return json.loads(json.dumps(x))
        except Exception:
            return x

    s = x
    # common cleanup
    s = s.replace("\r", "")
    s = _strip_code_fence(s)
    s = s.strip()

    # sometimes model outputs leading/trailing text, attempt to extract json array/object
    # try direct loads first
    try:
        return json.loads(s)
    except Exception:
        pass

    # attempt: find first JSON array/object substring
    # array
    m = re.search(r"(\[.*\])", s, flags=re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # object
    m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError(f"Cannot parse JSON from: {x[:200]}...")


# ----------------------------
# Normalizers
# ----------------------------
def _lower(x: Any) -> str:
    return str(x).strip().lower()


def _norm_type(t: str) -> str:
    t = str(t).strip()
    t = t.strip("<>").strip()
    return t.lower()


def _norm_rel(r: str) -> str:
    r = str(r).strip()
    # handle "xxx/yyy" -> yyy (your old logic)
    if "/" in r:
        r = r.split("/")[-1]
    r = r.strip("<>").strip()
    return r.lower()


def _as_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


# ----------------------------
# Task parsers
# ----------------------------
# We support two representations:
# 1) With offsets: (start, end, type) for NER; (start,end,type) for triggers; RE triples uses head/tail spans
# 2) Without offsets: use surface text as identifier (less strict than the paper, but works if files lack offsets)

# ---- NER ----
# expected possible formats per item label/predict:
# A) list of dict: [{"start": 0, "end": 4, "type": "person", "text": "John"}, ...]
# B) list of list/tuple: [[start, end, type], ...] OR [[text, type], ...] OR ["type, text; type, text"]
# C) string like "location, Texas; location, Oklahoma" (as in your sample)
def parse_ner(obj: Any) -> Tuple[Set[Tuple], bool]:
    """
    return: (set of entities, has_offset)
      if has_offset: entities are (start, end, type)
      else:          entities are (text, type)
    """
    entities: Set[Tuple] = set()
    has_offset = False

    if obj is None:
        return entities, has_offset

    # if it's a string like: "location, Texas; location, Oklahoma"
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return entities, has_offset
        parts = [p.strip() for p in re.split(r"[;\n]+", s) if p.strip()]
        for p in parts:
            # try "type, text"
            if "," in p:
                t, txt = p.split(",", 1)
                entities.add((_lower(txt), _norm_type(t)))
            else:
                # fallback: whole chunk as text, unknown type
                entities.add((_lower(p), "unknown"))
        return entities, has_offset

    # list
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                # offset style
                if ("start" in it and "end" in it and "type" in it):
                    st = _as_int(it["start"])
                    ed = _as_int(it["end"])
                    if st is not None and ed is not None:
                        entities.add((st, ed, _norm_type(it["type"])))
                        has_offset = True
                elif ("text" in it and "type" in it):
                    entities.add((_lower(it["text"]), _norm_type(it["type"])))
                else:
                    # best effort
                    if "type" in it and "name" in it:
                        entities.add((_lower(it["name"]), _norm_type(it["type"])))
            elif isinstance(it, (list, tuple)):
                if len(it) == 3:
                    st = _as_int(it[0]); ed = _as_int(it[1])
                    if st is not None and ed is not None:
                        entities.add((st, ed, _norm_type(it[2])))
                        has_offset = True
                    else:
                        # maybe [text, type, ...] weird
                        entities.add((_lower(it[0]), _norm_type(it[1])))
                elif len(it) == 2:
                    # [text, type] OR [type, text]
                    a, b = it
                    # heuristic: if a looks like a type
                    if str(a).lower() in {"person", "location", "organization", "miscellaneous", "misc"}:
                        entities.add((_lower(b), _norm_type(a)))
                    else:
                        entities.add((_lower(a), _norm_type(b)))
                else:
                    # ignore
                    continue
            elif isinstance(it, str):
                # same as string chunk
                chunk = it.strip()
                if not chunk:
                    continue
                if "," in chunk:
                    t, txt = chunk.split(",", 1)
                    entities.add((_lower(txt), _norm_type(t)))
                else:
                    entities.add((_lower(chunk), "unknown"))
        return entities, has_offset

    # unknown
    return entities, has_offset


# ---- RE ----
# expected possible formats:
# A) list of dict: [{"head": {...}, "tail": {...}, "relation": "member_of"}]
#   where head/tail might include offsets: {"start":..,"end":..,"type":..} or text
# B) list of 5-tuples like your old JMERE: [e1, t1, e2, t2, r]
# C) string like: "WWE <spot> none <spot> PWStream"
def parse_re(obj: Any) -> Tuple[Set[Tuple], bool]:
    """
    return: (set of relations, has_offset)
      if has_offset: rels are (h_start,h_end,h_type, t_start,t_end,t_type, rel)
      else:          rels are (h_text,h_type, t_text,t_type, rel) OR (h_text,t_text,rel)
    """
    rels: Set[Tuple] = set()
    has_offset = False

    if obj is None:
        return rels, has_offset

    # string style with <spot>
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return rels, has_offset
        # possibly multiple lines
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        for ln in lines:
            if "<spot>" in ln:
                parts = [p.strip() for p in ln.split("<spot>")]
                if len(parts) >= 3:
                    h = parts[0]
                    r = parts[1]
                    t = parts[2]
                    rels.add((_lower(h), _lower(t), _norm_rel(r)))
                else:
                    # fallback
                    rels.add((_lower(ln), "", "unknown"))
            else:
                # fallback: no structure
                rels.add((_lower(ln), "", "unknown"))
        return rels, has_offset

    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                rel = it.get("relation", it.get("rel", it.get("r", "none")))
                head = it.get("head", it.get("h", {}))
                tail = it.get("tail", it.get("t", {}))

                # head/tail dict may carry offsets
                if isinstance(head, dict) and isinstance(tail, dict) and \
                   ("start" in head and "end" in head and "start" in tail and "end" in tail):
                    hs = _as_int(head["start"]); he = _as_int(head["end"])
                    ts = _as_int(tail["start"]); te = _as_int(tail["end"])
                    if None not in (hs, he, ts, te):
                        ht = _norm_type(head.get("type", head.get("t", "unknown")))
                        tt = _norm_type(tail.get("type", tail.get("t", "unknown")))
                        rels.add((hs, he, ht, ts, te, tt, _norm_rel(rel)))
                        has_offset = True
                        continue

                # text-based
                h_text = head.get("text", head.get("name", head)) if isinstance(head, dict) else head
                t_text = tail.get("text", tail.get("name", tail)) if isinstance(tail, dict) else tail
                h_type = head.get("type", "unknown") if isinstance(head, dict) else "unknown"
                t_type = tail.get("type", "unknown") if isinstance(tail, dict) else "unknown"
                rels.add((_lower(h_text), _norm_type(h_type), _lower(t_text), _norm_type(t_type), _norm_rel(rel)))

            elif isinstance(it, (list, tuple)):
                # 5-tuple style: [e1, t1, e2, t2, r]
                if len(it) == 5:
                    e1, t1, e2, t2, r = it
                    rels.add((_lower(e1), _norm_type(t1), _lower(e2), _norm_type(t2), _norm_rel(r)))
                # 3-tuple style: [head, rel, tail]
                elif len(it) == 3:
                    h, r, t = it
                    rels.add((_lower(h), _lower(t), _norm_rel(r)))
                else:
                    continue
            elif isinstance(it, str):
                # line style
                ln = it.strip()
                if "<spot>" in ln:
                    parts = [p.strip() for p in ln.split("<spot>")]
                    if len(parts) >= 3:
                        rels.add((_lower(parts[0]), _lower(parts[2]), _norm_rel(parts[1])))
                else:
                    rels.add((_lower(ln), "", "unknown"))

        return rels, has_offset

    return rels, has_offset


# ---- ED (Event Detection) ----
# expected possible formats:
# A) list of dict: [{"trigger": {"start":..,"end":..,"text":..}, "event_type":"Conflict:Attack"}]
# B) list of list/tuple: [[start,end,event_type], ...] OR [[trigger_text,event_type], ...]
# C) string like: "Justice:Arrest-Jail, take; Conflict:Attack, killing"
def parse_ed(obj: Any) -> Tuple[Set[Tuple], bool]:
    """
    return: (set of triggers, has_offset)
      if has_offset: triggers are (start, end, event_type)
      else:          triggers are (trigger_text, event_type)
    """
    triggers: Set[Tuple] = set()
    has_offset = False

    if obj is None:
        return triggers, has_offset

    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return triggers, has_offset
        # format: "EventType, trigger; EventType, trigger"
        parts = [p.strip() for p in re.split(r"[;\n]+", s) if p.strip()]
        for p in parts:
            if "," in p:
                et, trig = p.split(",", 1)
                triggers.add((_lower(trig), _norm_type(et)))
            else:
                # only event type?
                triggers.add(("", _norm_type(p)))
        return triggers, has_offset

    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                et = it.get("event_type", it.get("type", it.get("event", "unknown")))
                tri = it.get("trigger", it.get("trg", it.get("trigger_span", None)))

                if isinstance(tri, dict) and "start" in tri and "end" in tri:
                    st = _as_int(tri["start"]); ed = _as_int(tri["end"])
                    if st is not None and ed is not None:
                        triggers.add((st, ed, _norm_type(et)))
                        has_offset = True
                        continue
                # text trigger
                if isinstance(tri, dict):
                    trig_text = tri.get("text", "")
                else:
                    trig_text = tri if tri is not None else ""
                triggers.add((_lower(trig_text), _norm_type(et)))

            elif isinstance(it, (list, tuple)):
                if len(it) == 3:
                    st = _as_int(it[0]); ed = _as_int(it[1])
                    if st is not None and ed is not None:
                        triggers.add((st, ed, _norm_type(it[2])))
                        has_offset = True
                    else:
                        # maybe [trigger_text, event_type, ...]
                        triggers.add((_lower(it[0]), _norm_type(it[1])))
                elif len(it) == 2:
                    triggers.add((_lower(it[0]), _norm_type(it[1])))
            elif isinstance(it, str):
                ln = it.strip()
                if "," in ln:
                    et, trig = ln.split(",", 1)
                    triggers.add((_lower(trig), _norm_type(et)))
        return triggers, has_offset

    return triggers, has_offset


# ----------------------------
# Metrics
# ----------------------------
def micro_prf(correct: int, pred: int, gold: int) -> Dict[str, float]:
    p = correct / pred if pred > 0 else 0.0
    r = correct / gold if gold > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def eval_file(
    file_path: str,
    task: str,
    strict_offset_required: bool = False,
) -> Dict[str, Any]:
    """
    task: "ner" | "re" | "ed"
    strict_offset_required:
      - True: if any sample lacks offsets in either gold or pred, count as format wrong and skip
      - False: allow fallback to text-based matching (will be less strict than paper if offsets missing)
    """
    data = load_jsonl(file_path)
    format_wrong = 0
    used_offset = 0
    used_text_fallback = 0

    correct = 0
    total_pred = 0
    total_gold = 0

    for item in data:
        gold_raw = item.get("label", None)
        pred_raw = item.get("predict", None)

        try:
            gold_obj = safe_json_loads(gold_raw)
        except Exception:
            # some files store label as plain string already
            gold_obj = gold_raw

        try:
            pred_obj = safe_json_loads(pred_raw)
        except Exception:
            pred_obj = pred_raw

        try:
            if task == "ner":
                gold_set, gold_has_off = parse_ner(gold_obj)
                pred_set, pred_has_off = parse_ner(pred_obj)
            elif task == "re":
                gold_set, gold_has_off = parse_re(gold_obj)
                pred_set, pred_has_off = parse_re(pred_obj)
            elif task == "ed":
                gold_set, gold_has_off = parse_ed(gold_obj)
                pred_set, pred_has_off = parse_ed(pred_obj)
            else:
                raise ValueError(f"Unknown task: {task}")
        except Exception as e:
            format_wrong += 1
            continue

        # decide matching representation
        if gold_has_off and pred_has_off:
            used_offset += 1
        else:
            if strict_offset_required:
                format_wrong += 1
                continue
            used_text_fallback += 1

        correct += len(gold_set & pred_set)
        total_pred += len(pred_set)
        total_gold += len(gold_set)

    prf = micro_prf(correct, total_pred, total_gold)
    return {
        "file": file_path,
        "task": task,
        "correct": correct,
        "pred": total_pred,
        "gold": total_gold,
        "format_wrong": format_wrong,
        "used_offset_samples": used_offset,
        "used_text_fallback_samples": used_text_fallback,
        "micro_precision": prf["precision"],
        "micro_recall": prf["recall"],
        "micro_f1": prf["f1"],
        "num_samples": len(data),
        "strict_offset_required": strict_offset_required,
    }

def missing_result(file_path: str, task: str, strict_offset_required: bool) -> Dict[str, Any]:
    return {
        "file": file_path,
        "task": task,
        "missing_file": True,
        "error": f"File not found: {file_path}",
        "correct": 0,
        "pred": 0,
        "gold": 0,
        "format_wrong": 0,
        "used_offset_samples": 0,
        "used_text_fallback_samples": 0,
        "micro_precision": 0.0,
        "micro_recall": 0.0,
        "micro_f1": 0.0,
        "num_samples": 0,
        "strict_offset_required": strict_offset_required,
    }


def safe_eval_file(file_path: str, task: str, strict_offset_required: bool) -> Dict[str, Any]:
    """
    Evaluate if file exists; otherwise return a placeholder result.
    If evaluation throws unexpected error, also return placeholder with error message.
    """
    if not file_path or not os.path.exists(file_path):
        return missing_result(file_path, task, strict_offset_required)

    try:
        res = eval_file(file_path, task=task, strict_offset_required=strict_offset_required)
        res["missing_file"] = False
        return res
    except Exception as e:
        r = missing_result(file_path, task, strict_offset_required)
        r["error"] = f"Eval error: {repr(e)}"
        return r
def pretty_print(res: Dict[str, Any], name: str):
    print("=" * 80)
    print(f"[{name}]  task={res['task']}")
    print(f"file: {res['file']}")
    if res.get("missing_file", False):
        print(f"WARNING: {res.get('error', 'missing file')}")
    print(f"samples: {res['num_samples']}")
    print(f"correct: {res['correct']}, pred: {res['pred']}, gold: {res['gold']}")
    print(f"format_wrong: {res['format_wrong']}")
    print(f"used_offset_samples: {res['used_offset_samples']}")
    print(f"used_text_fallback_samples: {res['used_text_fallback_samples']}")
    print(f"Micro Precision: {res['micro_precision']:.4f}")
    print(f"Micro Recall:    {res['micro_recall']:.4f}")
    print(f"Micro F1:        {res['micro_f1']:.4f}")
    if res["used_text_fallback_samples"] > 0:
        print("NOTE: Some samples lack offsets; fell back to text-based matching (less strict than paper).")
    print("=" * 80)



def main():
    parser = argparse.ArgumentParser("Evaluate MEE/MNER/MRE jsonl results (micro-F1, span-based when available)")
    parser.add_argument("--mee_file", type=str, required=True, help="ED/MEE result jsonl")
    parser.add_argument("--mner_file", type=str, required=True, help="NER result jsonl")
    parser.add_argument("--mre_file", type=str, required=True, help="RE result jsonl")
    parser.add_argument(
        "--strict_offset",
        action="store_true",
        help="Require offsets for span-based eval; otherwise fallback to text-based when offsets missing."
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="",
        help="Optional path to save metrics as JSON (default: empty, do not save)."
    )
    args = parser.parse_args()

    # 关键修改：用 safe_eval_file，缺文件也返回占位结果而不是崩溃退出
    mee_res = safe_eval_file(args.mee_file, task="ed", strict_offset_required=args.strict_offset)
    mner_res = safe_eval_file(args.mner_file, task="ner", strict_offset_required=args.strict_offset)
    mre_res = safe_eval_file(args.mre_file, task="re", strict_offset_required=args.strict_offset)

    pretty_print(mee_res, "MEE/ED")
    pretty_print(mner_res, "MNER/NER")
    pretty_print(mre_res, "MRE/RE")

    if args.out_file:
        save_metrics(args.out_file, mee_res, mner_res, mre_res)
        print(f"[SAVED] metrics -> {args.out_file}")

if __name__ == "__main__":
    main()