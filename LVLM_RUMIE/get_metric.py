import json
import re
import argparse
from typing import Any, Dict, List, Tuple, Optional, Set
import os

def _is_none_relation(rel: Any) -> bool:
    """
    Return True if rel indicates 'no relation' / none.
    """
    if rel is None:
        return True
    r = _norm_rel(rel)
    return r in {
        "", "none", "no_relation", "no relation", "norelation",
        "na", "n/a", "null", "nil", "other", "unknown"
    }
# ----------------------------
# Save
# ----------------------------
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

    # try direct loads first
    try:
        return json.loads(s)
    except Exception:
        pass

    # attempt: find first JSON array/object substring
    m = re.search(r"(\[.*\])", s, flags=re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError(f"Cannot parse JSON from: {x[:200]}...")


# ----------------------------
# Normalizers / Cleaners
# ----------------------------
def _lower(x: Any) -> str:
    return str(x).strip().lower()


def _norm_type(t: str) -> str:
    t = str(t).strip()
    t = t.strip("<>").strip()
    return t.lower()


def _norm_rel(r: str) -> str:
    r = str(r).strip()
    # handle "xxx/yyy" -> yyy
    if "/" in r:
        r = r.split("/")[-1]
    r = r.strip("<>").strip()
    return r.lower()


def _as_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _clean_model_text(s: str) -> str:
    """
    Remove model artifacts like <think>...</think>, and role prefixes.
    Keep only content that may contain labels/predictions.
    """
    if not s:
        return ""

    s = s.replace("\r", "")

    # remove <think> ... </think> blocks
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.IGNORECASE | re.DOTALL)

    # remove standalone role lines (common prompt formatting)
    lines: List[str] = []
    for ln in s.splitlines():
        t = ln.strip()
        if not t:
            continue
        if t.lower() in {"user", "assistant", "system"}:
            continue
        if t.lower() in {"<think>", "</think>"}:
            continue
        lines.append(ln)

    return "\n".join(lines).strip()


# ----------------------------
# Task parsers
# ----------------------------
# We support two representations:
# 1) With offsets: (start, end, type) for NER; (start,end,type) for triggers; RE triples uses head/tail spans
# 2) Without offsets: use surface text as identifier

# ---- NER ----
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

    # string: "location, Texas; location, Oklahoma"
    if isinstance(obj, str):
        s = _clean_model_text(obj)
        if not s:
            return entities, has_offset
        parts = [p.strip() for p in re.split(r"[;\n]+", s) if p.strip()]
        for p in parts:
            if "," in p:
                t, txt = p.split(",", 1)
                t = _norm_type(t)
                txt = _lower(txt)
                if t and txt:
                    entities.add((txt, t))
            else:
                # stricter: ignore noisy fragments without comma
                continue
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
                    txt = _lower(it["text"])
                    tp = _norm_type(it["type"])
                    if txt and tp:
                        entities.add((txt, tp))
                else:
                    if "type" in it and "name" in it:
                        txt = _lower(it["name"])
                        tp = _norm_type(it["type"])
                        if txt and tp:
                            entities.add((txt, tp))

            elif isinstance(it, (list, tuple)):
                if len(it) == 3:
                    st = _as_int(it[0]); ed = _as_int(it[1])
                    if st is not None and ed is not None:
                        entities.add((st, ed, _norm_type(it[2])))
                        has_offset = True
                    else:
                        # maybe [text, type, ...]
                        txt = _lower(it[0]); tp = _norm_type(it[1])
                        if txt and tp:
                            entities.add((txt, tp))

                elif len(it) == 2:
                    a, b = it
                    # heuristic: if a looks like a type
                    if str(a).lower() in {"person", "location", "organization", "miscellaneous", "misc"}:
                        txt = _lower(b); tp = _norm_type(a)
                    else:
                        txt = _lower(a); tp = _norm_type(b)
                    if txt and tp:
                        entities.add((txt, tp))

            elif isinstance(it, str):
                chunk = _clean_model_text(it)
                if not chunk:
                    continue
                if "," in chunk:
                    t, txt = chunk.split(",", 1)
                    t = _norm_type(t); txt = _lower(txt)
                    if t and txt:
                        entities.add((txt, t))
                else:
                    continue

        return entities, has_offset

    return entities, has_offset


# ---- RE ----
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
        s = _clean_model_text(obj)
        if not s:
            return rels, has_offset
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        for ln in lines:
            if "<spot>" in ln:
                parts = [p.strip() for p in ln.split("<spot>")]
                if len(parts) >= 3:
                    h = parts[0]
                    r = parts[1]
                    t = parts[2]
                    nr = _norm_rel(r)
                    if _is_none_relation(nr):
                        continue
                    rels.add((_lower(h), _lower(t), nr))
            else:
                # stricter: ignore noisy fragments without <spot>
                continue
        return rels, has_offset

    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                rel = it.get("relation", it.get("rel", it.get("r", "none")))
                if _is_none_relation(rel):
                    continue
                head = it.get("head", it.get("h", {}))
                tail = it.get("tail", it.get("t", {}))

                # offset style
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
                rel_norm = _norm_rel(rel)
                if _is_none_relation(rel_norm):
                    continue
                rels.add((_lower(h_text), _norm_type(h_type), _lower(t_text), _norm_type(t_type), rel_norm))

            elif isinstance(it, (list, tuple)):
                if len(it) == 5:
                    e1, t1, e2, t2, r = it
                    rel_norm = _norm_rel(r)
                    if _is_none_relation(rel_norm):
                        continue
                    rels.add((_lower(e1), _norm_type(t1), _lower(e2), _norm_type(t2), rel_norm))
                
                elif len(it) == 3:
                    h, r, t = it
                    rel_norm = _norm_rel(r)
                    if _is_none_relation(rel_norm):
                        continue
                    rels.add((_lower(h_text), _norm_type(h_type), _lower(t_text), _norm_type(t_type), rel_norm))
            elif isinstance(it, str):
                ln = _clean_model_text(it)
                if not ln:
                    continue
                if "<spot>" in ln:
                    parts = [p.strip() for p in ln.split("<spot>")]
                    if len(parts) >= 3:
                        rels.add((_lower(parts[0]), _lower(parts[2]), _norm_rel(parts[1])))
                else:
                    continue

        return rels, has_offset

    return rels, has_offset


# ---- ED (Event Detection) ----
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
        s = _clean_model_text(obj)
        if not s:
            return triggers, has_offset

        parts = [p.strip() for p in re.split(r"[;\n]+", s) if p.strip()]
        for p in parts:
            if "," in p:
                et, trig = p.split(",", 1)
                et = _norm_type(et)
                trig = _lower(trig)
                if et and trig:
                    triggers.add((trig, et))
            else:
                # stricter: ignore fragments without "event_type, trigger"
                continue

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

                if isinstance(tri, dict):
                    trig_text = tri.get("text", "")
                else:
                    trig_text = tri if tri is not None else ""
                trig_text = _lower(trig_text)
                et_norm = _norm_type(et)
                if trig_text and et_norm:
                    triggers.add((trig_text, et_norm))

            elif isinstance(it, (list, tuple)):
                if len(it) == 3:
                    st = _as_int(it[0]); ed = _as_int(it[1])
                    if st is not None and ed is not None:
                        triggers.add((st, ed, _norm_type(it[2])))
                        has_offset = True
                    else:
                        txt = _lower(it[0]); tp = _norm_type(it[1])
                        if txt and tp:
                            triggers.add((txt, tp))
                elif len(it) == 2:
                    txt = _lower(it[0]); tp = _norm_type(it[1])
                    if txt and tp:
                        triggers.add((txt, tp))

            elif isinstance(it, str):
                ln = _clean_model_text(it)
                if not ln:
                    continue
                if "," in ln:
                    et, trig = ln.split(",", 1)
                    et = _norm_type(et); trig = _lower(trig)
                    if et and trig:
                        triggers.add((trig, et))
                else:
                    continue

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
      - False: allow fallback to text-based matching
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
        except Exception:
            format_wrong += 1
            continue

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
    print(f"[{name}]  task={res.get('task')}")
    print(f"file: {res.get('file')}")
    if res.get("missing_file", False):
        print(f"WARNING: {res.get('error', 'missing file')}")
    print(f"samples: {res.get('num_samples', 0)}")
    print(f"correct: {res.get('correct', 0)}, pred: {res.get('pred', 0)}, gold: {res.get('gold', 0)}")
    print(f"format_wrong: {res.get('format_wrong', 0)}")
    print(f"used_offset_samples: {res.get('used_offset_samples', 0)}")
    print(f"used_text_fallback_samples: {res.get('used_text_fallback_samples', 0)}")
    print(f"Micro Precision: {res.get('micro_precision', 0.0):.4f}")
    print(f"Micro Recall:    {res.get('micro_recall', 0.0):.4f}")
    print(f"Micro F1:        {res.get('micro_f1', 0.0):.4f}")
    if res.get("used_text_fallback_samples", 0) > 0:
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
