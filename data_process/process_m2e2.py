"""
Build M2E2 ED/EAE test sets in UMIE-style formats.

Input:
  1) text_multimedia_event.json
     - list of sentence records
     - fields:
         sentence_id, sentence, golden-event-mentions[
             {event_type, trigger{text}, arguments[{role,text}, ...]}
         ]
  2) image_multimedia_event.json
     - dict keyed by image_id WITHOUT extension (e.g., "xxx_1")
     - value:
         {event_type, role: {RoleName: [[flag, x1, y1, x2, y2], ...], ...}}
  3) crossmedia_coref.txt
     - TSV lines: sentence_id \t image_filename(with .jpg) \t event_type

Output:
  - m2e2_test_ED.jsonl / .json
  - m2e2_test_EAE.jsonl / .json
  - m2e2_mismatch_image_event_type.tsv (optional report)

Notes:
  - We select the FIRST event mention in the sentence whose event_type matches crossmedia_coref.
  - We include visual roles/bboxes only when image_event.event_type == coref_event_type
    (to avoid leaking wrong-type visual arguments). Mismatches are written to TSV.
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Any


def simplify_event_type(event_type: str, keep_prefix: bool = False) -> str:
    """
    Convert "Conflict:Attack" -> "attack" (default) or keep full if keep_prefix=True.
    """
    if not event_type:
        return ""
    et = event_type.strip()
    if (not keep_prefix) and (":" in et):
        et = et.split(":", 1)[1]
    return et.strip().lower()


def format_role(role: str, lower_roles: bool = True) -> str:
    r = (role or "").strip()
    return r.lower() if lower_roles else r


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


def load_text_events(text_json_path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], List[Dict[str, Any]]]]:
    """
    Returns:
      sent_lookup: sentence_id -> sentence_record
      ev_lookup: (sentence_id, event_type) -> [event_mentions...]
    """
    with open(text_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sent_lookup = {r["sentence_id"]: r for r in data}

    ev_lookup = defaultdict(list)
    for r in data:
        sid = r.get("sentence_id")
        for ev in r.get("golden-event-mentions", []):
            ev_lookup[(sid, ev.get("event_type"))].append(ev)

    return sent_lookup, ev_lookup


def load_image_events(image_json_path: str) -> Dict[str, Dict[str, Any]]:
    with open(image_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_bbox_list(box: List[Any]) -> List[int]:
    """
    Image JSON box format is usually: [flag, x1, y1, x2, y2].
    We return [x1, y1, x2, y2] as int.
    """
    if len(box) >= 5:
        return [int(box[1]), int(box[2]), int(box[3]), int(box[4])]
    if len(box) == 4:
        return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
    raise ValueError(f"Unsupported bbox format: {box}")


def build_ed_eae(
    coref_rows: List[Tuple[str, str, str]],
    sent_lookup: Dict[str, Dict[str, Any]],
    ev_lookup: Dict[Tuple[str, str], List[Dict[str, Any]]],
    img_lookup: Dict[str, Dict[str, Any]],
    keep_event_prefix: bool = False,
    lower_roles: bool = True,
    include_visual_if_mismatch: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
    """
    Returns:
      ed_records, eae_records, mismatch_rows
    mismatch_rows contain rows where image_event.event_type != coref_event_type
    """
    ed_records = []
    eae_records = []
    mismatches = []

    for sent_id, image_file, coref_event_type in coref_rows:
        srec = sent_lookup.get(sent_id)
        if not srec:
            # If needed, you can choose to skip or keep an empty record.
            continue

        # Find event mention matching coref_event_type
        evs = ev_lookup.get((sent_id, coref_event_type), [])
        if not evs:
            # Strict mode: skip if no matching mention
            # You could also fallback to first mention:
            # evs = srec.get("golden-event-mentions", [])
            continue
        ev = evs[0]

        text = srec.get("sentence", "")
        trigger = (ev.get("trigger") or {}).get("text", "").strip()

        et_simple = simplify_event_type(coref_event_type, keep_prefix=keep_event_prefix)

        # ---- ED ----
        ed_label = f"{et_simple}, {trigger}".strip()
        ed_records.append(
            {"text": text, "label": ed_label, "image_id": image_file}
        )

        # ---- EAE ----
        parts = [et_simple]

        # Text arguments
        for arg in ev.get("arguments", []):
            r = format_role(arg.get("role"), lower_roles=lower_roles)
            v = (arg.get("text") or "").strip()
            if r and v:
                parts.append(f"{r}, {v}")

        out = {"text": text, "image_id": image_file}

        # Visual arguments (bbox)
        img_key = os.path.splitext(image_file)[0]  # remove .jpg
        img_rec = img_lookup.get(img_key)

        obj_idx = 1
        if img_rec:
            img_event_type = img_rec.get("event_type")
            ok = (img_event_type == coref_event_type)
            if (not ok) and (not include_visual_if_mismatch):
                mismatches.append(
                    {
                        "sentence_id": sent_id,
                        "image_id": image_file,
                        "coref_event_type": coref_event_type,
                        "image_event_type": str(img_event_type),
                    }
                )
            if ok or include_visual_if_mismatch:
                role_dict = img_rec.get("role") or {}
                for role_name, boxes in role_dict.items():
                    r = format_role(role_name, lower_roles=lower_roles)
                    for box in boxes:
                        try:
                            coords = extract_bbox_list(box)
                        except Exception:
                            continue
                        oname = f"O{obj_idx}"
                        out[oname] = coords
                        parts.append(f"{r}, {oname}")
                        obj_idx += 1

        out["label"] = " <spot> ".join(parts)
        eae_records.append(out)

    return ed_records, eae_records, mismatches


def dump_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dump_json(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def dump_tsv(path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(h, "")) for h in headers) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_event_json", default="datasets/m2e2/data/m2e2_annotations/text_multimedia_event.json", help="path to text_multimedia_event.json")
    ap.add_argument("--image_event_json",default="datasets/m2e2/data/m2e2_annotations/image_multimedia_event.json", help="path to image_multimedia_event.json")
    ap.add_argument("--coref_txt", default="datasets/m2e2/data/m2e2_annotations/crossmedia_coref.txt", help="path to crossmedia_coref.txt")
    ap.add_argument("--out_dir", default="data_process/processed_data/event", help="output directory")
    ap.add_argument("--keep_event_prefix", default=True,
                    help="keep full event type like Conflict:Attack instead of attack")
    ap.add_argument("--no_lower_roles", default=True, action="store_true",
                    help="do NOT lowercase roles (default is lowercase)")
    ap.add_argument("--include_visual_if_mismatch",default=False, 
                    help="include visual bboxes even if image_event_type != coref_event_type (NOT recommended)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    coref_rows = load_crossmedia_coref(args.coref_txt)
    sent_lookup, ev_lookup = load_text_events(args.text_event_json)
    img_lookup = load_image_events(args.image_event_json)

    ed_records, eae_records, mismatches = build_ed_eae(
        coref_rows=coref_rows,
        sent_lookup=sent_lookup,
        ev_lookup=ev_lookup,
        img_lookup=img_lookup,
        keep_event_prefix=args.keep_event_prefix,
        lower_roles=(not args.no_lower_roles),
        include_visual_if_mismatch=args.include_visual_if_mismatch,
    )

    # Write outputs
    ed_jsonl = os.path.join(args.out_dir, "m2e2_test_ED.jsonl")
    eae_jsonl = os.path.join(args.out_dir, "m2e2_test_EAE.jsonl")
    ed_json = os.path.join(args.out_dir, "m2e2_test_ED.json")
    eae_json = os.path.join(args.out_dir, "m2e2_test_EAE.json")
    mismatch_tsv = os.path.join(args.out_dir, "m2e2_mismatch_image_event_type.tsv")

    dump_jsonl(ed_jsonl, ed_records)
    dump_jsonl(eae_jsonl, eae_records)
    dump_json(ed_json, ed_records)
    dump_json(eae_json, eae_records)
    dump_tsv(mismatch_tsv, mismatches)

    print("Done.")
    print(f"ED:  {len(ed_records)} records -> {ed_jsonl}")
    print(f"EAE: {len(eae_records)} records -> {eae_jsonl}")
    print(f"Mismatch rows: {len(mismatches)} -> {mismatch_tsv if mismatches else '(none)'}")


if __name__ == "__main__":
    main()