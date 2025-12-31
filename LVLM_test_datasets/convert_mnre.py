import argparse
import os
import json
import cv2
from tqdm import tqdm

INSTRUCTION = (
    "Please extract the following relation between [head] and [tail]: "
    "part of, contain, present in, none, held on, member of, peer, "
    "place of residence, locate at, alternate names, neighbor, subsidiary, "
    "awarded, couple parent, nationality, place of birth, charges, siblings, "
    "religion, race."
)

def resolve_image_path(image_folder: str, image_id: str):
    """
    Try image_id as-is; if not found, try appending .jpg.
    Returns (resolved_path, resolved_image_id) or (None, None) if not found.
    """
    if not image_id:
        return None, None

    cand1 = os.path.join(image_folder, image_id)
    img = cv2.imread(cand1)
    if img is not None:
        return cand1, image_id

    cand2 = os.path.join(image_folder, f"{image_id}.jpg")
    img = cv2.imread(cand2)
    if img is not None:
        return cand2, f"{image_id}.jpg"

    return None, None


def process_re_data(input_file: str, output_file: str, image_folder: str):
    results = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            data = json.loads(line.strip())

            text = data.get("text", "")
            entities = data.get("entity", [])
            relations = data.get("relation", [])
            image_id = data.get("image_id", "")

            # Basic entity sanity check
            if not isinstance(entities, list) or len(entities) < 2:
                continue

            head = entities[0].get("text", "")
            tail = entities[1].get("text", "")
            if not head or not tail:
                continue

            # relation type
            relation_type = relations[0].get("type", "none") if relations else "none"

            # Resolve image
            resolved_img_path, resolved_img_id = resolve_image_path(image_folder, image_id)
            if image_id and resolved_img_path is None:
                # keep your original behavior: log and skip
                print(f"找不到图片: {os.path.join(image_folder, image_id)}")
                continue

            # User message
            user_content = (
                f"{INSTRUCTION} "
                f"<image> text: {text} "
                f"Head entity: {head} ; Tail entity: {tail}"
            )

            # Assistant message (strict format)
            assistant_content = f"{head} <spot> {relation_type} <spot> {tail}"

            sample = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "images": [resolved_img_path] if resolved_img_path else []
            }

            results.append(sample)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert MNRE-V2 RE jsonl to LVLM message-format json."
    )
    parser.add_argument(
        "--input_file",
        default="data_process/processed_data/relation/MNRE-V1/test.fixed.jsonl",
        help="Path to input jsonl (default: %(default)s)"
    )
    parser.add_argument(
        "--output_file",
        default="LVLM_test_datasets/MNRE-V1/test.json",
        help="Path to output json (default: %(default)s)"
    )
    parser.add_argument(
        "--image_folder",
        default="datasets/MNRE-V2/mnre_image/img_org/test",
        help="Folder containing images; image_id will be joined to this path (default: %(default)s)"
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    process_re_data(
        input_file=args.input_file,
        output_file=args.output_file,
        image_folder=args.image_folder
    )


if __name__ == "__main__":
    main()