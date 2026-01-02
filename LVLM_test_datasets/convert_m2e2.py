import argparse
import json
import os

INSTRUCTION = (
    "Please extract the following event type: "
    "Justice:Arrest-Jail, Life:Die, Movement:Transport, Conflict:Attack, "
    "Conflict:Demonstrate, Contact:Meet, Contact:Phone-Write, "
    "Transaction:Transfer-Money."
)

def process_voa_style_to_messages(input_path: str, output_path: str, image_base_dir: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # expected: list

    results = []
    for item in data:
        text = item.get("text", "")
        label = item.get("label", "")
        image_id = item.get("image_id", "")

        # Skip samples without label or image_id (adjust if you want to keep them)
        if not label or not image_id:
            continue

        user_content = f"<image>{INSTRUCTION} text: {text}"

        sample = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": label}
            ],
            "images": [os.path.join(image_base_dir, image_id)]
        }

        results.append(sample)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert processed M2E2 event json into LVLM (VOA-style) message format."
    )
    parser.add_argument(
        "--input_file",
        default="data_process/processed_data/event/m2e2_test_ED.json",
        help="Path to processed M2E2 event json (default: %(default)s)"
    )
    parser.add_argument(
        "--output_file",
        default="LVLM_test_datasets/m2e2/rule_vision_noise/Image-Side-Contradictory_Perturbation_clip/test.json",
        help="Output path for VOA-style json (default: %(default)s)"
    )
    parser.add_argument(
        "--image_base_dir",
        default="rumie_datasets/m2e2/rule_vision_noise/Image-Side-Contradictory_Perturbation_clip",
        help="Base directory for images; image_id will be joined to this path (default: %(default)s)"
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    process_voa_style_to_messages(
        input_path=args.input_file,
        output_path=args.output_file,
        image_base_dir=args.image_base_dir
    )


if __name__ == "__main__":
    main()