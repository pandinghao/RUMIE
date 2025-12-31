import argparse
import json
import os

INSTRUCTION = "Please extract the following entity type: person, location, miscellaneous, organization."

def process_data(input_file: str, output_file: str, image_folder: str):
    processed_data = []

    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            data = json.loads(line.strip())

            text = data.get("text", "")
            entities = data.get("entity", [])
            image_id = data.get("image_id", "")

            # Prepare messages
            messages = [
                {
                    "content": f"{INSTRUCTION} <image> text: {text}",
                    "role": "user"
                }
            ]

            # Build assistant response
            extracted_entities = []
            for entity in entities:
                entity_type = entity.get("type", "")
                entity_text = entity.get("text", "")
                if entity_type in ["person", "location", "miscellaneous", "organization"]:
                    extracted_entities.append(f"{entity_type}, {entity_text}")

            response = "; ".join(extracted_entities)

            messages.append({
                "content": response,
                "role": "assistant"
            })

            # Image path
            images = [os.path.join(image_folder, image_id)] if image_id else []

            processed_data.append({
                "messages": messages,
                "images": images
            })

    # Ensure output dir exists
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(processed_data, outfile, ensure_ascii=False, indent=4)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Twitter15 NER jsonl to LVLM message-format json."
    )
    parser.add_argument(
        "--input_file",
        default="rumie_datasets/twitter17/text_noise/change_context/test.jsonl",
        help="Path to input jsonl (default: %(default)s)"
    )
    parser.add_argument(
        "--output_file",
        default="LVLM_test_datasets/twitter17/change_context/test.json",
        help="Path to output json (default: %(default)s)"
    )
    parser.add_argument(
        "--image_folder",
        default="datasets/twitter17_data/twitter17_images",
        help="Folder containing images; image_id will be joined to this path (default: %(default)s)"
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    process_data(
        input_file=args.input_file,
        output_file=args.output_file,
        image_folder=args.image_folder
    )


if __name__ == "__main__":
    main()