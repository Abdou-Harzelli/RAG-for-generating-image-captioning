import argparse
import json
from pathlib import Path

from PIL import Image


IMAGES_DIR = Path("data/coco_rag_data/processed_images_batch")
IDS_FILE = Path("data/coco_rag_data/train_kb_ids.json")
EXPECTED_SIZE = (224, 224)
EXPECTED_MODE = "RGB"
VALID_EXTENSIONS = {".jpg", ".jpeg"}


def load_allowed_filenames(ids_file: Path) -> set[str]:
    with ids_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"IDs file must contain a JSON list: {ids_file}")

    ids = []
    for item in data:
        value = item.get("id") if isinstance(item, dict) else item
        ids.append(int(value))

    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs found in: {ids_file}")

    return {f"{image_id:012d}.jpg" for image_id in ids}


def main():
    parser = argparse.ArgumentParser(description="Validate processed KB images.")
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--ids-file", type=Path, default=IDS_FILE)
    args = parser.parse_args()

    images_dir = args.images_dir
    ids_file = args.ids_file

    if not images_dir.is_dir():
        raise SystemExit(f"ERROR: Directory not found: {images_dir}")
    if not ids_file.is_file():
        raise SystemExit(f"ERROR: IDs file not found: {ids_file}")

    allowed_filenames = load_allowed_filenames(ids_file)

    all_paths = sorted(
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )

    image_paths = [path for path in all_paths if path.name in allowed_filenames]

    invalid_format = 0
    invalid_dimensions = 0
    non_rgb = 0
    unreadable = 0

    for index, path in enumerate(image_paths, start=1):
        try:
            with Image.open(path) as image:
                if image.format != "JPEG":
                    invalid_format += 1

                if image.size != EXPECTED_SIZE:
                    invalid_dimensions += 1

                if image.mode != EXPECTED_MODE:
                    non_rgb += 1

                image.verify()

        except Exception as error:
            unreadable += 1
            print(f"Unreadable: {path} - {error}")

        if index % 1000 == 0 or index == len(image_paths):
            print(f"Validated: {index}/{len(image_paths)}")

    print("\n========== VALIDATION RESULTS ==========")
    print(f"IDs file used: {ids_file}")
    print(f"Total JPEG files discovered: {len(all_paths)}")
    print(f"Total JPEG files selected by IDs: {len(image_paths)}")
    print(f"Invalid format: {invalid_format}")
    print(f"Invalid dimensions: {invalid_dimensions}")
    print(f"Non-RGB images: {non_rgb}")
    print(f"Unreadable images: {unreadable}")

    if image_paths and not any((
        invalid_format,
        invalid_dimensions,
        non_rgb,
        unreadable,
    )):
        print("\nSUCCESS: All images passed validation!")
    else:
        print("\nVALIDATION FAILED: Check the results above.")

if __name__ == "__main__":
    main()