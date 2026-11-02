from pathlib import Path
from PIL import Image

IMAGES_DIR = Path("data/coco_rag_data/processed_images_batch")
EXPECTED_SIZE = (224, 224)
EXPECTED_MODE = "RGB"
VALID_EXTENSIONS = {".jpg", ".jpeg"}

def main():
    if not IMAGES_DIR.is_dir():
        raise SystemExit(f"ERROR: Directory not found: {IMAGES_DIR}")

    image_paths = sorted(
        path for path in IMAGES_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )

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
            print(f"Unreadable: {path} — {error}")

        if index % 1000 == 0 or index == len(image_paths):
            print(f"Validated: {index}/{len(image_paths)}")

    print("\n========== VALIDATION RESULTS ==========")
    print(f"Total JPEG files found: {len(image_paths)}")
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