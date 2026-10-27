import json
import argparse
from pathlib import Path
from PIL import Image


def load_expected_images(selection_file: Path) -> list[str]:

    with selection_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "images" not in data or not isinstance(data["images"], list):
        raise ValueError(
            "Invalid selection JSON: expected an 'images' list."
        )

    expected_filenames = []

    for image in data["images"]:
        filename = image.get("file_name")

        if not filename:
            raise ValueError(
                f"An image entry has no 'file_name': {image}"
            )

        expected_filenames.append(filename)

    if len(expected_filenames) != len(set(expected_filenames)):
        raise ValueError(
            "The selection JSON contains duplicate image filenames."
        )

    return expected_filenames


def check_image(image_path: Path) -> str | None:
    """
    Verify the image structure and try to decode its contents.

    Returns None if valid, or an error description if invalid.
    """
    try:
        with Image.open(image_path) as image:
            image.verify()

        # Reouverte de l'image pour le chargement en cas de besoin.
        with Image.open(image_path) as image:
            image.load()

        return None

    except Exception as error:
        return f"{type(error).__name__}: {error}"


def diagnose_dataset(
    images_dir: Path,
    selection_file: Path,
    report_path: Path,
) -> dict:
    """check the images in the dataset against the expected list and generate a report."""

    expected_filenames = load_expected_images(selection_file)

    # Map filenames to their paths for lookup.
    actual_files = {
        path.name: path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }

    expected_set = set(expected_filenames)
    actual_set = set(actual_files)

    missing_images = sorted(expected_set - actual_set)
    unexpected_images = sorted(actual_set - expected_set)

    corrupted_images = []
    valid_images = 0

    # Check every expected image that exists.
    existing_expected = sorted(expected_set & actual_set)
    total_to_check = len(existing_expected)

    for index, filename in enumerate(existing_expected, start=1):
        error = check_image(actual_files[filename])

        if error:
            corrupted_images.append({
                "file_name": filename,
                "error": error,
            })
        else:
            valid_images += 1

        if index % 500 == 0 or index == total_to_check:
            print(f"Checked {index}/{total_to_check} images...")

    status = (
        "PASS"
        if not missing_images
        and not corrupted_images
        and not unexpected_images
        else "ISSUES_FOUND"
    )

    report = {
        "selection_seed": None,
        "expected_images": len(expected_filenames),
        "image_files_found": len(actual_files),
        "valid_images": valid_images,
        "missing_images_count": len(missing_images),
        "missing_images": missing_images,
        "corrupted_images_count": len(corrupted_images),
        "corrupted_images": corrupted_images,
        "unexpected_images_count": len(unexpected_images),
        "unexpected_images": unexpected_images,
        "status": status,
    }

    # Include the selection seed when it exists in the JSON.
    with selection_file.open("r", encoding="utf-8") as file:
        selection_data = json.load(file)

    report["selection_seed"] = selection_data.get("seed")
    report["selection_file"] = str(selection_file)
    report["images_directory"] = str(images_dir)

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose missing and corrupted COCO dataset images."
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/coco_rag_data/train2017_subset"),
        help="Directory containing downloaded images.",
    )

    parser.add_argument(
        "--selection-file",
        type=Path,
        default=Path("data/coco_rag_data/train_subset_10000.json"),
        help="JSON manifest containing the expected images.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/coco_rag_data/diagnostic_report.json"),
        help="Path for the output report.",
    )

    args = parser.parse_args()

    if not args.images_dir.is_dir():
        parser.error(f"Image directory does not exist: {args.images_dir}")

    if not args.selection_file.is_file():
        parser.error(f"Selection JSON does not exist: {args.selection_file}")

    try:
        report = diagnose_dataset(
            images_dir=args.images_dir,
            selection_file=args.selection_file,
            report_path=args.report,
        )
    except (ValueError, json.JSONDecodeError, OSError) as error:
        parser.error(f"Diagnostic failed: {error}")

    print("\n Diagnostic Report Summary:")
    print(f"Expected images:    {report['expected_images']}")
    print(f"Image files found:  {report['image_files_found']}")
    print(f"Valid images:       {report['valid_images']}")
    print(f"Missing images:     {report['missing_images_count']}")
    print(f"Corrupted images:   {report['corrupted_images_count']}")
    print(f"Unexpected images:  {report['unexpected_images_count']}")
    print(f"Final status:       {report['status']}")
    print(f"Report saved to:    {args.report}")


if __name__ == "__main__":
    main()