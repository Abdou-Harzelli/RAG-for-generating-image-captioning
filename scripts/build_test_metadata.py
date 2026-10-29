
import argparse
import json
from collections import defaultdict
from pathlib import Path
''' c un fichier special pour la verification des 100 imgs de test ainsi que la creation d'un fichier json de metadata pour les images de test.'''

def load_json(file_path: Path) -> dict:
    """Load a JSON file."""
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_test_ids(test_ids_file: Path) -> list[int]:
    """Load and validate the test image IDs."""

    data = load_json(test_ids_file)

    if not isinstance(data, list):
        raise ValueError(
            "test_ids.json must contain a JSON list of image IDs."
        )

    if any(
        isinstance(image_id, bool)
        or not isinstance(image_id, int)
        for image_id in data
    ):
        raise ValueError(
            "Every test image ID must be an integer."
        )

    if len(data) != len(set(data)):
        raise ValueError(
            "Duplicate test IDs detected in test_ids.json."
        )

    return data


def build_test_metadata(
    test_ids_file: Path,
    selection_file: Path,
    annotations_dir: Path,
) -> dict:
    """Build metadata for test images using official COCO annotations."""

    test_ids = load_test_ids(test_ids_file)
    test_id_set = set(test_ids)

    # Load training IDs to check train/test overlap.
    selection_data = load_json(selection_file)

    if not isinstance(selection_data.get("images"), list):
        raise ValueError(
            "The training selection JSON must contain an 'images' list."
        )

    train_ids = {
        image["id"]
        for image in selection_data["images"]
    }

    overlap_ids = sorted(test_id_set & train_ids)

    # Support both COCO 2017 train and validation annotations.
    captions_files = [
        annotations_dir / "captions_train2017.json",
        annotations_dir / "captions_val2017.json",
    ]

    instances_files = [
        annotations_dir / "instances_train2017.json",
        annotations_dir / "instances_val2017.json",
    ]

    captions_by_image = defaultdict(list)
    objects_by_image = defaultdict(set)
    filenames_by_image = {}

    available_caption_files = [
        path for path in captions_files if path.is_file()
    ]

    available_instance_files = [
        path for path in instances_files if path.is_file()
    ]

    if not available_caption_files:
        raise FileNotFoundError(
            "No COCO caption annotation files were found."
        )

    if not available_instance_files:
        raise FileNotFoundError(
            "No COCO instance annotation files were found."
        )

    # Read official captions and image filenames.
    for file_path in available_caption_files:
        data = load_json(file_path)

        for image in data.get("images", []):
            image_id = image["id"]

            if image_id in test_id_set:
                filenames_by_image[image_id] = image["file_name"]

        for annotation in data.get("annotations", []):
            image_id = annotation["image_id"]

            if image_id in test_id_set:
                captions_by_image[image_id].append(
                    annotation["caption"]
                )

    # Read official object labels and image filenames.
    for file_path in available_instance_files:
        data = load_json(file_path)

        category_names = {
            category["id"]: category["name"]
            for category in data.get("categories", [])
        }

        for image in data.get("images", []):
            image_id = image["id"]

            if image_id in test_id_set:
                filenames_by_image[image_id] = image["file_name"]

        for annotation in data.get("annotations", []):
            image_id = annotation["image_id"]

            if image_id not in test_id_set:
                continue

            category_name = category_names.get(
                annotation["category_id"]
            )

            if category_name:
                objects_by_image[image_id].add(category_name)

    # Build one entry for each test ID.
    test_images = []

    for image_id in test_ids:
        test_images.append({
            "id": image_id,
            "file_name": filenames_by_image.get(image_id),
            "captions": captions_by_image.get(image_id, []),
            "objects": sorted(
                objects_by_image.get(image_id, set())
            ),
        })

    images_with_captions = sum(
        bool(image["captions"])
        for image in test_images
    )

    images_with_objects = sum(
        bool(image["objects"])
        for image in test_images
    )

    images_without_filename = sum(
        image["file_name"] is None
        for image in test_images
    )

    total_captions = sum(
        len(image["captions"])
        for image in test_images
    )

    # Store both image-level metadata and validation statistics.
    return {
        "count": len(test_images),
        "source": "COCO 2017 annotations",
        "diagnostics": {
            "unique_test_images": len(test_ids),
            "duplicate_test_ids": 0,
            "train_test_overlap_count": len(overlap_ids),
            "train_test_overlap_ids": overlap_ids,
            "images_with_captions": images_with_captions,
            "images_without_captions": (
                len(test_images) - images_with_captions
            ),
            "images_with_objects": images_with_objects,
            "images_without_objects": (
                len(test_images) - images_with_objects
            ),
            "images_without_filename": images_without_filename,
            "total_captions": total_captions,
        },
        "images": test_images,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build metadata for COCO test images."
    )

    parser.add_argument(
        "--test-ids-file",
        type=Path,
        default=Path(
            "data/coco_rag_data/test_ids.json"
        ),
    )

    parser.add_argument(
        "--selection-file",
        type=Path,
        default=Path(
            "data/coco_rag_data/train_subset_10000.json"
        ),
    )

    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path(
            "data/annotations_trainval2017/annotations"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/coco_rag_data/test_images_metadata.json"
        ),
    )

    args = parser.parse_args()

    required_files = [
        args.test_ids_file,
        args.selection_file,
    ]

    for file_path in required_files:
        if not file_path.is_file():
            parser.error(
                f"Required file not found: {file_path}"
            )

    try:
        metadata = build_test_metadata(
            test_ids_file=args.test_ids_file,
            selection_file=args.selection_file,
            annotations_dir=args.annotations_dir,
        )

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.output.open("w", encoding="utf-8") as file:
            json.dump(
                metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )

    except (
        KeyError,
        ValueError,
        json.JSONDecodeError,
        OSError,
    ) as error:
        parser.error(f"Failed to build test metadata: {error}")

    diagnostics = metadata["diagnostics"]

    print("\n========== TEST METADATA SUMMARY ==========")
    print(f"Test images:              {metadata['count']}")
    print(
        f"Images with captions:     "
        f"{diagnostics['images_with_captions']}"
    )
    print(
        f"Images with objects:      "
        f"{diagnostics['images_with_objects']}"
    )
    print(
        f"Images without captions:  "
        f"{diagnostics['images_without_captions']}"
    )
    print(
        f"Images without objects:   "
        f"{diagnostics['images_without_objects']}"
    )
    print(
        f"Images without filename:  "
        f"{diagnostics['images_without_filename']}"
    )
    print(
        f"Train/test overlap:       "
        f"{diagnostics['train_test_overlap_count']}"
    )
    print(f"Total captions:           {diagnostics['total_captions']}")
    print(f"Output file:              {args.output}")

    if diagnostics["train_test_overlap_count"] > 0:
        print(
            "\nWARNING: Some test IDs also occur in the "
            "training selection. Check the overlap IDs in the JSON."
        )

    if diagnostics["images_without_filename"] > 0:
        print(
            "\nWARNING: Some IDs were not found in the available "
            "COCO image metadata."
        )

    print("============================================")


if __name__ == "__main__":
    main()
