import argparse
import json
from collections import defaultdict
from pathlib import Path
'''
 le but f had le fichier c que ndiro lookup li fiha les infos tae l'image + objects li fiha ldakhl
 based on captions li kayn f json file (tae images)

'''

def load_json(file_path: Path) -> dict:
    """Load a JSON file."""
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_unified_lookup(
    selection_file: Path,
    captions_file: Path,
    instances_file: Path,
) -> dict:
    """Build a unified lookup for selected COCO training images."""

    selection_data = load_json(selection_file)

    if not isinstance(selection_data.get("images"), list):
        raise ValueError(
            "The selection JSON must contain an 'images' list."
        )

    selected_images = {}

    for image in selection_data["images"]:
        image_id = image["id"]

        if image_id in selected_images:
            raise ValueError(
                f"Duplicate image ID in training selection: {image_id}"
            )

        selected_images[image_id] = {
            "file_name": image["file_name"],
            "url": image.get("url"),
        }

    selected_ids = set(selected_images)

    # Load official captions.
    captions_data = load_json(captions_file)
    captions_by_image = defaultdict(list)

    for annotation in captions_data.get("annotations", []):
        image_id = annotation["image_id"]

        if image_id in selected_ids:
            captions_by_image[image_id].append(
                annotation["caption"]
            )

    # Load official object annotations.
    instances_data = load_json(instances_file)

    category_names = {
        category["id"]: category["name"]
        for category in instances_data.get("categories", [])
    }

    objects_by_image = defaultdict(set)

    for annotation in instances_data.get("annotations", []):
        image_id = annotation["image_id"]

        if image_id not in selected_ids:
            continue

        category_name = category_names.get(
            annotation["category_id"]
        )

        if category_name:
            objects_by_image[image_id].add(category_name)

    # Build the unified lookup.
    unified_lookup = {}

    for image_id, metadata in selected_images.items():
        unified_lookup[str(image_id)] = {
            "file_name": metadata["file_name"],
            "url": metadata["url"],
            "captions": captions_by_image.get(image_id, []),
            "objects": sorted(objects_by_image.get(image_id, set())),
        }

    return unified_lookup


def main():
    parser = argparse.ArgumentParser(
        description="Build a unified lookup for COCO training images."
    )

    parser.add_argument(
        "--selection-file",
        type=Path,
        default=Path(
            "data/coco_rag_data/train_subset_10000.json"
        ),
    )

    parser.add_argument(
        "--captions-file",
        type=Path,
        default=Path(
            "data/annotations_trainval2017/annotations/"
            "captions_train2017.json"
        ),
    )

    parser.add_argument(
        "--instances-file",
        type=Path,
        default=Path(
            "data/annotations_trainval2017/annotations/"
            "instances_train2017.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/coco_rag_data/unified_lookup.json"
        ),
    )

    args = parser.parse_args()

    required_files = [
        args.selection_file,
        args.captions_file,
        args.instances_file,
    ]

    for file_path in required_files:
        if not file_path.is_file():
            parser.error(f"Required file not found: {file_path}")

    try:
        lookup = build_unified_lookup(
            selection_file=args.selection_file,
            captions_file=args.captions_file,
            instances_file=args.instances_file,
        )

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.output.open("w", encoding="utf-8") as file:
            json.dump(
                lookup,
                file,
                indent=2,
                ensure_ascii=False,
            )

    except (KeyError, ValueError, json.JSONDecodeError, OSError) as error:
        parser.error(f"Failed to build training lookup: {error}")

    total_images = len(lookup)

    images_with_captions = sum(
        bool(item["captions"])
        for item in lookup.values()
    )

    images_with_objects = sum(
        bool(item["objects"])
        for item in lookup.values()
    )

    total_captions = sum(
        len(item["captions"])
        for item in lookup.values()
    )

    print("\n========== UNIFIED LOOKUP SUMMARY ==========")
    print(f"Selected images:           {total_images}")
    print(f"Images with captions:      {images_with_captions}")
    print(f"Images with object labels: {images_with_objects}")
    print(f"Total captions:            {total_captions}")
    print(f"Output file:               {args.output}")
    print("=============================================")


if __name__ == "__main__":
    main()
