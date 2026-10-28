
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
    """
    Build a unified lookup dictionary for the selected COCO images.

    Each image is indexed by its COCO image ID and contains:
    - filename
    - image URL
    - ground-truth captions
    - annotated object labels
    """

    # Load the team's selected image manifest.
    selection_data = load_json(selection_file)

    if not isinstance(selection_data.get("images"), list):
        raise ValueError(
            "The selection JSON must contain an 'images' list."
        )

    selected_images = {
        image["id"]: {
            "file_name": image["file_name"],
            "url": image.get("url"),
        }
        for image in selection_data["images"]
    }

    selected_ids = set(selected_images)

    # Load official caption annotations.
    captions_data = load_json(captions_file)

    captions_by_image = defaultdict(list)

    for annotation in captions_data.get("annotations", []):
        image_id = annotation["image_id"]

        if image_id in selected_ids:
            captions_by_image[image_id].append(
                annotation["caption"]
            )

    # Load official object-instance annotations.
    instances_data = load_json(instances_file)

    category_names = {
        category["id"]: category["name"]
        for category in instances_data.get("categories", [])
    }

    objects_by_image = defaultdict(list)

    for annotation in instances_data.get("annotations", []):
        image_id = annotation["image_id"]

        if image_id not in selected_ids:
            continue

        category_id = annotation["category_id"]
        category_name = category_names.get(category_id)

        if category_name is not None:
            objects_by_image[image_id].append(category_name)

    # Combine the information into one lookup dictionary.
    unified_lookup = {}

    for image_id, metadata in selected_images.items():
        unified_lookup[str(image_id)] = {
            "file_name": metadata["file_name"],
            "url": metadata["url"],
            "captions": captions_by_image.get(image_id, []),
            "objects": sorted(
                set(objects_by_image.get(image_id, []))
            ),
        }

    return unified_lookup


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a unified caption and object lookup "
            "for the selected COCO images."
        )
    )

    parser.add_argument(
        "--selection-file",
        type=Path,
        default=Path("data/coco_rag_data/train_subset_10000.json"),
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

    for file_path in (
        args.selection_file,
        args.captions_file,
        args.instances_file,
    ):
        if not file_path.is_file():
            parser.error(f"Required file not found: {file_path}")

    try:
        lookup = build_unified_lookup(
            selection_file=args.selection_file,
            captions_file=args.captions_file,
            instances_file=args.instances_file,
        )
    except (KeyError, ValueError, json.JSONDecodeError, OSError) as error:
        parser.error(f"Failed to build lookup: {error}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        json.dump(
            lookup,
            file,
            indent=2,
            ensure_ascii=False,
        )

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

    print("\n FINAL Unified Lookup Summary:")
    print(f"Selected images:          {total_images}")
    print(f"Images with captions:     {images_with_captions}")
    print(f"Images with object labels:{images_with_objects:>8}")
    print(f"Total captions:           {total_captions}")
    print(f"Output file:              {args.output}")


if __name__ == "__main__":
    main()
