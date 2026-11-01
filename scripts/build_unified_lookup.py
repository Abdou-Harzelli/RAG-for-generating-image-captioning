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


def save_json(data: dict, file_path: Path) -> None:
    """Save data to a JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def build_unified_lookup(
    selection_file: Path,
    captions_file: Path,
    instances_file: Path,
) -> dict:
    """
    Build a unified lookup containing image metadata, captions,
    and object labels for the selected training images.
    """

    # 1. Load the selected images.
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
                f"Duplicate image ID in selection file: {image_id}"
            )

        selected_images[image_id] = {
            "file_name": image["file_name"],
            "url": image.get("url"),
        }

    selected_ids = set(selected_images.keys())

    # 2. Load captions and group them by image ID.
    captions_data = load_json(captions_file)
    captions_by_image = defaultdict(list)

    for annotation in captions_data.get("annotations", []):
        image_id = annotation["image_id"]

        if image_id in selected_ids:
            captions_by_image[image_id].append(
                annotation["caption"]
            )

    # 3. Load object categories.
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

    # 4. Build the unified lookup.
    unified_lookup = {}

    for image_id, metadata in selected_images.items():
        unified_lookup[str(image_id)] = {
            "file_name": metadata["file_name"],
            "url": metadata["url"],
            "captions": captions_by_image.get(image_id, []),
            "objects": sorted(
                objects_by_image.get(image_id, set())
            ),
        }

    return unified_lookup


def clean_captions(
    unified_lookup: dict,
    max_captions: int = 5,
) -> tuple[dict, dict, list]:
    """
    Limit each image to max_captions and record all modifications.

    The first captions in COCO annotation order are preserved.

    Images with fewer captions are not given artificial captions.

    Returns:
        cleaned_lookup: Metadata after cleaning.
        cleaning_report: Cleaning statistics.
        changes: Before-and-after details for modified images.
    """

    cleaned_lookup = {}
    changes = []

    total_captions_before = 0
    total_captions_after = 0

    images_with_fewer_than_target = 0

    for image_id, metadata in unified_lookup.items():
        original_captions = metadata.get("captions", [])

        total_captions_before += len(original_captions)

        # Preserve the first max_captions captions.
        cleaned_captions = original_captions[:max_captions]

        total_captions_after += len(cleaned_captions)

        if len(original_captions) < max_captions:
            images_with_fewer_than_target += 1

        # Preserve the original metadata and replace its captions.
        cleaned_metadata = {
            **metadata,
            "captions": cleaned_captions,
        }

        cleaned_lookup[image_id] = cleaned_metadata

        # Record only images whose captions actually changed.
        if len(original_captions) > max_captions:
            removed_captions = original_captions[max_captions:]

            changes.append({
                "image_id": int(image_id),
                "file_name": metadata.get("file_name"),
                "captions_count_before": len(original_captions),
                "captions_count_after": len(cleaned_captions),
                "captions_before": original_captions,
                "captions_after": cleaned_captions,
                "captions_removed": removed_captions,
            })

    cleaning_report = {
        "total_images": len(unified_lookup),
        "max_captions_per_image": max_captions,
        "total_captions_before": total_captions_before,
        "total_captions_after": total_captions_after,
        "images_modified": len(changes),
        "captions_removed": (
            total_captions_before - total_captions_after
        ),
        "images_with_fewer_than_target": (
            images_with_fewer_than_target
        ),
        "images_with_exactly_target_captions": sum(
            len(metadata.get("captions", [])) == max_captions
            for metadata in cleaned_lookup.values()
        ),
        "all_images_have_exactly_target_captions": all(
            len(metadata.get("captions", [])) == max_captions
            for metadata in cleaned_lookup.values()
        ),
        "cleaning_strategy": (
            f"Keep the first {max_captions} captions in "
            "COCO annotation order; record removed captions."
        ),
    }

    return cleaned_lookup, cleaning_report, changes


def validate_lookup(
    unified_lookup: dict,
    expected_captions: int = 5,
) -> dict:
    """Validate caption counts in the cleaned lookup."""

    caption_counts = [
        len(metadata.get("captions", []))
        for metadata in unified_lookup.values()
    ]

    images_with_captions = sum(
        count > 0 for count in caption_counts
    )

    images_with_objects = sum(
        bool(metadata.get("objects", []))
        for metadata in unified_lookup.values()
    )

    return {
        "total_images": len(unified_lookup),
        "images_with_captions": images_with_captions,
        "images_without_captions": (
            len(unified_lookup) - images_with_captions
        ),
        "images_with_objects": images_with_objects,
        "images_without_objects": (
            len(unified_lookup) - images_with_objects
        ),
        "total_captions": sum(caption_counts),
        "images_with_expected_caption_count": sum(
            count == expected_captions
            for count in caption_counts
        ),
        "images_with_unexpected_caption_count": sum(
            count != expected_captions
            for count in caption_counts
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build and clean the COCO unified image lookup, "
            "and generate a before-and-after caption report."
        )
    )

    parser.add_argument(
        "--selection-file",
        type=Path,
        default=Path(
            "data/coco_rag_data/train_subset_10000.json"
        ),
        help="JSON file containing selected training images.",
    )

    parser.add_argument(
        "--captions-file",
        type=Path,
        default=Path(
            "data/annotations_trainval2017/annotations/"
            "captions_train2017.json"
        ),
        help="Official COCO captions annotations.",
    )

    parser.add_argument(
        "--instances-file",
        type=Path,
        default=Path(
            "data/annotations_trainval2017/annotations/"
            "instances_train2017.json"
        ),
        help="Official COCO object instance annotations.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/coco_rag_data/unified_lookup.json"
        ),
        help="Output path for the cleaned unified lookup.",
    )

    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path(
            "data/coco_rag_data/caption_cleaning_report.json"
        ),
        help="Output path for cleaning statistics.",
    )

    parser.add_argument(
        "--changes-output",
        type=Path,
        default=Path(
            "data/coco_rag_data/caption_changes.json"
        ),
        help="Output path for before-and-after comparisons.",
    )

    parser.add_argument(
        "--max-captions",
        type=int,
        default=5,
        help="Maximum number of captions per image.",
    )

    args = parser.parse_args()

    if args.max_captions < 1:
        parser.error("--max-captions must be at least 1.")

    # Step 1: Build the original lookup.
    print("Building the unified lookup...")

    original_lookup = build_unified_lookup(
        selection_file=args.selection_file,
        captions_file=args.captions_file,
        instances_file=args.instances_file,
    )

    print(
        f"Selected images: {len(original_lookup)}"
    )

    # Step 2: Clean captions and record modifications.
    print("\nCleaning captions...")

    cleaned_lookup, cleaning_report, changes = clean_captions(
        unified_lookup=original_lookup,
        max_captions=args.max_captions,
    )

    # Step 3: Validate the cleaned lookup.
    validation_report = validate_lookup(
        cleaned_lookup,
        expected_captions=args.max_captions,
    )

    # Step 4: Save the cleaned lookup.
    save_json(cleaned_lookup, args.output)

    # Step 5: Save the cleaning report.
    cleaning_report["validation"] = validation_report

    save_json(cleaning_report, args.report_output)

    # Step 6: Save before-and-after comparisons.
    changes_data = {
        "total_modified_images": len(changes),
        "changes": changes,
    }

    save_json(changes_data, args.changes_output)

    # Step 7: Display a summary.
    print("\n========== CLEANING SUMMARY ==========")
    print(f"Total images: {cleaning_report['total_images']}")
    print(
        "Captions before: "
        f"{cleaning_report['total_captions_before']}"
    )
    print(
        "Captions after: "
        f"{cleaning_report['total_captions_after']}"
    )
    print(
        "Images modified: "
        f"{cleaning_report['images_modified']}"
    )
    print(
        "Captions removed: "
        f"{cleaning_report['captions_removed']}"
    )
    print(
        "Images with fewer than "
        f"{args.max_captions} captions: "
        f"{cleaning_report['images_with_fewer_than_target']}"
    )
    print(
        "Images with unexpected caption counts: "
        f"{validation_report['images_with_unexpected_caption_count']}"
    )

    print("\n========== OUTPUT FILES ==========")
    print(f"Cleaned lookup: {args.output}")
    print(f"Cleaning report: {args.report_output}")
    print(f"Before/after comparisons: {args.changes_output}")


if __name__ == "__main__":
    main()
