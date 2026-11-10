import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


# Project root: the parent directory of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SOURCE = PROJECT_ROOT / "data" / "coco_rag_data" / "train2017_subset"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "coco_rag_data" / "processed_images_batch"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_SIZE = (224, 224)
DEFAULT_IDS_FILE = PROJECT_ROOT / "data" / "coco_rag_data" / "train_kb_ids.json"


def load_allowed_filenames(ids_file: Path) -> set[str]:
    with ids_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"IDs file must contain a JSON list: {ids_file}")

    ids: list[int] = []
    for item in data:
        value = item.get("id") if isinstance(item, dict) else item
        try:
            ids.append(int(value))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid image ID in {ids_file}: {value!r}") from error

    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs found in {ids_file}")

    return {f"{image_id:012d}.jpg" for image_id in ids}


def parse_arguments():
    """Read configurable batch-processing options."""
    parser = argparse.ArgumentParser(
        description="Batch-preprocess COCO images into 224x224 RGB JPEG files."
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Directory containing the original images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory where processed images will be saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of images handled in each batch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of source images to process; useful for testing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip output images that already pass validation.",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=DEFAULT_IDS_FILE,
        help="JSON file with image IDs to preprocess (default: train_kb_ids.json).",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero.")

    return args


def discover_images(source_dir: Path, output_dir: Path) -> list[Path]:
    """Find supported images recursively without scanning the output folder."""
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Source directory does not exist: {source_dir}"
        )

    images = []

    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        # Do not accidentally process generated files as source files.
        if path.resolve().is_relative_to(output_dir):
            continue

        images.append(path)

    return sorted(images)


def output_path_for(image_path: Path, source_dir: Path, output_dir: Path) -> Path:
    """
    Preserve subdirectories and filenames while changing the extension to .jpg.

    Example:
      source/subfolder/image.png -> output/subfolder/image.jpg
    """
    relative_path = image_path.relative_to(source_dir)
    return (output_dir / relative_path).with_suffix(".jpg")


def is_valid_output(path: Path) -> bool:
    """Return True only when an existing output is a readable JPEG, RGB, 224x224."""
    if not path.is_file():
        return False

    try:
        with Image.open(path) as image:
            if image.format != "JPEG":
                return False

            if image.size != TARGET_SIZE or image.mode != "RGB":
                return False

            image.verify()

        return True

    except (OSError, ValueError, UnidentifiedImageError):
        return False


def preprocess_one(source_path: Path, destination_path: Path) -> None:
    """Convert one image into a 224x224 RGB JPEG."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    # Correct the image orientation using EXIF metadata before resizing.
    with Image.open(source_path) as original:
        image = ImageOps.exif_transpose(original)
        image = image.convert("RGB")
        image = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

        # Write to a temporary file first to avoid leaving a partial JPEG.
        temporary_path = destination_path.with_name(
            destination_path.stem + ".tmp.jpg"
        )

        try:
            image.save(
                temporary_path,
                format="JPEG",
                quality=95,
                optimize=True,
            )
            temporary_path.replace(destination_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def main():
    args = parse_arguments()

    source_dir = args.source.resolve()
    output_dir = args.output.resolve()

    if source_dir == output_dir or source_dir in output_dir.parents:
        raise ValueError(
            "The output directory must not be the source directory "
            "or a directory inside it."
        )

    images = discover_images(source_dir, output_dir)
    images_before_filter = len(images)

    ids_file = args.ids_file.resolve()
    if not ids_file.is_file():
        raise FileNotFoundError(
            f"IDs file does not exist: {ids_file}"
        )

    allowed_filenames = load_allowed_filenames(ids_file)
    images = [image for image in images if image.name in allowed_filenames]

    if args.limit is not None:
        images = images[:args.limit]

    if not images:
        print(f"No supported images found in: {source_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(images)
    total_batches = (total + args.batch_size - 1) // args.batch_size

    processed = 0
    skipped = 0
    failed = []
    start_time = time.perf_counter()

    print("=" * 55)
    print("BATCH IMAGE PREPROCESSING")
    print("=" * 55)
    print(f"Source directory : {source_dir}")
    print(f"Output directory : {output_dir}")
    print(f"IDs file         : {ids_file}")
    print(f"Images discovered: {images_before_filter}")
    print(f"IDs requested    : {len(allowed_filenames)}")
    print(f"Images selected  : {total}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Total batches    : {total_batches}")
    print(f"Resume enabled   : {args.resume}")
    print("=" * 55)

    for batch_index in range(total_batches):
        start = batch_index * args.batch_size
        end = min(start + args.batch_size, total)
        batch = images[start:end]

        print(
            f"\nBatch {batch_index + 1}/{total_batches} "
            f"({len(batch)} images)"
        )

        for image_path in batch:
            destination = output_path_for(
                image_path, source_dir, output_dir
            )

            try:
                if args.resume and is_valid_output(destination):
                    skipped += 1
                    continue

                preprocess_one(image_path, destination)

                # Verify the result immediately after writing it.
                if not is_valid_output(destination):
                    raise ValueError(
                        "Output validation failed: expected JPEG, "
                        "RGB mode, and 224x224 dimensions."
                    )

                processed += 1

            except Exception as error:
                failed.append(
                    {
                        "source": str(image_path.relative_to(source_dir)),
                        "error": str(error),
                    }
                )

        completed = end
        percentage = completed / total * 100

        print(
            f"Progress: {completed}/{total} "
            f"({percentage:.1f}%) | "
            f"Processed: {processed} | "
            f"Skipped: {skipped} | "
            f"Failed: {len(failed)}"
        )

    elapsed_seconds = round(time.perf_counter() - start_time, 2)

    if failed:
        status = "COMPLETED_WITH_ERRORS"
    else:
        status = "SUCCESS"

    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source_dir),
        "output_directory": str(output_dir),
        "ids_filter_file": str(ids_file),
        "images_discovered_before_filter": images_before_filter,
        "requested_ids_count": len(allowed_filenames),
        "total_images_selected": total,
        "batch_size": args.batch_size,
        "total_batches": total_batches,
        "processed_successfully": processed,
        "skipped_valid_outputs": skipped,
        "failed": len(failed),
        "elapsed_seconds": elapsed_seconds,
        "output_format": "JPEG",
        "output_dimensions": list(TARGET_SIZE),
        "output_mode": "RGB",
        "status": status,
        "errors": failed,
    }

    report_path = output_dir.parent / "batch_processing_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 55)
    print("FINAL BATCH PROCESSING SUMMARY")
    print("=" * 55)
    print(f"Selected images : {total}")
    print(f"Processed       : {processed}")
    print(f"Skipped         : {skipped}")
    print(f"Failed          : {len(failed)}")
    print(f"Elapsed seconds : {elapsed_seconds}")
    print(f"Status          : {status}")
    print(f"JSON report     : {report_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()