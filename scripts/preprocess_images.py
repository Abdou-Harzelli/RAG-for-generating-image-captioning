import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


# Supported image extensions.
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def get_exif_orientation(image: Image.Image) -> int:
    """
    Return the EXIF orientation value.

    Orientation 1 means the image is already in its normal
    orientation. If EXIF data is missing, return 1.
    """
    try:
        exif = image.getexif()
        return exif.get(274, 1)
    except (AttributeError, TypeError, ValueError):
        return 1


def transform_image(
    image_path: Path,
    output_path: Path,
    target_size: tuple[int, int],
) -> dict:
    """
    Correct orientation, convert to RGB, resize, and center-crop.

    The original image is never modified.

    Returns statistics describing the transformations.
    """

    with Image.open(image_path) as original:
        original_format = original.format
        original_mode = original.mode
        original_size = original.size

        # Step 1: Check EXIF orientation before applying correction.
        original_orientation = get_exif_orientation(original)

        # Step 2: Correct orientation using EXIF metadata.
        image = ImageOps.exif_transpose(original)

        # Step 3: Convert to RGB.
        image = image.convert("RGB")

        # Step 4: Resize and center-crop to the target dimensions.
        # ImageOps.fit resizes the image and crops it to the
        # requested aspect ratio and dimensions.
        image = ImageOps.fit(
            image,
            target_size,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )

        # Save processed images as JPEG.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image.save(
            output_path,
            format="JPEG",
            quality=95,
            optimize=True,
        )

        # Step 5: Validate the saved image.
        with Image.open(output_path) as processed:
            processed.verify()

        return {
            "original_format": original_format,
            "original_mode": original_mode,
            "original_size": list(original_size),
            "exif_orientation_corrected": original_orientation != 1,
            "rgb_conversion_applied": original_mode != "RGB",
            "processed_size": list(target_size),
            "processed_mode": "RGB",
            "output_file": str(output_path),
        }


def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    report_path: Path,
    target_size: tuple[int, int],
) -> dict:
    """
    Process every supported image in the input directory.

    Successfully processed images are saved separately.
    Failures are recorded without stopping the entire batch.
    """

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: {input_dir}"
        )

    # Find images recursively and ignore generated outputs.
    image_paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and output_dir.resolve() not in path.resolve().parents
    )

    if not image_paths:
        raise ValueError(
            f"No supported images found in: {input_dir}"
        )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "target_size": list(target_size),
        "output_format": "JPEG",
        "output_mode": "RGB",
        "total_images_found": len(image_paths),
        "processed_successfully": 0,
        "failed_images": 0,
        "exif_orientation_corrected_count": 0,
        "rgb_conversion_count": 0,
        "errors": [],
        "images": [],
    }

    print(f"Images found: {len(image_paths)}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target dimensions: {target_size[0]} x {target_size[1]}")
    print("\nStarting image preprocessing...\n")

    for index, image_path in enumerate(image_paths, start=1):
        # Preserve the relative directory structure if present.
        relative_path = image_path.relative_to(input_dir)

        # Normalize output extension because all outputs are JPEG.
        output_path = (
            output_dir / relative_path
        ).with_suffix(".jpg")

        try:
            result = transform_image(
                image_path=image_path,
                output_path=output_path,
                target_size=target_size,
            )

            report["processed_successfully"] += 1

            if result["exif_orientation_corrected"]:
                report["exif_orientation_corrected_count"] += 1

            if result["rgb_conversion_applied"]:
                report["rgb_conversion_count"] += 1

            report["images"].append({
                "input_file": str(image_path),
                **result,
            })

            print(
                f"[{index}/{len(image_paths)}] "
                f"OK: {image_path.name}"
            )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            SyntaxError,
        ) as error:
            report["failed_images"] += 1

            error_details = {
                "file_name": str(image_path),
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

            report["errors"].append(error_details)

            print(
                f"[{index}/{len(image_paths)}] "
                f"FAILED: {image_path.name} - {error}"
            )

    # Save the report as JSON.
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n========== PREPROCESSING SUMMARY ==========")
    print(f"Images found: {report['total_images_found']}")
    print(
        "Processed successfully: "
        f"{report['processed_successfully']}"
    )
    print(f"Failed images: {report['failed_images']}")
    print(
        "EXIF orientation corrections: "
        f"{report['exif_orientation_corrected_count']}"
    )
    print(
        "Images converted to RGB: "
        f"{report['rgb_conversion_count']}"
    )
    print(f"Output dimensions: {target_size}")
    print(f"JSON report: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess COCO images: correct EXIF orientation, "
            "convert to RGB, resize, and center-crop."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/coco_rag_data/train2017_subset"),
        help="Directory containing the original images.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/coco_rag_data/processed_images"),
        help="Directory where processed images will be saved.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/coco_rag_data/image_transformation_report.json"
        ),
        help="Path for the JSON processing report.",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Target width and height in pixels.",
    )

    args = parser.parse_args()

    if args.size < 1:
        parser.error("--size must be a positive integer.")

    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error(
            "Input and output directories must be different."
        )

    preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        target_size=(args.size, args.size),
    )


if __name__ == "__main__":
    main()