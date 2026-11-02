import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_IMAGE_DIR = Path("data/coco_rag_data/processed_images_batch")
DEFAULT_OUTPUT_DIR = Path("data/coco_rag_data/image_embeddings")


# --------------------------------------------------
# 1. Load CLIP model and processor
# --------------------------------------------------

def load_model(device):
    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {device}")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    return model, processor


# --------------------------------------------------
# 2. Find and sort images
# --------------------------------------------------

def get_image_paths(image_dir, limit=None):
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted(image_dir.glob("*.jpg"))

    if limit is not None:
        image_paths = image_paths[:limit]

    if not image_paths:
        raise ValueError(f"No JPG images found in {image_dir}")

    print(f"Found {len(image_paths)} images.")

    return image_paths


# --------------------------------------------------
# 3. Load images safely
# --------------------------------------------------

def load_batch_images(batch_paths):
    valid_images = []
    valid_paths = []
    errors = []

    for path in batch_paths:
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                valid_images.append(image.copy())
                valid_paths.append(path)

        except Exception as error:
            errors.append({
                "file_name": path.name,
                "error": str(error)
            })

    return valid_images, valid_paths, errors


# --------------------------------------------------
# 4. Extract embeddings from one batch
# --------------------------------------------------

def extract_batch_embeddings(
    model,
    processor,
    images,
    device,
    normalize=True
):
    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        vision_outputs = model.vision_model(
            pixel_values=pixel_values
        )

        image_features = model.visual_projection(
            vision_outputs.pooler_output
        )

        if normalize:
            image_features = F.normalize(
                image_features,
                p=2,
                dim=-1
            )

    return image_features.cpu().numpy().astype(np.float32)


# --------------------------------------------------
# 5. Save embeddings
# --------------------------------------------------

def save_embeddings(embeddings, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, embeddings)

    print(f"Embeddings saved to: {output_path}")


# --------------------------------------------------
# 6. Save image-to-embedding mapping
# --------------------------------------------------

def save_mapping(mapping, output_path):
    output_path = Path(output_path)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(mapping, file, indent=4, ensure_ascii=False)

    print(f"Mapping saved to: {output_path}")


# --------------------------------------------------
# 7. Main extraction pipeline
# --------------------------------------------------

def generate_image_embeddings(
    image_dir,
    output_dir,
    batch_size=16,
    normalize=True,
    limit=None
):
    start_time = time.perf_counter()
    started_at = datetime.now(timezone.utc)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, processor = load_model(device)
    image_paths = get_image_paths(image_dir, limit)

    all_embeddings = []
    mapping = []
    errors = []

    total_batches = (
        len(image_paths) + batch_size - 1
    ) // batch_size

    for batch_index in tqdm(
        range(0, len(image_paths), batch_size),
        total=total_batches,
        desc="Extracting embeddings"
    ):
        batch_paths = image_paths[
            batch_index:batch_index + batch_size
        ]

        images, valid_paths, batch_errors = load_batch_images(
            batch_paths
        )

        errors.extend(batch_errors)

        if not images:
            continue

        try:
            batch_embeddings = extract_batch_embeddings(
                model=model,
                processor=processor,
                images=images,
                device=device,
                normalize=normalize
            )

            all_embeddings.append(batch_embeddings)

            for path in valid_paths:
                try:
                    image_id = int(path.stem)
                except ValueError:
                    image_id = None

                mapping.append({
                    "index": len(mapping),
                    "image_id": image_id,
                    "file_name": path.name
                })

        except Exception as error:
            for path in valid_paths:
                errors.append({
                    "file_name": path.name,
                    "error": str(error)
                })

    if not all_embeddings:
        raise RuntimeError("No embeddings were successfully extracted.")

    final_embeddings = np.concatenate(
        all_embeddings,
        axis=0
    ).astype(np.float32)

    # --------------------------------------------------
    # 8. Quality checks
    # --------------------------------------------------

    expected_dimension = 512

    if final_embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix.")

    if final_embeddings.shape[1] != expected_dimension:
        raise ValueError(
            f"Expected {expected_dimension} dimensions, "
            f"got {final_embeddings.shape[1]}"
        )

    if not np.isfinite(final_embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    norms = np.linalg.norm(final_embeddings, axis=1)

    if normalize and not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("L2 normalization validation failed.")

    # --------------------------------------------------
    # 9. Save outputs
    # --------------------------------------------------

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "image_embeddings.npy"
    mapping_path = output_dir / "image_embeddings_mapping.json"
    metadata_path = output_dir / "image_embeddings_metadata.json"
    report_path = output_dir / "embedding_report.json"

    save_embeddings(final_embeddings, embeddings_path)
    save_mapping(mapping, mapping_path)

    elapsed_seconds = round(
        time.perf_counter() - start_time, 2
    )

    metadata = {
        "model_name": MODEL_NAME,
        "framework": "PyTorch + Hugging Face Transformers",
        "embedding_dimension": expected_dimension,
        "number_of_embeddings": len(final_embeddings),
        "dtype": str(final_embeddings.dtype),
        "normalized": normalize,
        "normalization_method": "L2" if normalize else None,
        "device": str(device),
        "input_directory": str(image_dir),
        "output_shape": list(final_embeddings.shape),
        "created_at_utc": started_at.isoformat()
    }

    report = {
        "status": "SUCCESS" if not errors else "COMPLETED_WITH_ERRORS",
        "total_images_found": len(image_paths),
        "images_processed": len(final_embeddings),
        "images_failed": len(errors),
        "embedding_dimension": expected_dimension,
        "embedding_shape": list(final_embeddings.shape),
        "dtype": str(final_embeddings.dtype),
        "all_values_finite": bool(
            np.isfinite(final_embeddings).all()
        ),
        "normalized": normalize,
        "minimum_vector_norm": float(norms.min()),
        "maximum_vector_norm": float(norms.max()),
        "elapsed_seconds": elapsed_seconds,
        "errors": errors
    }

    save_mapping(metadata, metadata_path)
    save_mapping(report, report_path)

    # --------------------------------------------------
    # 10. Final summary
    # --------------------------------------------------

    print("\n========== EMBEDDING EXTRACTION SUMMARY ==========")
    print(f"Images found       : {len(image_paths)}")
    print(f"Images processed   : {len(final_embeddings)}")
    print(f"Images failed      : {len(errors)}")
    print(f"Embedding shape    : {final_embeddings.shape}")
    print(f"Embedding dtype    : {final_embeddings.dtype}")
    print(f"Normalized         : {normalize}")
    print(f"Elapsed seconds    : {elapsed_seconds}")
    print(f"Status             : {report['status']}")
    print(f"Output directory   : {output_dir}")
    print("==================================================")


# --------------------------------------------------
# 11. Command-line arguments
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CLIP image embeddings from COCO images."
    )

    parser.add_argument(
        "--image-dir",
        default=str(DEFAULT_IMAGE_DIR)
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR)
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of images (useful for testing)."
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization."
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero.")

    generate_image_embeddings(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
        limit=args.limit
    )