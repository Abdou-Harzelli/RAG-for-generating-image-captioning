import argparse
import json
import random
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SEED = 20261027
NUM_TRAIN_IMAGES = 10_000
BASE_URL = "http://images.cocodataset.org/train2017"
USER_AGENT = "COCO-RAG-Student-Project/1.0"


def find_train_captions(root: Path) -> Path:
    candidates = [
        root / "annotations" / "captions_train2017.json",
        root / "captions_train2017.json",
        root / "annotations_trainval2017" / "annotations" / "captions_train2017.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find captions_train2017.json. Extract "
        "annotations_trainval2017.zip, then pass the folder containing "
        "the annotations/ directory (or the JSON file's parent folder)."
    )


def load_shared_selection(captions_path: Path, output_dir: Path) -> list[dict]:
    selection_path = output_dir / "train_subset_10000.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the saved selection so reruns never silently change the dataset.
    if selection_path.exists():
        with selection_path.open("r", encoding="utf-8") as f:
            saved = json.load(f)
        print(f"Using existing selection: {selection_path}")
        return saved["images"]

    with captions_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    captions = data.get("annotations", [])
    if len(images) < NUM_TRAIN_IMAGES:
        raise ValueError(
            f"Expected at least {NUM_TRAIN_IMAGES} training images in "
            f"{captions_path}, found {len(images)}."
        )

    # Sort before seeded sampling so every member gets the same selection.
    images = sorted(images, key=lambda item: int(item["id"]))
    rng = random.Random(SEED)
    selected = sorted(rng.sample(images, NUM_TRAIN_IMAGES), key=lambda item: int(item["id"]))
    selected_ids = {int(item["id"]) for item in selected}

    captions_by_id: dict[int, list[str]] = {image_id: [] for image_id in selected_ids}
    for ann in captions:
        image_id = int(ann["image_id"])
        if image_id in captions_by_id:
            captions_by_id[image_id].append(ann["caption"])

    records = []
    for image in selected:
        image_id = int(image["id"])
        filename = image.get("file_name") or f"{image_id:012d}.jpg"
        records.append({
            "id": image_id,
            "file_name": filename,
            "url": f"{BASE_URL}/{filename}",
            "captions": captions_by_id[image_id],
        })

    with selection_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"seed": SEED, "count": len(records), "source": str(captions_path),
             "images": records},
            f, ensure_ascii=False, indent=2
        )

    # Also export a simple CSV-like ID list, one filename per line.
    with (output_dir / "train_subset_10000_filenames.txt").open("w", encoding="utf-8") as f:
        for item in records:
            f.write(item["file_name"] + "\n")

    print(f"Saved shared selection: {selection_path}")
    print(f"Selection seed: {SEED}; images: {len(records)}")
    return records


def download_one(url: str, destination: Path, timeout: int = 45) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP status {response.status}")
        content = response.read()
    if not content:
        raise RuntimeError("Downloaded an empty file")
    destination.write_bytes(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir", type=Path, default=Path("."),
        help="Folder containing annotations/captions_train2017.json (default: current folder)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("coco_rag_data"),
        help="Output folder (default: ./coco_rag_data)"
    )
    parser.add_argument(
        "--limit", type=int, default=NUM_TRAIN_IMAGES,
        help="Download only the first N records from the shared 10,000-image selection "
             "(for testing; use 10000 for the project)"
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Only create the deterministic selection and caption metadata"
    )
    parser.add_argument(
        "--retries", type=int, default=4,
        help="Retries per image (default: 4)"
    )
    args = parser.parse_args()

    if not 1 <= args.limit <= NUM_TRAIN_IMAGES:
        parser.error(f"--limit must be between 1 and {NUM_TRAIN_IMAGES}")

    try:
        captions_path = find_train_captions(args.annotations_dir)
        records = load_shared_selection(captions_path, args.output_dir)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Keep a compact metadata file for the images actually requested.
    chosen = records[:args.limit]
    images_dir = args.output_dir / "train2017_subset"
    images_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.output_dir / f"captions_metadata_{len(chosen)}.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(chosen, f, ensure_ascii=False, indent=2)

    print(f"Metadata written to: {metadata_path}")
    print(f"Images will be saved in: {images_dir}")

    if args.no_download:
        print("No image downloads requested.")
        return 0

    success = 0
    failed = []
    total = len(chosen)

    for index, item in enumerate(chosen, start=1):
        filename = item["file_name"]
        destination = images_dir / filename
        if destination.exists() and destination.stat().st_size > 0:
            success += 1
            continue

        downloaded = False
        for attempt in range(1, args.retries + 1):
            try:
                download_one(item["url"], destination)
                downloaded = True
                success += 1
                break
            except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
                if attempt == args.retries:
                    failed.append({"id": item["id"], "file_name": filename, "error": str(exc)})
                else:
                    time.sleep(min(2 * attempt, 10))

        if index % 100 == 0 or index == total:
            print(f"Progress: {index}/{total}; available: {success}; failed: {len(failed)}")

    if failed:
        failed_path = args.output_dir / "failed_downloads.json"
        failed_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
        print(f"{len(failed)} downloads failed. Details: {failed_path}")
        print("Run the same command again; existing non-empty files will be skipped.")
        return 2

    print(f"Finished: {success}/{total} selected images are available.")
    print("Share train_subset_10000.json and this exact script with all team members.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
