import argparse
import concurrent.futures
import json
import math
import os
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLOWorld


# ============================================================
# JSON / ID UTILITIES
# ============================================================

def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def normalize_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ID d'image invalide : {value!r}") from exc


def load_split_ids(path: Path) -> list[int]:
    data = load_json(path)
    if isinstance(data, list):
        values = data
    elif isinstance(data, dict):
        values = next(
            (data[k] for k in ("ids", "image_ids", "train_kb_ids", "val_ids", "test_ids") if k in data),
            None,
        )
        if values is None:
            raise ValueError(f"Aucune liste d'IDs reconnue dans {path}")
    else:
        raise ValueError(f"Format de split non supporté : {path}")

    if not isinstance(values, list):
        raise ValueError(f"Les IDs doivent être une liste dans {path}")

    ids = [normalize_id(x["id"] if isinstance(x, dict) else x) for x in values]
    if len(ids) != len(set(ids)):
        raise ValueError(f"IDs dupliqués dans {path}")
    return ids


def validate_splits(train: list[int], val: list[int], test: list[int]) -> None:
    sets = {"train/val": set(train) & set(val),
            "train/test": set(train) & set(test),
            "val/test": set(val) & set(test)}
    overlaps = [f"{name}: {len(ids)}" for name, ids in sets.items() if ids]
    if overlaps:
        raise ValueError("FUITE DE DONNÉES : " + ", ".join(overlaps))
    print(">>> Anti-leakage : PASS (aucun chevauchement). <<<")


# ============================================================
# COCO METADATA AND CAPTIONS
# ============================================================

def load_coco_image_metadata(train_data: dict, val_data: dict) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for split_name, dataset in (("train2017", train_data), ("val2017", val_data)):
        for image in dataset.get("images", []):
            image_id = normalize_id(image["id"])
            if image_id in result:
                raise ValueError(f"ID COCO dupliqué entre métadonnées : {image_id}")
            result[image_id] = {
                "file_name": image["file_name"],
                "url": image.get("coco_url") or image.get("flickr_url"),
                "annotation_split": split_name,
            }
    return result


def build_captions_by_image(datasets: list[dict], selected_ids: set[int]) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for dataset in datasets:
        for ann in dataset.get("annotations", []):
            image_id = normalize_id(ann["image_id"])
            caption = ann.get("caption")
            if image_id in selected_ids and isinstance(caption, str) and caption.strip():
                grouped[image_id].append(caption.strip())
    return dict(grouped)


def clean_text_for_retrieval(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# IMAGE DISCOVERY, DOWNLOAD AND VALIDATION
# ============================================================

def is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            return im.width > 0 and im.height > 0
    except (OSError, UnidentifiedImageError, ValueError):
        return False


def resolve_image_path(image_root: Path, file_name: str, annotation_split: str) -> Path | None:
    # Original COCO images first (better for detection);
    # processed_images (224x224) is only a fallback.
    candidates = [
        image_root / "train2017_subset" / file_name,
        image_root / annotation_split / file_name,
        image_root / "images" / annotation_split / file_name,
        image_root / file_name,
        image_root / "images" / file_name,
        image_root / "downloaded_coco" / annotation_split / file_name,
        image_root / "processed_images" / file_name,
        image_root.parent / annotation_split / file_name,
        image_root.parent / "images" / annotation_split / file_name,
    ]
    for candidate in candidates:
        if candidate.is_file() and is_readable_image(candidate):
            return candidate
    return None


def download_one_image(image_id: int, meta: dict, destination: Path, timeout: int, retries: int) -> tuple[int, str | None]:
    url = meta.get("url")
    if not url:
        return image_id, "Aucune URL COCO/Flickr dans les métadonnées"

    destination.parent.mkdir(parents=True, exist_ok=True)
    urls = [url]
    if url.startswith("http://"):
        urls.insert(0, "https://" + url[len("http://"):])

    last_error = "Téléchargement impossible"
    for attempt in range(retries):
        for candidate_url in urls:
            tmp = destination.with_suffix(destination.suffix + ".part")
            try:
                request = urllib.request.Request(
                    candidate_url,
                    headers={"User-Agent": "COCO-RAG-image-downloader/1.0"},
                )
                with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)

                if not is_readable_image(tmp):
                    raise ValueError("Le fichier téléchargé n'est pas une image lisible")

                os.replace(tmp, destination)
                return image_id, None
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
        time.sleep(min(2 ** attempt, 8))
    return image_id, last_error


def prepare_all_images(
    image_root: Path,
    selected_images: dict[int, dict],
    ordered_ids: list[int],
    download_workers: int,
    timeout: int,
    retries: int,
) -> dict[int, Path]:
    print("\n========== IMAGE DISCOVERY ==========")
    resolved: dict[int, Path] = {}
    missing: list[int] = []

    for image_id in ordered_ids:
        meta = selected_images[image_id]
        path = resolve_image_path(image_root, meta["file_name"], meta["annotation_split"])
        if path is None:
            missing.append(image_id)
        else:
            resolved[image_id] = path

    print(f"Images requises : {len(ordered_ids)}")
    print(f"Images déjà disponibles et lisibles : {len(resolved)}")
    print(f"Images à télécharger : {len(missing)}")

    download_root = image_root / "downloaded_coco"
    failures: list[dict] = []

    if missing:
        print("\n========== DOWNLOADING MISSING COCO IMAGES ==========")
        jobs = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=download_workers) as executor:
            for image_id in missing:
                meta = selected_images[image_id]
                target = download_root / meta["annotation_split"] / meta["file_name"]
                jobs[executor.submit(
                    download_one_image, image_id, meta, target, timeout, retries
                )] = (image_id, target)

            completed = 0
            for future in concurrent.futures.as_completed(jobs):
                image_id, target = jobs[future]
                try:
                    returned_id, error = future.result()
                except Exception as exc:
                    returned_id, error = image_id, f"{type(exc).__name__}: {exc}"

                completed += 1
                if error is None and target.is_file() and is_readable_image(target):
                    resolved[returned_id] = target
                else:
                    failures.append({
                        "image_id": returned_id,
                        "file_name": selected_images[returned_id]["file_name"],
                        "url": selected_images[returned_id].get("url"),
                        "error": error or "Image téléchargée mais invalide",
                    })

                if completed % 50 == 0 or completed == len(missing):
                    print(f"Download checked: {completed}/{len(missing)}")

    unresolved = [image_id for image_id in ordered_ids if image_id not in resolved]
    report = {
        "required": len(ordered_ids),
        "found_or_downloaded": len(resolved),
        "missing_or_failed": len(unresolved),
        "failures": failures,
    }
    save_json(report, image_root / "image_download_report.json")

    if unresolved:
        save_json(
            [{"image_id": i, "file_name": selected_images[i]["file_name"],
              "url": selected_images[i].get("url")} for i in unresolved],
            image_root / "missing_yolo_images.json",
        )
        examples = "\n".join(
            f"ID {i}: {selected_images[i]['file_name']}" for i in unresolved[:20]
        )
        raise RuntimeError(
            f"{len(unresolved)} images restent absentes après téléchargement.\n"
            f"Exemples:\n{examples}\n"
            f"Consulte : {image_root / 'image_download_report.json'}\n"
            "Le modèle YOLO-World n'a pas été lancé."
        )

    print(f"Images prêtes : {len(resolved)}/{len(ordered_ids)}")
    return {image_id: resolved[image_id] for image_id in ordered_ids}


# ============================================================
# YOLO-WORLD METADATA EXTRACTION WITH CHECKPOINTS
# ============================================================

MODEL_TYPE = "YOLO-World"

# NOTE: the object/attribute split is a simple word-list heuristic applied to
# the detected label. It is NOT a guaranteed semantic classification.
# Every detection is kept in `detections` with its original label.
ATTRIBUTE_WORDS = {
    "red", "blue", "green", "yellow", "black", "white", "brown",
    "orange", "pink", "purple", "gray", "grey", "colorful",
    "wooden", "metal", "metallic", "plastic", "glass", "stone",
    "small", "large", "big", "tiny", "young", "old", "open",
    "closed", "broken", "dirty", "clean", "striped", "spotted",
    "transparent", "bright", "dark",
}

CHECKPOINT_ENTRY_KEYS = (
    "objects", "attributes", "detector", "confidence_threshold",
    "queries_prompt", "detections", "relationships",
    "scene_context_tags", "scene_context_primary", "image_path_used",
)


def label_is_attribute(label: str) -> bool:
    words = set(re.findall(r"[a-z]+", label.lower().replace("-", " ")))
    return bool(words & ATTRIBUTE_WORDS)


SCENE_CONTEXT_RULES = {
    "transport": {"car", "truck", "bus", "bicycle", "motorcycle", "train", "boat", "traffic light"},
    "sports_outdoor": {"person", "bicycle", "horse", "dog", "frisbee", "skateboard", "surfboard"},
    "urban_outdoor": {"car", "bus", "truck", "traffic light", "person"},
    "nature_outdoor": {"tree", "plant", "flower", "bird", "horse"},
    "indoor_home": {"chair", "table", "sofa", "bed", "bottle", "cup", "bowl", "laptop", "television"},
}


def _box_center_and_area(box_xyxy: list[float]) -> tuple[float, float, float]:
    x1, y1, x2, y2 = box_xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + (w / 2.0)
    cy = y1 + (h / 2.0)
    return cx, cy, w * h


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0

    _, _, area_a = _box_center_and_area(box_a)
    _, _, area_b = _box_center_and_area(box_b)
    union = max(area_a + area_b - inter, 1e-9)
    return inter / union


def infer_relationships(detections: list[dict], max_relationships: int = 24) -> list[dict]:
    """
    Build lightweight spatial relationships from object boxes.
    We only use object-type detections to avoid noisy attribute boxes.
    """
    object_dets = [d for d in detections if d.get("type") == "object" and d.get("box_xyxy")]
    object_dets = sorted(object_dets, key=lambda d: d.get("score", 0.0), reverse=True)
    object_dets = object_dets[:12]

    relationships: list[dict] = []
    seen = set()

    for i, src in enumerate(object_dets):
        for j, dst in enumerate(object_dets):
            if i == j:
                continue

            src_box = src["box_xyxy"]
            dst_box = dst["box_xyxy"]

            src_cx, src_cy, _ = _box_center_and_area(src_box)
            dst_cx, dst_cy, _ = _box_center_and_area(dst_box)
            iou = _box_iou(src_box, dst_box)

            if iou >= 0.2:
                relation = "overlapping"
            else:
                dx = dst_cx - src_cx
                dy = dst_cy - src_cy
                if abs(dx) >= abs(dy):
                    relation = "left_of" if dx > 0 else "right_of"
                else:
                    relation = "above" if dy > 0 else "below"

                dist = math.hypot(dx, dy)
                if dist < 80:
                    relation = "near"

            key = (src["label"], relation, dst["label"])
            if key in seen:
                continue
            seen.add(key)

            relationships.append({
                "subject": src["label"],
                "relation": relation,
                "object": dst["label"],
                "subject_score": round(float(src.get("score", 0.0)), 4),
                "object_score": round(float(dst.get("score", 0.0)), 4),
            })

            if len(relationships) >= max_relationships:
                return relationships

    return relationships


def infer_scene_context(object_labels: set[str]) -> list[str]:
    tags = []
    for tag, needed in SCENE_CONTEXT_RULES.items():
        if object_labels & needed:
            tags.append(tag)
    return tags if tags else ["unknown"]


def is_valid_checkpoint_entry(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and all(k in item for k in CHECKPOINT_ENTRY_KEYS)
        and isinstance(item["objects"], list)
        and isinstance(item["attributes"], list)
        and isinstance(item["detections"], list)
        and isinstance(item["relationships"], list)
        and isinstance(item["scene_context_tags"], list)
        and isinstance(item["scene_context_primary"], str)
    )


def load_checkpoint(path: Path, config: dict, expected_ids: set[int]) -> tuple[dict, float]:
    """Return (valid_results, previously_elapsed_seconds)."""
    if not path.is_file():
        return {}, 0.0
    try:
        saved = load_json(path)
        if saved.get("config") != config:
            print("Checkpoint ignoré : configuration différente (modèle, seuil, classes ou device).")
            return {}, 0.0

        raw_results = saved.get("results", {})
        if not isinstance(raw_results, dict):
            print("Checkpoint ignoré : format de résultats invalide.")
            return {}, 0.0

        valid: dict[str, dict] = {}
        foreign = invalid = 0
        for key, item in raw_results.items():
            try:
                image_id = int(key)
            except (TypeError, ValueError):
                foreign += 1
                continue
            if image_id not in expected_ids:
                foreign += 1
            elif not is_valid_checkpoint_entry(item):
                invalid += 1
            else:
                valid[str(image_id)] = item

        if foreign or invalid:
            print(f"Checkpoint : {foreign} entrée(s) étrangère(s) et {invalid} entrée(s) invalide(s) ignorée(s).")
        prior_elapsed = float(saved.get("elapsed_seconds", 0.0) or 0.0)
        print(f"Checkpoint compatible : {len(valid)} images déjà traitées.")
        return valid, prior_elapsed
    except Exception as exc:
        print(f"Checkpoint illisible, nouveau traitement : {exc}")
        return {}, 0.0


def extract_metadata_with_yolo_world(
    image_paths: dict[int, Path],
    model_id: str,
    confidence_threshold: float,
    text_queries: list[str],
    device: str,
    checkpoint_path: Path,
    checkpoint_every: int,
) -> tuple[dict, dict, dict, dict]:
    config = {
        "model_type": MODEL_TYPE,
        "model_id": model_id,
        "confidence_threshold": confidence_threshold,
        "text_queries": text_queries,
        "device": device,
    }
    results_by_id, prior_elapsed = load_checkpoint(checkpoint_path, config, set(image_paths))

    print(f"\nChargement de {MODEL_TYPE} : {model_id}")
    print(f"Device : {device}")

    model = YOLOWorld(model_id)
    model.to(device)

    print(f"Définition des classes textuelles personnalisées ({len(text_queries)} classes)...")
    model.set_classes(text_queries)

    ordered_ids = list(image_paths)
    total = len(ordered_ids)
    pending = [i for i in ordered_ids if str(i) not in results_by_id]
    already_done = total - len(pending)
    print(f"Images totales : {total}; déjà traitées : {already_done}; restantes : {len(pending)}")

    def save_checkpoint(session_seconds: float) -> None:
        save_json(
            {
                "config": config,
                "elapsed_seconds": round(prior_elapsed + session_seconds, 2),
                "results": results_by_id,
            },
            checkpoint_path,
        )

    started_at = time.time()
    for index, image_id in enumerate(pending, start=1):
        image_path = image_paths[image_id]
        global_completed = already_done + index
        image_started_at = time.time()

        print(
            f"\n[YOLO-WORLD {global_completed}/{total}] "
            f"Analyse de l'image {image_id} : {image_path.name}",
            flush=True,
        )

        try:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Image illisible pour ID {image_id}: {image_path}") from exc

        results = model.predict(
            image,
            conf=confidence_threshold,
            device=device,
            verbose=False,
        )
        r = results[0]

        objects, attributes, detections = set(), set(), []

        if r.boxes is not None and len(r.boxes) > 0:
            boxes_xyxy = r.boxes.xyxy.cpu().tolist()
            scores = r.boxes.conf.cpu().tolist()
            class_ids = r.boxes.cls.cpu().tolist()
            names = r.names

            for box, score, cls_id in zip(boxes_xyxy, scores, class_ids):
                label = str(names.get(int(cls_id), "")).strip().lower()
                if not label:
                    continue

                is_attr = label_is_attribute(label)
                (attributes if is_attr else objects).add(label)
                # Every detection is kept, with its original label.
                detections.append({
                    "label": label,
                    "type": "attribute" if is_attr else "object",
                    "score": round(float(score), 4),
                    "box_xyxy": [round(float(v), 2) for v in box],
                })

        relationships = infer_relationships(detections)
        scene_context_tags = infer_scene_context(objects)
        scene_context_primary = scene_context_tags[0] if scene_context_tags else "unknown"

        results_by_id[str(image_id)] = {
            "objects": sorted(objects),
            "attributes": sorted(attributes),
            "detector": model_id,
            "confidence_threshold": confidence_threshold,
            "queries_prompt": text_queries,
            "detections": detections,
            "relationships": relationships,
            "scene_context_tags": scene_context_tags,
            "scene_context_primary": scene_context_primary,
            "image_path_used": str(image_path),
        }

        session_seconds = time.time() - started_at
        average = session_seconds / index
        eta_seconds = int(average * (len(pending) - index))
        image_seconds = time.time() - image_started_at

        print(
            f"[YOLO-WORLD {global_completed}/{total}] Terminé "
            f"en {image_seconds:.1f}s | objets={len(objects)} | "
            f"attributs={len(attributes)} | ETA restante≈{eta_seconds // 60} min",
            flush=True,
        )

        if index % checkpoint_every == 0 or index == len(pending):
            save_checkpoint(session_seconds)
            print(f"Checkpoint sauvegardé : {global_completed}/{total}", flush=True)

    session_seconds = time.time() - started_at
    save_checkpoint(session_seconds)

    objects_by_image, attributes_by_image, detection_metadata = {}, {}, {}
    for key, item in results_by_id.items():
        image_id = int(key)
        objects_by_image[image_id] = item["objects"]
        attributes_by_image[image_id] = item["attributes"]
        detection_metadata[image_id] = {k: item[k] for k in CHECKPOINT_ENTRY_KEYS
                                        if k not in ("objects", "attributes")}

    missing_results = set(image_paths) - set(objects_by_image)
    if missing_results:
        raise RuntimeError(f"Checkpoint incomplet : {len(missing_results)} images sans résultat YOLO")

    run_stats = {
        "processed_this_session": len(pending),
        "session_seconds": round(session_seconds, 2),
        "total_elapsed_seconds": round(prior_elapsed + session_seconds, 2),
    }
    return objects_by_image, attributes_by_image, detection_metadata, run_stats


# ============================================================
# LOOKUP BUILDING
# ============================================================

def build_lookup_for_split(
    split_ids: list[int],
    selected_images: dict[int, dict],
    captions_by_image: dict[int, list[str]],
    objects_by_image: dict[int, list[str]],
    attributes_by_image: dict[int, list[str]],
    detection_metadata: dict[int, dict],
    max_captions: int,
) -> tuple[dict, dict]:
    lookup = {}
    for image_id in split_ids:
        meta = selected_images[image_id]
        captions = captions_by_image.get(image_id, [])[:max_captions]
        det = detection_metadata[image_id]
        lookup[str(image_id)] = {
            "file_name": meta["file_name"],
            "url": meta["url"],
            "raw_captions": captions,
            "retrieval_captions": [clean_text_for_retrieval(c) for c in captions],
            "captions": captions,
            "objects": objects_by_image[image_id],
            "attributes": attributes_by_image[image_id],
            "relationships": det["relationships"],
            "scene_context": det["scene_context_primary"],
            "annotation_info": {
                "source": "YOLO-World (Open-Vocabulary Detections)",
                "detector": det["detector"],
                "confidence_threshold": det["confidence_threshold"],
                "queries_prompt": det["queries_prompt"],
                "attribute_split_method": "word-list heuristic (not a semantic guarantee)",
                "relationship_method": "box-based spatial heuristics",
                "image_split": meta["annotation_split"],
                "caption_annotation_file": f"captions_{meta['annotation_split']}.json",
                "detections": det["detections"],
                "scene_context_tags": det["scene_context_tags"],
                "image_path_used": det["image_path_used"],
            },
        }

    counts = [len(x["raw_captions"]) for x in lookup.values()]
    report = {
        "total_images": len(lookup),
        "images_with_captions": sum(n > 0 for n in counts),
        "images_without_captions": sum(n == 0 for n in counts),
        "total_captions": sum(counts),
        "images_with_expected_caption_count": sum(n == max_captions for n in counts),
        "images_with_unexpected_caption_count": sum(n != max_captions for n in counts),
        "images_with_objects": sum(bool(x["objects"]) for x in lookup.values()),
        "images_without_objects": sum(not x["objects"] for x in lookup.values()),
        "images_with_attributes": sum(bool(x["attributes"]) for x in lookup.values()),
        "images_with_relationships": sum(bool(x["relationships"]) for x in lookup.values()),
        "total_relationships": sum(len(x["relationships"]) for x in lookup.values()),
        "images_with_known_scene_context": sum(x["scene_context"] != "unknown" for x in lookup.values()),
        "total_detections": sum(len(x["annotation_info"]["detections"]) for x in lookup.values()),
        "images_from_train2017": sum(x["annotation_info"]["image_split"] == "train2017" for x in lookup.values()),
        "images_from_val2017": sum(x["annotation_info"]["image_split"] == "val2017" for x in lookup.values()),
    }
    return lookup, report


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build isolated Train KB / Validation / Test lookups with COCO captions and YOLO-World."
    )
    data_dir = Path("data/coco_rag_data")
    annotations_dir = Path("data/annotations_trainval2017/annotations")

    parser.add_argument("--captions-file", type=Path, default=annotations_dir / "captions_train2017.json")
    parser.add_argument("--captions-val-file", type=Path, default=annotations_dir / "captions_val2017.json")
    parser.add_argument("--image-dir", type=Path, default=data_dir)
    parser.add_argument("--train-ids", type=Path, default=data_dir / "train_kb_ids.json")
    parser.add_argument("--val-ids", type=Path, default=data_dir / "val_ids.json")
    parser.add_argument("--test-ids", type=Path, default=data_dir / "test_ids.json")
    parser.add_argument("--output-dir", type=Path, default=data_dir)
    parser.add_argument("--max-captions", type=int, default=5)
    parser.add_argument("--yolo-model", default="yolov8s-world.pt")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument(
        "--text-queries",
        nargs="+",
        default=[
            "person", "man", "woman", "child", "dog", "cat", "bird", "horse",
            "car", "bus", "truck", "bicycle", "motorcycle", "train", "boat",
            "chair", "table", "sofa", "bed", "bottle", "cup", "bowl", "backpack",
            "handbag", "suitcase", "umbrella", "traffic light", "tree", "plant",
            "flower", "food", "laptop", "phone", "television", "red", "blue",
            "green", "yellow", "black", "white", "brown", "wooden", "metal"
        ],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--download-timeout", type=int, default=45)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--benchmark", type=int, default=0, metavar="N",
        help="Run only on N images spread across all splits, report speed, then exit "
             "(no lookups written, separate checkpoint). Example: --benchmark 100",
    )
    args = parser.parse_args()

    if args.max_captions < 1:
        parser.error("--max-captions doit être >= 1")
    if not 0 <= args.conf_threshold <= 1:
        parser.error("Le seuil de confiance doit être compris entre 0 et 1")
    if args.benchmark < 0:
        parser.error("--benchmark doit être >= 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA demandé mais indisponible. Utilise --device cpu.")

    print("\n========== LOADING DATASET SPLITS ==========")
    train_ids = load_split_ids(args.train_ids)
    val_ids = load_split_ids(args.val_ids)
    test_ids = load_split_ids(args.test_ids)
    print(f"Train KB IDs : {len(train_ids)}")
    print(f"Validation IDs: {len(val_ids)}")
    print(f"Test IDs     : {len(test_ids)}")
    validate_splits(train_ids, val_ids, test_ids)

    ordered_ids = list(dict.fromkeys(train_ids + val_ids + test_ids))
    all_ids = set(ordered_ids)
    full_total = len(ordered_ids)
    print(f"Total unique IDs: {full_total}")

    print("\n========== LOADING COCO ANNOTATIONS ==========")
    captions_train = load_json(args.captions_file)
    captions_val = load_json(args.captions_val_file)
    selected_images = load_coco_image_metadata(captions_train, captions_val)

    absent_metadata = all_ids - set(selected_images)
    if absent_metadata:
        raise ValueError(
            f"{len(absent_metadata)} IDs absents des métadonnées COCO. "
            f"Exemples : {sorted(absent_metadata)[:15]}"
        )
    print(f"COCO image metadata loaded: {len(selected_images)}")

    print("\n========== GROUPING CAPTIONS ==========")
    captions_by_image = build_captions_by_image([captions_train, captions_val], all_ids)
    print(f"Images with at least one caption: {len(captions_by_image)}")

    # Benchmark mode: evenly spaced sample across train/val/test.
    benchmark_mode = args.benchmark > 0
    if benchmark_mode:
        step = max(1, full_total // args.benchmark)
        ordered_ids = ordered_ids[::step][: args.benchmark]
        print(f"\n>>> MODE BENCHMARK : {len(ordered_ids)} images <<<")

    image_paths = prepare_all_images(
        image_root=args.image_dir,
        selected_images=selected_images,
        ordered_ids=ordered_ids,
        download_workers=args.download_workers,
        timeout=args.download_timeout,
        retries=args.download_retries,
    )

    print("\n========== YOLO-WORLD EXTRACTION ==========")
    if benchmark_mode:
        checkpoint_path = args.output_dir / "yolo_world_benchmark_checkpoint.json"
        checkpoint_path.unlink(missing_ok=True)  # always measure from scratch
    else:
        checkpoint_path = args.output_dir / "yolo_world_checkpoint.json"

    objects, attributes, detection_info, run_stats = extract_metadata_with_yolo_world(
        image_paths=image_paths,
        model_id=args.yolo_model,
        confidence_threshold=args.conf_threshold,
        text_queries=args.text_queries,
        device=args.device,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
    )

    total_detections = sum(len(d["detections"]) for d in detection_info.values())
    images_with_objects = sum(bool(v) for v in objects.values())
    images_with_attributes = sum(bool(v) for v in attributes.values())

    detection_run = {
        "model_type": MODEL_TYPE,
        "model": args.yolo_model,
        "confidence_threshold": args.conf_threshold,
        "text_queries": args.text_queries,
        "device": args.device,
        "images_analyzed": len(detection_info),
        "images_with_objects_detected": images_with_objects,
        "images_with_attributes_detected": images_with_attributes,
        "total_detections": total_detections,
        "session_seconds": run_stats["session_seconds"],
        "total_elapsed_seconds": run_stats["total_elapsed_seconds"],
        "avg_seconds_per_image_this_session": (
            round(run_stats["session_seconds"] / run_stats["processed_this_session"], 3)
            if run_stats["processed_this_session"] else None
        ),
    }

    if benchmark_mode:
        avg = detection_run["avg_seconds_per_image_this_session"] or 0.0
        projected_min = avg * full_total / 60
        benchmark_report = {
            **detection_run,
            "full_dataset_size": full_total,
            "projected_full_run_minutes": round(projected_min, 1),
            "projected_full_run_hours": round(projected_min / 60, 2),
        }
        report_path = args.output_dir / "yolo_world_benchmark_report.json"
        save_json(benchmark_report, report_path)
        print("\n========== BENCHMARK RESULT ==========")
        print(f"Model            : {args.yolo_model} ({args.device})")
        print(f"Images           : {len(detection_info)}")
        print(f"Avg per image    : {avg:.2f}s")
        print(f"Total detections : {total_detections}")
        print(f"Images w/ objects: {images_with_objects} | w/ attributes: {images_with_attributes}")
        print(f"Projection for {full_total} images: ≈{projected_min / 60:.1f} h")
        print(f"Report: {report_path}")
        return

    print("\n========== BUILDING SEPARATE LOOKUPS ==========")
    split_map = {"train_kb": train_ids, "val": val_ids, "test": test_ids}
    output_names = {
        "train_kb": "train_kb_lookup.json",
        "val": "val_lookup.json",
        "test": "test_lookup.json",
    }
    reports = {}

    for split_name, ids in split_map.items():
        lookup, report = build_lookup_for_split(
            ids, selected_images, captions_by_image, objects, attributes,
            detection_info, args.max_captions,
        )
        output_path = args.output_dir / output_names[split_name]
        save_json(lookup, output_path)
        reports[split_name] = report
        print(
            f"{split_name}: {report['total_images']} images, "
            f"{report['total_captions']} captions -> {output_path}"
        )

    report_path = args.output_dir / "unified_lookup_validation_report.json"
    save_json({
        "project_split_counts": {name: len(ids) for name, ids in split_map.items()},
        "total_unique_ids": len(ordered_ids),
        "anti_leakage": "PASS",
        "image_count": len(image_paths),
        "yolo_checkpoint": str(checkpoint_path),
        "detection_run": detection_run,
        "splits": reports,
        "important_note": (
            "Only train_kb_lookup.json may be used to build the retrieval Knowledge Base. "
            "Validation and Test lookups must remain outside the retrieval index."
        ),
    }, report_path)

    print("\n========== FINAL SUMMARY ==========")
    print(f"Train KB : {len(train_ids)}")
    print(f"Validation: {len(val_ids)}")
    print(f"Test     : {len(test_ids)}")
    print(f"Images analyzed by YOLO-World: {len(detection_info)}")
    print(f"Total detections: {total_detections}")
    print(f"Total processing time: {run_stats['total_elapsed_seconds'] / 60:.1f} min")
    print(f"Report: {report_path}")
    print("SUCCESS: les trois lookups indépendants ont été générés.")


if __name__ == "__main__":
    main()