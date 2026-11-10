import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_int_id_set(values: list[Any], name: str) -> set[int]:
    ids: list[int] = []
    for v in values:
        if isinstance(v, dict):
            v = v.get("id")
        try:
            ids.append(int(v))
        except Exception as exc:
            raise ValueError(f"Invalid ID in {name}: {v!r}") from exc
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs detected in {name}")
    return set(ids)


def parse_lookup_ids(lookup: dict[str, Any]) -> set[int]:
    out = set()
    for k in lookup:
        out.add(int(k))
    return out


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, details: dict[str, Any]) -> None:
    checks.append({"name": name, "passed": passed, "details": details})


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 quality control for preprocessing outputs")

    data_dir = Path("data/coco_rag_data")
    parser.add_argument("--train-ids", type=Path, default=data_dir / "train_kb_ids.json")
    parser.add_argument("--val-ids", type=Path, default=data_dir / "val_ids.json")
    parser.add_argument("--test-ids", type=Path, default=data_dir / "test_ids.json")

    parser.add_argument("--train-lookup", type=Path, default=data_dir / "train_kb_lookup.json")
    parser.add_argument("--val-lookup", type=Path, default=data_dir / "val_lookup.json")
    parser.add_argument("--test-lookup", type=Path, default=data_dir / "test_lookup.json")

    parser.add_argument("--text-embeddings", type=Path, default=data_dir / "text_embeddings.npy")
    parser.add_argument("--text-mapping", type=Path, default=data_dir / "text_embeddings_mapping.json")
    parser.add_argument("--text-index", type=Path, default=data_dir / "text_index.faiss")

    parser.add_argument("--expected-captions-per-image", type=int, default=5)
    parser.add_argument("--self-retrieval-samples", type=int, default=100)
    parser.add_argument("--report", type=Path, default=data_dir / "phase8_quality_control_report.json")

    args = parser.parse_args()

    checks: list[dict[str, Any]] = []

    required_paths = [
        args.train_ids,
        args.val_ids,
        args.test_ids,
        args.train_lookup,
        args.val_lookup,
        args.test_lookup,
        args.text_embeddings,
        args.text_mapping,
    ]
    missing = [str(p) for p in required_paths if not p.is_file()]
    add_check(
        checks,
        "required_files_present",
        len(missing) == 0,
        {"missing": missing},
    )

    if missing:
        status = "FAIL"
        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "checks": checks,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    train_ids = as_int_id_set(load_json(args.train_ids), "train_ids")
    val_ids = as_int_id_set(load_json(args.val_ids), "val_ids")
    test_ids = as_int_id_set(load_json(args.test_ids), "test_ids")

    overlap_train_val = len(train_ids & val_ids)
    overlap_train_test = len(train_ids & test_ids)
    overlap_val_test = len(val_ids & test_ids)
    add_check(
        checks,
        "split_anti_leakage",
        overlap_train_val == 0 and overlap_train_test == 0 and overlap_val_test == 0,
        {
            "train_val_overlap": overlap_train_val,
            "train_test_overlap": overlap_train_test,
            "val_test_overlap": overlap_val_test,
        },
    )

    train_lookup = load_json(args.train_lookup)
    val_lookup = load_json(args.val_lookup)
    test_lookup = load_json(args.test_lookup)

    train_lookup_ids = parse_lookup_ids(train_lookup)
    val_lookup_ids = parse_lookup_ids(val_lookup)
    test_lookup_ids = parse_lookup_ids(test_lookup)

    add_check(
        checks,
        "lookup_id_coverage",
        train_ids <= train_lookup_ids and val_ids <= val_lookup_ids and test_ids <= test_lookup_ids,
        {
            "missing_train_ids_in_lookup": len(train_ids - train_lookup_ids),
            "missing_val_ids_in_lookup": len(val_ids - val_lookup_ids),
            "missing_test_ids_in_lookup": len(test_ids - test_lookup_ids),
        },
    )

    caption_count_mismatches = []
    for image_id_str, payload in train_lookup.items():
        n = len(payload.get("captions", []))
        if n != args.expected_captions_per_image:
            caption_count_mismatches.append({"image_id": int(image_id_str), "caption_count": n})

    add_check(
        checks,
        "train_caption_count_policy",
        len(caption_count_mismatches) == 0,
        {
            "expected_captions_per_image": args.expected_captions_per_image,
            "mismatch_count": len(caption_count_mismatches),
            "mismatch_examples": caption_count_mismatches[:20],
        },
    )

    text_embeddings = np.load(args.text_embeddings)
    text_mapping = load_json(args.text_mapping)

    shape_ok = text_embeddings.ndim == 2
    finite_ok = bool(np.isfinite(text_embeddings).all())
    dtype_ok = text_embeddings.dtype == np.float32
    rows_match_mapping = shape_ok and (text_embeddings.shape[0] == len(text_mapping))

    add_check(
        checks,
        "text_embedding_matrix_integrity",
        shape_ok and finite_ok and dtype_ok and rows_match_mapping,
        {
            "shape": list(text_embeddings.shape),
            "dtype": str(text_embeddings.dtype),
            "all_finite": finite_ok,
            "mapping_rows": len(text_mapping),
            "rows_match_mapping": rows_match_mapping,
        },
    )

    if shape_ok and text_embeddings.shape[0] > 0:
        norms = np.linalg.norm(text_embeddings, axis=1)
        norm_ok = bool(np.allclose(norms, 1.0, atol=1e-3))
        add_check(
            checks,
            "text_embedding_l2_normalization",
            norm_ok,
            {
                "min_norm": float(norms.min()),
                "max_norm": float(norms.max()),
            },
        )

    mapping_image_ids = []
    caption_ids = []
    caption_indices = []
    bad_link_rows = 0

    for i, row in enumerate(text_mapping):
        image_id = row.get("image_id")
        try:
            image_id = int(image_id)
            mapping_image_ids.append(image_id)
        except Exception:
            bad_link_rows += 1
            continue

        if image_id not in train_lookup_ids:
            bad_link_rows += 1

        if "caption_id" in row:
            caption_ids.append(str(row["caption_id"]))
        if "caption_index" in row:
            try:
                caption_indices.append(int(row["caption_index"]))
            except Exception:
                bad_link_rows += 1

    duplicate_caption_ids = len(caption_ids) - len(set(caption_ids))
    duplicate_caption_indices = len(caption_indices) - len(set(caption_indices))

    image_counter = Counter(mapping_image_ids)
    expected_rows = sum(len(item.get("captions", [])) for item in train_lookup.values())

    add_check(
        checks,
        "text_mapping_linkage",
        bad_link_rows == 0 and duplicate_caption_ids == 0 and duplicate_caption_indices == 0,
        {
            "bad_link_rows": bad_link_rows,
            "duplicate_caption_ids": duplicate_caption_ids,
            "duplicate_caption_indices": duplicate_caption_indices,
            "mapped_unique_images": len(image_counter),
            "expected_caption_rows_from_lookup": expected_rows,
            "actual_mapping_rows": len(text_mapping),
        },
    )

    add_check(
        checks,
        "missing_embedding_rows",
        len(text_mapping) == expected_rows,
        {
            "expected_rows": expected_rows,
            "actual_rows": len(text_mapping),
            "difference": expected_rows - len(text_mapping),
        },
    )

    if faiss is None:
        add_check(
            checks,
            "faiss_available",
            False,
            {"reason": "faiss import failed"},
        )
    elif not args.text_index.is_file():
        add_check(
            checks,
            "faiss_index_file_present",
            False,
            {"missing": str(args.text_index)},
        )
    else:
        index = faiss.read_index(str(args.text_index))
        ntotal_ok = index.ntotal == len(text_mapping)
        add_check(
            checks,
            "faiss_index_size_match",
            ntotal_ok,
            {
                "index_ntotal": int(index.ntotal),
                "mapping_rows": len(text_mapping),
                "embedding_rows": int(text_embeddings.shape[0]),
                "index_dimension": int(index.d),
            },
        )

        sample_size = min(args.self_retrieval_samples, text_embeddings.shape[0])
        if sample_size > 0:
            rng = np.random.default_rng(0)
            sample_rows = rng.choice(text_embeddings.shape[0], size=sample_size, replace=False)
            scores, ids = index.search(np.ascontiguousarray(text_embeddings[sample_rows]), 1)
            top1_rate = float((ids[:, 0] == sample_rows).mean())
            add_check(
                checks,
                "faiss_self_retrieval_smoke_test",
                top1_rate >= 0.95,
                {
                    "sample_size": int(sample_size),
                    "top1_exact_row_rate": round(top1_rate, 4),
                    "min_top1_score": float(scores[:, 0].min()) if sample_size else None,
                },
            )

    passed_all = all(c["passed"] for c in checks)
    status = "PASS" if passed_all else "FAIL"

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks_passed": sum(1 for c in checks if c["passed"]),
        "checks_failed": sum(1 for c in checks if not c["passed"]),
        "checks": checks,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "status": status,
        "report": str(args.report),
        "checks_passed": report["checks_passed"],
        "checks_failed": report["checks_failed"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
