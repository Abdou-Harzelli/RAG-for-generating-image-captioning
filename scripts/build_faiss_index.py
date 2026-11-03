import argparse
import json
import os
import random
import time

import faiss
import numpy as np

DATA_DIR = os.path.join("data", "coco_rag_data")


def main():
    p = argparse.ArgumentParser(description="Build a FAISS IndexFlatIP.")
    p.add_argument("--name", required=True, help="index name, e.g. text or image")
    p.add_argument("--embeddings", default=None, help="default: data/coco_rag_data/<name>_embeddings.npy")
    p.add_argument("--mapping", default=None, help="default: data/coco_rag_data/<name>_embeddings_mapping.json")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--k", type=int, default=5)
    a = p.parse_args()

    emb_path = a.embeddings or os.path.join(DATA_DIR, f"{a.name}_embeddings.npy")
    map_path = a.mapping or os.path.join(DATA_DIR, f"{a.name}_embeddings_mapping.json")
    index_out = os.path.join(DATA_DIR, f"{a.name}_index.faiss")
    report_out = os.path.join(DATA_DIR, f"{a.name}_index_report.json")

    # 1. Load and validate the embeddings BEFORE touching FAISS
    emb = np.load(emb_path)
    assert emb.ndim == 2 and emb.shape[1] == a.dim, f"bad shape {emb.shape}, expected (N, {a.dim})"
    assert emb.dtype == np.float32, f"dtype must be float32, got {emb.dtype}"
    assert np.isfinite(emb).all(), "NaN/Inf in embeddings"
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"not L2-normalized: [{norms.min()}, {norms.max()}]"
    emb = np.ascontiguousarray(emb)
    n = emb.shape[0]

    with open(map_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    assert len(mapping) == n, f"mapping has {len(mapping)} entries but embeddings have {n} rows"
    if "caption_index" in mapping[0]:
        assert all(m["caption_index"] == i for i, m in enumerate(mapping)), "mapping order != row order"

    # 2. Build: exact search, inner product (= cosine because vectors are unit length)
    t0 = time.time()
    index = faiss.IndexFlatIP(a.dim)
    index.add(emb)
    build_seconds = round(time.time() - t0, 4)
    assert index.ntotal == n, (index.ntotal, n)

    # 3. Save, then reload from disk so we test the saved file, not the one in memory
    faiss.write_index(index, index_out)
    index = faiss.read_index(index_out)
    assert index.ntotal == n and index.d == a.dim

    # 4. Self-retrieval test: each vector must find itself with score ~1.0
    rows = random.Random(0).sample(range(n), min(100, n))
    queries = emb[rows]
    t0 = time.time()
    scores, ids = index.search(queries, a.k)
    ms_per_query = round((time.time() - t0) / len(rows) * 1000, 3)
    assert scores[:, 0].min() > 0.999, f"top-1 score too low: {scores[:, 0].min()}"
    exact_top1 = float((ids[:, 0] == np.array(rows)).mean())   # <1.0 only when identical vectors tie

    duplicates = int(n - len(np.unique(emb, axis=0)))

    report = {
        "index_type": "IndexFlatIP",
        "metric": "inner product (= cosine on L2-normalized vectors)",
        "embedding_dim": a.dim,
        "vectors_in_index": index.ntotal,
        "source_embeddings": os.path.basename(emb_path),
        "source_mapping": os.path.basename(map_path),
        "index_file": os.path.basename(index_out),
        "build_time_seconds": build_seconds,
        "search_ms_per_query": ms_per_query,
        "self_retrieval_top1_exact_id_rate": exact_top1,
        "identical_vectors_in_matrix": duplicates,
        "faiss_version": faiss.__version__,
        "status": "SUCCESS",
    }
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved index  : {index_out}")
    print(f"Saved report : {report_out}")


if __name__ == "__main__":
    main()