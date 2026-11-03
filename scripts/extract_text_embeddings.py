# scripts/extract_text_embeddings.py
import os
import json
import time
import numpy as np
import torch
import transformers
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

DATA_DIR = os.path.join("data", "coco_rag_data")
FLAT_PATH = os.path.join(DATA_DIR, "captions_kb_flat.jsonl")
NPY_OUT = os.path.join(DATA_DIR, "text_embeddings.npy")
MAPPING_OUT = os.path.join(DATA_DIR, "text_embeddings_mapping.json")
REPORT_OUT = os.path.join(DATA_DIR, "text_embedding_report.json")

MODEL_NAME = "openai/clip-vit-base-patch32"
EMBED_DIM = 512
EXPECTED_CAPTIONS = 50_000
BATCH_SIZE = 64


def load_flat_rows(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run first: "
            "python src/preprocessing/caption_pipeline.py --name kb "
            "--forbid-ids data/coco_rag_data/test_ids.json"
        )
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def pick_device():
    """Use the GPU only if it can really run PyTorch kernels; otherwise fall back to CPU."""
    if torch.cuda.is_available():
        try:
            (torch.zeros(1, device="cuda") + 1).sum().item()   # forces a real kernel launch
            return "cuda"
        except Exception as exc:
            print(f"CUDA GPU found but not usable with this PyTorch build ({type(exc).__name__}); using CPU.")
    return "cpu"


def extract_text_embeddings():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Load the flat caption file: single source of truth for row order
    print(f"Loading captions from {FLAT_PATH}...")
    flat_rows = load_flat_rows(FLAT_PATH)

    # 2. Embed the CLEANED caption, in file order (row i = FAISS vector i)
    all_captions = [row["caption_clean"] for row in flat_rows]
    total_captions = len(all_captions)
    assert total_captions == EXPECTED_CAPTIONS, (
        f"Expected {EXPECTED_CAPTIONS} captions, got {total_captions}"
    )
    assert all(all_captions), "Found an empty cleaned caption"

    mapping_metadata = [
        {
            "caption_index": pos,                       # global row = FAISS vector id
            "image_id": row["image_id"],                # real integer ID
            "caption_id": row["caption_id"],
            "file_name": f"{row['image_id']:012d}.jpg",
            "caption_sub_id": row["caption_index"],     # 0..4 inside the image
            "caption_text": row["caption_clean"],
        }
        for pos, row in enumerate(flat_rows)
    ]
    print(f"Loaded {total_captions} captions.")

    # 3. Initialize CLIP text encoder
    device = pick_device()
    print(f"Loading text encoder '{MODEL_NAME}' on {device}...")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
    model = CLIPTextModelWithProjection.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # 4. Extract embeddings in batches
    embeddings_list = []
    start_time = time.time()
    print(f"Generating embeddings in batches of {BATCH_SIZE}...")

    for i in range(0, total_captions, BATCH_SIZE):
        batch = all_captions[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            embeds = model(**inputs).text_embeds                       # (batch, 512)
            embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)   # L2 normalize
        embeddings_list.append(embeds.cpu().numpy())

        done = min(i + BATCH_SIZE, total_captions)
        if (i // BATCH_SIZE) % 50 == 0 or done == total_captions:
            print(f"Processed {done}/{total_captions} captions...")

    embedding_matrix = np.vstack(embeddings_list).astype(np.float32)
    elapsed_time = round(time.time() - start_time, 2)

    # 5. Sanity checks (fail loudly instead of saving bad data)
    assert embedding_matrix.shape == (total_captions, EMBED_DIM), embedding_matrix.shape
    assert np.isfinite(embedding_matrix).all(), "NaN/Inf found in embeddings"
    assert np.allclose(np.linalg.norm(embedding_matrix, axis=1), 1.0, atol=1e-3), \
        "Embeddings are not L2-normalized"

    # 6. Save deliverables
    np.save(NPY_OUT, embedding_matrix)
    print(f"Saved numpy embeddings array to: {NPY_OUT}")

    with open(MAPPING_OUT, "w", encoding="utf-8") as f:
        json.dump(mapping_metadata, f, indent=2)
    print(f"Saved mapping metadata to: {MAPPING_OUT}")

    report = {
        "model": MODEL_NAME,
        "device": device,
        "source_file": os.path.basename(FLAT_PATH),
        "text_field": "caption_clean",
        "total_captions_processed": total_captions,
        "embedding_shape": list(embedding_matrix.shape),
        "dtype": str(embedding_matrix.dtype),
        "l2_normalization": True,
        "batch_size": BATCH_SIZE,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "processing_time_seconds": elapsed_time,
        "status": "SUCCESS",
    }
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved embedding report to: {REPORT_OUT}")

    print("\n=============================================")
    print("Step 4 Member 2 Complete: Text Embeddings Ready!")
    print(f"Matrix Shape: {embedding_matrix.shape}")
    print("=============================================")


if __name__ == "__main__":
    extract_text_embeddings()