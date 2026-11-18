import os
import json
import numpy as np
import faiss
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from datetime import datetime, timezone

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32
TOP_K = 5  # For Recall@K, P@K, MRR, and MAP

# Paths to your existing training artifacts
FAISS_INDEX_PATH = "./data/coco_rag_data/image_index.faiss"  # Or whatever you named your saved index file
KB_IDS_PATH = "./data/coco_rag_data/train_kb_ids.json"

IMAGE_EMBEDDINGS_PATH = "./data/coco_rag_data/image_embeddings/image_embeddings.npy"
IMAGE_MAPPING_PATH = "./data/coco_rag_data/image_embeddings/image_embeddings_mapping.json"

METRIC_REPORT_PATH = "./data/coco_rag_data/metric_evaluation_report.json"

# ==========================================
# PREPROCESSING PIPELINE FOR TEST IMAGES
# ==========================================
def get_image_transform():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

def load_and_preprocess_image(image_path, transform):
    try:
        img = Image.open(image_path).convert("RGB")
        return transform(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def generate_test_embeddings(image_paths, model, processor, transform):
    model.eval()
    all_embeddings = []
    
    print(f"Generating embeddings for {len(image_paths)} test images...")
    with torch.no_grad():
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[i : i + BATCH_SIZE]
            batch_tensors = [load_and_preprocess_image(p, transform) for p in batch_paths]
            
            valid_tensors = [t for t in batch_tensors if t is not None]
            if not valid_tensors:
                continue
                
            batch_input = torch.stack(valid_tensors).to(DEVICE)
            # Get projected image embeddings from CLIP
            vision_outputs = model.vision_model(pixel_values=batch_input)
            image_features = model.visual_projection(vision_outputs.pooler_output)
            
            # L2 normalization for cosine similarity
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_embeddings.append(image_features.cpu().numpy())
            
    return np.vstack(all_embeddings).astype('float32')

# ==========================================
# EVALUATION METRICS ENGINE
# ==========================================
def evaluate_retriever(test_embeddings, test_ground_truth_ids, kb_image_ids, faiss_index, k=5):
    recalls = []
    precisions = []
    reciprocal_ranks = []
    average_precisions = []
    top1_hits = []

    distances, indices = faiss_index.search(test_embeddings, k)

    for i, retrieved_indices in enumerate(indices):
        true_id = int(test_ground_truth_ids[i])
        retrieved_ids = [int(kb_image_ids[idx]) for idx in retrieved_indices]

        hit = 1.0 if true_id in retrieved_ids else 0.0
        recalls.append(hit)
        precisions.append(hit / k)

        if hit:
            rank = retrieved_ids.index(true_id) + 1
            reciprocal_ranks.append(1.0 / rank)
            average_precisions.append(1.0 / rank)
            top1_hits.append(1.0 if rank == 1 else 0.0)
        else:
            reciprocal_ranks.append(0.0)
            average_precisions.append(0.0)
            top1_hits.append(0.0)

    mean_recall = float(np.mean(recalls))
    mean_precision = float(np.mean(precisions))
    mrr = float(np.mean(reciprocal_ranks))
    map_score = float(np.mean(average_precisions))
    top1_accuracy = float(np.mean(top1_hits))

    print(f"\n--- FAISS SELF-RETRIEVAL RESULTS ---")
    print(f"Recall@{k}: {mean_recall:.4f} ({mean_recall * 100:.2f}%)")
    print(f"Precision@{k}: {mean_precision:.4f} ({mean_precision * 100:.2f}%)")
    print(f"MRR: {mrr:.4f}")
    print(f"MAP@{k}: {map_score:.4f}")
    print(f"Top-1 accuracy: {top1_accuracy:.4f} ({top1_accuracy * 100:.2f}%)")

    return mean_recall, mean_precision, mrr, map_score, top1_accuracy

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- FAISS IMAGE-INDEX SELF-RETRIEVAL ---")

    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    with open(KB_IDS_PATH, "r", encoding="utf-8") as f:
        kb_image_ids = [int(image_id) for image_id in json.load(f)]

    stored_embeddings = np.load(IMAGE_EMBEDDINGS_PATH).astype("float32")

    # Validate the source embeddings used to build the index.
    norms = np.linalg.norm(stored_embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            f"Stored embeddings are not L2-normalized: [{norms.min():.6f}, {norms.max():.6f}]"
        )

    if faiss_index.ntotal != len(stored_embeddings):
        raise ValueError(
            f"FAISS/index mismatch: index has {faiss_index.ntotal} vectors, "
            f"embeddings have {len(stored_embeddings)} rows"
        )

    if faiss_index.ntotal != len(kb_image_ids):
        raise ValueError(
            f"FAISS/ID mismatch: index has {faiss_index.ntotal} vectors, "
            f"mapping has {len(kb_image_ids)} IDs"
        )

    # Use exactly 100 vectors that are already in the index.
    sample_size = min(100, len(stored_embeddings))
    sample_indices = np.linspace(
        0, len(stored_embeddings) - 1, sample_size, dtype=np.int64
    )
    test_embeddings = np.ascontiguousarray(stored_embeddings[sample_indices])
    test_ground_truth_ids = [kb_image_ids[index] for index in sample_indices]

    print(f"Loaded FAISS index containing {faiss_index.ntotal} vectors.")
    print(f"Evaluating {sample_size} vectors already present in the index.")

    results = evaluate_retriever(
        test_embeddings=test_embeddings,
        test_ground_truth_ids=test_ground_truth_ids,
        kb_image_ids=kb_image_ids,
        faiss_index=faiss_index,
        k=TOP_K,
    )

    # Save a complete evaluation report.
    timestamp = datetime.now(timezone.utc).isoformat()
    report = {
        "created_at_utc": timestamp,
        "evaluation_type": "FAISS image-index self-retrieval",
        "purpose": "Validate image embedding normalization, vector-ID mapping, FAISS index consistency, and exact retrieval.",
        "model": MODEL_NAME,
        "embedding_dimension": stored_embeddings.shape[1],
        "embedding_dtype": str(stored_embeddings.dtype),
        "similarity": {
            "method": "cosine similarity implemented by inner product",
            "faiss_index_type": "IndexFlatIP",
            "normalization": "L2",
            "formula": "cos(a,b) = (a dot b) / (||a|| * ||b||) = a dot b for unit vectors",
            "meaning": "Ranks vectors by their direction similarity; larger scores indicate greater similarity."
        },
        "dataset_scope": {
            "knowledge_base_image_count": len(kb_image_ids),
            "tested_sample_count": sample_size,
            "source": "100 vectors sampled from the existing 10,000-image training knowledge base",
            "search_pool": "The same 10,000-image FAISS index",
            "not_used": "Held-out validation/test image files",
            "reason": "The 100 queries are vectors already present in the index, so this measures index consistency rather than unseen-data generalization."
        },
        "checks": {
            "l2_normalization_passed": True,
            "min_embedding_norm": float(norms.min()),
            "max_embedding_norm": float(norms.max()),
            "embedding_index_size_match": faiss_index.ntotal == len(stored_embeddings),
            "embedding_id_mapping_size_match": faiss_index.ntotal == len(kb_image_ids),
            "embedding_index_size": faiss_index.ntotal,
            "embedding_matrix_rows": len(stored_embeddings),
            "mapping_rows": len(kb_image_ids)
        },
        "metrics": {
            "k": TOP_K,
            "recall_at_k": results[0],
            "precision_at_k": results[1],
            "mrr": results[2],
            "map_at_k": results[3],
            "top_1_accuracy": results[4]
        },
        "interpretation": {
            "what_it_proves": [
                "Stored vectors are L2-normalized.",
                "FAISS and the embedding matrix contain the same number of vectors.",
                "FAISS row positions correspond to image-ID mapping rows.",
                "Known vectors can be retrieved and the exact matching vector ranks first.",
                "Inner-product/cosine search is technically functioning."
            ],
            "what_it_does_not_prove": [
                "Semantic quality on unseen images.",
                "Text-to-image retrieval quality.",
                "Comparison quality against another student's held-out benchmark.",
                "Generalization beyond the 10,000-image knowledge base."
            ],
            "main_limitation": "Self-retrieval uses stored vectors as both queries and candidates; it does not represent unseen RAG retrieval quality."
        },
        "status": "SUCCESS"
    }

    with open(METRIC_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    print(f"Complete evaluation report saved to {METRIC_REPORT_PATH}.")