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

    # Save the evaluation results as a JSON report
    timestamp = datetime.now(timezone.utc).isoformat()
    report = {
        "timestamp": timestamp,
        "total_vectors": faiss_index.ntotal,
        "sample_size": sample_size,
        "recall": results[0],
        "precision": results[1],
        "mrr": results[2],
        "map": results[3],
        "top_1_accuracy": results[4],
    }

    report_path = "./data/coco_rag_data/evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    print(f"Evaluation report saved to {report_path}.")