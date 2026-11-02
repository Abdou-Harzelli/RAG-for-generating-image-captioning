import os
import json
import time
import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

def extract_text_embeddings():
    # 1. Setup paths matching README conventions
    lookup_path = os.path.join("data", "coco_rag_data", "unified_lookup.json")
    output_dir = os.path.join("data", "coco_rag_data")
    os.makedirs(output_dir, exist_ok=True)

    npy_out = os.path.join(output_dir, "text_embeddings.npy")
    mapping_out = os.path.join(output_dir, "text_embeddings_mapping.json")
    report_out = os.path.join(output_dir, "text_embedding_report.json")

    if not os.path.exists(lookup_path):
        print(f"Error: Unified lookup file not found at {lookup_path}")
        return

    # 2. Load unified lookup
    print(f"Loading captions from {lookup_path}...")
    with open(lookup_path, "r", encoding="utf-8") as f:
        unified_lookup = json.load(f)

    # --- THE FIX: Handle any JSON structure Member 4 used ---
    if isinstance(unified_lookup, dict):
        if "captions" in unified_lookup:
            unified_lookup = [unified_lookup]  # Wrapped single object
        elif "images" in unified_lookup:
            unified_lookup = unified_lookup["images"]  # Wrapped in 'images' root key
        else:
            unified_lookup = list(unified_lookup.values())  # Dictionary mapping keys to objects

    # 3. Flatten all 50,000 captions into an ordered list with metadata
    all_captions = []
    mapping_metadata = []

    for item in unified_lookup:
        # Failsafe in case of weird formatting
        if not isinstance(item, dict):
            continue
            
        image_id = item.get("image_id", item.get("file_name"))
        file_name = item.get("file_name", "")
        captions = item.get("captions", [])

        for idx, cap in enumerate(captions):
            all_captions.append(cap)
            mapping_metadata.append({
                "caption_index": len(all_captions) - 1,
                "image_id": image_id,
                "file_name": file_name,
                "caption_sub_id": idx,
                "caption_text": cap
            })

    total_captions = len(all_captions)
    print(f"Extracted {total_captions} captions across {len(unified_lookup)} images.")

    # 4. Initialize CLIP Model
    model_name = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading text encoder '{model_name}' on {device}...")

    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device)
    model.eval()

    # 5. Extract Embeddings in Batches
    batch_size = 64
    embeddings_list = []
    start_time = time.time()

    print(f"Generating embeddings in batches of {batch_size}...")
    for i in range(0, total_captions, batch_size):
        batch_captions = all_captions[i:i + batch_size]

        inputs = tokenizer(batch_captions, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Text projection layer to match image embedding dimension (512)
            embeds = outputs.text_embeds

            # Mandatory L2 Normalization for FAISS IndexFlatIP
            embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
            embeddings_list.append(embeds.cpu().numpy())

        if (i + batch_size) % 3200 < batch_size or (i + batch_size) >= total_captions:
            processed = min(i + batch_size, total_captions)
            print(f"Processed {processed}/{total_captions} captions...")

    # 6. Save Deliverables
    if not embeddings_list:
        print("No embeddings were generated. Check your JSON format.")
        return

    embedding_matrix = np.vstack(embeddings_list).astype(np.float32)
    elapsed_time = round(time.time() - start_time, 2)

    np.save(npy_out, embedding_matrix)
    print(f"Saved numpy embeddings array to: {npy_out}")

    with open(mapping_out, "w", encoding="utf-8") as f:
        json.dump(mapping_metadata, f, indent=2)
    print(f"Saved mapping metadata to: {mapping_out}")

    report = {
        "model": model_name,
        "device": device,
        "total_captions_processed": total_captions,
        "embedding_shape": list(embedding_matrix.shape),
        "dtype": str(embedding_matrix.dtype),
        "l2_normalization": True,
        "processing_time_seconds": elapsed_time,
        "status": "SUCCESS"
    }

    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved embedding report to: {report_out}")

    print("\n=============================================")
    print("Step 4 Member 2 Complete: Text Embeddings Ready!")
    print(f"Matrix Shape: {embedding_matrix.shape}")
    print("=============================================")

if __name__ == "__main__":
    extract_text_embeddings()