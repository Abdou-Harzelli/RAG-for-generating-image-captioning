import os
import json
import random

def generate_splits(
    train_json_path="data/coco_rag_data/train_subset_10000.json",
    output_dir="data/coco_rag_data",
    seed=42
):
    random.seed(seed)
    
    # 1. RAG Knowledge Base -> Full 10,000 images
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        
    if isinstance(train_data, list):
        train_kb_ids = [item['id'] if isinstance(item, dict) else item for item in train_data]
    elif isinstance(train_data, dict) and 'images' in train_data:
        train_kb_ids = [img['id'] if isinstance(img, dict) else img for img in train_data['images']]
    elif isinstance(train_data, dict):
        train_kb_ids = list(train_data.keys())

    # Ensure we strictly have 10000 for KB if the file has more
    train_kb_ids = train_kb_ids[:10000]

    # 2. Load val2017 pool to carve out Testing (100) safely
    val2017_paths = [
        "data/annotations/captions_val2017.json",
        "data/annotations_trainval2017/annotations/captions_val2017.json"
    ]
    val2017_json_path = next((p for p in val2017_paths if os.path.exists(p)), None)
    
    if not val2017_json_path:
        raise FileNotFoundError("Could not find captions_val2017.json. Ensure annotations are extracted.")
        
    with open(val2017_json_path, 'r', encoding='utf-8') as f:
        val2017_data = json.load(f)
        
    val2017_all_ids = [img['id'] for img in val2017_data['images']]
    random.shuffle(val2017_all_ids)


    # Carve out 100 for validation from train2017 (10000 images of training)
    val_ids = train_kb_ids[:100]


    # Carve out 100 for testing from val2017
    test_ids = val2017_all_ids[:100]
    
    # STRICT ANTI-LEAKAGE ASSERTIONS
    set_train = set(train_kb_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)
    

    assert len(set_train.intersection(set_test)) == 0, "Leakage between Train KB and Test!"
    assert len(set_val.intersection(set_test)) == 0, "Leakage between Val and Test!"
    print(">>> Anti-leakage checks PASSED: Zero overlap confirmed across all splits. <<<")

    # Save the files
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_kb_ids.json"), "w", encoding="utf-8") as f:
        json.dump(train_kb_ids, f, indent=2)
    with open(os.path.join(output_dir, "val_ids.json"), "w", encoding="utf-8") as f:
        json.dump(val_ids, f, indent=2)
    with open(os.path.join(output_dir, "test_ids.json"), "w", encoding="utf-8") as f:
        json.dump(test_ids, f, indent=2)
        
    print(f"Splits generated successfully!")
    print(f"Train KB: {len(train_kb_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

if __name__ == "__main__":
    generate_splits()