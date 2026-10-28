import os
import json
import random

def generate_splits(
    train_json_path="data/coco_rag_data/train_subset_10000.json",
    output_dir="data/coco_rag_data",
    seed=42
):
    random.seed(seed)
    
    # 1. RAG Knowledge Base (Training) -> All 10,000 images
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        
    # PROPERLY PARSE THE JSON (This is what I broke earlier!)
    if isinstance(train_data, list):
        train_kb_ids = [item['id'] if isinstance(item, dict) else item for item in train_data]
    elif isinstance(train_data, dict) and 'images' in train_data:
        train_kb_ids = [img['id'] if isinstance(img, dict) else img for img in train_data['images']]
    elif isinstance(train_data, dict):
        # In case it's a dict mapping IDs directly
        train_kb_ids = list(train_data.keys())
        
    # 2. Validation -> 100 images from the 10,000 training set
    val_ids = random.sample(train_kb_ids, 100)
    
    # 3. Testing -> 100 images from the untouched COCO val2017 dataset
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
    test_ids = random.sample(val2017_all_ids, 100)
    
    # Save the files
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_kb_ids.json"), "w", encoding="utf-8") as f:
        json.dump(train_kb_ids, f, indent=2)
    with open(os.path.join(output_dir, "val_ids.json"), "w", encoding="utf-8") as f:
        json.dump(val_ids, f, indent=2)
    with open(os.path.join(output_dir, "test_ids.json"), "w", encoding="utf-8") as f:
        json.dump(test_ids, f, indent=2)
        
    print(f"Splits corrected successfully!")
    print(f"Train KB: {len(train_kb_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

if __name__ == "__main__":
    generate_splits()