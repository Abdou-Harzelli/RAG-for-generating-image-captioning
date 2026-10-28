import os
import json
import random

def generate_splits(
    subset_json_path="data/coco_rag_data/train_subset_10000.json",
    output_dir="data/coco_rag_data",
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42
):
    # Fix random seed for exact reproducibility across all team members
    random.seed(seed)
    
    if not os.path.exists(subset_json_path):
        raise FileNotFoundError(f"Could not find subset file at {subset_json_path}")
        
    with open(subset_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Get all image IDs (or filenames) from the downloaded subset
    if isinstance(data, list):
        image_ids = [item['id'] if isinstance(item, dict) else item for item in data]
    elif isinstance(data, dict) and 'images' in data:
        image_ids = [img['id'] for img in data['images']]
    elif isinstance(data, dict):
        image_ids = list(data.keys())
    else:
        raise ValueError("Unrecognized JSON format in train_subset_10000.json")

    # Shuffle deterministically
    random.shuffle(image_ids)
    
    total = len(image_ids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_kb_ids = image_ids[:train_end]
    val_ids = image_ids[train_end:val_end]
    test_ids = image_ids[val_end:]
    
    # STRICT ANTI-LEAKAGE ASSERTION CHECKS
    set_train = set(train_kb_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)
    
    # 1. Check for overlaps between Train KB and Validation
    train_val_overlap = set_train.intersection(set_val)
    assert len(train_val_overlap) == 0, f"LEAKAGE DETECTED: {len(train_val_overlap)} IDs overlap between Train KB and Val!"
    
    # 2. Check for overlaps between Train KB and Test 
    train_test_overlap = set_train.intersection(set_test)
    assert len(train_test_overlap) == 0, f"LEAKAGE DETECTED: {len(train_test_overlap)} IDs overlap between Train KB and Test!"
    
    # 3. Check for overlaps between Val and Test
    val_test_overlap = set_val.intersection(set_test)
    assert len(val_test_overlap) == 0, f"LEAKAGE DETECTED: {len(val_test_overlap)} IDs overlap between Val and Test!"
    
    print(">>> Anti-leakage checks PASSED: Zero overlap confirmed across all splits. <<<")


    os.makedirs(output_dir, exist_ok=True)
    
    # Save the 3 required split JSON files
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