import json
import os

def run_integration_tests():
    print("=" * 60)
    print("PHASE 1 - STEP 2.4: INTEGRATION & METADATA MAPPING AUDIT")
    print("=" * 60)

    # 1. Verify existence of required files
    required_files = [
        "data/coco_rag_data/train_kb_ids.json", 
        "data/coco_rag_data/val_ids.json", 
        "data/coco_rag_data/test_ids.json", 
        "data/coco_rag_data/unified_lookup.json",
        "data/coco_rag_data/test_images_metadata.json"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"❌ ERROR: Missing required file '{file}'. Ensure previous steps are completed.")
    print("✓ All split and metadata JSON files are present.")

    # 2. Load Split JSON files
    with open("data/coco_rag_data/train_kb_ids.json", "r") as f:
        train_data = json.load(f)
    with open("data/coco_rag_data/val_ids.json", "r") as f:
        val_data = json.load(f)
    with open("data/coco_rag_data/test_ids.json", "r") as f:
        test_data = json.load(f)

    train_ids = set(train_data)
    val_ids = set(val_data)
    test_ids = set(test_data)

    print(f"\n📊 Loaded Split Counts:")
    print(f"   - Training Knowledge Base : {len(train_ids)} images")
    print(f"   - Validation Split        : {len(val_ids)} images")
    print(f"   - Test Split              : {len(test_ids)} images")

    # 3. Size Validations (10k KB, 100 Val, 100 Test)
    assert len(train_ids) == 10000, f"Train KB size is {len(train_ids)}, expected 10,000."
    assert len(val_ids) == 100, f"Validation size is {len(val_ids)}, expected 100."
    assert len(test_ids) == 100, f"Test size is {len(test_ids)}, expected 100."
    print("✓ Split sizes match project requirements perfectly.")

    # 4. Anti-Leakage Audit (Test Split MUST NOT intersect with Training KB)
    # Note: Validation can share a parent folder with training, but Test must be strictly isolated.
    test_kb_overlap = train_ids.intersection(test_ids)
    assert len(test_kb_overlap) == 0, f"🚨 CRITICAL LEAKAGE: Test images found inside Knowledge Base: {test_kb_overlap}"
    print("✓ Anti-Leakage Audit Passed: Zero intersection between Test split and Training Knowledge Base[cite: 3].")

    # 5. Load Metadata Mapping File (which includes Test images too)
    with open("data/coco_rag_data/unified_lookup.json", "r") as f:
        metadata_map = json.load(f)
    ## load test metadata
    with open("data/coco_rag_data/test_images_metadata.json", "r") as f :
        test_metadata_map = json.load(f)

    print("✓ Metadata mapping file loaded successfully.")

    # 6. Completeness Check: Do ALL split IDs (Train, Val, AND Test) have metadata entries?
    all_target_ids = train_ids.union(val_ids).union(test_ids)
    missing_metadata = []

    test_image_ids = {item["id"] for item in test_metadata_map.get("images", [])}

    for img_id in all_target_ids:
        in_unified_lookup = str(img_id) in metadata_map or img_id in metadata_map
        in_test_metadata = img_id in test_image_ids

        if not (in_unified_lookup or in_test_metadata):
            missing_metadata.append(img_id)

    assert len(missing_metadata) == 0, f"❌ ERROR: {len(missing_metadata)} image IDs are missing metadata mappings!"
    print(f"✓ Completeness Check Passed: All {len(all_target_ids)} images (including Test images) have verified metadata tags.")

    print("\n" + "=" * 60)
    print("🎉 SUCCESS: ALL INTEGRATION & METADATA TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    run_integration_tests()