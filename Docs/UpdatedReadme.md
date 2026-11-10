
## 4. Test the download script

Download 10 images first:

```bash
python scripts/download_coco_rag_subset.py \
  --annotations-dir data/annotations_trainval2017 \
  --output-dir data/coco_rag_data \
  --limit 10
```

## 5. Download the 10,100 training images

After the test succeeds, run:

```bash
python scripts/download_coco_rag_subset.py \
  --annotations-dir data/annotations_trainval2017 \
  --output-dir data/coco_rag_data
```

## Since we updated the number of data :

remove the old file 

``` bash
remove train_subset_10000_filenames.txt , 
and    train_subset_10000.json
```



## Dataset Splits
When running `python scripts/create_splits.py`, the script outputs the following local JSON files inside `data/coco_rag_data/`:
- `train_kb_ids.json`: Contains 1,0000 image IDs for the knowledge base.
- `val_ids.json`: Contains 100 image IDs for validation.
- `test_ids.json`: Contains 1,000 image IDs for testing.


``` bash expected output : 
└─$ python scripts/create_splits.py
>>> Anti-leakage checks PASSED: Zero overlap confirmed across all splits. <<<
Splits generated successfully!
Train KB: 10000 | Val: 100 | Test: 1000
```


## step 3 , check of captions in the 10k imgs et la generation de json special de 10k img
*nom de fichier est: build_unified_lookup.py*
the result must look like this :
========== UNIFIED LOOKUP SUMMARY ==========
Selected images:          10100
Images with captions:     10100
Images with object labels:    9914
Total captions:           50029
Output file:              data\coco_rag_data\unified_lookup.json
=============================================


*la commande pour executer le fichier est :*
python .\scripts\build_unified_lookup.py




## step 3 , check the 100 test img 

ouptupt exemple: 
*+fichier json nom:test_images_metadata.json*
--- Test Summary ---
Selected images:            1000
Images with captions:       1000
Images with object labels:  99 (this doesnt exist in the output)
Total captions:             500
Images without captions:    0
Images without object labels: 1 (this too doesnt )
Test IDs file:              data\coco_rag_data\test_ids.json
=============================================
*la commande d'execution est:*
python .\scripts\build_test_metadata.py



## new version of build_unified_lookup (VOL2)

## Extraction of Objects, Attributes, and Relationships from Images
In Phase 1 of our RAG pipeline for image captioning, step 5 focuses on extracting structured semantic metadata—specifically objects, attributes, and relationships—from the MS COCO dataset to support subsequent vector retrieval and hallucination analysis

## Why We Switched from Grounding DINO to YOLO
During our initial prototyping, we evaluated Grounding DINO (IDEA-Research/grounding-dino-base) for open-vocabulary detection. However, we encountered severe performance bottlenecks:

Excessive CPU & Memory Overhead: Grounding DINO relies on a heavy transformer architecture and dense cross-attention mechanisms that overloaded local CPU and RAM resources.

Time Constraints: With our strict 8-day sprint timeline for Phase 1, processing over 11,000 images with DINO became impractically slow.

To overcome this, we transitioned our image decoder to YOLO-World (yolov8s-world.pt). YOLO-World retains open-vocabulary capabilities (allowing us to query custom text prompts dynamically) while leveraging an optimized convolutional backbone that delivers blazing-fast inference speeds and reliable checkpoint recovery.

## Detailed Comparison: Grounding DINO vs. YOLO-World
Architecture & Backbone: Grounding DINO uses a heavy transformer-based encoder (Swin Transformer + text encoder), whereas YOLO-World enhances the lightweight YOLOv8 architecture with prompt-based region-text contrastive learning.

## Inference Speed & Efficiency: 
Grounding DINO is computationally heavy and prone to severe latency on standard student or local hardware. YOLO-World is optimized for speed, allowing us to process large batches rapidly with minimal resource strain.

## Prompt Integration: 
Both models support open-vocabulary detection via text prompts, but YOLO-World handles dynamic class embedding injections (model.set_classes()) much more efficiently during large-scale loops.

## The Role of the Confidence Threshold (--conf-threshold)
The confidence threshold (set to 0.25 in our runner script) serves as the primary mathematical filter for all model predictions.

## How it functions: 
When YOLO-World scans an image against our text queries, it outputs probability scores ranging from 0.0 to 1.0 for each bounding box. Any detection falling below --conf-threshold is automatically discarded.

## Tuning balance: A higher threshold increases precision at the risk of missing subtle or occluded objects, while a lower threshold increases recall but introduces background noise. An empirical threshold of 0.25 provides the ideal balance for capturing primary scene elements while keeping our metadata clean and structured.

*before running the script :*
 python -m pip install ultralytics       

*command to run it*:
  python .\scripts\build_unified_lookup.py

## it will generate 3 folders .json 
*1-val_lookup : for imgs of validations*
*2-test_lookup: for testing img*
*3-train_kb_lookup: for the whole imgs*