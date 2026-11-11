
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


*Run this command to execute : INTEGRATION TEST Metadata AND Captions with images*
 python3 scripts/test_phase1_integration.py





## Phase 1 Statistics and Anlysis : 
``` bash 
pip install nltk 
```

run every cell cuz they're connected espcaially variables 







## Preprocessing results (step1 from STEP3):
ex of output:

Total images processed: 10,000

Output format: JPEG (.jpg)

Output dimensions: 224 × 224

Color mode: RGB

Successfully processed: 10,000

Failed images: 0

Validation results:

JPEG files found: 10,000

Invalid formats: 0

Invalid dimensions: 0

Non-RGB images: 0

Unreadable images: 0


*new strecture:*
data/
└── coco_rag_data/
    └── processed_images/
        ├── 000000073653.jpg
        ├── ...
        └── (10,000 processed images)

## How to validate the images:

Run the preprocessing script:

python .\scripts\preprocess_images.py        




### Image Batch Processing & Validation (Member 4)

Implemented a batch-processing pipeline to preprocess 10,000 COCO images efficiently in batches of 100.

**Features:**
- Batch processing with progress tracking.
- Image resizing to 224 × 224 pixels.
- Conversion to RGB and JPEG format.
- Error handling and JSON processing reports.
- Image integrity and output validation.

**Results:**
- Images processed: 10,000
- Batch size: 100 (100 batches)
- Processing time: 206.83 seconds
- Failed images: 0
- Corrupted images: 0
- Invalid dimensions: 0
- Non-RGB images: 0

**Validation status:** `SUCCESS`

**Scripts:**
- `scripts/run_preprocessing_batch.py` — Batch preprocessing.
- `scripts/diagnose_images.py` — Checks missing and corrupted images.
- `scripts/validate_processed_images.py` — Validates format, dimensions, and RGB mode.

**Reports:**
- `data/coco_rag_data/batch_processing_report.json`
- `data/coco_rag_data/processed_images_batch_diagnostic.json`











## Caption preprocessing which is the same for batch processing 

### Run it

**Linux**
```bash
python3 scripts/caption_pipeline.py --self-test
python3 scripts/caption_pipeline.py --name kb --forbid-ids data/coco_rag_data/test_ids.json
python3 scripts/caption_pipeline.py --name test --input data/coco_rag_data/test_lookup.json
```

**Windows**
```powershell
python src\preprocessing\caption_pipeline.py --self-test
python src\preprocessing\caption_pipeline.py --name kb --forbid-ids data\coco_rag_data\test_ids.json
python src\preprocessing\caption_pipeline.py --name test --input data\coco_rag_data\test_images_metadata.json
```




**Member 1 — Image Embedding Extraction: COMPLETED**

I successfully extracted CLIP image embeddings for all 10,000 preprocessed COCO images.

**Configuration:**
- Model: `openai/clip-vit-base-patch32`
- Framework: PyTorch + Hugging Face Transformers
- Device: CPU
- Batch size: 16
- Embedding dimension: 512
- Data type: float32
- L2 normalization: enabled

**Results:**
- Images processed: 10,000
- Failed images: 0
- Embedding matrix: `(10000, 512)`
- Processing time: 1,377.62 seconds
- Status: SUCCESS

*to run the script*
`python .\scripts\extract_image_embeddings.py--batch-size16`                                                         

**Generated deliverables:**
- `image_embeddings.npy`
- `image_embeddings_mapping.json`
- `image_embeddings_metadata.json`
- `embedding_report.json`

The image embeddings are ready for the FAISS indexing stage. Please confirm the expected similarity metric and normalization convention before integration.

**Member 2 — Text Embedding Extraction: COMPLETED**

Successfully extracted normalized CLIP text embeddings for all 50,000 reference captions in `unified_lookup.json`.

**Configuration:**
- Model: `openai/clip-vit-base-patch32`
- Framework: PyTorch + Hugging Face Transformers
- Device: CPU
- Batch size: 64
- Embedding dimension: 512
- Data type: float32
- L2 normalization: enabled (FAISS IndexFlatIP ready)

**Results:**
- Captions processed: 50,000
- Embedding matrix: `(50000, 512)`
- Status: SUCCESS

*To run the script:*
`python .\scripts\extract_text_embeddings.py`

**Generated deliverables:**
- `data/coco_rag_data/text_embeddings.npy`
- `data/coco_rag_data/text_embeddings_mapping.json`
- `data/coco_rag_data/text_embedding_report.json`



# FAISS Index (Step 4, task 3)


```bash
python3 scripts/build_faiss_index.py --name text
python3 scripts/build_faiss_index.py --name image --embeddings data/coco_rag_data/image_embeddings/image_embeddings.npy --mapping data/coco_rag_data/image_embeddings/image_embeddings_mapping.json
```





## phase 8 quality control final Q & A: 

What I added for Phase 8 (quality control code)  
I created a full Phase 8 checker script:

- [phase8_quality_control.py](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

It checks:

1. Required artifacts exist.
2. Split anti-leakage.
3. Lookup ID coverage.
4. Caption count policy.
5. Embedding integrity (shape, dtype, NaN/Inf).
6. L2 normalization.
7. Vector-to-metadata linkage.
8. Missing embedding rows.
9. FAISS index size consistency.
10. FAISS self-retrieval smoke test.

It writes:

- [phase8_quality_control_report.json]
Run command  
python [phase8_quality_control.py]









1. Required files exist
    
    - Makes sure training, validation, test IDs, lookup files, embeddings, mapping JSON, and FAISS index all exist.
2. Split anti-leakage
    
    - Ensures the same image IDs do not appear across train/val/test sets.
3. Lookup ID coverage
    
    - Confirms every image ID in the split appears in its corresponding lookup table.
4. Caption count policy
    
    - Checks each training image has exactly 5 captions, as expected.
5. Embedding matrix integrity
    
    - Validates shape, dtype, finite values, and that the embedding count matches the mapping length.
6. L2 normalization
    
    - Verifies text embeddings are normalized to unit length, which is important for FAISS similarity search.
7. Text mapping linkage
    
    - Confirms each mapping row points to a valid image, has no duplicate caption IDs/indices, and matches the expected number of caption rows.
8. Missing embedding rows
    
    - Checks there are no missing rows between the lookup caption count and the embedding mapping.
9. FAISS index size match
    
    - Ensures the FAISS index has the same number of vectors and dimension as the embeddings.
10. FAISS self-retrieval smoke test
    

- Queries a sample of vectors against the index and verifies they retrieve themselves with high accuracy.