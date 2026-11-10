# RAG for Image Captioning

## 1. Clone the repository

```bash
git clone https://github.com/Abdou-Harzelli/RAG-for-generating-image-captioning.git

cd RAG-for-generating-image-captioning
```

## 2. Download the COCO dataset

Download these two files:

- [COCO 2017 Val Images (1 GB)](http://images.cocodataset.org/zips/val2017.zip)
- [COCO 2017 Train/Val Annotations (241 MB)](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

Extract both ZIP files into the `data/` directory. (inside the cloning repo create data folder and put the Extracted ZIP data)

**Do not download the full Train Images ZIP (18 GB).** We will download only 10,000 training images using our script.

## 3. Prepare the data

The directory structure should be:

```text
data/
├── val2017/
└── annotations_trainval2017/
    └── annotations/
        └── captions_train2017.json
```

Create the output directory:

```bash
mkdir -p data/coco_rag_data
```

## 4. Test the download script

Download 10 images first:

```bash
python scripts/download_coco_rag_subset.py \
  --annotations-dir data/annotations_trainval2017 \
  --output-dir data/coco_rag_data \
  --limit 10
```

## 5. Download the 10,000 training images

After the test succeeds, run:

```bash
python scripts/download_coco_rag_subset.py \
  --annotations-dir data/annotations_trainval2017 \
  --output-dir data/coco_rag_data
```

## 6. Team rules

- Everyone uses the same script and image selection file.
- Keep dataset files locally; do not upload images or ZIP files to GitHub.


## Project Proposed Structure : 
```text
RAG-for-generating-image-captioning/
├── data/                 # Local data storage (ignored by git)
├── scripts/              # Downloader scripts, audit scripts
├── src/                  # Core pipeline code (preprocessing, embeddings, etc.)
├── outputs/              # Generated .npy arrays FAISS indices (ignored by git)
├── requirements.txt      # Python dependencies
├── .gitignore            # Keeping large files out of git
└── README.md             # Project documentation
```

## Requirements installation :
- Simple :
```bash
pip install -r requirements.txt
```
- if u want to work with an environment : 
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Splits
When running `python scripts/create_splits.py`, the script outputs the following local JSON files inside `data/coco_rag_data/`:
- `train_kb_ids.json`: Contains 1,0000 image IDs for the knowledge base.
- `val_ids.json`: Contains 1,000 image IDs for validation.
- `test_ids.json`: Contains 1,000 image IDs for testing.

*Note: Ensure your `data/coco_rag_data/` directory exists before running the script.*

## For Data split and anti leackage only 
*Note: either pull with :*
```bash
 git pull --force origin main 
```    
 
*Or  Remove the 3 files :*
- test_ids.json
- train_kb_ids.json
- val_ids.json 
and then run 
```bash 
python3 scripts/create_splits.py 
```

## step 3 , check of captions in the 10k imgs et la generation de json special de 10k img
*nom de fichier est: build_unified_lookup.py*
the result must look like this :
========== UNIFIED LOOKUP SUMMARY ==========
Selected images:          10000
Images with captions:     10000
Images with object labels:    9914
Total captions:           50029
Output file:              data\coco_rag_data\unified_lookup.json
=============================================

========== TEST IMAGES DIAGNOSTIC ==========
Test IDs in file:           100
Unique test images:         100
Duplicate test IDs:         0
Train/test overlap:         0


*la commande pour executer le fichier est :*
python .\scripts\build_unified_lookup.py


## step 3 , check the 100 test img 

ouptupt exemple: 
*+fichier json nom:test_images_metadata.json*
--- Test Summary ---
Selected images:            100
Images with captions:       100
Images with object labels:  99
Total captions:             500
Images without captions:    0
Images without object labels: 1
Test IDs file:              data\coco_rag_data\test_ids.json
=============================================
*la commande d'execution est:*
python .\scripts\build_test_metadata.py

*Run this command to execute : INTEGRATION TEST Metadata AND Captions with images*
 python3 scripts/test_phase1_integration.py







## Caption Cleaning and Before/After Comparison (NEW STEP OF CLEANING THE CAPTIONS , ALL the img ARE NOW WITH 5 CAPTIONS)

### What's New?

The `scripts/build_unified_lookup.py` script has been updated with a caption-cleaning feature and a mechanism to track changes.

Previously, the script combined image metadata, captions, and object labels into `unified_lookup.json`. However, some images had more than five captions.

**The new version introduces three improvements:**

1. **Caption cleaning:** Keeps a maximum of five captions per image.
2. **Cleaning report:** Records statistics about the cleaning process.
3. **Before/after comparison:** Saves the original captions, retained captions, and removed captions for every modified image.

### Example: What Happens to an Image?

Suppose an image originally has six captions.

**Before cleaning:**
```json
{
  "file_name": "000000000309.jpg",
  "captions": [
    "A teddy bear lying in the grass.",
    "A stuffed bear resting outside.",
    "A bear toy lying on the ground.",
    "A teddy bear in a grassy area.",
    "A toy bear lying down.",
    "A teddy bear next to a tree."
  ]
}
```

**After cleaning — `unified_lookup.json`:**
```json
{
  "file_name": "000000000309.jpg",
  "captions": [
    "A teddy bear lying in the grass.",
    "A stuffed bear resting outside.",
    "A bear toy lying on the ground.",
    "A teddy bear in a grassy area.",
    "A toy bear lying down."
  ]
}
```

The first five captions are preserved in their original COCO annotation order. The sixth caption is removed because the configured maximum is five.

### New Output Files

Running the script generates three JSON files in `data/coco_rag_data/`.

#### 1. `unified_lookup.json` — Final Cleaned Data

Contains the metadata used by subsequent project stages.

Each image includes:
- `file_name`
- `url`
- `captions` — up to five captions
- `objects` — object category labels from COCO annotations

This is the main lookup after cleaning.

#### 2. `caption_cleaning_report.json` — Cleaning Statistics

Documents the overall effect of the cleaning process.

Example:
```json
{
  "total_images": 10000,
  "max_captions_per_image": 5,
  "total_captions_before": 50029,
  "total_captions_after": 50000,
  "images_modified": 29,
  "captions_removed": 29,
  "images_with_fewer_than_target": 0,
  "images_with_exactly_target_captions": 10000,
  "all_images_have_exactly_target_captions": true
}
```

These numbers illustrate the expected output based on the current dataset. The actual report may differ if the input data changes.

#### 3. `caption_changes.json` — Before/After History

**This is the main new file for tracking individual changes.**

Unlike the cleaning report, which contains aggregate statistics, this file identifies every modified image and shows exactly what changed.

Example:
```json
{
  "total_modified_images": 1,
  "changes": [
    {
      "image_id": 309,
      "file_name": "000000000309.jpg",
      "captions_count_before": 6,
      "captions_count_after": 5,
      "captions_before": [
        "Caption 1",
        "Caption 2",
        "Caption 3",
        "Caption 4",
        "Caption 5",
        "Caption 6"
      ],
      "captions_after": [
        "Caption 1",
        "Caption 2",
        "Caption 3",
        "Caption 4",
        "Caption 5"
      ],
      "captions_removed": [
        "Caption 6"
      ]
    }
  ]
}
```

**How to read this file:**
- `captions_before`: All captions before cleaning.
- `captions_after`: Captions retained in the final lookup.
- `captions_removed`: Captions excluded during cleaning.
- `total_modified_images`: Number of images whose captions were changed.

The example above is illustrative; the actual file contains the real image IDs and captions from our dataset.

### How to Run

From the repository root:

```bash
python scripts/build_unified_lookup.py
```

The script regenerates all three output files.

### Current Dataset Results

Based on our previous validation:

| Metric | Before cleaning | After cleaning |
|---|---:|---:|
| Selected images | 10,000 | 10,000 |
| Total captions | 50,029 | 50,000 |
| Images with six captions | 29 | 0 |
| Images modified | — | 29 |
| Captions removed | — | 29 |

The results should be confirmed by running the updated script.

### Important

- The script keeps the first five captions in COCO annotation order.
- It does not generate artificial captions for images with fewer than five captions.
- Images without object labels remain in the lookup.
- `caption_changes.json` records only images whose captions were modified.
- The cleaning report and comparison file help the team verify and reproduce the preprocessing results.

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