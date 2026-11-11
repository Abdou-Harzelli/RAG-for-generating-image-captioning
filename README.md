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