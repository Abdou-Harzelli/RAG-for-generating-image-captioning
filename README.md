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
- `train_kb_ids.json`: Contains 8,000 image IDs for the knowledge base.
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