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

