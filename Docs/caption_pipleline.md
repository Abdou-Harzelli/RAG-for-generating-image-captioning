## Caption processing (cleaning caption)

Script: `scripts/caption_pipeline.py`

### What it does

Every image has 5 human captions. We keep **two versions** of each:

| Field | Content | Used for |
|---|---|---|
| `caption_raw` | Exactly as COCO wrote it, never modified | **Evaluation only** (BLEU, F1, hallucination rate) |
| `caption_clean` | Lightly normalized text | **Retrieval only** (BM25, text embeddings, FAISS) |

Cleaning rules: lowercase, remove accents, straighten curly quotes, drop punctuation
and hyphens (`black-and-white` becomes `black and white`), collapse spaces and newlines.
We deliberately do **not** remove stopwords or stem words. That is left to the Phase 2 ablation.

### Example (test image 21465)

| caption_raw | caption_clean |
|---|---|
| `A dirty table has incense burners on it.` | `a dirty table has incense burners on it` |
| `There are various items on shelves tagged for sale. ` | `there are various items on shelves tagged for sale` |
| `a cabinet with some paint on top of it ` | `a cabinet with some paint on top of it` |
| `A blue shelf holds different sized silver vases.` | `a blue shelf holds different sized silver vases` |
| `A blue shelving unit has a vase and metal cups on it. ` | `a blue shelving unit has a vase and metal cups on it` |

Note the trailing spaces in the raw captions. They are kept in `caption_raw` and removed in `caption_clean`.

### Run it

**Linux**
```bash
python3 scripts/caption_pipeline.py --self-test
python3 scripts/caption_pipeline.py --name kb --forbid-ids data/coco_rag_data/test_ids.json
python3 scripts/caption_pipeline.py --name test --input data/coco_rag_data/test_images_metadata.json
```

**Windows**
```powershell
python scripts\caption_pipeline.py --self-test
python scripts\caption_pipeline.py --name kb --forbid-ids data\coco_rag_data\test_ids.json
python scripts\caption_pipeline.py --name test --input data\coco_rag_data\test_images_metadata.json
```

Results from our run:

| Split | Images | Captions | Wrong caption count | Empty after cleaning | Avg words raw -> clean |
|---|---|---|---|---|---|
| kb | 10,000 | 50,000 | 0 | 0 | 10.461 -> 10.469 |
| test | 100 | 500 | 0 | 0 | 10.572 -> 10.588 |
