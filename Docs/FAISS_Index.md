# FAISS Index (Step 4, task 3)

Script: `scripts/build_faiss_index.py`

## What is it?
A FAISS index is a fast search tool. You give it a vector (a list of 512 numbers), and it finds the most
similar vectors in milliseconds.

We build two indexes:

| Index | Made from | Size | Used to |
|---|---|---|---|
| `text_index.faiss` | `text_embeddings.npy` | 50,000 captions | find captions that match an image |
| `image_index.faiss` | image embeddings | 10,000 images | find images that look similar |

## How it works
- **Type:** `IndexFlatIP`. It compares the query with every vector, so the answer is exact.
- **Score:** inner product. Our vectors have length 1, so this is the same as cosine similarity.
  A higher score means more similar.
- **Vector ID = row number.** Row `i` in the `.npy` file is vector `i` in the index and item `i` in the
  mapping JSON. FAISS does not store names, so the mapping file tells us which caption or image each vector is.

## How to run
Run these commands from the project root folder.

Linux / macOS:
```bash
python3 scripts/build_faiss_index.py --name text
python3 scripts/build_faiss_index.py --name image --embeddings data/coco_rag_data/image_embeddings/image_embeddings.npy --mapping data/coco_rag_data/image_embeddings/image_embeddings_mapping.json
```

Windows:
```powershell
python scripts\build_faiss_index.py --name text
python scripts\build_faiss_index.py --name image --embeddings data\coco_rag_data\image_embeddings\image_embeddings.npy --mapping data\coco_rag_data\image_embeddings\image_embeddings_mapping.json
```

The image embeddings are saved in the `image_embeddings/` subfolder, so their paths must be passed.
The text command uses the default paths. Both indexes are saved in `data/coco_rag_data/`.

## What the script checks
1. The embeddings have the right shape `(N, 512)`, no bad numbers (NaN), and length 1.
2. The mapping has the same number of items as the embeddings, in the same order.
3. The saved index file can be loaded again and has the right number of vectors.
4. Searching with 100 vectors from the index finds each one with a score above 0.999.

## Results
Copy the numbers from `data/coco_rag_data/text_index_report.json` and `image_index_report.json`.

| | Text index | Image index |
|---|---|---|
| Vectors | TBD | 10,000 |
| Dimension | 512 | 512 |
| Type / score | IndexFlatIP / inner product | IndexFlatIP / inner product |
| Build time (seconds) | TBD | 0.0036 |
| Search time (ms per query) | TBD | 0.589 |
| Identical vectors | TBD | 0 |

"Identical vectors" counts captions that are exactly the same after cleaning, for example two images
with the caption "a man riding a wave". This is normal. It can cause ties when searching.