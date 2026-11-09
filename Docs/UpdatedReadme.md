
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
Train KB: 10000 | Val: 100 | Test: 1000 ```

