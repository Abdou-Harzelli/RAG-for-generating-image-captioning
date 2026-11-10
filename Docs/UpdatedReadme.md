
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

