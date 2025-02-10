# Region-aware Distribution Contrast: A Novel Approach to Multi-Task Partially Supervised Learning
This repo is the official implementation of Region-aware Distribution Contrast: A Novel Approach to Multi-Task Partially Supervised Learning.
## Requirements

use the provided requirements.txt to install the dependencies:

```bash
pip install -r requirements.txt
```

## Prepare dataset

We use the preprocessed NYUv2 dataset and Cityscapes dataset as "(CVPR22)Learning Multiple Dense Prediction Tasks from Partially Annotated Data".  Download the dataset and the label setting, then place the dataset folder in `./data/`

## Train our method

Modify the paths to dataset in nyu_mtl_region_cons_gaussian.py

Then run

```bash
python nyu_mtl_region_cons_gaussian.py
```

## Evaluation

We provide the pre-trained models in `./checkpoints/`. Modify the paths to checkpoint in eval_nyu_region_cons_gaussian.py

Then run

```bash
python eval_nyu_region_cons_gaussian.py
```

