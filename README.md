# Chest X-Ray VAE

Clean, modular PyTorch repo for training and evaluating a beta-VAE on the Kaggle Chest X-Ray Pneumonia dataset.

Dataset:

* Kaggle: Chest X-Ray Images (Pneumonia)
* Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Project structure

```text
chest\_xray\_vae\_repo/
├── configs/
│   ├── train.yaml
│   └── eval.yaml
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── src/
│   └── chest\_xray\_vae/
│       ├── \_\_init\_\_.py
│       ├── config.py
│       ├── data.py
│       ├── evaluate.py
│       ├── train.py
│       ├── models/
│       │   └── vae.py
│       └── utils/
│           ├── checkpoint.py
│           ├── io.py
│           ├── metrics.py
│           └── seed.py
├── outputs/
├── notebooks/
├── tests/
├── requirements.txt
├── .gitignore
└── pyproject.toml
```

## Expected dataset layout

After downloading and extracting the Kaggle dataset, place it like this:

```text
data/
└── chest\_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py --config configs/train.yaml
```

## Evaluate

```bash
python scripts/evaluate.py --config configs/eval.yaml
```

## Notes

* Images are converted to grayscale and resized to a fixed square resolution.
* The training script saves:

  * best checkpoint
  * last checkpoint
  * config snapshot
  * JSON metrics history
  * loss plots
  * reconstruction previews
  * generated sample grids
* The evaluation script computes:

  * reconstruction loss
  * KL loss
  * total loss
  * FID using 2048-d Inception features
  * Inception Score
  * generated sample exports

