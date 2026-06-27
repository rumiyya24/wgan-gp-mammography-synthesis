# WGAN-GP Mammography Synthesis

A lightweight Wasserstein GAN with Gradient Penalty (WGAN-GP) for generating synthetic mammograms, built without attention mechanisms, progressive growing, or style modulation. The goal is to test how much image quality a deliberately simple architecture can reach — and whether that's good enough for applications like medical education, algorithm stress-testing, and rare-pathology simulation, where full diagnostic-grade realism isn't required.

This repository contains the training/preprocessing/evaluation notebooks, sample outputs, and the human evaluation application used.

## Why a simple architecture?

State-of-the-art generative models (StyleGAN2, diffusion models) produce excellent results but need multi-GPU clusters and weeks of training — out of reach for many hospitals, universities, and labs in resource-constrained settings. This project asks whether a minimal WGAN-GP, trainable on a single consumer GPU in days, can still reach a "clinically plausible enough" bar for non-diagnostic use cases.

## Results summary

| Metric | This model | StyleGAN2 (reference) | DDPM/MAMBO (reference) |
|---|---|---|---|
| FID | 48.34 | ~41.6 | ~38.1 |
| SSIM | 0.536 | ~0.57 | ~0.60 |

- **Human evaluation:** Six medical professionals (radiologists, breast surgeons, and a senior medical student) took part in a three-alternative forced-choice test to spot the synthetic image among real ones. Aggregate accuracy was **80.6%** — meaningfully above chance, but still showing that roughly one in five synthetic mammograms fooled trained experts.
- **LLM-based evaluation:** GPT-4V was tested on the same task as an exploratory, scalable alternative to expert review, reaching **76.9%** accuracy.
- **Training/inference cost:** Roughly 10x faster training and 5x faster inference than StyleGAN2-class models, at a 14–20% gap in FID.

Full methodology, ablations, and discussion are in the paper (see [Citation](#citation) below).

## Repository structure

```
.
├── notebooks/
│   ├── 01_preprocessing.ipynb       # Mammogram preprocessing pipeline
│   ├── 02_training_main.ipynb       # Main WGAN-GP training notebook
│   ├── 03_evaluation_metrics.ipynb  # FID / SSIM / GLCM / Sobel evaluation
│   └── archive/                     # Earlier experiments and training variants (TPU, PyTorch/TF trials)
├── generated_samples/
│   └── 150th_epoch/                 # Sample generator outputs at epoch 150
├── denoising_samples/
│   ├── bm3d_denoised_images/        # BM3D-denoised generator outputs
│   └── nlm_denoised_images/         # NLM-denoised generator outputs
├── human_evaluation_app/
│   ├── app.py                       # Human-evaluation ("Turing test") application
│   ├── mi_results.txt               # Raw per-case results, one evaluator
│   └── test_results.txt             # Raw per-case results, another evaluator
└── README.md
```

## Datasets

Training used two public mammography datasets, which are **not included in this repository** due to size and access terms:

- **CBIS-DDSM** — Curated Breast Imaging Subset of the Digital Database for Screening Mammography, available via The Cancer Imaging Archive (TCIA).
- **RSNA Breast Cancer Screening Dataset** — available via the [RSNA Breast Cancer Detection Challenge on Kaggle](https://www.kaggle.com/competitions/rsna-breast-cancer-detection).

You'll need to obtain these directly from their sources and agree to their respective usage terms before running the preprocessing/training notebooks.

## Model weights

Trained checkpoints (generator and discriminator, both TensorFlow and PyTorch versions) are **not included** in this repository due to file size. If you need them for research purposes, feel free to open an issue or reach out directly.

## Getting started

1. Obtain CBIS-DDSM and RSNA datasets (see above) and place them according to the paths expected in `notebooks/01_preprocessing.ipynb`.
2. Run notebooks in order: preprocessing → training → evaluation.
3. To explore the human-evaluation tool, see `human_evaluation_app/app.py`.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Model architecture (brief)

- **Generator:** maps a 100-dim latent vector through three fully-connected layers directly to a 256×256 pixel grid, followed by two lightweight convolutional refinement blocks. No progressive growing or upsampling cascade.
- **Critic:** four strided convolutional layers downsampling from 256×256 to 16×16, followed by global average pooling and two dense layers producing an unbounded realness score. No batch normalization, per WGAN-GP best practice.
- **Loss:** standard WGAN-GP objective with gradient penalty coefficient λ=10, 5 critic updates per generator update.

See the paper for full architectural and training details.

## Limitations & ethical notes

- This model is for **non-diagnostic use only** (education, algorithm robustness testing, rare-case simulation). It has not been validated for training production diagnostic systems.
- Unconditional generation means specific pathology subtypes can't be requested on demand.
- Synthetic medical images carry real risks of misuse (e.g., fraudulent use in research/regulatory submissions) and of amplifying biases present in the training data.
  
## Citation

If you use this work, please cite the accompanying paper. (Citation details to be added once published — check back here or open an issue if you need a pre-publication reference.)

## License

MIT — see [LICENSE](LICENSE).
