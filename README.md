# Autonomous Fine-Tuning of Vision–Language Models for Turbulence-Degraded Text Recognition

[![JMLR 2025](https://img.shields.io/badge/JMLR-2025-blue.svg)](https://jmlr.org)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace Demo](https://img.shields.io/badge/🤗-Demo-orange.svg)](https://huggingface.co/spaces/Rafi123/qwen3b-autonomous-demo)
[![Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/fatehajannatayrin/turbulence-blur-text-extraction-book-images)

**[Project Page](https://ayrin21.github.io/asft) &nbsp;|&nbsp; [Paper](#) &nbsp;|&nbsp; [Demo](https://huggingface.co/spaces/Rafi123/qwen3b-autonomous-demo) &nbsp;|&nbsp; [Dataset](https://www.kaggle.com/datasets/fatehajannatayrin/turbulence-blur-text-extraction-book-images)**

> **Fateha Jannata Ayrin\*, Muhammad Anwarul Azim\*, Mohammad Khairul Islam**  
> Department of Computer Science and Engineering, University of Chittagong, Bangladesh  
> \* Equal Contribution

---

## Overview

This repository contains the code and resources for **ASFT (Autonomous Self-Supervised Fine-Tuning)**, a framework for improving OCR performance of Qwen2.5-VL-3B on book cover images degraded by strong atmospheric turbulence (α = 0.7).

ASFT achieves **93.27% text similarity** and **9.59% CER** on 50 held-out test samples, outperforming all evaluated supervision strategies and the zero-shot baseline by a large margin.

![System Interface](Screenshot%202026-04-08%20005848.png)

---

## Key Results

| Method | EM (%) | TS (%) | W-F1 (%) | CER (%) ↓ | WER (%) ↓ |
|---|---|---|---|---|---|
| Base (Zero-shot) | 0.00 | 45.09 | 28.74 | 72.80 | 89.30 |
| Standard QLoRA | 26.00 | 84.10 | 79.80 | 20.50 | 28.40 |
| PCL (Layer 6) | 34.00 | 88.97 | 85.11 | 13.47 | 19.58 |
| **ASFT (Ours)** | **38.00** | **93.27** | **88.18** | **9.59** | **14.57** |

- **+48.18 pp** text similarity over zero-shot baseline  
- **−63.21 pp** CER over zero-shot baseline  
- **4× data efficiency** — ASFT with 100 samples matches QLoRA with 400 samples  
- **Zero inference overhead** — all novel modules discarded after training  

---

## Method

ASFT integrates three novel training-only modules on top of a **QLoRA backbone** (Qwen2.5-VL-3B, 4-bit NF4, r = 32):

### 🔴 AQE — Autonomous Quality Evaluator
Four scalar quality scores (accuracy, coherence, completeness, confidence) are produced from mean-pooled final-layer hidden states via independent MLP heads (d→256→64→1). A meta-decision network routes each sample to one of three decisions:
- **Accept** → loss scale ×0.9 (prevent over-fitting on easy samples)
- **Refine** → loss scale ×1.0 (standard gradient)
- **Reject** → loss scale ×1.5 (hard-example emphasis)

### 🟢 SCM — Self-Critique Module
A single-layer Transformer encoder re-processes hidden states H to derive an improvement vector ΔH. Refined states H̃ = H + W_r[H; ΔH] are used in the consistency loss. Operates at training time only — zero inference overhead.

### 🟣 ACM — Adaptive Curriculum Manager
Maintains a rolling window of size 50 over quality scores and acceptance rates, dynamically adjusting the AQE quality threshold τ_t at each update step. Difficulty increases when the model is performing well and decreases when struggling.

### Composite Loss

```
L = L_CE + 0.35 * L_qual - 0.25 * L_cons
```

- `L_qual` drives all four quality scores toward 1
- `L_cons` enforces alignment between SCM-refined and original hidden states, weighted by confidence

---

## Installation

```bash
git clone https://github.com/ayrin21/asft.git
cd asft
pip install -r requirements.txt
```

**Requirements:**
- Python 3.9+
- PyTorch 2.0+
- Transformers 4.40+
- PEFT (for QLoRA)
- bitsandbytes
- Pillow, OpenCV (for turbulence simulation)

---

## Dataset

The dataset is built from book cover images sourced from **Project Gutenberg**, degraded using our atmospheric turbulence simulation pipeline.

- **Download:** [Kaggle Dataset](https://www.kaggle.com/datasets/fatehajannatayrin/turbulence-blur-text-extraction-book-images)
- **Size:** 5,000 clean + 500 turbulence-degraded book covers
- **Split:** 400 train / 50 validation / 50 test (seed 42)
- **Format:** LLaVA conversation JSON (UTF-8)

### Turbulence Simulation

Each image is degraded with α = 0.7:
1. Spatially correlated Gaussian displacement fields (Δx, Δy ~ N(0, (α·4.5)²))
2. Smoothed with 11×11 Gaussian kernel
3. Bilinear remapping
4. Final Gaussian blur (9×9, σ_b = 2.0)

---

## Training

```bash
python train_asft.py \
  --model_name Qwen/Qwen2.5-VL-3B-Instruct \
  --data_path data/train.json \
  --output_dir checkpoints/asft \
  --lora_r 32 \
  --lora_alpha 64 \
  --bits 4 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 2e-4 \
  --epochs 2 \
  --use_aqe \
  --use_scm \
  --use_acm
```

**Training hardware:** Dual Tesla T4 GPUs  
**Training time:** ~40 minutes (400 samples) / ~368 minutes (full ASFT run)  
**Extra VRAM:** 0.61 GB for AQE + SCM + ACM modules during training

---

## Inference

```bash
python inference.py \
  --model_name Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter_path checkpoints/asft \
  --image_path path/to/book_cover.jpg
```

Output format:
```
Title: The Adventures of Huckleberry Finn
Author: Mark Twain
Other: English
```

**Inference speed:** ~7.13 seconds/image (4-bit, Tesla T4)

---

## Repository Structure

```
asft/
├── train_asft.py          # Main training script
├── inference.py           # Inference script
├── turbulence_sim.py      # Turbulence simulation pipeline
├── modules/
│   ├── aqe.py             # Autonomous Quality Evaluator
│   ├── scm.py             # Self-Critique Module
│   └── acm.py             # Adaptive Curriculum Manager
├── data/
│   ├── train.json         # Training data (LLaVA format)
│   ├── val.json           # Validation data
│   └── test.json          # Test data
├── evaluate.py            # Evaluation metrics (TS, CER, WER, F1, EM)
├── requirements.txt
├── index.html             # Project page
└── README.md
```

---

## Demo

Try the live demo on HuggingFace Spaces — upload any turbulence-degraded book cover image and get structured OCR output instantly:

👉 **[https://huggingface.co/spaces/Rafi123/qwen3b-autonomous-demo](https://huggingface.co/spaces/Rafi123/qwen3b-autonomous-demo)**

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{ayrin2025asft,
  title   = {Autonomous Fine-Tuning of Vision-Language Models
             for Turbulence-Degraded Text Recognition},
  author  = {Ayrin, Fateha Jannata and
             Azim, Muhammad Anwarul and
             Islam, Mohammad Khairul},
  journal = {Journal of Machine Learning Research},
  year    = {2025},
  url     = {https://github.com/ayrin21/asft}
}
```

---

## Contact

- **Fateha Jannata Ayrin** — fjayrin@std.cu.ac.bd  
- **Muhammad Anwarul Azim** — azim@cu.ac.bd  
- **Mohammad Khairul Islam** — mkislam@cu.ac.bd  

Department of Computer Science and Engineering  
University of Chittagong, Chittagong 4331, Bangladesh

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
