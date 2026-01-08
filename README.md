# Project 2: Synthetic-to-Real Generalization for Chessboard Square and Board-State Classification

## Overview

This project explores **synthetic-to-real (sim-to-real) generalization** for chessboard perception.
We train a **per-square multi-class classifier** primarily on **synthetic chessboard images**
generated using Blender, and evaluate how well the trained model generalizes to **real chessboard images**.

The project examines:
- Zero-shot generalization (training on synthetic data only)
- Fine-tuning using a small amount of real data
- Joint training on synthetic and real data

The final output of the system is a **full chessboard state**, represented as a **FEN string**,
reconstructed from per-square classifications.

---

## Project Goals

- Train a per-square chessboard classifier using synthetic images
- Measure the domain gap between synthetic and real chessboard images
- Evaluate sim-to-real transfer effectiveness
- Compare zero-shot, fine-tuned, and combined training setups
- Produce a full board-state (FEN) prediction from a single static image

---

## Repository Structure

```text
.
├── data/
│   ├── synthetic/        # Synthetic images and labels
│   ├── real/             # Real images and labels
│   └── splits/           # Train / validation / test splits
├── blender/
│   ├── chess-set.blend
│   └── synthetic_chess.py
├── models/
│   ├── train.py
│   ├── finetune.py
│   └── evaluate.py
├── inference/
│   └── image_to_fen.py
├── utils/
│   ├── preprocessing.py
│   ├── perspective_transform.py
│   └── fen_utils.py
├── requirements.txt
└── README.md
