# Model Checkpoints

Pretrained model checkpoints are **not stored directly in this repository** due to file
size limitations.

The pretrained models used in this project are available under the **GitHub Releases**
section of this repository.

---

## Available Models

### `zero_shot.pth`
- Training setup: trained on synthetic data only
- Purpose: zero-shot evaluation on real images

### `fine_tuned.pth`
- Training setup: synthetic pretraining followed by fine-tuning on a small real dataset
- Purpose: improved generalization to real chessboard images

### `combined.pth`
- Training setup: training on a combination of synthetic and real data
- Purpose: final model used for inference (default)
