# Model Checkpoints

This directory contains pretrained model checkpoints used for inference.

## Available Models

### `zero_shot.pth`
- Training setup: trained on synthetic data only
- Purpose: zero-shot evaluation on real images

### `fine_tuned.pth`
- Training setup: synthetic pretraining followed by fine-tuning on a small real dataset
- Purpose: improved generalization to real chessboard images

### `combined_model.pth`
- Training setup: training on a combination of synthetic and real data
- Purpose: final model used for inference.
