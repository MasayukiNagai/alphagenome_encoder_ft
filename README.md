# AlphaGenome Encoder Fine-tuning

`alphagenome-encoder-ft` is a PyTorch implementation of the encoder-only MPRA fine-tuning workflow from [`alphagenome_FT_MPRA`](../alphagenome_FT_MPRA/README.md).

This repository is built on top of [`alphagenome-pytorch`](../alphagenome-pytorch/README.md) and focuses on a smaller scope than the original JAX-based project. In particular, it currently targets lentiMPRA-style scalar regression with AlphaGenome encoder features, reusable encoder-only training utilities, and an MPRA oracle API for inference. It does **not** aim to cover all of the features in `alphagenome_FT_MPRA`.

## Scope

The current codebase includes:

- a `LentiMPRADataset` for TSV-based lentiMPRA training data
- a lightweight `MPRAHead` for scalar prediction from encoder outputs
- reusable encoder-only training helpers with stage 1 head-only training and optional stage 2 encoder unfreezing
- an MPRA oracle with `core`, `flanked`, and `full` construct modes
- an MPRA-specific training script in [`scripts/train_mpra.py`](scripts/train_mpra.py)

The current codebase does not yet include the full feature surface of `alphagenome_FT_MPRA`, such as the broader JAX utilities, attribution pipelines, cached embedding workflows, or the full collection of benchmarking and downstream analysis scripts.

## Repository Layout

```text
alphagenome-encoder-ft/
├── src/alphagenome_encoder_ft/
│   ├── data.py      # lentiMPRA dataset and dataloader helpers
│   ├── heads.py     # MPRAHead
│   ├── oracle.py    # MPRAOracle and checkpoint loading
│   ├── train.py     # reusable encoder-only training utilities
│   └── __init__.py
├── scripts/
│   └── train_mpra.py
└── tests/
```

## Quick Start

```bash
cd alphagenome-encoder-ft
PYTHONPATH=src python scripts/train_mpra.py \
  --input_tsv /path/to/HepG2.tsv \
  --pretrained_weights /path/to/alphagenome.pt
```

For direct Python usage, import from `alphagenome_encoder_ft` after setting `PYTHONPATH=src` or installing the package in editable mode.
