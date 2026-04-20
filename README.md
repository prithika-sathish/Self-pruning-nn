# Self-Pruning Neural Network

## Overview

This project implements a neural network that learns to prune its own weights during training. Each weight is associated with a learnable gate, allowing the network to suppress less important connections dynamically instead of relying on post-training pruning.

---

## Project Structure

```
self-pruning-nn/
├── train.py                         # main script
├── self_pruning_nn_fixed_v2.ipynb   # notebook (experiments + outputs)
├── results/                         # generated outputs
├── requirements.txt
├── README.md
```

---

## Method

Each weight has a learnable gate:

```
gate = sigmoid(gate_score)
effective_weight = weight × gate
```

Training objective:

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
```

* CrossEntropyLoss → classification performance
* SparsityLoss (L1 on gates) → encourages pruning
* λ → controls sparsity vs accuracy trade-off

---

## Architecture

Feedforward network on CIFAR-10:

```
3072 → 512 → 256 → 128 → 10
```

All layers are implemented using custom `PrunableLinear`.

---

## Results

| Lambda | Test Accuracy | Sparsity (%) |
| ------ | ------------- | ------------ |
| 0.01   | 56.42%        | 1.71%        |
| 0.05   | 55.45%        | 1.87%        |
| 0.1    | 56.10%        | 6.39%        |

### Observations

* Increasing λ increases sparsity
* Accuracy remains relatively stable for small λ
* Higher λ (0.1) leads to noticeable pruning
* Deeper layers exhibit higher pruning compared to early layers

---

## Outputs

All outputs are stored in:

```
results/
```

Includes:

* `report.md` → explanation and analysis
* `results.json` → final metrics
* `gate_distribution.png` → gate value histogram
* `training_curves.png` → accuracy and loss curves
* `model_lambda_*.pt` → trained models

---

## How to Run

### Option 1: Script

```
pip install -r requirements.txt
python train.py
```

### Option 2: Notebook

Open `self_pruning_nn_fixed_v2.ipynb` in Jupyter or Colab and run all cells.

---

## Dataset

Uses CIFAR-10 via torchvision (automatically downloaded).

---

## Key Insight

The model performs **soft pruning during training** by continuously reducing gate values. A threshold is applied during evaluation to measure effective sparsity, enabling the network to learn a compact representation without a separate pruning stage.

---

## Conclusion

The model successfully:

* learns which connections are less important
* gradually suppresses them via gating
* maintains a balance between sparsity and accuracy

This demonstrates a practical approach to self-pruning neural networks.
