# Pessimistic Imaginary Rewards (PIR) - Supplementary Material

This repository is a fork of the DreamerV3 JAX implementation, modified to include **Pessimistic Imaginary Rewards (PIR)**. PIR stabilizes early-stage training by penalizing reward predictions in latent imagination based on epistemic uncertainty.

## 🌟 Core Modifications
The implementation of PIR is self-contained within the following files:

1.  **`dreamerv3/agent.py`**: 
    - Modified the `imagination` method.
    - Replaced single reward prediction with $N$ stochastic forward passes ($N=3$ by default).
    - Implemented the Lower Confidence Bound (LCB) reward: $\tilde{r} = \mu(r) - \lambda \sigma(r)$.
2.  **`dreamerv3/configs.yaml`**: 
    - Added `lambda_uncertainty` (default: 0.0).
    - Added `latent_samples` (default: 3).

---

## 🚀 Reproduction Guide

### 1. Requirements
Ensure you have JAX installed with GPU support.
```bash
pip install -r requirements.txt
