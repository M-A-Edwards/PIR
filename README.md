# Pessimistic Imaginary Rewards (PIR) - Supplementary Material

This repository is a fork of the DreamerV3 JAX implementation, modified to include **Pessimistic Imaginary Rewards (PIR)**. PIR stabilizes early-stage training by penalizing reward predictions in latent imagination based on epistemic uncertainty.

## Core Modifications
The implementation of PIR is self-contained within the following files:

1.  **`dreamerv3/agent.py`**: 
    - Modified the `imagination` method.
    - Replaced single reward prediction with $N$ stochastic forward passes ($N=3$ by default).
    - Implemented the Lower Confidence Bound (LCB) reward: $\tilde{r} = \mu(r) - \lambda \sigma(r)$.
2.  **`dreamerv3/configs.yaml`**: 
    - Added `lambda_uncertainty` (default: 0.0).
    - Added `latent_samples` (default: 3).

---

## Reproduction Guide

### 1. Requirements
Ensure you have JAX installed with GPU support.
```bash
pip install -r requirements.txt
```

## 2. Run PIR (Proposed Method)

Run the 1M parameter model on DMC Vision with the uncertainty penalty (`λ = 1.0`) and for seed 1/2/3, respectively (`--seed 1`):

```bash
python dreamerv3/main.py --configs dmc_vision --size 1m \
  --logdir ./logdirM1
  --run.steps 10000 \
  --agent.lambda_uncertainty 1.0 \
  --agent.latent_samples 3 \
  --run.debug false
```

## 3. Run Baseline (Vanilla DreamerV3)

Run the baseline without the uncertainty penalty and for seed 1/2/3, respectively (`--seed 1`):

```bash
python dreamerv3/main.py --configs dmc_vision --size 1m \
  --logdir ./logdirB1
  --run.steps 10000 \
  --agent.lambda_uncertainty 0.0 \
  --run.debug false
```


# Expected Empirical Results (10k Steps)

Based on our paper, you should observe the below in the metrics file for the log directory:

| Metric                 | Baseline (`λ = 0`) | PIR (`λ = 1.0`) |
| ---------------------- | ------------------ | --------------- |
| Reward Prediction Loss | `0.814 ± 0.025`    | `0.616 ± 0.003` |
| Return Stability (`σ`) | `0.018`            | `0.004`         |

PIR significantly reduces variance across seeds and lowers reward prediction error by grounding the policy in more **certain** imaginary trajectories.

-
# PIR Logic Snippet

The following logic was integrated into the latent imagination loop:

```python
# From dreamerv3/agent.py

num_samples = self.config.latent_samples 
lambda_uncertainty = self.config.lambda_uncertainty

rew_preds = []
for _ in range(num_samples):
    # self.rew(inp, 2).pred() handles the symlog/discrete conversion
    rew_preds.append(self.rew(inp, 2).pred())

rew_stack = jnp.stack(rew_preds, axis=0)
rew_mean = rew_stack.mean(0)
rew_std = rew_stack.std(0)

# Lower Confidence Bound (LCB) calculation
rew_adjusted = rew_mean - lambda_uncertainty * rew_std
)
```

---

# License

This code is based on the original DreamerV3 repository and is licensed under the MIT License.
