# Logistic Regression from Scratch

A NumPy-only implementation of binary logistic regression with **mini-batch SGD**, **L2 regularization**, and a **numerically stable** forward/backward pass — validated against `scikit-learn` on the Wisconsin Breast Cancer dataset.

The point of this project isn't to beat scikit-learn. It's to show that I can derive the gradients by hand, reason about the optimizer, and match a battle-tested library's solution to within optimizer noise — and explain every line of why.

---

## Results

Trained for 100 epochs (`lr=0.1`, mini-batch size 32) with the L2 strength matched to scikit-learn's `C=1.0`. 80/20 split, `random_state=42`, features standardized on the train split only.

| Metric | Scratch | scikit-learn |
|---|---|---|
| Test accuracy | **99.12%** | 97.37% |
| Weight vector correlation (vs sklearn) | **0.97** | — |
| Final regularized train loss | 0.072 (from 0.210) | — |

The scratch model edging ahead on this particular split is split variance, not a real quality gap — the meaningful signal is the **0.97 correlation between the two weight vectors**, which says both optimizers converged to essentially the same decision boundary from completely different code.

![Training loss curve](assets/loss_curve.png)

---

## What's implemented from scratch

- **Sigmoid** with overflow guarding (`z` clipped to `[-500, 500]` before `exp`).
- **Binary cross-entropy** loss with `eps`-clipping inside the `log` to avoid `log(0)`.
- **L2 regularization** on the weights only — the bias is deliberately left unregularized.
- **Mini-batch SGD** with per-epoch reshuffling.
- Analytic gradients (no autograd), derived below.
- Loss history tracking for the convergence plot.

---

## The math

**Forward pass.** For features $x$, weights $w$, bias $b$:

$$z = x w + b, \qquad \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

**Objective.** Mean binary cross-entropy plus an L2 penalty:

$$\mathcal{L}(w, b) = -\frac{1}{m}\sum_{i=1}^{m}\Big[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\Big] + \frac{\lambda}{2}\lVert w \rVert_2^2$$

**Gradients.** The reason logistic regression is clean is that the sigmoid and the cross-entropy cancel almost everything. With $\sigma'(z) = \sigma(z)(1-\sigma(z))$, the chain rule collapses to:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{m} X^\top (\hat{y} - y) + \lambda w$$

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)$$

The error term $(\hat{y} - y)$ is the only thing the loss derivative leaves behind — no sigmoid derivative survives in the final expression. The $\lambda w$ term comes straight from differentiating $\tfrac{\lambda}{2}\lVert w\rVert^2$, and note it's absent from the bias gradient because the bias isn't penalized.

---

## Matching scikit-learn's regularization

scikit-learn minimizes (for `lbfgs`/`liblinear`):

$$\frac{1}{2} w^\top w + C \sum_{i=1}^{m} \text{logloss}_i$$

Dividing through by $C \cdot m$ rewrites this in the same per-sample-mean form this project uses, with the penalty coefficient

$$\frac{\lambda}{2} = \frac{1}{2 C m} \;\Longrightarrow\; \lambda = \frac{1}{C \, m}$$

So `C=1.0` corresponds to `lambda_param = 1 / n_samples`, which is exactly how the comparison is set up. Getting this mapping right is what makes the weight-correlation comparison fair rather than apples-to-oranges.

---

## A known subtlety (and why I left it in)

The L2 gradient term $\lambda w$ is applied **once per mini-batch**, not once per epoch. With ~15 batches per epoch, the weights get decayed ~15× more often than the matched objective implies, so the realized regularization is stronger than the nominal $\lambda$. Measurably:

| Penalty application | Weight-norm ratio (scratch / sklearn) | Correlation |
|---|---|---|
| Per batch (current) | 0.84 | 0.9704 |
| Per batch, scaled by `1/n_batches` | 0.95 | 0.9721 |

The current behavior is the standard SGD-with-weight-decay convention (each step nudges weights toward zero), so it's a legitimate choice rather than a bug — but it's the reason the scratch weights come out slightly shrunk relative to sklearn's full-batch solution. The one-line fix, if you want the realized penalty to match the stated objective exactly, is to scale `lambda_param` by `1/n_batches` inside the batch loop.

---

## Usage

```bash
pip install numpy scikit-learn matplotlib
python compare.py   # trains both models, prints the metrics table
```

```python
from logistic_regression import LogisticRegressionScratch

model = LogisticRegressionScratch(
    learning_rate=0.1,
    n_iters=100,
    lambda_param=1 / n_samples,   # matches sklearn C=1.0
    batch_size=32,
)
model.fit(X_train, y_train)        # expects standardized features
preds = model.predict(X_test)      # default threshold 0.5
```

**Note:** features must be standardized before training — the fixed learning rate assumes roughly unit-scale inputs, and the sigmoid clipping will mask exploding activations rather than fix them.

---

## Project structure

```
logistic-regression-from-scratch/
├── logistic_regression.py   # the LogisticRegressionScratch class
├── compare.py               # trains + benchmarks against sklearn
├── assets/
│   └── loss_curve.png       # convergence plot
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Possible extensions

- Multiclass via one-vs-rest or a softmax head.
- Early stopping on a validation-loss plateau instead of a fixed epoch count.
- L1 / elastic-net penalties.
- A learning-rate schedule (the loss curve flattens hard after ~epoch 40).
