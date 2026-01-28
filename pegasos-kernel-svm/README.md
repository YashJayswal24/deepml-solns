# Pegasos Kernel SVM Implementation

A deterministic implementation of the Pegasos algorithm for training kernel SVMs from scratch.

## 🧠 Theory: The Math Behind SVMs

SVMs are mathematically intense. There are two main approaches:

1. **Primal SVM (Pegasos):** Uses sub-gradient descent directly on the hinge loss objective.
2. **Dual SVM (SMO):** Uses Lagrange multipliers and the KKT conditions to solve the dual optimization problem.

### The Kernel Trick
Instead of computing transformations to higher dimensions explicitly, kernels compute the dot product in that space directly:
- **Linear:** `K(x, y) = x · y`
- **RBF:** `K(x, y) = exp(-||x - y||² / 2σ²)`

### Pegasos Update Rules (Deterministic)
The algorithm I implemented:
1. **Learning Rate:** `η_t = 1 / (λ * t)` (decreases over time)
2. **Margin Check:** If `y_i * f(x_i) < 1` (margin violated):
   - `α_i ← (1 - 1/t) * α_i + η_t * y_i`
   - `b ← b + η_t * y_i`

> **Note:** The original Deep-ML description uses `α_i ← α_i + η_t(y_i - λ*α_i)`, but my solution simplifies the decay term to `(1 - 1/t)` which is equivalent to `(1 - η_t * λ)`. Both achieve the same result.

---

## 💡 Key Learnings & Insights

This was by far the **hardest concept mathematically** for me to understand.

- **Dual vs. Primal:** The SMO (Dual) method has complex update rules involving error terms (`E_i = f(x_i) - y_i`) and kernel similarities.
- **Lagrangian Multipliers:** Required for solving constrained optimization, a concept heavily used in physics too.
- **Recommended Resource:** [An Idiot's Guide to SVMs (MIT)](https://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf) — An excellent resource, though don't be fooled by the title; this requires serious mathematical effort!

---
*Solved as part of my deep-learning journey on Deep-ML.*
