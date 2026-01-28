# DenseNet Dense Block Implementation

A from-scratch implementation of a DenseNet dense block using NumPy.

## 💡 Key Learnings

- **Type Matters:** Forgot to use `0.0` instead of `0` in `np.maximum`, causing type conversion issues. Small details matter!
- **DenseNet's Philosophy:** Concatenate *all* previous features into each layer. Brute-force feature reuse.

## ⚖️ DenseNet vs. ResNet

| Feature | DenseNet | ResNet |
|---|---|---|
| Connection | Concatenation (adds channels) | Addition (merges channels) |
| Parameters | Fewer (efficient reuse) | More |
| GPU Memory | **High** | Low/Moderate |
| Training Speed | **Slow** | Faster |
| Best For | Max accuracy & feature reuse | Speed & lower memory |

**Verdict:** They achieve similar goals (gradient flow), but ResNet is more practical for real-world deployment.

---
*Solved on Deep-ML.*
