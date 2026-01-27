# Basic Autograd Operations (Micrograd style)

Special thanks to **Andrej Karpathy** for his amazing [YouTube tutorial](https://youtu.be/VMj-3S1tku0?si=gjlnFP4o3JRN9dTg) on building Micrograd. This implementation is inspired by his work.

## 🚀 The Idea

The most beautiful thing about this implementation is how the **differentiation chain** is split into systematic, step-by-step commands. Instead of manual calculus, we define how each operation (Addition, Multiplication, ReLU) contributes to the gradient locally, and let the chain rule do the rest.

### 🧩 Topological Sorting: From LeetCode to Real Life

I used to think of **Topological Sorting** as just another algorithm for LeetCode or CodeForces problems. Seeing it used here was an "aha!" moment.

In a computational graph, some gradients *must* be computed before others. If you have `d = a + b`, you can't accurately find the gradient of `a` until you've finished with `d`. Topological sorting ensures that we process the graph in the exact order required by the dependencies, allowing a single `.backward()` call to trigger the entire chain—just like in PyTorch.

### 🐍 Python Simplicity

It's incredible how Python allows us to write such systematic code. By overriding magic methods like `__add__` and `__mul__`, we can build complex expression trees that track their own history (`_prev`) and know how to propagate gradients back (`_backward`).

---
*Implementation details found in `autograd.py`. Truly a masterclass in elegant software design for ML.*
