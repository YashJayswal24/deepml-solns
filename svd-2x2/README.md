# Singular Value Decomposition (SVD) for 2x2

Implementation of SVD for a simple 2x2 matrix using basic linear algebra properties.

## 🧮 Theory & Insights

SVD is arguably the most famous ML technique before the Deep Learning era.

- **The Math:** We use the property that the headers of `V` (right singular vectors) are the eigenvectors of `A^T A`, and the singular values `S` are the square roots of its eigenvalues.
- **Dimensionality Reduction:** SVD finds the "principal components" just like PCA, identifying the axes of greatest variance.

### Key Learnings
Implementing this refreshed my memory from college linear algebra. It's fascinating how one elegant decomposition can capture the "essence" of a matrix and change spaces so effectively.

---
*Solved on Deep-ML.*
