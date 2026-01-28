# GPT-2 Text Generation (Simplified)

A simplified implementation of the GPT-2 text generation loop, focusing on the core architecture components like embeddings, layer normalization, and the autoregressive generation process.

## 🧠 Theory: Transformers & Text Generation

The Transformer architecture, introduced in "Attention is All You Need", revolutionized NLP by allowing parallel processing of sequences. GPT-2 takes this further with a decoder-only stack trained to predict the next word in a sequence.

### Core Components implemented:
1.  **Embeddings (Token + Positional):** We combine the meaning of the word (`wte`) with its position in the sentence (`wpe`).
2.  **Layer Normalization:** Crucial for stabilizing deep networks. It ensures that inputs to each layer have a consistent distribution.
3.  **Autoregressive Generation:** The model predicts one token at a time, and that prediction is fed back as input for the next step.

---

## 💡 Key Learnings & Insights

Merging individual deep learning concepts into a primitive GPT was an amazing experience. It really highlights how complex systems are built from simple, understandable blocks.

-   **Dimensionality is Key:** I was initially confused by the tensor shapes, but simply noting down the dimensions of each component (e.g., `(seq_len, d_model)`) made the implementation straightforward.
-   **Layer Norm Axis:** I got stuck on which axis to normalize. Realizing that Layer Norm is about letting each *feature* have a competitive say in gradient descent helped me understand why we normalize across the last axis (features), unlike Batch Norm.
-   **Autoregression & Caching:** The implementation shows how redundant the computation is—we re-process the entire `hello hello` sequence just to add one more `hello`. This redundancy perfectly sets the stage for understanding why **KV Caching** is so important in production LLMs.

---
*Solved as part of my deep-learning journey on Deep-ML.*
