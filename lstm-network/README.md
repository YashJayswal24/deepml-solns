# Long Short-Term Memory (LSTM) Network - Intuition

A manual implementation of an LSTM cell using NumPy, inspired by the classic [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah.

## 🤔 The Intuition behind the Gates

Instead of just formulas, think of an LSTM as a **highly controlled memory stream**. 

### The "Math" of Intuition
- **`*` (Multiplication):** This is the **Selector**. It decides how much of something passes through.
- **`+` (Addition):** This is the **Knowledge Adder**. It allows for gradients to flow much easier (the "highway" for memory) without vanishing.
- **`Sigmoid` (0 to 1):** The **Gatekeeper**. Used for selecting. 0 means "forget/block everything," 1 means "keep/let everything through."
- **`Tanh` (-1 to 1):** The **Featurizer**. It creates new candidate information. It’s perfect for "extracting features" because it normalizes values while preserving the "direction" of the signal.

---

### What each gate "feels" like:

1.  **Forget Gate (`f`):** *The Eraser.* It looks at the past memory and the new input and asks: "Is the old stuff still relevant?" High value means "Keep it," low value means "Wipe it."
2.  **Input Gate (`i`):** *The Filter.* It decides which parts of the *new* information are actually worth remembering.
3.  **Candidate Memory (`g`):** *The Writer.* This is the actual new information being prepared. It uses `tanh` to turn the raw data into useful features.
4.  **Cell State Update (`C_t`):** This is the **Memory Stream**. It's the old memory (filtered by the Eraser) **plus** the new information (filtered by the Writer). The `+` ensures we don't lose the signal over time.
5.  **Output Gate (`o`):** *The Presenter.* It takes the updated long-term memory and decides what parts of it should be shown as the current output (hidden state).

---
*Implementation found in `lstm.py`. Ready for the next sequence!*
