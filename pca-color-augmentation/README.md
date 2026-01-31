# PCA Color Augmentation

Implementation of the semantic color augmentation technique from the legendary AlexNet paper.

## 🎨 Theory & Insights

This technique is a perfect example of mathematical elegance in Deep Learning.

- **The Idea:** Instead of adding random RGB noise (which looks like static), we analyze the covariance of the image's colors.
- **PCA:** We find the "principal directions" of color variation in the image.
- **Semantic Noise:** We add noise *along* these principal directions. This simulates changes in lighting intensity and color temperature (e.g., warmer or cooler light) rather than just random pixel corruption.

### Key Learnings
I love this idea—it keeps the semantic content of the image intact while making the model robust to lighting changes. This subtle but powerful data augmentation was likely a key factor in AlexNet's massive success.

---
*Solved on Deep-ML.*
