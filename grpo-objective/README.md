# GRPO Objective Function

Implementation of the Group Relative Policy Optimization (GRPO) objective function, as used in models like DeepSeekMath.

## 🧠 Theory & Insights

Reinforcement Learning (RL) math can look intimidating, but at its core, GRPO is about control.

- **The Core Idea:** It's just a loss function designed to prevent the model from "going berserk" when updating weights.
- **PPO Heritage:** It uses the same clipped surrogate trick as PPO to ensure updates are conservative (within `epsilon` bounds).
- **KL Penalty:** Adds a penalty term based on the KL divergence from a reference model, ensuring the new policy doesn't drift too far from a safe baseline.

### Key Learnings
Revisiting these concepts reminded me of my Stanford course days. The equations (likelihood ratios, advantages, KL divergence) are just tools to mathematically operationalize "learn efficiently but don't break what's already working."

---
*Solved on Deep-ML.*
