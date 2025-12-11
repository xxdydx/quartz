This is meant to be a summary of [PPO for LLMs: A Guide for Normal People](https://cameronrwolfe.substack.com/p/ppo-llm)
_Complementary notes to [[proximal policy optimisation]]_

## Mapping RL Concepts to Transformers

To apply PPO, we must rigorously define the Language Generation process as a Markov Decision Process (MDP) tuple $\langle S, A, R, P \rangle$.

- **State Space ($S$):** The full context window.
    - $s_t = \{x, y_1, y_2, ..., y_{t-1}\}$.
    - This includes the original prompt ($x$) plus all tokens generated up to time $t$.
- **Action Space ($A$):** The Token Vocabulary ($V$).
    - Dimension is defined by vocab size (e.g., ~50k for GPT-2, ~32k for LLaMA).
    - Action $a_t$ corresponds to selecting the next token $y_t$.
- **Transition Function ($P$):** **Deterministic.**
    - Unlike robotics, the environment does not change stochastically.
    - $s_{t+1} = \text{concat}(s_t, a_t)$. We simply append the chosen token to the context.
- **Episode Horizon ($T$):** The maximum generation length (e.g., 1024 tokens) or the `<EOS>` token.

## The 4-Model Architecture

Standard PPO uses one Actor and one Critic. PPO for LLMs requires keeping **four** distinct models (or sets of weights) in memory during training.

| **Model**                      | **Role**                                                   | **Trainable?** | **Memory Impact**                   |
| ------------------------------ | ---------------------------------------------------------- | -------------- | ----------------------------------- |
| **1. Actor ($\pi_\theta$)**    | The active LLM being optimized. Generates the response.    | **Yes**        | High (Gradients + Optimizer States) |
| **2. Critic ($V_\phi$)**       | Estimates expected return (Scalar output).                 | **Yes**        | High (Gradients + Optimizer States) |
| **3. Reference ($\pi_{ref}$)** | Frozen copy of initial SFT model. Used for KL calculation. | **No**         | Moderate (Inference weights only)   |
| **4. Reward Model ($r$)**      | Scores the final sequence.                                 | **No**         | Moderate (Inference weights only)   |

### The "Hydra" Head Variation

To save memory, the **Actor** and **Critic** often share the same backbone (Transformer layers) and only diverge at the final layer.

- **Actor Head:** Linear layer projecting to `vocab_size` (Logits).
- **Critic Head:** Linear layer projecting to `scalar` (Value).
- _Risk:_ Updates to the backbone to improve the Value function might degrade the Policy (and vice versa).

## 3. The Computational Lifecycle (Rollout vs. Update)

PPO for LLMs is split into two distinct computational phases with different bottlenecks.

### Phase 1: The Rollout (Inference Bottleneck)

The model must interact with the "environment" to generate data.

1. **Sampling:** We sample a batch of prompts $x$.
2. **Generation:** The Actor generates full responses autoregressively (token by token).
3. **Log-Prob Calculation:**
    - We forward pass the **Actor** to store log-probs of generated tokens.
    - We forward pass the **Reference Model** to store baseline log-probs (for KL).
4. **Scoring:**
    - We forward pass the **Reward Model** on the full text $(x, y)$.
    - We forward pass the **Critic** to estimate values for every step (for GAE).

> **Note:** This phase is slow because autoregressive generation cannot be parallelized ($O(T)$ sequential steps).

### Phase 2: The Update (Memory Bottleneck)

Once data is collected, we treat it as a static batch (like Supervised Learning) for $K$ epochs.

1. **Graph Construction:** We compute the composite loss ($L^{CLIP} + L^{VF} + S$).
2. **Backpropagation:** Gradients flow through the Actor and Critic.
3. **Constraint:** We do _not_ backpropagate through the Reference or Reward models (they are frozen).

## 4. Parameter Efficient Fine-Tuning (PEFT)

Because loading 4 models (plus optimizer states for 2 of them) requires massive VRAM, full fine-tuning is rarely done for 7B+ parameter models.

**LoRA (Low-Rank Adaptation) for PPO:**

Instead of updating all weights $\theta$, we freeze the main transformer and inject trainable rank-decomposition matrices.

- **Memory Win:** We only store gradients/optimizer states for the LoRA adapters (<1% of parameters).
- **Reference Model Optimization:** If using LoRA, the "Reference Model" is simply the _frozen base model_ without the active LoRA adapters enabled. We don't need a separate copy in VRAM; we just disable the LoRA outcome to get the reference logits.

## 5. Distributed Training Nuances

- **Collocation:** Usually, the Actor and Critic are placed on the same GPU (or sharded across the same group) because they require gradients.
- **Offloading:** The Reward Model and Reference Model can be offloaded to CPU (or different GPUs) and queried via inference API, since they don't participate in the backward pass.


## How GPT-o1 was trained

### abstract

In standard RLHF, the model was trained to _imitate_ human preferences. In **o1**, the model is trained to **get the right answer**, even if it takes a long time. The "behaviour" being reinforced is not the final text, but the **hidden "Chain of Thought" (CoT)** that precedes it.

### what data was used?

OpenAI has not released the raw dataset, but based on the system architecture and open-source replications, the data consists of three specific categories that differ from standard ChatGPT training:
- **The Data:** Tens of thousands of high-quality **(Prompt, Reasoning Trace, Final Answer)** triplets.
- **Source:** Likely created by human experts (PhDs in Math/Physics).
- **Goal:** This "bootstraps" the model. It learns that when asked a hard question, it should output a `<reasoning>` block before the `<answer>`.

![[Pasted image 20251211182852.png]]

During training, the model generates its _own_ data.
- It is given a prompt (e.g., a hard integral).
- It generates 10 different "thoughts" (reasoning paths).
- 3 paths lead to the wrong answer. 7 paths lead to the correct answer.
- **The Data:** The _failed_ paths become negative training examples (punishment), and the _successful_ paths become positive examples (reinforcement).

### the process

**Phase 1: Supervised Fine-Tuning (The Setup)**

The base model (likely GPT-4o) is fine-tuned on the "Cold Start" data. As a result, the model stops jumping to conclusions. It learns to say _"Let's analyse this step by step..."_ and produce a long sequence of intermediate tokens before the final answer.

**Phase 2: Outcome-Based Reinforcement Learning**

This is the "Secret Sauce." It is different from standard RLHF (PPO) in one major way: **The Reward Function.**
- **Standard RLHF:** A human looks at the output and clicks "Thumbs Up." (Noisy, subjective, expensive).
- **o1 RL:** The system uses a **Programmatic Verifier**.
    1. **Prompt:** "Solve for x..."
    2. **Action:** The model generates a long chain of thought.
    3. **Check:** A compiler runs the code, or a math engine checks the final number.
    4. **Reward:**
        - If **Correct**: The _entire_ reasoning chain receives a positive reward ($+1$).
        - If **Incorrect**: The chain receives a negative reward ($-1$).

**Phase 3: Learning to "Backtrack" (The Emergent Behaviour)**

Because the RL algorithm (PPO or similar) wants to maximise the reward, the model learns that **verifying its own work** increases the success rate.
- It learns patterns like: _"Wait, that calculation looks too simple. Let me double-check."_
- Because in previous training runs, "double-checking" trajectories led to correct answers more often than "rushing" trajectories.
- **Result:** The model "learns to think" not because it is conscious, but because "thinking" (generating more tokens to process information) is the strategy that maximises the reward.