Lessons from how leading social media companies design and train models for semantic search.

## accuracy vs. latency
In modern search systems, there's an inherent trade-off.
- Accuracy: Larger models (more parameters, deeper layers) capture complex semantic nuances.
- Latency: We must score thousands of documents in milliseconds. Large models are too slow for real-time inference.

## knowledge distillation
[[natural language processing/knowledge distillation]]

To solve the trade-off, these companies use a **Teacher-Student Architecture**. The student (worse) model tries to mimic the teacher (better) model.
- **Teacher Model (`XLarge`):**    
    - **Structure:** 48 Layers, Hidden Dimension $d_{model} = 1536$.
    - **Role:** The "Oracle." It is computationally expensive and runs **offline**. It trains on massive, noisy datasets (User Clicks) to learn the "truth."
- **Student Model (`Large`):**
    - **Structure:** 24 Layers, Hidden Dimension $d_{model} = 1024$.
    - **Role:** The "inference engine." It runs **online**. It is trained to approximate the Teacher's function $f_T(x)$ using a specialised loss function.

### the maths

Standard training uses **Hard Labels** (Ground Truth).
$$y \in \{0, 1\}$$
If a user clicks a video, $y=1$. If not, $y=0$. This is noisy; a user might skip a relevant video just because they were busy.

Distillation uses **Soft Targets** (logits from the Teacher).
$$z_T = f_{Teacher}(x)$$
The Teacher might output $0.85$ for a clicked video and $0.15$ for a skipped one. The Student minimizes the **KL-Divergence** (difference in probability distributions) between its output $z_S$ and the Teacher's output $z_T$.
$$\mathcal{L}_{distill} = \text{KL}( \sigma(z_T / \tau) || \sigma(z_S / \tau) )$$
_(Where $\tau$ is a temperature parameter to smooth the distribution)._

**Engineering Gain:**
By decoupling training, the `XLarge` Teacher handles the heavy lifting of processing billions of Click-Through Rate (CTR) logs. The `Large` Student effectively "downloads" this knowledge, saving **73% of downstream training costs** compared to training the Student from scratch.


## pre-training objectives

Before the model can rank search results, it must understand the underlying structure of language. This is achieved via **Self-Supervised Learning**.

In this case scenario, a multi-task learning approach combining two objectives is utilised.

1. **Masked Language Modelling (MLM - BERT Style):**
    - **Concept:** Contextual Reconstruction.
    - **Input:** `The quick [MASK] fox.`
    - **Target:** Predict `brown`.
    - **Intuition:** Forces the model to understand bidirectional context.
2. **Replaced Token Detection (RTD - ELECTRA Style):**
    - **Concept:** Discriminative Analysis.
    - **Mechanism:** A small "generator" network creates plausible fake tokens.
    - **Input:** `The quick red fox.` (Where "red" is a fake generated to replace "brown").
    - **Task:** The Discriminator must classify every token as `REAL` or `FAKE`.
    - **Advantage:** **Sample Efficiency.** MLM only learns from the 15% masked tokens. RTD learns from _every_ token in the sequence.

### catastrophic forgetting

When we move from Stage 1 (General Web Data) to Stage 2 (domain data) during training, the model adapts to the domain data but might possibly forget standard English grammar. This is **Catastrophic Forgetting**.

To solve this, during Stage 2 training, we mix in a percentage of Stage 1 data. This "Replay" strategy constrains the optimisation landscape, ensuring the model improves on the domain task without degrading on the general language task.

## ranking architectures

### twin-tower vs cross-encoder

- **Twin-Tower (Bi-Encoder):** Processes Query and Doc separately. Fast but shallow. The interaction only happens at the very end (Dot Product).
- **Cross-Encoder (Interaction):** Concatenates Query and Doc: `[CLS] Query [SEP] Doc`. The self-attention mechanism allows every Query token to "attend" to every Doc token. **This is much more accurate but computationally heavier.**

### QQ/QD cross-encoders

To maximise accuracy, we use a Cross-Encoder with a specialised topology called **QQQD**.

**The Inputs:**
The model accepts **two** parallel inputs sharing the **same** BERT backbone weights:
1. **Matching Branch ($QQ$):** Input is `[CLS] Q [SEP] Q`.
2. **Semantic Branch ($QD$):** Input is `[CLS] Q [SEP] D`.

$$Score_{final} = \alpha \cdot \text{Net}(QQ) + \beta \cdot \text{Net}(QD)$$

The gradients from the $QQ$ branch flow back into the shared backbone, forcing the encoder to learn extremely sharp representations of the Query tokens.

**Why do this?**

Using **Integrated Gradients** (an explainability technique), engineers observed that without the $QQ$ branch, the model often ignored the Query and focused solely on the Document content. The $QQ$ branch acts as a **Regulariser**, forcing the model to "pay attention" to the user's intent.

**Inference Optimisation:**
During online serving, we **drop the $QQ$ branch**. We only run the $QD$ branch. The model retains the "Query Awareness" learned during training but incurs zero additional inference cost.

### BERT-ESIM

Standard BERT uses the `[CLS]` token for classification. We enhance this with **ESIM (Enhanced Sequential Inference Model)** operations to explicitly capture alignment.

Given the encoded Query vector $\mathbf{q}$ and Document vector $\mathbf{d}$:
1. **Difference:** $|\mathbf{q} - \mathbf{d}|$ (Captures mismatching information, or how different they are).
2. **Product:** $\mathbf{q} \odot \mathbf{d}$ (Captures matching information, or how similar they are).
3. **Concatenation:** $[\mathbf{q}; \mathbf{d}; |\mathbf{q} - \mathbf{d}|; \mathbf{q} \odot \mathbf{d}]$.

This explicit modelling of "what matches" and "what doesn't" significantly outperforms standard `[CLS]` pooling ()


## data engineering

### listwise learning (loss function)

**Pairwise Approach (Legacy):**
- Input: Pairs `(Query, Doc+)` and `(Query, Doc-)`.
- Inefficiency: `Doc+` is re-encoded every time it is paired with a different negative sample.

**Listwise Approach (Current):**

- Input: A list `(Query, [Doc1, Doc2, Doc3, Doc4])`.
- **Mechanism:** The Query is encoded once. Each Doc is encoded once. The Loss Function (e.g., Softmax Cross-Entropy over the list) optimizes the _entire ranking order_ simultaneously.
- **Gain:** **30% Training Speedup** due to reduced redundant computation.

### signal supplementation

Many videos have no text (captions/titles). A text-based model fails here. We "inject" text signals:

1. **CQ (Click Query):**
    - If users searching for "funny cat" click Video X, we assign "funny cat" as a hidden text feature for Video X.
    - **Smoothing:** This signal is _too_ strong (circular logic). If the model relies too heavily on CQ, we artificially cap the score or smooth the label to prevent overfitting.

2. **OCR & ASR:**
    - Optical Character Recognition (Pixel-to-Text) and Automatic Speech Recognition (Audio-to-Text).
    - **Summarisation:** Raw ASR is too long. We use TF-IDF or a lightweight BERT-Sum model to extract the top 40 keywords (e.g., "recipe," "pasta," "boil") to feed into the relevance model.

## advanced training stability

### layerwise learning rate decay

Deep networks are hierarchical.
- **Bottom Layers:** Learn syntax (universal English).
- **Top Layers:** Learn semantics (e.g. domain knowledge or ranking logic).

We do not want to aggressively update the bottom layers, or the model will lose its language capabilities, resulting in catastrophic forgetting.
$$LR_{layer} = LR_{base} \times (\lambda)^{N - layer}$$
Where $\lambda$ (decay factor) is typically 0.9.
- Top Layer ($N$): Gets 100% of the Learning Rate.
- Bottom Layer (0): Gets a tiny fraction.
    This ensures **Training Stability** for massive models like the 48-layer XLarge Teacher.

### query augmentation

To prevent the model from memorising exact strings, we use **Data Augmentation** on the Query.
In the $QQ$ branch, we input `[CLS] Q [SEP] Q'`, where $Q'$ is:
- Back-translated (English $\to$ French $\to$ English).
- Rewritten by a generative model.
- Noisy (random token drops).
This forces the model to learn the **semantic intent** of the query, not just the token sequence.