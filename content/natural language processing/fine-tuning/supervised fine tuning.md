## 1. The High-Level Concept

What is it?

SFT is the process of taking a pre-trained "Base Model" (which predicts the next token from internet data) and forcing it to follow specific user instructions. It bridges the gap between "Autocomplete" and "Assistant."

**Where it fits:**
1. **Pre-Training:** Learn language (Predict next token on 10T tokens). Result: Base Model.
2. **SFT:** Learn to follow instructions (Predict next token on 100k high-quality pairs). Result: Chat Model.
3. **RLHF (PPO/DPO):** Learn to align with preferences (Maximise reward). Result: Aligned Model.

---

## 2. The Objective Function (The Math)

This is the most common technical screen question.

The Loss: Standard Cross-Entropy Loss.

$$L = - \sum_{t=1}^{T} \log P(x_t | x_{<t})$$

The Critical "Interview Twist": Loss Masking

In SFT, we do not calculate loss on the entire sequence. We only calculate loss on the Assistant's Response.

- **Why?** We don't want the model to "learn" the user prompt. We only want it to learn how to _respond_ to the prompt.
    
- **Implementation:**
    
    - Create a `labels` tensor identical to `input_ids`.
        
    - Set the indices corresponding to the **Prompt** (User instruction) to `-100`.
        
    - PyTorch's `CrossEntropyLoss` ignores indices with `-100` by default.
        

**Deep Dive: Prompt Loss Weighting (PLW)**

- _Question:_ "Should we ever train on the prompt?"
    
- _Answer:_ Usually **No**. Research shows that non-zero PLW (e.g., 0.1) creates a "quadratic negative impact" on performance for short-response tasks. It confuses the model's distribution. Standard practice is strict masking (PLW = 0).
    

---

## 3. Data Engineering (The "Real Work")

### A. Data Format

The input is not just text; it is a structured prompt.

- **Chat Templates:** You must use a consistent template (e.g., ChatML).
    
    ```
    <|im_start|>user
    Hello<|im_end|>
    <|im_start|>assistant
    Hi there!<|im_end|>
    ```
    
- **The EOS Token Trap:** If you forget to add the `<EOS>` token at the end of the training sample, the model will never stop generating. It will ramble forever during inference.
    

### B. Efficiency: Packing (Sequence Packing)

**Problem:** Instructions vary wildly in length (seq length 100 vs 2048).

- **Naive Approach (Padding):** Pad every sequence to 2048. Result: 90% of your GPU compute is wasted calculating attention on zeros.
    
- **SFT Solution (Packing):** Concatenate multiple short examples into one long 2048-token sequence.
    
    - _Sample 1:_ [User: Hi, Asst: Hello] (Length 20)
        
    - _Sample 2:_ [User: Code?, Asst: Print(x)] (Length 50)
        
    - _Packed Input:_ [Sample 1] [EOS] [Sample 2] [EOS] ...
        
    - **Attention Masking:** You must use "Block Diagonal Attention Masking" so Sample 1 cannot "see" Sample 2, even though they are in the same tensor.
        
    - **Impact:** 10x-20x speedup in training throughput.
        

---

## 4. Training Mechanics (Hyperparameters)

### A. Learning Rate

- **Pre-Training:** High LR (e.g., `1e-3` or `1e-4`).
    
- **SFT:** Low LR (e.g., `1e-5` or `2e-5`).
    
- **Reason:** The model already knows English. We are doing "surgery," not "construction." A high LR will cause "Catastrophic Forgetting" (lobotomizing the model).
    

### B. Batch Size

- **Small Batch Size (e.g., 1-8):** Works surprisingly well for SFT. It introduces noise that can help regularization.
    
- **Large Batch Size (e.g., 128+):** Required for stability, but often achieved via **Gradient Accumulation** to save VRAM.
    

### C. Epochs

- **Standard:** 1-3 Epochs.
    
- **Risk:** SFT models overfit _very_ quickly. If you train for 10 epochs, the model will just memorize the answers and lose its ability to generalize.
    

---

## 5. PEFT: LoRA vs. Full Fine-Tuning

You will be asked when to use which.

|**Feature**|**Full Fine-Tuning**|**LoRA (Low-Rank Adaptation)**|**QLoRA**|
|---|---|---|---|
|**Weights Updated**|100% (7B params)|< 1% (Adapter params only)|< 1% (Adapters + Quantized Base)|
|**VRAM Usage**|Massive (needs 80GB+ for 7B)|Low (can fit on consumer GPU)|Lowest (fits 7B on 12GB GPU)|
|**Performance**|The "Gold Standard"|98-99% of Full FT performance|Slightly lower than LoRA due to quantization noise|
|**Use Case**|Foundation Model Training|Specialized Tasks / Rapid Prototyping|Hobbyist / Extreme Resource Constraints|

LoRA Technical Detail:

$$W_{new} = W_{frozen} + (A \times B)$$

- We freeze the big matrix $W$.
    
- We learn two tiny matrices $A$ and $B$.
    
- This makes checkpointing instant (you only save the 100MB adapter, not the 14GB model).
    

---

## 6. Common Interview "Gotchas"

**1. "Why does my model refuse to stop generating?"**

- **Cause:** You masked the EOS token in the loss calculation, or your training data didn't have EOS tokens. The model never learned _when_ to stop.
    

**2. "Why is the loss decreasing but the model getting dumber?"**

- **Cause:** Overfitting or Catastrophic Forgetting. The model is memorizing the SFT data but forgetting general logic.
    
- **Fix:** Early stopping, reduce learning rate, or use LoRA (which naturally prevents forgetting by freezing the base model).
    

**3. "How do you handle multi-turn chat in SFT?"**

- **Method:** Concatenate the entire history.
    
    - _Input:_ [User: A] [Asst: B] [User: C]
        
    - _Target:_ [Asst: D]
        
- **Loss Masking:** You must calculate loss **ONLY** on [Asst: B] and [Asst: D]. You must mask out [User: A] and [User: C]. If you train on user turns, the model will learn to impersonate the user.
    

Efficient Training on a Single GPU

Why relevant: This video from Andrej Karpathy (though general) touches on the specific engineering constraints of training effectively on limited hardware, which directly applies to the efficiency tradeoffs (Packing, LoRA) discussed above.