## the problem with SFT

SFT updates every weight in the model.
- **Storage Cost:** A 7B model checkpoint is ~14GB. If you fine-tune for 50 different clients, you need $50 \times 14\text{GB} = 700\text{GB}$ of storage.
- **VRAM Cost:** You need to store **Optimiser States** (Adam) for every parameter. For a 7B model, optimiser states alone take ~42GB of VRAM (in float32), pushing total requirements to >80GB.

Hence, we freeze the main weights ($W_0$) and train _only_ a tiny "adapter" circuit that gets added on top.

## linear algebra prerequisite knowledge

To understand LoRA, these linear algebra concepts must be understood.

### matrix rank

The **Rank** of a matrix tells you how much "unique" information it holds.
The rank can be determined by counting the number of non-zero rows in the row echelon form of the matrix.

**Example**

Say we have matrix, $W$.
$$W = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 10 & 20 & 30 \end{bmatrix}$$
The REF of the matrix below:
$$W_{REF} = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \\ \mathbf{0} & \mathbf{0} & \mathbf{0} \end{bmatrix}$$

Hence, $\text{rank(W) = 1}$.


### matrix decomposition 

Any massive matrix can be broken down into two tiny matrices multiplied together, **IF** the massive matrix is Low Rank.

- If $W$ is a $100 \times 100$ matrix (10,000 numbers), but it is Rank 2:
    - We can represent it as $A \times B$.
    - $A$ is $100 \times 2$ (200 numbers).
    - $B$ is $2 \times 100$ (200 numbers).
    - **Total:** We only store 400 numbers instead of 10,000 to represent the exact same information.
- We don't need to learn a massive 10,000-parameter update. We just learn the two tiny matrices $A$ and $B$.


### matrix multiplication

Inner dimensions must match.
$$[d \times \mathbf{r}] \cdot [\mathbf{r} \times k] \rightarrow [d \times k]$$
This is why we can "merge" LoRA back into the main model without changing the model's architecture.


## the math

$$W = W_0 + \Delta W$$
We constrain $\Delta W$ to be low-rank using decomposition:
$$\Delta W = B \times A$$
- **$W_0$:** The Frozen Pre-trained Weights ($d \times k$ matrix).
- **$B$:** The "Up-Projection" matrix ($d \times r$). Initialised to **Zero**.
- **$A$:** The "Down-Projection" matrix ($r \times k$). Initialised to **Random Gaussian**.
- **$r$ (Rank):** The bottleneck dimension (e.g., 8, 16, 64). $r \ll \min(d, k)$.

**Why initialise B to 0?**
- At step 0, $\Delta W = B \times A = 0 \times \text{Random} = 0$.
- **Result:** The model starts _exactly_ as the pre-trained model. We don't break the pre-trained knowledge instantly. _Note: Recent research suggests non-zero initialisation can be robust, but standard practice is still zero B._

## the forward pass

In the forward pass, the update is scaled:

$$h = W_0 x + \frac{\alpha}{r} (BAx)$$

- **$r$ (Rank):** Determines the capacity (complexity) of the adaptation.
- **$\alpha$ (Alpha):** A constant scaling factor (like a "learning rate multiplier").
- **Why use $\frac{\alpha}{r}$?**
    - It makes hyper-parameter tuning robust to changes in Rank ($r$).
    - If you double $r$, you don't need to retrain your Learning Rate because the $\frac{1}{r}$ term automatically scales down the magnitude of the matrix product.
    - **Rule of Thumb:** Set $\alpha = 2 \times r$ (or sometimes $\alpha = r$).

## implementation details

### memory savings

**Scenario:** Llama-3-8B.
- **Full Fine-Tuning:** Needs gradients + optimiser states for 8 billion params.
    - Memory $\approx$ 16GB (Weights) + 16GB (Grads) + 32GB (Optimiser) = **64GB+**.
- **LoRA ($r=16$):**
    - Trainable params $\approx$ 0.1% (e.g., 8 Million).
    - Memory $\approx$ 16GB (Frozen Weights) + 0.05GB (Grads) + 0.1GB (Optimiser) = **~17GB**.
    - **Result:** You can train a 7B model on a single consumer GPU (24GB VRAM).

### inference speed

During training, we keep $W_0$, $A$, and $B$ separate.
During inference, we merge them to get zero latency penalty:
$$W_{merged} = W_0 + \frac{\alpha}{r}BA$$

- **Latency:** Exactly the same as the base model. No extra matrix multiplications.
- **Constraint:** Once merged, you can't easily "unmerge" or switch adapters.


## multi-LoRA serving

Standard LoRA is great because we can **merge** the weights ($W_{new} = W_0 + BA$) to get zero latency. However, we cannot just switch and re-insert adapters as and when we wish. Not on the standard architecture, anyway.

Instead of merging the weights, we keep the base model ($W_0$) frozen and shared. We apply the specific LoRA math ($B \times A$) on-the-fly for each specific request.

**The Architecture**
Imagine a GPU processing a batch of 3 requests.
- **Shared:** The massive Base Model (70GB).
- **Unique:** The tiny LoRA Adapters (10MB each) living in a "Memory Pool."


### the math

We write a custom CUDA kernel that does **batch-specific** multiplication.

<u>Standard Batching:</u>
$$Y = W_0 X$$
_(Everyone gets the same weights)_

<u>Multi-LoRA Batching:</u>
$$Y = W_0 X + \text{AddLoRA}(X, \text{indices})$$

**The Algorithm (Step-by-Step):**

1. **Base Pass:** Compute $Y_{base} = W_0 X$ for the whole batch (Fast, massive matrix mult).
2. **Adapter Pass (The Kernel):**
    - Look at **Row 0** (User A): Fetch `Adapter_Poetry` pointers ($A_1, B_1$). Compute $B_1 A_1 x_0$. Add to Row 0.
    - Look at **Row 1** (User B): Fetch `Adapter_SQL` pointers ($A_2, B_2$). Compute $B_2 A_2 x_1$. Add to Row 1.
    - Look at **Row 2** (User C): No adapter? Do nothing.


### difficulties in implementation
#### memory fragmentation

- **The Contiguous Requirement:** Standard PyTorch tensors must be stored in a **single, unbroken block** of physical RAM (contiguous memory). If you need 10MB, you cannot use two separate 5MB blocks; you need one 10MB hole.
- **Problem:** As variable-sized adapters are loaded and unloaded, the VRAM develops "holes" between allocated blocks. Eventually, you might have 1GB of free space but it is scattered in tiny 1MB gaps. If a new user needs 10MB, you get an OOM (Out of Memory) error despite having 1GB free.

The Solution (Page Tables): We treat VRAM like an Operating System treats RAM.
1. **Pages:** We slice VRAM into fixed-size blocks (e.g., 4KB).
2. **Non-Contiguous Storage:** We split a user's adapter into many pages and scatter them wherever space exists.
3. **The Page Table:** A lookup table that maps **Logical Memory** (Row 0, Row 1...) to **Physical Memory** (Page Address 0x4F, Page Address 0x9A...). The tensor _looks_ contiguous to the math operations, but is physically scattered.
#### kernel launch overhead (solved via SGMV)

**The Bottleneck:** The latency of dispatching a CUDA kernel from the CPU to the GPU (Launch Overhead) is roughly 10μs. If we iterate through a batch of users in Python (`for user in batch: apply_adapter()`), the total dispatch time scales linearly with batch size (N×10μs). For small adapters where computation takes <1μs, the system becomes **CPU-bound**, leaving the GPU idle for >90% of the time.

**The Solution: SGMV (Segmented Gather Matrix-Vector)** We replace the Python loop with a single monolithic CUDA kernel call:
$$Y = W_0 X + \text{SGMV}(X, \mathcal{A}, \mathcal{I})$$

- **A (Adapter Pool):** A pointer to the flattened memory pool containing all active adapter weights.
- **I (Segment Indices):** A metadata tensor indicating which rows belong to which Adapter ID.

**Mechanism:** The SGMV kernel launches thread blocks in parallel. Each thread block:

1. Reads its assigned row index i.
2. **Gathers** the specific weight parameters for that row from A using indices I.
3. Performs the matrix-vector multiplication. This reduces kernel launch overhead from O(N) to O(1).

### summary
|**Feature**|**Single Merged LoRA**|**Multi-LoRA (S-LoRA/vLLM)**|
|---|---|---|
|**Throughput**|High (for 1 task)|High (for mixed tasks)|
|**VRAM Usage**|1 Model|1 Model + Tiny Buffer|
|**Switching Cost**|**High** (Seconds)|**Zero** (Microseconds)|
|**Use Case**|Specialized Chatbot|ChatGPT / TikTok Feed (Many users, many intents)|


## interview questions

#### Why does LoRA work with a low rank of 1 or 2?

Research (specifically [Aghajanyan et al., 2020](https://arxiv.org/abs/2012.13255)) showed a fascinating property of Neural Networks:
> **"As models get larger, the 'intrinsic dimension' of learning a new task gets smaller."**

A pre-trained model ($W_0$) has billions of parameters ($d$). However, optimisation for a specific task (like "Code Generation") doesn't happen randomly in that space. It happens along a very low-dimensional manifold.

**Hypothesis:** $\theta_{task} = \theta_{pretrain} + P \cdot z$
    - $P$: A random projection matrix to a tiny subspace.
    - $z$: A tiny vector (e.g., dimension 100).

**LoRA Connection:** LoRA creates this subspace explicitly using matrices $A$ and $B$. If the task is simple (e.g., sentiment analysis), the intrinsic dimension is extremely low ($r=1$ or $2$).


#### Which modules should I target? Just Q and V?

**The Old Wisdom (Attention Only):** Originally, people only targeted `q_proj` and `v_proj`. Logic: "Attention determines _where_ the model looks. Changing attention changes context processing."

**The New SOTA (Target All Linear):** We now know that **MLPs (Feed Forward Networks)** act as the "Key-Value Memories" or "Knowledge Stores" of the LLM.
- **Attention Layers:** Route information (Syntactic/Contextual).
- **MLP Layers:** Store facts and logic (Semantic/Knowledge).

**Evidence:** When fine-tuning for Coding or Math, you are teaching the model _new logic_, which requires updating the "Knowledge Store" (MLPs). Targeting MLPs increases parameter count but significantly boosts performance.

#### How is LoRA different from Series Adapters (Houlsby)?

**1. Architecture (Parallel vs. Series)**
- **Adapters (Houlsby et al., 2019):** Insert tiny layers _between_ the Transformer blocks.
    - Flow: `Input -> Linear -> Nonlinearity -> Adapter -> Output`
- **LoRA:** Adds a branch _parallel_ to the existing weights.
    - Flow: `Input -> (W0 + BA) -> Output`

**2. Latency (The Dealbreaker)**
- **Adapters:** Because they are inserted sequentially, they increase the depth of the network. Even small adapters introduce **Inference Latency** because the GPU has to wait for the adapter layer to finish before moving to the next block. They cannot be merged easily due to the non-linearity (ReLU) inside them.
- **LoRA:** Because $W_{new} = W_0 + BA$ is a linear operation, we can **merge** weights during inference. The model architecture remains identical to the base model. **Zero Inference Latency.**


## technical implementation of LoRA

### [[loss functions]]

### hyperparameters

| **Hyperparameter**   | **Typical Value** | **Technical Explanation**                                                                                                                                                                                                                        |
| -------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Rank ($r$)**       | 8, 16, 64         | The inner dimension of matrices $A$ and $B$.<br>• **Low ($r=8$):** Good for simple style transfer.<br>• **High ($r=64+$):** Required for learning "New Knowledge" (e.g., a new programming language).                                            |
| **Alpha ($\alpha$)** | $2 \times r$      | The scaling factor. The update is scaled by $\frac{\alpha}{r}$.<br>• **Heuristic:** Set $\alpha = 2r$. This amplifies the LoRA signal.<br>• **Why?** It acts like a specific learning rate for the adapter. High $\alpha$ = Stronger adaptation. |
| **Dropout**          | 0.05 (5%)         | Randomly zeros out 5% of neurons in the adapter during training. Prevents the adapter from overfitting to specific keywords in the training data.                                                                                                |
| **Target Modules**   | `all-linear`      | Which layers get wrappers?<br>• **Old Way:** Just `q_proj`, `v_proj` (Attention).<br>• **New Way:** All linear layers (MLPs + Attention). This yields better results because MLPs store "Knowledge."                                             |
### skeleton code (PyTorch)

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        
        # 1. The Frozen Pre-trained Weight (Simulated here)
        # In a real library, this references the existing model's weight
        self.pretrained_weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        
        # 2. The LoRA Matrices (Trainable)
        # A: Gaussian Initialization (The "Down" projection)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # B: Zero Initialization (The "Up" projection)
        # CRITICAL: We init B to zero so the training starts with NO change to the model.
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # 3. Scaling Factor
        self.scaling = alpha / rank
        
        # 4. Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, In_Features]
        
        # Path 1: The "Frozen" Highway
        # Standard matrix multiplication: x @ W.T
        result_frozen = torch.nn.functional.linear(x, self.pretrained_weight)
        
        # Path 2: The "LoRA" Detour
        # Apply dropout first
        x_dropped = self.dropout(x)
        
        # x @ A.T -> [Batch, Seq_Len, Rank]
        # result @ B.T -> [Batch, Seq_Len, Out_Features]
        result_adapter = (x_dropped @ self.lora_A.T) @ self.lora_B.T
        
        # Scaling
        result_adapter = result_adapter * self.scaling
        
        # Merge
        return result_frozen + result_adapter
```


## references

https://nebius.com/blog/posts/fine-tuning/lora-low-rank-adaptation
https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html
https://medium.com/@amodwrites/a-definitive-guide-to-qlora-fine-tuning-falcon-7b-with-peft-78f500a1f337