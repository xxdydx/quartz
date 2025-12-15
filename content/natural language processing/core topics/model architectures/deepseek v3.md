## specifications
- **Total Parameters:** **671 Billion**
- **Active Parameters (per token):** **37 Billion** (Only ~5.5% active)
- **Training Data:** **14.8 Trillion** tokens (diverse, multilingual, code, math).
- **Training Cost:** **2.788M H800 GPU Hours** (Extremely low compared to Llama 3 405B).
- **Context Window:** **128K** tokens.
- **Architecture Dimensions:**
    - **Layers:** 61
    - **Hidden Dimension ($d_{model}$):** 7168
    - **Attention Heads:** 128
    - **Head Dimension:** 128

## architecture

The model uses the "Fine-Grained + Shared" architecture with specific tuning.
### expert configuration
- **Total Experts:** **257** per layer.
    - **Routed Experts:** **256** (Fine-grained, selectable).
    - **Shared Experts:** **1** (Always active, handles common knowledge).
- **Active Experts ($K$):** **8** routed experts + **1** shared expert = **9** total active per token.
- **Expert Size:** Each expert is small (FFN intermediate dimension is lower than standard).
    - _Standard Dense FFN Inter-dim:_ $4 \times 7168 \approx 28k$.
    - _DeepSeek Routed Expert Inter-dim:_ **2,048**. (Slicing the FFN into ~14 slices).

**Analysis**

- Old models like Mixtral had 8 massive experts, and these experts might possibly be combing different areas of expertise. Say an expert is proficient in math, physics, and chemistry. If the token is just "1+1", you are wasting compute activating the Physics neurons.
- Deepseek has 256 experts— the token can activate multiple experts as and when needed. For example, if the sentence is "The derivatives in financial markets," the model can activate **Expert 4 (Calculus)** + **Expert 99 (Economics)** + **Expert 200 (History)**.

### routing mechanism: sigmoid + bias

^c58f83
#### Expert Selection

DeepSeek V3 changes the math slightly to allow independent expert selection.
$$G(x)_i = \frac{\sigma(x \cdot u_i)}{\sum_{j \in \mathcal{T}} \sigma(x \cdot u_j)}$$
- **Sigmoid ($\sigma$):** Instead of Softmax (where experts compete), each expert gets an independent score between 0 and 1.
- **Selection:** We still pick the Top-K scores.
- **Normalization:** We still divide by the sum of the selected scores to ensure stability.

**Rationale**

- Let's say we've 4 experts: **Expert A:** $\sigma(0.0) = \mathbf{0.50}$, **Expert B:** $\sigma(6.0) \approx \mathbf{0.99}$, **Expert C:** $\sigma(-3.0) \approx \mathbf{0.05}$, **Expert D:** $\sigma(2.0) \approx \mathbf{0.88}$
- If Expert v is super strong ($0.99$), it would push Expert D down to near zero (e.g., $0.01$), even though Expert D was actually quite relevant.
- **DeepSeek's Sigmoid approach** allows multiple experts to have high scores simultaneously (e.g., 0.99 and 0.88), acknowledging that a token might truly need two strong experts.

#### Load Balancing of Experts

DeepSeek separates the "Semantic Match" from the "Load Balancing" using a standalone bias term ($b_i$).
$$Logit_i = (x \cdot u_i) + b_i$$
- **Mechanism:** - The **Semantic Score** $(x \cdot u_i)$ remains pure. The router still calculates a high match for "def" $\leftrightarrow$ "Python". If Expert $i$ is overloaded, the system decreases $b_i$. The expert becomes "less attractive" to the router.
- The final score drops, and the token might go to the 2nd best expert. **Crucially, the router weights ($u_i$) are not touched.**
- **Why it's better:** It changes the _threshold_ for selection without changing the _direction_ of the expert's embedding ($u_i$). The expert retains its semantic meaning; it just becomes "more expensive" to call.



### node-limited routing

<u>Hardware layout of an AI training cluster</u>
- Node: A single physical server containing 8 GPUs (e.g. NVIDIA H800s)
	- Inside this node, the GPUs communicate with one another using NVLink (ultra fast speeds).
- Cluster: Thousands of nodes connected by InfiniBand cables (not so fast)
	- Inter-node communication is significantly slower and has higher latency compared to intra-node communication.

**Constraint:** The router forces the model to select at most **4 nodes** (out of the total cluster) for any given token's dispatch.

**Reasoning:** In a cluster of 16,000 GPUs, crossing the network cables between server racks (Inter-Node) is slow. Staying within the same server (Intra-Node) is fast. This constraint minimizes network traffic.



## multi-head latent attention

Designed to fix the **KV Cache memory bottleneck** for long-context inference.

### problem
In standard attention (like Llama 2), for every token $t$, a Key vector ($k_t$) and Value vector ($v_t$) is created and stored for **every attention head**.

- **Input:** Hidden state $h_t$ (dimension $d_{model} = 7168$).
- Generation:$$k_t = h_t W_K \quad \text{and} \quad v_t = h_t W_V$$
- Storage Cost: Storing $k_t$ and $v_t$ for all $n_h$ heads.$$\text{Cache Size} = 2 \times n_{heads} \times d_{head} \times \text{Layers} \times \text{Context}$$For DeepSeek V3, this would be $2 \times 128 \times 128 = 32,768$ floats per token. This is massive.

### the solution (low-rank compression)
DeepSeek replaces the huge $K$ and $V$ matrices with a **Compressed Latent Vector ($c_{KV}$)**.

#### Step 1: Down-Projection (Compression)

Instead of generating 128 heads worth of keys/values directly, we project the input $h_t$ down to a tiny bottleneck vector.

$$c_{KV} = h_t W_{DKV}$$

- **$c_{KV}$:** The Compressed Latent Vector.
- **Dimension:** $d_c = 512$ (DeepSeek V3 spec).
- **$W_{DKV}$:** The Down-Projection Matrix.

**Storage Savings:** instead of storing $32,768$ numbers per token, we only store **512** numbers ($c_{KV}$).

#### Step 2: Up-Projection (Decompression)

When the model needs to calculate attention scores, it mathematically "unzips" this vector to generate the full keys and values for all 128 heads.

$$k_{content} = c_{KV} W_{UK}$$

$$v_{content} = c_{KV} W_{UV}$$

- **$W_{UK}, W_{UV}$:** Up-Projection Matrices that expand 512 dims back to the full $n_h \times d_h$ dimension.

### the "decoupled" rope strategy

The Issue: Rotary Positional Embeddings (RoPE) can't be applied to the compressed latent vector $c_{KV}$ because RoPE requires the full head dimension to rotate effectively.

The Fix: DeepSeek splits the Key/Query vectors into two parts: "Content" and "RoPE".

- **Content Part:** Compressed heavily (as seen above).
- RoPE Part: Kept uncompressed but very small ($d_{rope} = 64$).$$k_{final} = [k_{content}, k_{rope}]$$
- The tiny $k_{rope}$ is cached separately.
- So we get memory savings of compression + positional accuracy of high-dimension RoPE.

## multi-token prediction (mtp)

### how it works
DeepSeek V3 uses MTP for 2 purposes
- This improves the quality of the model during training, by forcing it to plan ahead.
- This also accelerates the speed of the model during inference via speculative decoding.

In a standard LLM (like Llama or GPT-4), the training objective is simple: given a sequence of tokens $t_1, t_2, ... t_t$, predict $t_{t+1}$. 
- However, this "greedy" local focus can prevent the model from learning long-range dependencies or to "plan" the structure of a sentence before generation starts.

DeepSeek V3, however, predicts not just $t_{t+1}$, but also $t_{t+2}$, $t_{t+3}$, and so on (up to a depth $D$).
- The model can understand how the current token influences the immediate future, leading to denser training signals and better representation learning.

### implementation

Many previous attempts at multi-token prediction used **independent/parallel heads** (predicting $t+1$ and $t+2$ simultaneously from the same hidden state).

However, DeepSeek-V3 uses a **Sequential MTP** architecture.

DeepSeek-V3 attaches additional "MTP Modules" to the main model. They are lightweight Transformer blocks.
1. **Main Model:** Processes input and outputs a hidden state $h_t$ to predict $t_{t+1}$.
2. **MTP Module 1:** Takes the hidden state $h_t$ (from the main model) and the embedding of the _actual_ next token $t_{t+1}$. It processes them through a Transformer layer to output a new hidden state $h^1_t$, which is used to predict **$t_{t+2}$**.
3. **MTP Module 2 (and so on):** Takes $h^1_t$ and the embedding of $t_{t+2}$ to predict **$t_{t+3}$**.

At inference time, the MTP module is usually **discarded**. It is a training scaffold that makes the main weights smarter. However, it _can_ be used for [[decoding strategies|Speculative Decoding]] to speed up generation.

## hardware-level details

### fp8 mixed precision training

To train a 671B model cheaply, they used FP8 (8-bit floating point), but with a twist to prevent "loss of precision."

- **Standard FP8:** Quantizes the whole tensor. If one number is huge (outlier), it skews the scale, and small numbers become zero.
    
- **DeepSeek's Fine-Grained Quantization:**
    - **Activations:** Quantized on **1x128** tiles (per token).
    - **Weights:** Quantized on **128x128** blocks.
    - This isolates outliers. A single large number only "ruins" its own little 128-block, not the whole matrix.

### dual-pipe scheduling

Addressing the "Pipeline Bubble" (the idle time when GPUs wait for data from other GPUs).

- **Standard:** Forward Pass $\to$ Wait $\to$ Backward Pass.
- **Dual-Pipe:** DeepSeek overlaps the computation and communication.
    - While one chunk of the model is computing the **Forward** pass of batch A, it is simultaneously receiving the **Backward** gradients of batch B.
    - **Result:** Near-zero communication overhead. The network cables are busy exactly when the GPU cores are busy.
