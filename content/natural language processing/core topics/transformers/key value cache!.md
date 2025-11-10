the key-value cache greatly accelerates inference!

## the purpose of a k-v cache

### the problem
- transformers autoregressively generate one token at a time— to generate token `t`, the attention mechanism needs the **Query** from token `t` to attend to the **Keys** and **Values** from all the previous tokens.
- when the model generates token 100, it calculates `Q_100`. it then needs `K_1 ... K_100` and `V_1 ... V_100`
- in a naive implementation, the model would have to reprocess the entire input sequence to recompute the same key & value vectors that it had already computed earlier.
- the recomputation for each step $t$ will be $O(t^2)$ leading to an overall generation time complexity of $O(T^3)$ for a sequence of length $T$.

### the solution
- the `K` and `V` vectors for any given token _never change_ once they are computed.
- instead of recomputing them, we can **store (cache) them in GPU memory** after their first calculation.
- when generating token `t`:
	- feed the _single new token_ (`t`) into the model.
	- calculate its _new_ `Q_t`, `K_t`, and `V_t`
	- retrieve the _cached_ `K_1 ... K_{t-1}` and `V_1 ... V_{t-1}` from memory.
	- concatenate them: `K_all = [K_{cache}, K_t]` and `V_all = [V_{cache}, V_t]`.
	- perform attention: `Attention(Q_t, K_all, V_all)`.
	- update the cache with the new `K_t` and `V_t` for the next step.
- this changes the computation at each step `t` from $O(t^2)$ to $O(t)$ (dominated by the attention calculation over `t` items). This reduces the _overall_ generation complexity from $O(T^3)$ to $O(T^2)$, a massive speedup.

## the core mechanism

- a separate K and V cache is required for every self-attention layer in the model.
	- **what is stored**: `[batch_size, num_heads, seq_len, d_head]` tensors for K and V.
	- **where**: stored in VRAM (GPU memory), as it needs to be accessed at every generation step.
- this means that the KV cache will consume a huge amount of memory

$$\text{Cache Size} \approx 2 \times (\text{batch\_size}) \times (\text{seq\_len}) \times (\text{num\_layers}) \times (d_{\text{model}}) \times (\text{precision})$$
- **2:** For storing both K and V.
- $d_{model}$: `num_heads * d_head`.11
- **precision:** 2 bytes for FP16, 4 bytes for FP32.12

**Example: Llama 2 7B at FP16**
- `batch_size` = 1
- `seq_len` (context) = 4096
- `num_layers` = 3213
- `d_model` = 4096
- `precision` = 2 bytes (for FP16)
$$\text{Cache Size} = 2 \times 1 \times 4096 \times 32 \times 4096 \times 2 \approx 2,147,483,648 \text{ bytes} \approx \textbf{2.15 GB}$$
This 2.15 GB is _per sequence_ and is _in addition_ to the 14 GB needed to store the model weights. For a large batch size (e.g., 32), the KV cache would require `32 * 2.15 GB = 68.8 GB` of VRAM, which is more than most high-end GPUs have.


## Recent Advancements in KV Caching

The 2.15 GB per-sequence memory cost of the standard KV cache is a major bottleneck, limiting context length and batch size. Recent advancements focus on two things:

1. **Reducing the size** of the K and V tensors _before_ they are cached.
2. **Managing the cache memory** more efficiently.

### 1. Architectural Changes (GQA & SWA)

These are changes to the model's _design_ to make the KV cache inherently smaller or more manageable.

#### The Problem: Head Count & Context Length

- **Head Count:** In standard Multi-Head Attention (MHA), every Query (Q) head has its own corresponding Key (K) and Value (V) head. The cache size scales _linearly_ with `num_heads`.
- **Context Length:** The cache size scales _linearly_ with `seq_len`, making extremely long contexts (e.g., 100k+ tokens) impractical.

#### The Solution: Grouped-Query & Sliding Window Attention

- **Grouped-Query Attention (GQA)**
    - A compromise between Multi-Head (MHA) and Multi-Query (MQA) attention.
    - **MQA:** Has `N` query heads but only **one** K head and **one** V head that are shared by all Q heads. This drastically cuts cache size but can sometimes reduce model quality.
    - **GQA:** Has `N` query heads, but groups them. For example, 8 Q heads might share a _single_ K/V head. This is the "grouped" part.
    - **Impact:** Models like **Llama 2/3** and **Mistral** use GQA. It achieves almost the same quality as MHA but reduces the K and V cache size by the number of groups (e.g., a 8x reduction), making inference much faster and more memory-efficient.

- **Sliding Window Attention (SWA)**
    
    - Instead of caching _all_ previous tokens from `1 ... t`, the model only caches the **last `W` tokens** (e.g., a window of 4096 or 8192 tokens).
    - The cache for each layer now has a **fixed size** (`[batch_size, num_heads, W, d_head]`).
    - When token `t` is generated, it only attends to tokens `K_{t-W} ... K_t`.
    - **Impact:** This allows models like **Mistral 7B** to handle theoretically infinite sequences while only paying the memory cost for a _fixed_ 8k-token cache. The trade-off is that the model cannot "see" tokens that are further back than the window size.

### 2. Memory Management (PagedAttention)

This is a change to the _inference engine_ (like vLLM) that manages how the cache is stored in VRAM.

#### The Problem: Memory Fragmentation & Waste

- The standard implementation pre-allocates a single, _contiguous_ tensor for the cache: `[batch_size, seq_len, ...]`.
- This is extremely wasteful due to **internal fragmentation**.
- **Example:** A batch of two sequences:
    - Sequence A: 1000 tokens
    - Sequence B: 100 tokens
- To batch them, you must allocate for the _longest_ sequence (1000 tokens) for _both_. Sequence B's cache allocation is 90% empty, wasting massive amounts of VRAM. This makes batching inefficient.

#### The Solution: PagedAttention (vLLM)

- **Core Idea:** Borrows the concept of **virtual memory and paging** from operating systems.
- The KV cache VRAM is divided into small, fixed-size **"blocks"** (like pages).
- A sequence's KV cache is **not stored contiguously**. Instead, it is stored in a non-contiguous set of blocks, allocated as needed.
- A "page table" maps the logical token index to its physical block in VRAM.

- **Key Benefits:**
    - **Near-zero fragmentation:** A 100-token sequence uses _only_ the blocks it needs. A 1000-token sequence uses 10x as many blocks. There is no wasted, pre-allocated space.
    - **Efficient Batching:** Allows for much larger and more dynamic batches, dramatically increasing GPU throughput (requests/second).
    - **Shared Prefixes:** If multiple requests in a batch share a common prefix (e.g., the same system prompt), they can _all point to the same physical blocks_ for that prefix (Copy-on-Write), saving even more memory.

### 3. Compression (Cache Quantization)

This is a change to the _data format_ of the cache itself.

#### The Problem: High Precision is Expensive

- The previous example used FP16 (2 bytes per number), which is already a standard.
- This 2.15 GB cost is still too high.

#### The Solution: KV Cache Quantization

- **Core Idea:** Store the K and V values in the cache at a lower precision, like **INT8** (1 byte), **FP8** (1 byte), or even **INT4/FP4** (0.5 bytes).
- **Mechanism:**
    1. The model computes the K and V vectors in its normal precision (e.g., FP16).
    2. Just before writing to the cache, the numbers are **quantized** (converted) to INT8.
    3. When reading from the cache for the next step, the numbers are **de-quantized** (converted back) to FP16 to be used in the attention calculation.

- **Impact:**
    - **INT8 Quantization:** Instantly **halves** the KV cache memory footprint (e.g., your 2.15 GB becomes ~1.08 GB).
    - **INT4/FP4 Quantization:** Reduces the cache memory by **75%** (to ~0.54 GB).

- This adds a very small computational overhead (for the quant/de-quant step) but the memory savings almost always make it worthwhile, with minimal impact on model accuracy.



## References & Extra Material
vLLM by UC Berkeley — https://blog.vllm.ai/2023/06/20/vllm.html
PagedAttention — https://www.hopsworks.ai/dictionary/pagedattention