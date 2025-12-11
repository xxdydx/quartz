### why have feed-forward layers at all?
- attention only mixes information _between_ tokens and does so through mostly linear transformations.
- without non-linearity, the model cannot form new representations, only re-weight and recombine what already exists.
- the FFN introduces non-linear transformations at every token, which allows for abstraction and feature construction.

### the core structure

the feed-forward network in each transformer block is a **two-layer MLP applied independently to every token**.
$$
FFN(x)=W_2​g(W_1​x+b_1​)+b_2​
$$

| Component | Meaning                                       |
| --------- | --------------------------------------------- |
| x         | Input token embedding after attention         |
| W₁        | Expands x to a higher-dimensional space       |
| g         | Non-linearity (ReLU, GELU, or gated variants) |
| W₂        | Projects the expanded vector back down        |
| b₁, b₂    | Biases                                        |

### expanding the dimensional space
$$\text{dim}(W_1 x) = d_{\text{ff}} \quad \text{where typically } d_{\text{ff}} \approx 4 \, d_{\text{model}}$$

​
widening of dimensions is done:
- to explore more feature combinations
- for more expressive non-linear transformations

|Model|$d_{model}$|$d_{ff}$ (approx)|
|---|---|---|
|Transformer-Base|512|2048|
|BERT-Base|768|3072|
|GPT-2 Small|768|3072|
|GPT-3 (175B)|12288|49152|
the tokens are projected into a high-dimensional space to make linearly separable patterns easier to detect, then are projected back into the model's working space.


## residual connection and layer normalisation

each of the two sub-layers (MHA and FFN) in a transformer block has an identical "wrapper" around it.

### the sub-layer wrapper

the full operation for _each_ sub-layer (both attention and FFN) follows this pattern:

$$\text{SubLayerOutput} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

where `x` is the input to the sub-layer, and `SubLayer` is the function itself (e.g., the FFN). this process happens in two steps:

### 1. residual connection (`x + ...`)

- **what it is:** the input to the sub-layer (`x`) is added directly to the output of the sub-layer (`SubLayer(x)`). this is also known as a **skip connection**.
- **why it's needed:**
    - **combats vanishing gradients:** this is the primary benefit. in very deep networks (like a 12-layer bert), gradients can become tiny as they are backpropagated. the residual connection provides a direct, "shortcut" path for the gradient to flow, bypassing the sub-layer's transformations and allowing deep models to train effectively.
    - **preserves information:** it allows the model to easily learn an **identity function** (i.e., just pass the input `x` through unchanged). this means a layer only needs to learn the _change_ (the "residual") it wants to make to the token's representation, rather than learning the entire transformation from scratch.

### 2. layer normalisation (`LayerNorm(...)`)

- **what it is:** a normalisation technique applied _after_ the residual connection, before the data is passed to the next layer.
- **how it works:**
    - it computes the mean and variance **across all features ($d_{model}$) for a single token**.
    - it then normalises that token's vector to have a mean of 0 and a variance of 1.
    - **note:** this is different from batch normalisation, which normalises _across the batch_ for a single feature. layer normalisation is independent of the batch size, which is vital for NLP tasks with varying sequence lengths.
- **why it's needed:**
    - **stabilises training:** it ensures that the inputs to every layer are always in a consistent, predictable distribution (i.e., not too large or too small). this speeds up and stabilises the training process.
    - **reduces covariate shift:** it reduces the problem where the distribution of one layer's activations changes as the parameters of the previous layers are updated during training.

### the full encoder block flow

putting it all together, the full data flow for a single token `x` through one complete encoder block is:

1. **mha sub-layer:**
    - `attn_output = MultiHeadAttention(x)`
    - `add_norm_1 = LayerNorm(x + attn_output)` (residual connection + layernorm)
2. **ffn sub-layer:**
    - `ffn_output = FFN(add_norm_1)`
    - `output = LayerNorm(add_norm_1 + ffn_output)` (residual connection + layernorm)

this final `output` is then passed as the input `x` to the _next_ transformer block in the stack.
