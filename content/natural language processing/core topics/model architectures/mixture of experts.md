*Referenced from Stanford CS336 [Lecture](https://www.youtube.com/watch?v=LPv1KfUXLCo) on Mixture of Experts*

MoE allows for scaling of model parameters massively without increasing the compute cost (FLOPs) proportionally.

**The problem with dense models**: In a standard Transformer (like Llama 2 or GPT-3), every single parameter is active for every single token. To make the model "smarter" (more parameters), you must proportionally increase the compute (FLOPs) required for every forward pass.

**The MoE Solution (Conditional Computation):** We decouple parameter count from compute. We can scale a model to **Total Parameters** of 671B (like DeepSeek V3), but only use **Active Parameters** of 37B per token.

## the breakdown

In a standard transformer (dense model), every input token passes through every single neuron in the Feed-Forward Network (FFN) layers. In a MoE transformer, the dense FFN layers are replaced by a **Sparse MoE layer**.

**The Experts**

Instead of one giant FFN, there are $N$ smaller FFNs. These are called **Experts**. These FFN layers do not share weights and they specialise in different patterns (e.g. one expert might be better at maths, another might be better at coding, etc.)

**The Router**

This is a network that determines which tokens are sent to which expert. How to route a token to an expert is one of the big decisions when working with MoEs - the router - is typically a learnable linear projection followed by a non-linearity (Softmax or Sigmoid) to produce probabilities/scores.

## how to choose experts

### token choice routing
- **Mechanism:** Each _token_ chooses its top $k$ experts independently.
- **Pros:** Simple to implement.
- **Cons:** **Load Imbalance**. If the token is "The", and Expert #5 is the "Grammar Expert," Expert #5 gets hammered with millions of "The" tokens, while the "Physics Expert" sits idle. This causes compute bottlenecks (one GPU works, others wait).
### expert choice routing
- **Mechanism:** Each _expert_ selects the top $k$ tokens it wants to process from the batch.
- **Pros:** Guarantees perfect load balancing (each expert takes exactly its capacity).
- **Cons:** Hard to implement for auto-regressive decoding (generation) because you don't have the full batch of future tokens.
### hash routing
- **Mechanism:** Instead of learning a router, use a fixed random hash function to assign tokens to experts.
- **Insight:** Hash Routing outperforms dense models of equivalent inference cost. This proves that **Parameters > Intelligence**. Simply having more "memory slots" (parameters) helps the model, even if the routing logic is dumb. However, learned routing is significantly better.

## the math (token choice routing)

The Gating Network (Router) produces a probability distribution over all $N$ experts. If we computed all of them, the model would be too slow. We need a specific algorithm to select only a few.

The industry standard is **Top-K Gating**.

### softmax top-k algorithm
**Used in:** Switch Transformer, Mixtral 8x7B.

1. The router projects the input token $x$ (dimension $d$) into a score for each expert.
$$H(x) = x \cdot W_g$$
2. We find the indices of the $k$ highest values in $H(x)$.$$\mathcal{T} = \text{TopK}(H(x), k)$$
3. We apply Softmax **only** to the Top-K elements to get the final gating weights. Then, we re-normalise the probabilities for the top-K experts chosen. $$G(x)_i = \frac{e^{H(x)_i}}{\sum_{j \in \mathcal{T}} e^{H(x)_j}} \quad \text{for } i \in \mathcal{T}$$
4. Output $y$ for a token $x$ is the weighted sum of the selected experts. $E_i(x)$ is the output vector from expert $i$. $\mathcal{T}$ is the set of selected "Top-K" experts.$$y = \sum_{i \in \mathcal{T}} G(x)_i \cdot E_i(x)$$
### sigmoid top-k algorithm
**Used in:** [[deepseek v3#^c58f83|DeepSeek V3]]

## problems with MoE

### mode collapse (inequality)

**The Problem**

1. **Initialization:** Weights are random. Expert #1 gets slightly better random weights for common words.
2. **The Loop:** The router notices Expert #1 has lower loss. It sends _more_ tokens to Expert #1. Expert #1 gets _more_ gradient updates. It becomes even smarter. The router sends _even more_ tokens.
3. **The Collapse:** Eventually, Expert #1 processes 100% of tokens. Experts #2–64 process 0%. The trained model essentially becomes a Dense model with a lot of wasted memory.

**The Solution**

We add a penalty to the loss function to punish the router for choosing the same expert too often.$$L_{total} = L_{CrossEntropy} + \alpha \cdot N \sum_{i=1}^N f_i \cdot P_i$$
- $f_i$: The fraction of tokens _actually_ sent to Expert $i$.
- $P_i$: The probability fraction the router _assigned_ to Expert $i$.
- **Intuition:** We want the vector $f$ (actual usage) and vector $P$ (predicted probability) to be uniform. This dot product is minimised when usage is uniform.
- **The Downside:** This loss **fights** the main objective. The model _wants_ to use the Star Expert (it gives better answers!), but the Auxiliary Loss forces it to use the bad experts. The penalty causes the routing weights to be altered and could potentially corrupt the semantic understanding of the router. This degrades performance.

A better solution to this can be found in [[deepseek v3#^c58f83|DeepSeek V3's routing mechanism]].

## deepseek-v3 MoE architecture (covered in CS336)
[[deepseek v3|DeepSeek v3]]

## when to use MoE?

If you are designing a system, use MoE if:
1. **Inference Constraints:** You need the intelligence of a 100B+ model but can only afford the latency/cost of a 10B model.
2. **Massive Scale:** You have enough data (trillions of tokens) to train the experts. MoE overfits easily on small datasets.
3. **Multi-Tasking:** You are training on code, math, and literature simultaneously. MoE separates these domains better than Dense models

