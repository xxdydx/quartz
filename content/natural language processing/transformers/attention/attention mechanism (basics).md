
# introduction
## why attention?
- RNNs compressed the entire sequence into a single vector, which caused long sentences to lose nuance
- attention solves this by allowing model to look back at all tokens dynamically, and allowing it to decide which tokens (out of all the other tokens) are more relevant than others for this specific token's representation

## the math behind attention

$$
\text{Attention}(Q, K, V)
  = \text{softmax}\!\left( \frac{Q K^{\top}}{\sqrt{d_k}} \right) V

$$
- **QKᵀ**: measures similarity between two tokens in the sequence.
- **√dₖ**: scaling factor to stabilise gradients.    
- **softmax**: normalises relevance scores.
- **V**: weighted sum of values (information content).


### what is Q, K, and V?
- **query** — what is this token looking for?
- **key** — what does this token offer to others to find?
- **value** — what does this token contain as information to be shared?

imagine a sentence "the cat sat"

| Role          | Analogy                          | Example in the sentence         |
| ------------- | -------------------------------- | ------------------------------- |
| **Query (Q)** | The question they’re asking      | “Whose action am I part of?”    |
| **Key (K)**   | The label on their name tag      | “I’m a noun”, “I’m a verb”      |
| **Value (V)** | The information they can provide | Their actual meaning or context |
|               |                                  |                                 |

so the token "cat" might ask ("query"):  _who is doing something with me?_  
“the” might say (“key”): _i'm a determiner describing someone._  
“sat” might say (“key”): _i’m the action verb, here’s what’s happening._

the model compares these queries and keys to see which token should "attend" to whom.
when two tokens' Q and K align, they exchange information, which is included in the "Value" representation.


# different types of attention

## multi-head attention

### why is self-head attention limiting?
- in self-attention, every token in a sequence compares its query with every other token's key to produce weights, then takes a weighted average of all the values.
- this single-head version learns one *kind* of relationship pattern across the whole sequence, which is not enough to fully comprehend the semantic meaning of the sequence.
- a single head projects everything into one subspace, which is limiting as it must simultaneously encode: short-range dependencies ("the cat), long-range dependencies ("sat on the mat), and syntactic/semantic cues— too much for a single dot-product vector to capture effectively.

### what is multi-head attention?
- multi-head attention fixes this by giving the model several independent attention mechanisms in parallel.
- each head has its own learned projections.
- each head performs its own self-attention.
- then, the outputs of all heads are concatenated and linearly mixed.

$$
\text{head}_i = \text{softmax}\!\left( \frac{(X W_Q^{(i)}) (X W_K^{(i)})^{\top}}{\sqrt{d_k}} \right) (X W_V^{(i)})
$$
$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h) W_O
$$

### why multiple heads?
each head can learn a different pattern of relationships—

for example,
- one head might focus on nearby words ("the " -> "cat")
- another might focus on long-distance syntax ("sat" -> "on")
- another on punctuation or positional cues

after concatenation, the model fuses all those "views" into a richer representation.

### dimensions
$$
d_k = \frac{d_{model}}{h}
$$
where h = number of heads and $d_k$ is the dimensions per head.


**example values**

| Model            | $d_{\text{model}}$ | $h$ | $d_k$ |
| ---------------- | ------------------ | --- | ----- |
| Transformer-Base | 512                | 8   | 64    |
| BERT-Base        | 768                | 12  | 64    |
| BERT-Large       | 1024               | 16  | 64    |
| GPT-3            | 12288              | 96  | 128   |
| LLaMA-2 7B       | 4096               | 32  | 128   |

### why this ratio matters
- if $d_k$​ is **too small**, each head lacks representational power — it can’t learn rich relations.
- if $d_k$​ is **too large**, fewer heads fit into the same model size, reducing diversity.
- keeping $d_k$​ roughly 64–128 strikes a balance between _per-head expressiveness_ and _number of independent relation channels_.

### complexity and scaling
- time complexity per layer: $O(n^2 \cdot d_{model})$, because pairwise dot products for each head is computed
- splitting into multiple heads doesn’t change the asymptotic cost, since $h \cdot d_k = d_{\text{model}}$​.  
- however, it increases _implementation parallelism_ — each head is smaller, so the operations fit GPU tensor cores more efficiently.

## masked self-attention

### what is it?
- used in the decoder layer (either in decoder-only models like GPT or in the decoder layer of encoder-decoder models like T5, BART)
- same as self-attention, but with a causal mask added to prevent tokens from seeing future positions (especially during the training process)

$$
\text{MaskedAttention(Q,K,V)}=softmax(\frac{​​QK^{⊤​}}{\sqrt{d_k}}+M_{causal}​)
$$
where $M_{ij} = -\infty \text{ if } j>i.$
- when predicting token $i$, the model only looks at tokens $1…i$ (it can’t peek ahead).

example:
when generating "the cat sat",
- to predict "cat", the model only looks at "the"
- to predict "sat", the model only looks at "the cat"
- it never sees the future words

### direction
- the attention direction is thus **unidirectional**.
- this allows the model to process the sentence in a **causal order** and allows for **autoregressive** generation (generating one token at a time)
- once token is generated, the token is **frozen** and thus cannot be updated later on based on future context. e.g. let's say 4 tokens are generated already, and we're trying to predict the 5th token. the 3rd token in the generated sequence cannot be updated as the 5th token is being predicted.
  
## cross-attention

### what is it?
- this attention mechanism is for two different sequences— typically, the encoder output (source sentence) and the decoder hidden states (target sentence).
- for encoder-decoder models, like translation models: T5/Bart
$$

\text{CrossAttention}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}})
  = \text{softmax}\!\left(
      \frac{Q_{\text{dec}} K_{\text{enc}}^{\top}}{\sqrt{d_k}}
    \right)
    V_{\text{enc}}

$$

using an example of a translation of this sentence "the cat sat on the mat" to french. 

encoder output, "the cat sat on the mat"
decoder output so far, "Le chat s’est"

when generating the next token "assis":
- there is **masked self-attention**
	- the current decoder query attends to the previous french tokens generated, to build a context of what is already said in french
	- there's a casual mask, so model cannot see future tokens
- there is **cross-attention**
	- using the above context from what is already said in french, the decoder now attends to the encoder's representation of the english sentence
	- it asks, "which parts of the english sentence are relevant to predict the next token, given whatever i have already generated in french?"
	- the queries come from the french decoder side, the keys and values come from the english sentence
- there is **output projection, token selection, and loop**
    - the decoder takes the result of masked self-attention and cross-attention, applies residual connections and LayerNorm, then passes it through the position-wise feed-forward network to produce the next hidden state hₜ
    - a linear projection maps $hₜ$ into vocabulary logits: $logits = hₜ W_{out} + b$, typically with $W_{out}$ tied to the embedding matrix
    - softmax over logits yields $P(\text{token} | \text{French prefix, English source})$
    - a decoding strategy selects the next token: greedy (argmax), beam search, top-k, or nucleus sampling
    - the chosen token **“assis”** is appended to the French prefix, key–value caches are updated, the positional index advances, and the decoder repeats for the next position













