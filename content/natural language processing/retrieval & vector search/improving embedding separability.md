in high dimensional embedding spaces, distinct inputs can map to very similar vectors. this can cause *semantic overlap*.

to counter this, we can try a couple of methods to push similar vectors away in an embedding space.

## hard negative sampling

- introduces negatives that are _semantically close_, but not relevant
- forces the model to learn more subtle distinctions

example:
positive: "the capital of france is paris",
hard negative: "the population of france is 67 million"

mathematical form:
```python
L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```
the loss function minimises distance between anchor and positive while pushing negatives away by at least `margin`.

### in-batch negatives

instead of finding one specific hard negative, this method uses _every other sample_ in the batch as a negative.

- **how it works:** in a batch of 128, for one `(anchor, positive)` pair, you use the other 126 samples as negatives.
- **improvements:**
    - **Efficiency:** Massively faster, no complex sampling required.
    - **Automatic Hardness:** In a large batch, some of the 126 negatives will be "hard" by pure chance.
- **Used In:** This is the standard for modern contrastive learning (e.g., in models like CLIP) and is the foundation for the **InfoNCE Loss**.

### offline-hard negative mining (production technique)

this is a key technique for production search systems (like in the **DPR paper**).

1. **Train v1:** Train a model using simple `in-batch` negatives.
2. **Run Inference:** Use this v1 model to find the _top 100 "wrong" answers_ for your training queries.
3. **Store:** Save these "wrong" answers. They are now your curated **hard negatives**.
4. **Re-Train v2:** Re-train the model, but this time, explicitly feed it these pre-computed hard negatives to force it to learn from its own specific mistakes.


## larger margin in contrastive / triplet loss

- increases the **minimum distance** between positive and negative pairs
- encourages the model to create **wider gaps** between dissimilar samples

referring to the mathematical form of triplet loss above,
where `margin` controls how far apart negatives must be from the anchor.
- small margin → overlap between classes
- large margin → clearer separation, but harder optimisation

**practical range:**  
`margin = 0.1 – 0.5` (typical for cosine distance)


## regularisation and normalisation

regularisation shapes the **geometry** of embedding space.

methods:
- **L2 normalisation**: keep all embeddings on the unit hypersphere
- **weight decay**: limits overfitting of projection layers
- **dropout or noise**: improves robustness
- **orthogonal regularisation**: encourages less correlated dimensions

these constraints keep embeddings evenly distributed and reduce collapse (where many points overlap).

example (L2 normalisation):
```python
x = x / x.norm(dim=1, keepdim=True)
```


## topic or label supervision

- adds **explicit class or topic labels** to reinforce structure in embedding space
- used in _multi-task learning_, combining contrastive and classification losses

example:
$$
L_{total} = L_{contrastive} + λ * L_{classification}
$$

this teaches the encoder to preserve semantic similarity _and_ categorical distinction.  
useful for domain-specific retrieval (e.g. finance vs medicine vs news).


## References

- **[Paper] SimCLR (InfoNCE, Temperature, Projector Head):** Chen, et. al. (2020). `A Simple Framework for Contrastive Learning of Visual Representations`.
    - [https://arxiv.org/abs/2002.05709](https://arxiv.org/abs/2002.05709)

- **[Paper] DPR (Offline Hard Negative Mining):** Karpukhin, et. al. (2020). `Dense Passage Retrieval for Open-Domain Question Answering`.
    - [https://arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)

- **[Article] SBERT (The Anisotropy/Collapse Problem):** Reimers, N. (SBERT authors). `The Problem: Bad Sentence Embeddings`.
    - [https://sbert.net/docs/avg_word_embeddings.html](https://www.google.com/search?q=https://sbert.net/docs/avg_word_embeddings.html) (See Section 2)

- **[Article] Two-Tower Models (The Architecture):** Google Developers (2020). `Dual-Encoder models for Recommendation`.
    - [https://developers.google.com/machine-learning/recommendation/dual-encoders](https://www.google.com/search?q=https://developers.google.com/machine-learning/recommendation/dual-encoders)