## what are embeddings?

- a way to represent **words, sentences, images, or other data** as points in a **high-dimensional vector space**.
- each dimension captures some **latent feature** (e.g. gender, tense, topic, sentiment).
- **similar meanings → vectors point in similar directions** (small angle between them).
- the goal: convert discrete data (like text) into **continuous numerical representations** that models can compute with.

**example:**  
“the cat sits on the mat” → [0.2, 0.5, 0.8, …] (a 384-dimensional vector)  
“dog on carpet” → another vector close in direction to the above.


## why embeddings matter

- computers can’t process words directly — only numbers.
- embeddings capture **semantic similarity** numerically.
- useful for:
    - search and retrieval        
    - recommendation systems
    - clustering and visualisation
    - semantic matching (e.g. duplicate detection, intent classification)        

## word2vec (static embeddings)

### what it does
- learns **one fixed vector per word** by predicting word co-occurrence in text.
- words with similar contexts end up close together in vector space.

### training objectives

| model                              | objective                                              | example                                                         |
| ---------------------------------- | ------------------------------------------------------ | --------------------------------------------------------------- |
| **skip-gram**                      | predict surrounding words given a centre word.         | input: “king” → output: likely words “queen”, “crown”, “royal”. |
| **cbow** (continuous bag of words) | predict the centre word given its surrounding context. | context: “the ___ sits on the throne” → predicts “king”.        |

### key idea

- the model shifts word vectors to maximise prediction success.
- as a result, **semantic relationships** emerge geometrically.

**famous relation:**  
`vector("king") - vector("man") + vector("woman") ≈ vector("queen")`
that happens because the “royal” feature direction is learned implicitly.

### limitations

- **one vector per word** → fails for polysemy (e.g. “bank” = river vs finance).
- cannot use sentence context.
- works well for analogy reasoning but poorly for nuanced meaning.

## glove (global vectors)

- another static embedding model, but instead of predicting neighbours, it uses **word co-occurrence statistics** over the whole corpus.
- learns embeddings so that **dot products approximate log co-occurrence probabilities**.
- generally smoother, more globally consistent than word2vec, but still context-independent.

## contextual embeddings (elmo, bert)

- unlike word2vec, **contextual models** produce _different vectors_ for the same word depending on its sentence.

### elmo
- bi-directional lstm over sentences.
- combines multiple layers → gives token representation conditioned on its context.

### bert (bidirectional encoder representations from transformers)

- uses transformer architecture (self-attention).
- each token “looks at” other tokens in the sentence to build its meaning.
- outputs a contextual vector for each token or the whole sentence (e.g. `[cls]` token).

**example:**
- “bank” in “river bank” and “bank account” → totally different vectors.

this solves word2vec’s ambiguity issue.

## sentence and document embeddings

- for retrieval or classification, we need embeddings that represent **entire sentences** or **paragraphs**.
- average token embeddings → crude but simple.
- better: use pre-trained **bi-encoders** (e.g. `sentence-transformers`), which are trained for similarity tasks.

## measuring similarity

### cosine similarity
- measures **angle** between vectors.
- formula:  
    $sim(a,b) = \frac{a·b}{‖a‖‖b‖}$
- range: –1 to 1.
- high value → vectors point in the same direction → semantically close.

### **euclidean distance**
- measures **straight-line distance** between points.
- sensitive to vector length (magnitude).

**why cosine is used more:**
- in embeddings, magnitude (vector length) is often arbitrary; direction captures meaning better.
- normalisation removes scale differences between vectors.

## visual intuition

imagine all words as dots in 3d (for simplicity):
- “cat”, “dog”, “rabbit” cluster together — the _animal_ region.
- “stock”, “investment”, “market” cluster elsewhere — the _finance_ region.
- distances and angles reflect relationships.

when trained correctly, embeddings form a **semantic map of language**.

## evaluation metrics

- **intrinsic** (measure embedding quality itself):
    - word similarity (correlation with human-judged pairs)
    - analogy tests (king-man+woman=queen)
- **extrinsic** (measure task performance):
    - use embeddings as input for classification or retrieval, see if they help.






