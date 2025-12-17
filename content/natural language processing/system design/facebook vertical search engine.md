*Referenced from [Embedding-Based Retrieval in Facebook Search](https://www.alphaxiv.org/abs/2006.11632) paper*

## purpose
- A social search engine; relevance doesn't just depend on the text query, also requires understanding of the searcher's context (e.g. location, social graph, prev searches, videos recently watched, etc.)
- Their old search system is called *Unicorn*, based on boolean matching.
	- Each document was represented as a "bag of terms", e.g. `text:john` or `location:seattle`
	- Queries were executed as boolean expressions, e.g. (`and (term text:john) (term location:seattle))`to find exact matches
- The goal is to integrate EBR to this old method seamlessly.

## overview
Say we have **millions or even billions** of documents.

Retrieval Layer (recall) — This is the bottom layer of the stack. Its goal is to retrieve a relevant set of documents from an index containing billions of items with low latency and computational cost. Often the bottleneck of a system.

Ranking Layer (precision) — This layer processes the retrieved set using complex algorithms to rank the most desired results at the top.

Applying embeddings in the retrieval layer offer the best opportunity to improve the system.

### proposed framework

![[Pasted image 20251217114843.png]]

## model
The task is formulated as a recall optimisation problem. The objective is to maximise the number of relevant results found within the top $K$ results returned by the model.

![[Pasted image 20251217121846.png]]

### unified embedding architecture
- Two neural networks (encoder) that transforms inputs into low-dimensional dense embeddings.
	- **Query Encoder** — It incorporates the query text, the searcher's information, and the search context (e.g. searcher's location, social connections, etc.)
	- **Document Encoder** — It incorporates document metadata and context. (e.g. for a FB group, inputs can include the explicit group location or related social clusters)
	
- **Similarity function** — uses **Cosine Similarity** $$S(Q, D) = \cos(E_Q, E_D) = \frac{E_Q \cdot E_D}{||E_Q|| [cite_start]||E_D||}$$
- **Distance for Training:** $D = 1 - \cos(E_Q, E_D)$, used for triplet loss later.

### loss function, triplet loss
- the goal is to separate a positive pair (query + relevant result) from a negative pair (query + irrelevant pair) by a specific distance margin, $m$.$$L = \sum \max(0, D(q, d_+) - D(q, d_-) + m)$$
- Tuning the margin $m$ is critical. The optimal value varies by task, and different values can result in 5-10% variance in recall performance.

### training data mining

#### negative labels
- Team had 2 options, either they select random documents from the pool or they select results that were shown to the user AND ignored.
- They found out that random samples were superior (55% drop in recall for the latter option) 
	- Extreme negative results bias the model towards "hard cases", however the vast majority of retrieval index consists of "easy cases". 
#### positive labels
- Team had 2 options, clicks vs impressions, both achieved similar recall, so they decided to go with using clicks. 
- By clicks I mean results the user actually clicked.


### eval metrics
- Online metrics (e.g. A/B tests) and offline metrics (e.g. recall@K) used. 
- They sampled 10,000 search sessions and run a K-Nearest Neighbour (KNN) search across the _entire_ index to measure how often the target result appears in the top K results.
	- KNN is used instead of ANN as accuracy is more important than speed for this experiment, besides only 10,000 sessions are being used as the sample size.


## feature engineering

Moving from text embeddings to **unified embeddings** yielded significant recall improvements (e.g,. +18% for Events, +16% for Groups)

### how does it help?
- By feeding diverse inputs (text, location, social) into the network, the model will learn via backpropagation which signals matter for different types of queries.
- It'll also help resolve ambiguity where text fails.
- Allows for hard negative mining training method.

### text features

#### Character N-grams
- _Example:_ For the word **"apple"**, a 3-character N-gram (trigram) split would be: `app`, `ppl`, `ple`
- So if a user misspells the word as "aple", a few chunks might still match, which will output a close similarity score to the real "apple" document.
- The model relies primarily on these character chunks rather than full words.
	- There are fewer unique 3-letter combinations than there are unique words in the English language. This keeps the "vocabulary" small, making the model faster to train.
	- It solves the "Out-of-Vocabulary" problem. On Facebook, people have unique names or use slang that isn't in a standard dictionary. Character N-grams can represent _any_ word, even one the model has never seen before, by breaking it into familiar chunks.


#### Word N-grams
- _Example:_ In the query "hot dog stand," the word N-grams capture that "hot" and "dog" belong together ("hot dog"), rather than just being a temperature and an animal.
- Adding this on top of character chunks improved recall by **+1.5%**. It helps the model understand specific phrases better than characters alone.
- However, there are too many possible word combinations (e.g., **352 million** unique trigrams in queries) to store in a simple list.
- The system uses hashing to solve this, possible downside of hash collision.

#### Fuzzy Matching
- The model understands that two words are practically the same even if spelled differently.
- It maps the user query **"kacis creations"** to the page **"Kasie's creations"**. Boolean matching fails here because "kacis" $\neq$ "Kasie's", but the embedding model sees they are close in vector space.

#### Optionalization
- The model learns which words in a query are important and which are noise (optional).
- For the query **"mini cooper nw"**, the user likely wants the "Mini Cooper Owner's Club." The term "nw" (perhaps "northwest") is too specific and blocks exact matches. The embedding model learns to "drop" or ignore "nw" to retrieve the correct group, recognising the core intent is "Mini Cooper".


### location features

Adding geographical data to the embeddings will help the model understand "where" the user is, without explicitly typing it into the query.
- **Query Side:** The model inputs the searcher’s city, region, country, and language.
- **Document Side:** It inputs explicit location tags provided by group admins.
- As a result, the model learns **Implicit Location Matching**.
    - _Example:_ A user in **Louisville, KY** searches for **"equipment for sale."** The model automatically ranks results like **"Kentucky Farm Equipment"** higher, even though the user never typed "Kentucky." It learned the relationship between the user's GPS and the text in the result.


### social embedding features

Using the massive web of connections on Facebook (who is friends with whom, who likes what page) as a feature.
- A separate model is trained to understand the social graph (users & entities).
- The output of the model is then fed into the Unified Embedding model as an extra input. This will allow the search model to understand social relevance better.


## serving

### ANN Strategy
- Exact kNN is not feasible for production systems as it has a time complexity of $O(n\cdot d)$, which is impractical if you want to serve results within milliseconds.
- The system uses an **Inverted Index-based ANN**
	- Inverted index has lower storage costs through vector quantisation.
- The system uses the FAISS library to quantise vectors and implement efficient NN search.

**Inverted index**
- 
### tuning ANN for performance
This is to balance between recall and performance in the production environment.

**Metric Shift: Percentage of Index Scanned**

- The standard parameter `nprobe` controls how many clusters to search. However, clusters are often **imbalanced** (some contain many more documents than others), especially with the IMI algorithm.
- Tuning `nprobe` alone is unreliable because scanning 10 small clusters is faster than 10 large ones. Instead, they tune based on the **percentage of the index scanned**, which is a truer measure of system cost.
- This ensures consistent latency. A setting that scans 0.5% of the index is predictable, whereas a setting of `nprobe=10` could scan 0.1% or 5% depending on the query.

**Data Transformation (OPQ)**

- The team applies Optimised Product Quantisation (OPQ) to rotate and transform the data before quantisation.
- OPQ consistently outperformed standard PCA (Principal Component Analysis). For example, it achieved **74.29% recall** compared to 67.54% for standard PQ (PCA performed worse than standard PQ).

**Compression Limit**

- Increasing the size of the product quantiser (`pq_bytes`) improves accuracy, but with diminishing returns.
- They found little benefit to setting the byte size larger than **$d/4$**, where $d$ is the dimension of the embedding vector.

### system implementation

- Instead of building a new vector database, Facebook modified their existing inverted-index engine ("Unicorn") to support embeddings directly.
- Document embeddings are quantized (compressed) and stored in the inverted index. The "Coarse Cluster ID" is stored as a searchable **Term**, and the vector details are stored as a **Payload**.
- They created a new Boolean operator `(nn <key> : radius <radius>)`. This allows embedding checks to be mixed freely with standard text/location filters in a single query.
- The document side is computed offline in batches and indexed; the query side is computed online in real-time.

Say the user searches for "john smithe" (typo) expecting to find "John Smith" in Seattle (where the user is at).

**Why not top-K?**

Top-K searches for the absolute best mathematical matches across the **entire index** before applying any filters.
- **Wasteful Computation:** A Top-K search might return 100 perfect matches from New York. If the user requires results in "Seattle," all 100 are discarded, leaving zero results.
- **Inefficient:** It forces the system to scan a huge portion of the index to ensure it has the "global best," only to throw most of it away later.

**Why not filter first and then run embedding search?**

Even after filtering for "Seattle," there might be 500,000 documents left. Running complex vector math (cosine similarity) on 500,000 vectors in real-time is too slow and CPU-intensive.


The system uses **Radius Mode** to achieve the best of both worlds.
- It treats the Embedding Cluster just like another filter term (e.g., `Cluster_42` -> people that the user is related to via social graph).
- The engine instantly finds the **intersection** of "Seattle" AND "Cluster_42".
- This reduces the pool to a tiny number (e.g., 50 docs), making the final distance calculation fast and efficient.

The query in the hybrid system looks something like this:
```lisp
(and
  (or (term location:seattle) (term location:menlo_park))  ; Boolean Constraint
  (or
     (and (term text:john) (term text:smithe))             ; Exact Text Match (Fails)
     (nn model-key radius 0.24 : nprobe 16)                ; Embedding Match (Succeeds)
  )
)
```

The term match `text:smithe` fails. However, the `(nn)` operator finds "John Smith" because his vector distance is within **0.24** of the query. The Boolean `AND` ensures the result is strictly from Seattle or Menlo Park.

### architecture
- Online: The query encoder is deployed as a real-time service. It computes the vector for the user's request on the fly.
- Offline: Since document features (titles, locations) change less frequently, their embeddings are computed in **batch offline** using Spark. - - These embeddings are quantised (Coarse + PQ) and published into the inverted index alongside standard text terms.


## later-stage optimisation

- While EBR improves **Recall** (finding missed results), it generally has lower **Precision** than exact term matching (it finds more irrelevant "fuzzy" matches).
- They established a closed feedback loop using human evaluation.
    1. **Log Results:** They record the actual results returned by the new EBR system.
    2. **Human Rating:** These results are sent to human raters to label as "Relevant" or "Not Relevant".
    3. **Retraining:** This labeled data is used to re-train the ranking model.
- **Outcome:** The ranker learns specifically how to filter out the "bad" embedding results while keeping the "good" ones, ensuring high precision in the final output.


## hard mining

The standard training method uses "Random Negatives" (e.g., Query: "Pizza", Positive: "Pizza Hut", Negative: "Tax Software"). This becomes too easy for the model. To improve further, the model must be challenged with **Hard Negatives**.
### the problem with "easy negatives"
- After a certain point, the model can easily distinguish "Pizza" from "Tax Software" just by looking at basic text.'
- When the model gets the answer right with high confidence, the "Loss" approaches zero. The backpropagation signal disappears, and the model **stops learning**. It never learns to distinguish fine details (e.g., "Pizza Hut" vs. "Pizza Box Supplier").

### online hard negative mining
- The system looks at the batch of data currently in memory (e.g. ~50 query/answer pairs)
- E.g. For **Query #1**, there is:
	- **1** Correct Answer (Positive).
	- **49** Wrong Answers (Negatives).
- This method selects the ones with the highest vector similarity scores to the current query (out of the 49).
- Pros: Very efficient. No extra data processing required.
- Cons: The "hardness" is limited. The hardest negative in a random batch of 50 is still usually quite easy compared to the hardest negative in the entire database. This leads to lazy learning.

### offline hard negative mining
- Instead of just looking within the batch, the system looks through the entire database to find hard negatives.
- **The Workflow:**
    1. Take a snapshot of the current model.
    2. Run queries against the **entire index** (billions of items) to find the top results.
    3. Select results that are **visually/semantically similar** to the query but were **not** clicked by the user.
    4. Feed these specific failures back into the training data as negatives.


### false negatives

The hardest negatives are often actually **False Negatives**.
- _Example:_ User searches "Apple". They click the first result (Fruit). The second result was "Apple Computers."
- _Hard Mining Logic:_ "The user didn't click 'Apple Computers', so it is a Negative. Push it away."
- Hence, the model learns that "Apple Computers" is factually irrelevant to the query "Apple", which is factually wrong.

The solution is **semi-hard mining**:
- Instead of selecting the Top 1 (Hardest) result, they select items ranked between **101 and 500**.
- These items are hard enough to be challenging (better than random), but "bad" enough that they are likely truly irrelevant, minimising the risk of punishing valid results.


## deep shadow retrieval

Once the model is trained, deploying it is risky. If the new model is bad, users get terrible results. **Deep Shadow Retrieval (DSR)** is a technique to test and tune the model on live traffic without the user ever seeing the results.

### offline evaluation is difficult

Standard metrics (Recall@K) calculated on static datasets often fail to predict real-world success because:
- They don't account for **System Latency** (how long the query takes).
- They don't account for **Index State** (items being added/deleted in real-time).
- They don't reveal how the **Boolean Filters** interact with the embeddings.

### how shadow retrieval works

Instead of A/B testing (showing the new model to 1% of users), they run the model in "Shadow Mode."
1. **User Action:** User searches for "John".
2. **Live Path:** The existing production system returns results. User clicks "John Doe".
3. **Shadow Path (Invisible):** Simultaneously, the _new_ Embedding Model runs the search for "John".
4. **The Comparison:** The system checks: **"Did the Shadow Model retrieve the specific item ('John Doe') that the user clicked?"**

It is not enough to just check the Top 1 result.
- The Shadow system performs the full retrieval (ANN search -> Boolean Filtering).
- It logs the rank of the clicked item in the Shadow results.
- "Recall at Shadow." If the clicked item appears _anywhere_ in the Shadow retrieval set, it counts as a generic "hit.

### the purpose
This is primarily a **System Tuning** tool. It allows engineers to tweak dangerous parameters safely:
- **Tuning Search Radius:** They can adjust the radius (how fuzzy the match is) and see if Recall improves or drops in real-time.
- **Tuning `nprobe`:** They can lower `nprobe` (speeding up the system) and immediately see if it causes the system to miss the user's clicked item.
- **Latency Profiling:** They can measure exactly how many milliseconds the model adds to the server load under real traffic conditions (peak hours vs. off-hours).