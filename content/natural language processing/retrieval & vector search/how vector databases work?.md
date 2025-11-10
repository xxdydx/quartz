a vector db stores and searches through vector embeddings
when you store a text like:
> “The capital of France is Paris.”

you don’t store the string itself for similarity search. instead, you use an **embedding model** (e.g. OpenAI’s `text-embedding-3-large`) to map that text into a high-dimensional vector, such as:
$v = [0.12, -0.45, 0.78, …]$ (say, 1536 dimensions)

semantic similarity between two texts then becomes a geometric problem— measuring how close vectors are using cosine similarity or euclidean distance.

however, doing cosine similarity on large indexes is incredibly complex + time consuming.

**Brute force complexity:** $O(n·d)$ per query.  
For n = 10⁹, d = 768 → impossible in real-time.

### the curse of higher dimensionality

at higher dimensions (especially in the order of hundreds), points (or vectors, in this case) end up much closer to one another. hence, you can't even use data structures like k-d trees or ball trees (optimal for 10-20 dimensions) because they work by dividing space into neat regions. 

they're fast when you can rule out large areas, not when all the points are packed together in one space and almost every point is about the same distance away.

so, we use **Approximate Nearest Neighbour** (ANN) methods to ensure high recall at sublinear cost.


### ANN methods

| Type                    | Algorithm Examples  | Key Idea                                           |
| ----------------------- | ------------------- | -------------------------------------------------- |
| **Quantisation-based**  | IVF, PQ, OPQ, ScaNN | Compress vectors into centroids or subspaces       |
| **Graph-based**         | HNSW, NSG, KGraph   | Build small-world graphs linking nearby nodes      |
| **Tree-based / Hybrid** | Annoy, FLANN        | Space-partitioning, typically for lower dimensions |

at large scale, HNSW dominates as
- near-optimal recall-time trade-offs
- supports dynamic insertions
- cache-friendly & cpu-efficient


### how does HNSW work?
explained that in another article here: [[deep dive into HNSW]]

### practical systems implementation

#### adjacency list in HNSW
- each embedding (node) in space has neighbours at each level. for a node `i`, list of neighbour node IDs and (optionally) their precomputed distances is stored.

example
```bash
node i:
   level 0 neighbours: [j, k, l, …]
   level 1 neighbours: [p, q, …]
   …
```

- these lists form the graph connections that HNSW uses to navigate the vector space.
- neighbours are stored **contiguously** in memory so cpu can read them quickly (better cache locality)
- when searching, the algorithm follows these links to move closer to the query vector step by step.
- each level has fewer, longer connections — upper levels give faster jumps, lower levels give fine precision.
- this allows for **logarithmic-time search** while keeping memory overhead manageable.

#### memory layout and cache locality
- contiguous neighbour storage -> fewer cache misses
- systems like FAISS and Milvus use **prefetching** to anticipate which neighbours will be visited next, minimising memory latency.

#### vector compression & quantisation
- full 32-bit float embeddings take up copious amounts of memory
- product quantisation (PQ) splits each vector into smaller sub-vectors, replaces each with a 8-bit code referencing a codebook
- this cuts memory by 4-8x
- during search, the query remains full-precision, but stored vectors are approximated using lookup tables (asymmetric distance computation)
- https://www.pinecone.io/learn/series/faiss/product-quantization/






references


