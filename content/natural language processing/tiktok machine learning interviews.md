## abstract

this was my experience interning for the machine learning engineer internship role for 2026, at tiktok's vertical search department, specifically under the phototext & live team.

this is the [job description](https://lifeattiktok.com/search/7531689400276535559)
![[Pasted image 20251020011823.png]]
![[Pasted image 20251020011806.png]]


## interview 1
- leetcode question: [Minimum Number of Operations to Make Arrays Similar (Hard)](https://leetcode.com/problems/minimum-number-of-operations-to-make-arrays-similar/)
- discussion about past projects — did a search & retrieval pipeline at my previous internship, read my blogpost about it [here](https://blog.arul.me/posts/designing-semantic-search-engine)
- discussion about fine-tuning for that project— delved into different fine-tuning methods, [[low-rank adaptation (LoRA)]] vs [[supervised fine tuning]]
- discussion about transformer architecture (expected!, [[introduction to transformers]]), specifically about the attention mechanism, how it works and the math behind it.  [[attention mechanism, in depth!]] 
- discussion about self-supervised objectives in LLMs,  [[masked language modelling]] and [[causal language modelling]]
- discussion about fine-tuning, specifically about what types of inputs/outputs were used and loss functions that were used
- questions about how the retrieval module in RAG works
- question about bi-encoder vs cross-encoder
- discussion about vector similarity metrics, [[introduction to vector embeddings]]
- asked about the time complexity of brute-force vector search, and what are the methods to optimise it, [[how vector databases work?]]


## interview 2
- more coding heavy, was asked 2 leetcode medium questions — [Kth Missing Positive Number](https://leetcode.com/problems/kth-missing-positive-number) and [Delete and Earn](https://leetcode.com/problems/delete-and-earn/)
- was asked to explain the search & retrieval pipeline i made for my previous internship again
- discussion about metrics used to evaluate performance and the rationale for choosing them (specific metrics discussed were recall, precision, f1 score, accuracy)
- probability question asked: probability that three random points on a circle form a right or acute triangle. (answer: 1/4)


## interview 3
- leetcode question: [Implement Trie](https://leetcode.com/problems/implement-trie-prefix-tree/)
- bi-encoder vs cross-encoder
- discussion about negative mining and loss functions (e.g. triplet loss, contrastive loss, infonce)
- discussion about attention mechanisms, specifically self-attention vs cross-attention
- question on how to separate two embeddings that are similar to each other in the embedding space— add more hard negatives, diverse data, use larger margin in triplet/contrastive loss, apply regularisation, topic labels, hierarchical chunking [[improving embedding separability]]
- discussion about prompt finetuning— things like CoT, few-shot
