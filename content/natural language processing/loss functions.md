Most loss functions are derived from Maximum Likelihood Estimation (MLE). We try to maximize the probability of the data given our model parameters. Since maximizing a product of probabilities is hard (underflow), we take the negative log to turn it into a sum minimization problem.

## Classification (Discrete Categories)

**Goal:** Decide which "bucket" an input belongs to.

### A. Binary Cross-Entropy (BCE)

**Used for**: Binary Classification problems

**Input Format**: A single probability score $p \in [0, 1]$ (usually from a Sigmoid activation).

**The Formula**: $$L = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \cdot \log(p_i) + (1 - y_i) \cdot \log(1 - p_i) \right]$$
**Why this specific function?**
1. **Bernoulli Distribution:** We assume the data follows a Bernoulli distribution (coin flip). The likelihood of getting label $y$ is $p^y(1-p)^{1-y}$. Taking the negative log of this gives the formula above.
2. **Handling Both Cases:**
    - If $y=1$: The second term $(1-y)$ becomes 0. We maximise $\log(p)$.
    - If $y=0$: The first term $y$ becomes 0. We maximise $\log(1-p)$.
3. **Vanishing Gradient Protection:** When combined with Sigmoid, the derivative of BCE is simply $(p - y)$. This is linear and prevents gradients from vanishing when the model is wrong, unlike MSE which would multiply by a tiny slope near $p=0$ or $p=1$.

**When to Use & Rationale:**

- **Scenario:** Spam Detection (Spam/Not Spam).
    - **Rationale:** The output is a simple binary state.
- **Scenario:** Image Tagging (An image can contain "Cat", "Grass", AND "Sky").
    - **Rationale:** BCE treats each class as an **independent coin flip**. The presence of "Cat" does not reduce the probability of "Grass". (You run BCE separately for each of the $N$ tags).

### B. Categorical Cross-Entropy (CCE)

**Used for**: "Pick 1 of N" tasks (Multi-class).
**Input Format**: A vector of probabilities summing to 1 (from a Softmax activation).

**The Formula**:
$$L = - \sum_{c=1}^C y_{true, c} \cdot \log(y_{pred, c})$$

**How it differs from BCE:**
- **Structure:** BCE treats each class independently (is it A? Yes/No. Is it B? Yes/No). CCE assumes mutual exclusivity (if it is A, it _cannot_ be B).
- **One-Hot Encoding:** In CCE, $y_{true}$ is a one-hot vector (e.g., $[0, 0, 1]$). Because of the 0s, only the probability of the _correct_ class contributes to the loss.
- **Information Theory:** This minimises the **KL-Divergence** between the predicted distribution and the true distribution. It measures "how many extra bits do we need to encode the message using the wrong distribution?"

**When to Use & Rationale:**
- **Scenario:** Digit Classification (MNIST 0-9).
    - **Rationale:** A digit cannot be both a "3" and a "5". Softmax forces the probabilities to compete, so increasing confidence in "3" naturally lowers confidence in "5".
- **Scenario:** Next Token Prediction (LLMs).
    - **Rationale:** There is only one actual "next word" in the training text. We want to maximize the probability of that specific token.

### C. Focal Loss

**Used for**: Extreme Class Imbalance.

**The Formula**:$$L = - (1 - p_t)^\gamma \log(p_t)$$
**The Math Intuition**:
- It adds a **Modulating Factor** $(1 - p_t)^\gamma$.
- If the model is confident ($p_t \approx 0.9$), $(1-0.9)^2 = 0.01$. The loss is down-weighted by 100x.
- If the model is wrong/uncertain ($p_t \approx 0.1$), $(1-0.1)^2 = 0.81$. The loss stays high.
- **Result:** The gradients are dominated by the _hard_ examples, not the easy background.

**When to Use & Rationale:**
- **Scenario:** Object Detection (YOLO/RetinaNet).
    - **Rationale:** In an image, 99% of pixels are "Background" (easy negatives). A standard loss would be overwhelmed by these easy examples. Focal Loss forces the model to ignore the background and focus entirely on the hard, rare positive cases (the objects).
- **Scenario:** Fraud Detection (0.1% of transactions).
    - **Rationale:** Prevents the model from achieving 99.9% accuracy by simply predicting "Not Fraud" for everything.

### D. Label Smoothing

**Used for**: Preventing Overfitting/Overconfidence.

**The Math**: Instead of target $y=[0, 1]$, use $y=[0.1, 0.9]$.
$$y_{new} = (1 - \epsilon) \cdot y_{old} + \epsilon / K$$

Why: A model trained on pure CCE tries to push probabilities to infinite logits to reach exactly 1.0. This causes overfitting. 

## Regression (Continuous Numbers)

**Goal:** Predict a specific scalar value.

#### A. Mean Squared Error (MSE / L2)

The Formula:

$$L = \frac{1}{N} \sum (y - \hat{y})^2$$

**The Math Intuition:**

- **Gaussian Prior:** This assumes the noise in your data follows a **Normal (Gaussian) Distribution**. Maximizing the likelihood of Gaussian data is mathematically identical to minimizing squared error.
- **Convexity:** It is a smooth, convex parabola. It has a unique global minimum.
- **Sensitivity:** Because of the square, an error of 10 is $100\times$ worse than an error of 1.

#### B. Mean Absolute Error (MAE / L1)

The Formula:

$$L = \frac{1}{N} \sum |y - \hat{y}|$$

**The Math Intuition:**

- **Laplacian Prior:** Assumes noise follows a Laplacian distribution (sharper peak, fatter tails).
- **Constant Gradient:** The gradient is either $+1$ or $-1$, regardless of how close you are. It does not "slow down" as it approaches the target (requires learning rate decay).

#### C. Huber Loss (Smooth L1)

The Math:

$$L = \begin{cases} 0.5 (y-\hat{y})^2 & \text{if } |y-\hat{y}| < \delta \\ \delta |y-\hat{y}| - 0.5 \delta^2 & \text{otherwise} \end{cases}$$

Why: It fuses the best of both worlds. It is quadratic near 0 (for precise convergence like MSE) and linear far away (to be robust to outliers like MAE).



### 3. Metric Learning (Embeddings)

**Goal:** Distance optimisation in vector space.

#### A. Triplet Loss

The Formula:

$$L = \max(0, d(A, P)^2 - d(A, N)^2 + \alpha)$$

The Math Intuition:

- We want $d(A, P) + \alpha < d(A, N)$.
- **The Margin ($\alpha$):** Without $\alpha$, the model could put $A, P, N$ all at the exact same point (distance 0) and satisfy the condition $0 < 0$. The margin forces the negative to be _pushed away_ by a safety buffer.

#### B. InfoNCE (Contrastive)

The Formula:

$$L = - \log \frac{\exp(sim(A, P)/\tau)}{\sum_{i} \exp(sim(A, N_i)/\tau)}$$

The Math Intuition:

- **Softmax in Disguise:** This looks exactly like Categorical Cross-Entropy!
- The "True Class" is the Positive sample. The "False Classes" are all the Negative samples in the batch.
- **Temperature ($\tau$):** Controls how "peaky" the distribution is. Low $\tau$ makes the model focus only on the hardest negatives; high $\tau$ averages the gradient across all negatives.



### 4. Generative & LLMs

#### A. KL Divergence (Kullback-Leibler)

Used in: VAEs, RLHF.

The Formula:

$$D_{KL}(P || Q) = \sum P(x) \log \left( \frac{P(x)}{Q(x)} \right)$$

The Math Intuition:

- Measures the information lost when $Q$ is used to approximate $P$.
- **Asymmetry:** $KL(P||Q) \neq KL(Q||P)$.
- **In RLHF:** We add this as a penalty term: $Reward - \beta \cdot KL(\pi_{new} || \pi_{ref})$. This ensures the new fine-tuned model ($\pi_{new}$) doesn't drift too far from the original "grammatically correct" model ($\pi_{ref}$).

#### B. PPO Clip Loss

The Formula:

$$L = \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)$$

The Math Intuition:

- **Ratio $r_t(\theta)$:** The probability of taking action $a$ under the new policy vs the old policy.
- **The Clip:** If the new policy makes an action _way_ more likely than before (e.g., $r_t > 1.2$), we stop rewarding it.
- **Why:** Trust Region. In Reinforcement Learning, data is generated by the current policy. If you update the policy too much, the data you just collected becomes invalid, and training collapses. The clip forces "small steps."



### Summary: The "Why" Lookup Table

|**Function**|**Derived From (Probabilistic Prior)**|**Key Math Feature**|**Best For**|
|---|---|---|---|
|**BCE**|Bernoulli Distribution|Log cancellations prevent vanishing gradient|Binary Classification|
|**CCE**|Multinoulli Distribution|Softmax normalization (Classes compete)|Multi-class Classification|
|**MSE**|Gaussian (Normal) Distribution|Squaring penalizes outliers heavily|Clean Regression|
|**MAE**|Laplacian Distribution|Absolute value ignores outliers|Noisy Regression / Finance|
|**InfoNCE**|Categorical Cross-Entropy|Batch-based normalization|Semantic Search / Embeddings|