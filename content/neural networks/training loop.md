## loss functions

measures how wrong the current set of weights are. 
[[loss functions]]

**Mean squared error** — for regression (predicting a number):

> `L = (ŷ - y)²`


**Cross-entropy** — for classification (predicting a category). Your network outputs a probability for each class via softmax. If the correct class gets probability 0.9, your loss is low. If it only gets 0.01, loss is huge:

> `L = -log(p_correct)`

Why log? Because going from 0.01 → 0.1 is a much bigger improvement than 0.8 → 0.9, and log captures that asymmetry naturally.


## gradient descent

### update rule

All weights are updated by the same rule at every step:

> `w ← w - η · ∂L/∂w`
- `∂L/∂w` — partial derivative of loss w.r.t. this weight: "if I increase w by a tiny amount, how much does L change?"
- `η` — learning rate: step size
- Minus sign — move opposite to the gradient (downhill)

|η|What happens|
|---|---|
|Too large|Overshoot the minimum, bounce around, possibly diverge|
|Too small|Converge, but very slowly|
|Just right|Smooth convergence|

### the narrow valley problem

![[Pasted image 20260530140721.png]]

The **Hessian** H = ∇²L is the matrix of second derivatives. Its eigenvalues describe curvature in each direction of weight-space.

The **condition number** κ = λ_max / λ_min (i.e. steepest direction ÷ flattest direction) measures how unequal the curvatures are. When κ is large:
- Some directions are very steep (large eigenvalue)
- Some directions are very flat (small eigenvalue)
- The loss surface looks like a long narrow canyon

In this case, the gradient mostly points across the steep walls rather than along the floor toward the minimum. Gradient descent oscillates back and forth across the canyon while barely making progress forward.

This is the core geometric reason plain gradient descent struggles in practice.


### convergence rates

convergence = the point where the model finds the point of minima (and model stops getting better in terms of loss)
T = number of steps taken

![[Pasted image 20260530141425.png]]

|Assumption about L|Convergence rate|
|---|---|
|Convex, Lipschitz gradients|O(1/√T)|
|Strongly convex (λ_min > 0)|O(1/T)|
|Non-convex (real neural nets)|Convergence to ‖∇L‖ → 0 only — no global guarantee|

**Important:** convergence to ‖∇L‖ → 0 means you reach a point where the gradient is zero — but that could be a local minimum, a saddle point, or rarely a local maximum.


### mini-batch SGD

Computing the gradient over the entire dataset every step is too expensive. Instead, sample a random **mini-batch** of examples per step and estimate the gradient from that. 

![[Pasted image 20260530142336.png]]

This approach inevitably leads to noise in the calculation of the true gradient, but that could be a good thing.

![[Pasted image 20260530142413.png]]

Full gradient descent follows the terrain perfectly — which sounds good, but if it hits a flat saddle point, it just stops (gradient is ~0, so it thinks it's done). The noisy SGD path wiggles randomly, and that wiggle is just enough to bump it out of the flat zone and keep going.

**Linear scaling rule** (Goyal et al.): if you multiply batch size by k, multiply η by k. Intuition: k times more data per step means each gradient estimate is k times less noisy, so you can afford a larger step. This rule breaks down at very large batch sizes.


## backpropagation

The update rule `w ← w - η · ∂L/∂w` requires ∂L/∂w for every single weight. A network has millions of weights, organised in layers, with the loss only computed at the very end. Backpropagation computes all these gradients in one efficient pass.


If L depends on a, and a depends on w, then:

> `∂L/∂w = (∂L/∂a) · (∂a/∂w)`

In a network with layers a₁ → a₂ → ... → aₙ → L:

> `∂L/∂w = ∂L/∂aₙ · ∂aₙ/∂aₙ₋₁ · ... · ∂a₂/∂a₁ · ∂a₁/∂w`

This is a product of derivatives across all layers from the loss back to w. Backprop computes this product efficiently by working right to left, reusing intermediate results.

**Cost:** one forward pass + one backward pass. The backward pass costs roughly the same as the forward pass — not more. This is the key efficiency insight.

### vanishing and exploding gradients

The gradient at an early layer involves multiplying many terms together — one per layer between that layer and the output.

> If each term < 1: product shrinks exponentially → **vanishing gradients**. Early layers receive near-zero gradient and stop learning.

> If each term > 1: product grows exponentially → **exploding gradients**. Updates become huge and destabilise training.

To fix this, 

**Residual connections.** Instead of passing output through a transformation F(x), compute x + F(x). The gradient now has a direct path through the identity:

> `∂L/∂x = ∂L/∂(x + F(x)) · (1 + ∂F/∂x)`

The 1 guarantees a non-vanishing path for the gradient regardless of what F does.

**Careful initialisation.** Weight scale at initialisation determines the initial magnitude of these products.

- Xavier/Glorot: `Var(w) = 2/(nᵢₙ + nₒᵤₜ)` — for tanh/sigmoid activations
- He: `Var(w) = 2/nᵢₙ` — for ReLU, which zeroes ~half its inputs so the effective fan-in is halved

**Gradient clipping.** If gradient norm exceeds a threshold: rescale the entire gradient vector to have that norm. Preserves direction, bounds magnitude. Always clip by norm, not by value — clipping by value distorts the gradient direction.

### memory cost

Backprop needs intermediate activations from the forward pass to compute gradients. For a network with L layers, this is O(L) memory.

**Gradient checkpointing:** store activations only at certain checkpoint layers. Recompute everything else during the backward pass when needed. Memory cost drops from O(L) to O(√L), at the cost of one extra forward pass. Essential for training large models.


## Adam optimiser

### why we need an optimiser

In a real neural network, different weights behave very differently. Some get huge gradients every step (noisy, unstable). Some get tiny gradients rarely (like weights connected to uncommon words). Plain gradient descent uses the same learning rate η for all of them — which is wrong for both.


### momentum

Rather than using the raw gradient each step (which is noisy), keep a **running average** of recent gradients. This is called the first moment mₜ.

**mₜ = β₁ · mₜ₋₁ + (1 − β₁) · gₜ**

With β₁ = 0.9, this means: "90% of what I believed last step, plus 10% of what I'm seeing now." It's like a heavy ball that doesn't instantly change direction — it builds up speed in whatever direction gradients have _consistently_ been pointing.

### RMSProp

Track a running average of the **squared** gradient: **vₜ = β₂ · vₜ₋₁ + (1 − β₂) · gₜ²**

Squaring makes everything positive and amplifies large values. So vₜ is a measure of how wildly the gradient has been fluctuating. Then divide your step by √vₜ:

- Noisy parameter → large vₜ → divide by large number → **small step** (safe)
- Stable/rare parameter → small vₜ → divide by small number → **large step** (bold)

This is the "per-parameter learning rate" idea — it automatically calibrates each weight.

![[Pasted image 20260530180536.png]]



Adam computes two moving averages from the same raw gradient $g_t$ each step:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1),g_t \qquad \text{(momentum — tracks direction)}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2),g_t^2 \qquad \text{(RMSProp — tracks noise level)}$$

The update uses both together:

$$w_t = w_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

$\hat{m}_t$ supplies the direction and $\sqrt{\hat{v}_t}$ scales the step down where gradients are noisy. Each parameter gets its own $m$ and $v$, so the effective learning rate adapts independently per weight.



### Bias correction — why it exists

Both $m$ and $v$ are initialised to zero. Expanding $m_t$ by repeatedly substituting the recurrence:

$$m_t = (1-\beta_1)\sum_{i=1}^{t} \beta_1^{t-i}, g_i$$

The weights $\beta_1^{t-i}$ should sum to 1 for this to be a true average, but they actually sum to:

$$\sum_{i=1}^{t}(1-\beta_1),\beta_1^{t-i} = 1 - \beta_1^t$$

So the expected value of $m_t$ is:

$$\mathbb{E}[m_t] = \mathbb{E}[g]\cdot(1 - \beta_1^t)$$

It undershoots the true mean by a factor of $(1-\beta_1^t)$. The fix is to divide it out:

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

To see why the correction fades, consider what $(1-\beta_1^t)$ equals at different steps with $\beta_1 = 0.9$:

|$t$|$\beta_1^t$|$1 - \beta_1^t$|correction factor $\frac{1}{1-\beta_1^t}$|
|---|---|---|---|
|1|0.900|0.100|$\times 10.0$ — massive correction|
|5|0.590|0.410|$\times 2.44$|
|10|0.349|0.651|$\times 1.54$|
|50|0.005|0.995|$\times 1.005$ — essentially gone|
|100|$\approx 0$|$\approx 1$|$\times 1.000$|

At $t=1$, only one gradient has been fed in, worth $(1-\beta_1),g_1 = 0.1,g_1$, so the corrector multiplies by 10 to recover the true estimate. By step 50, $\beta_1^{50} \approx 0$ and the denominator $\approx 1$, so $\hat{m}_t \approx m_t$ — the correction has silently switched itself off. It only matters in the first few dozen steps when the running averages have not yet accumulated enough history to be trustworthy.