
## how does reinforcement learning differ from supervised learning?

### the optimisation objective
- **SFT (Supervised Fine-Tuning):** Optimises for **Imitation**. It treats the problem as a sequence of independent classification tasks.
    - **Loss Function:** Minimises Cross-Entropy Loss (Negative Log-Likelihood) against a fixed target sequence $y$.$$L_{CE} = - \sum_{t} \log P_\theta(y_t | y_{<t}, x)$$
    - **Goal:** Match the likelihood distribution of the training data.

- **RL (Reinforcement Learning):** Optimises for **Value Maximisation**. It treats the generation as a decision-making process where actions (tokens) lead to a final scalar outcome.
    - **Objective:** Maximises the Expected Return (Reward). $$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$
    - **Goal:** Maximise a utility function (Reward Model), potentially deviating from the training data distribution to find higher-value outputs.

### feedback granularity & signal
- **SFT (Dense Signal):** Feedback is provided at every token step $t$. The gradient $\nabla \log P(y_t)$ is computed immediately against the ground truth $y_t$. There is no ambiguity about _which_ action was incorrect.
- **RL (Sparse Signal):** Feedback is typically episodic (received only at the end of sequence $T$). The model receives a single scalar $R$.
    - **Credit Assignment Problem:** The optimisation algorithm (e.g., PPO) must estimate which specific tokens in the sequence contributed to the high/low reward, usually via Generalised Advantage Estimation (GAE).


### distributional putcome: "average" vs "peak"

- **SFT (Distribution Matching):** The model approximates the conditional probability $P_{data}(y|x)$. If the training data contains a mixture of high-quality and mediocre responses, the model learns the **mean** of that distribution.
- **RL (Mode Seeking):** RL shifts the policy mass $\pi_\theta$ toward regions of the output space that yield high $R$. If the Reward Model accurately ranks "superhuman" responses higher than "average" human responses, the policy can converge to the **peak** of the reward landscape, effectively outperforming the average demonstrator in the dataset.

## markov decision process

An MDP is defined by the tuple $\mathcal{M} = (S, A, P, R, \gamma)$.

**The components**
- **State ($S_t$):** The specific configuration of the world at time $t$.
- **Action ($A_t$):** The decision made by the agent.
- **Reward ($R_t$):** The immediate scalar signal received after taking action $A_t$ in state $S_t$.
- Transition Probability ($P$): The physics of the world.$$P(s' | s, a) = \mathbb{P}[S_{t+1}=s' | S_t=s, A_t=a]$$
    Note: In model-free RL (like PPO), we do not know $P$. We must learn strictly by interacting with the environment.
- **Discount Factor ($\gamma$):** A value between $[0, 1]$ that determines the present value of future rewards.

### markov property
The core assumption of RL is the **Markov Property**:
> "The future is independent of the past, given the present."

$$P(S_{t+1} | S_t) = P(S_{t+1} | S_t, S_{t-1}, \dots, S_0)$$
If the state $S_t$ does not contain all necessary information to make a decision, the system is **not** Markovian.

- _Example:_ In a moving image, a single frame is not a valid state because you can't tell the direction of velocity. You need two stacked frames (or an LSTM memory) to make it a valid Markov state.
- _Implication:_ This is why LLMs need the full context window. If you truncate the context, the model loses the "State," and the math of PPO collapses because the "Next Token" probability is no longer grounded in reality.

## the objective: cumulative returns

We do not optimize for the _reward_ ($r_t$). We optimise for the **Return** ($G_t$), which is the cumulative discounted reward.
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$
- **Reward ($R_t$):** The **immediate** feedback you get right now (at this specific millisecond).
- **Return ($G_t$):** The **cumulative** score you get from now until the end of the game.

### the role of $\gamma$
- $\gamma$ is a discount factor, guiding the agent to prioritise present rewards over future rewards.
- **$\gamma \approx 0$:** The agent is "myopic." It cares only about the immediate click/score. It will eat the marshmallow now.
- **$\gamma \approx 1$:** The agent is "farsighted." It will sacrifice immediate rewards for a massive payout 1,000 steps later.
- _Difficulty:_ As $\gamma \to 1$, training becomes unstable because credit assignment becomes harder (the butterfly effect).

## the three pillars: policy, value, and Q

To solve an MDP, we typically learn three functions.

## a. the policy ($\pi$) $\rightarrow$ "the controller"

The policy acts as the "brain" of the agent, by choosing which action to take next.

**the math**
The policy is a probability distribution over actions given states:

$$\pi(a|s) = \mathbb{P}[A_t = a | S_t = s]$$

In Deep RL (like PPO), this function is parameterised by weights $\theta$ (the neural network): $\pi_\theta(a|s)$.

### stochastic vs. deterministic

1. Deterministic Policy ($\mu$):$$a = \mu(s)$$
    - **Logic:** "Always do the exact same thing in this state."
    - **Use Case:** Robotics control (perfect precision), greedy exploitation.
    - **Limitation:** It cannot explore. If the policy says "Go Left," it will never try "Right" to see if it's better.

2. Stochastic Policy ($\pi$):$$a \sim \pi(\cdot|s)$$
    - **Logic:** "I am 80% sure 'Go Left' is best, but 20% of the time I'll try 'Go Right' just in case."
    - **Use Case:** **LLMs** (Next Token Prediction), PPO.
    - **Why it's vital:** The randomness (entropy) is what allows the agent to **explore** the environment without needing a separate mechanism like $\epsilon$-greedy.

## b. the value function ($V$) $\rightarrow$ "the predictor"

The Value function is a prediction of the future. It measures the long-term quality of a **state**, assuming we behave consistently from now on.

**the math**
$$V_\pi(s) = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \bigg| S_t = s \right]$$
**A state does not have an intrinsic value.** It only has value _relative_ to a specific policy.

- **Scenario:** You are standing on a cliff edge (State $s$).
- **Policy A (Suicidal):** Jumps off. $\to V_{\pi_A}(s) = -1000$.
- **Policy B (Safe):** Steps back. $\to V_{\pi_B}(s) = +10$.
- **Conclusion:** The "goodness" of the state depends entirely on what you plan to do next.

### the bellman expectation equation (for $V$)

We can write $V$ recursively. The value of _this_ state is the immediate reward plus the discounted value of the _next_ state.

$$V_\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V_\pi(s') \right]$$

- **In simple terms:** "My value now = Average Reward of my action + Discounted Average Value of where I land."

## c. the action-value function ($Q$) $\rightarrow$ "the analyzer"

The Value function ($V$) judges the _situation_. The Q-function judges the _decision_.

**the math**
$$Q_\pi(s, a) = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \bigg| S_t = s, A_t = a \right]$$

Notice the condition: We are locked into taking action $a$ at step $t$, but **after that**, we follow the policy $\pi$ forever.

**the decomposition**
We can break $Q$ down into:
1. The immediate outcome of the action.
2. The value of the future.
$$Q_\pi(s, a) = \mathbb{E} [R_{t+1} + \gamma V_\pi(S_{t+1})]$$

**why do we need both $V$ and $Q$?**
- If we have the model of the world (like in Chess), we can just look one step ahead using $V$.
- But in Model-Free RL (where we don't know the rules), we need $Q$.
	- **$V(s)$ tells us:** "We are winning."
	- **$Q(s, a)$ tells us:** "If I move the Knight, we are winning. If I move the Rook, we lose."
	- Without $Q$, the agent knows it's in a good spot but doesn't know which specific button press keeps it there.

## d. the synthesis: the advantage function (A)

This is how the pillars combine to drive modern algorithms like PPO.
We want to know: **"Is this specific action better or worse than the average action?"**
$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$
- **$Q^\pi(s, a)$**: How good is it to take action $a$?
- **$V^\pi(s)$**: How good is the state usually (the average of all actions)?

**the visual hierarchy**
1. $V(s)$ is the baseline (the average).
2. $Q(s,a)$ is the specific sample.
3. $A(s,a)$ is the difference.

If $A(s,a)>0$, the action $a$ outperformed the baseline. **We should increase $π(a∣s)$.** If $A(s,a)$<0, the action $a$ underperformed. **We should decrease $π(a∣s)$.**



## references

[An introduction to Reinforcement Learning](https://www.youtube.com/watch?v=JgvyzIkgxF0)
[Proximal Policy Optimization (PPO) for LLMs Explained Intuitively](https://www.youtube.com/watch?v=8jtAzxUwDj0)
[Reinforcement Learning with Human Feedback (RLHF) in 4 minutes](https://www.youtube.com/watch?v=vJ4SsfmeQlk)
[The Contextual Bandits Problem](https://www.youtube.com/watch?v=N5x48g2sp8M)
[How does DeepSeek learn? GRPO explained with Triangle Creatures](https://www.youtube.com/watch?v=wXEvvg4YJ9I)
