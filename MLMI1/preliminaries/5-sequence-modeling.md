# Introduction to Sequence Modeling 
Sequence modeling is a fundamental concept in statistical modeling and machine learning, dealing with data where order and context are crucial. 

---

## 1. Markov Models

Markov models are stochastic models that describe a sequence of possible events where the probability of each event depends only on the state attained in the previous event. This property is known as the **Markov property**.

### Discrete Data: n-gram Models

**First-Order Markov Model (Bi-gram Model)**  
In the context of sequence modeling, a first-order Markov model assumes that the probability of an event at time $t$ depends solely on the event at time $t - 1$. For a sequence of discrete variables $y_1, y_2, \dots, y_T$, this is expressed as:

$$
p(y_1, y_2, \dots, y_T) = p(y_1) \prod_{t=2}^{T} p(y_t \mid y_{t-1})
$$

where each $y_t$ belongs to a finite set $\{1, 2, \dots, K\}$.

#### Likelihood of a Markov Model

Given:
- An **emission probability vector** $\pi$, where $\pi_i = P(y_1 = i)$ is the probability of starting in state $i$,
- A **transition matrix** $T$ with entries $T_{ij} = P(y_t = i | y_{t-1} = j)$, representing the probability of transitioning from state $j$ to state $i$,

The likelihood of observing a sequence $y_{1:T} = \{y_1, y_2, \dots, y_T\}$ is given by:

$$
P(y_{1:T}) = \pi_{y_1} \prod_{t=2}^{T} T_{y_t y_{t-1}}
$$

The likelihood of the sequence $y_{1:T}$ can then be expressed as:

$$
P(y_{1:T}) = \pi_{y_1} \prod_{i,j} T_{ij}^{n_{ij}}
$$
where: 
- $n_{ij}$ is the **number of transitions** observed from state $j$ to state $i$ in the sequence $y_{1:T}$.

---
**Second-Order Markov Model (Tri-gram Model)**  
A second-order Markov model extends this dependency to two previous states:

$$
p(y_1, y_2, y_3, \dots, y_T) = p(y_1) p(y_2 \mid y_1) \prod_{t=3}^{T} p(y_t \mid y_{t-1}, y_{t-2})
$$

**Transition Probabilities:**

$$
T_{ijk} = p(y_t = k \mid y_{t-1} = j, y_{t-2} = i)
$$

This requires a three-dimensional array (tensor) to store all possible transitions, leading to computational and storage challenges as $K$ increases.

---

**Marginal Distribution Over the Second State**  
To compute the marginal distribution over the second state $y_2$:

$$
p(y_2 = k) = \sum_{l=1}^{K} p(y_2 = k \mid y_1 = l) p(y_1 = l) = \sum_{l=1}^{K} T_{lk} \pi_l^0
$$

Explanation: We sum over all possible initial states $l$, multiplying the probability of starting in state $l$ by the probability of transitioning from $l$ to $k$.

**Stationary Distribution**  
The stationary distribution $\pi^{\infty}$ represents the state probabilities as $t \to \infty$, where the distribution no longer changes over time:

$$
\pi_k^{\infty} = \lim_{t \to \infty} p(y_t = k)
$$

The stationary distribution satisfies the eigenvector equation:

$$
\pi_k^{\infty} = \sum_{l=1}^{K} \pi_l^{\infty} T_{lk}
$$

**Normalization Condition:**

$$
\sum_{k=1}^{K} \pi_k^{\infty} = 1
$$

Interpretation: The stationary distribution is the left eigenvector of the transition matrix $T$ associated with the eigenvalue 1.

### Continuous Data: Auto-Regressive (AR) Gaussian Models

For continuous data, we model the sequence using Gaussian distributions and linear relationships.

**First-Order Markov Model (AR(1))**  
An AR(1) model assumes that the current observation $y_t$ is a linear function of the previous observation $y_{t-1}$ plus Gaussian noise:

$$
y_t = A y_{t-1} + w_t
$$

where $y_t \in \mathbb{R}^D$ is a continuous vector, $A$ is a $D \times D$ matrix defining the linear relationship, and $w_t \sim N(0, \Sigma)$ is Gaussian noise with covariance $\Sigma$.

**Initial Distribution:**

$$
p(y_1) = N(y_1; \mu_0, \Sigma_0)
$$

**Conditional Distribution:**

$$
p(y_t \mid y_{t-1}) = N(y_t; A y_{t-1}, \Sigma)
$$

**Stationary Distribution**  
For scalar cases ($y_t \in \mathbb{R}$), the stationary variance $\sigma_{\infty}^2$ can be derived:

**Model Equation:**

$$
y_t = \lambda y_{t-1} + \sigma \epsilon_t
$$

where $\lambda$ is the autoregressive coefficient, $\sigma \epsilon_t$ represents Gaussian noise, and $\epsilon_t \sim N(0,1)$.

**Variance Recursion:**

$$
\langle y_t^2 \rangle = \lambda^2 \langle y_{t-1}^2 \rangle + \sigma^2
$$

At Stationarity ($\langle y_t^2 \rangle = \langle y_{t-1}^2 \rangle = \sigma_{\infty}^2$):

$$
\sigma_{\infty}^2 = \frac{\sigma^2}{1 - \lambda^2}
$$

Interpretation: The stationary variance depends on the noise variance $\sigma^2$ and the autoregressive coefficient $\lambda$. The process is stationary only if $|\lambda| < 1$.

---

## 2. Hidden Markov Models (HMMs)

An HMM extends the Markov model by introducing hidden states $x_t$, which are not directly observable. Instead, we observe $y_t$, which is probabilistically related to the hidden states.

### Discrete Hidden States

In an HMM with discrete hidden states:

**Emission Probabilities:**

$$
p(y_t \mid x_t = k) = S_{k, t}
$$

where $S_{k, t}$ defines the probability of observing $y_t$ given the hidden state $x_t = k$.

**Observation Model for Continuous Data:**

$$
p(y_t \mid x_t = k) = N(y_t; \mu_k, \Sigma_k)
$$

Each hidden state $k$ has an associated Gaussian distribution with mean $\mu_k$ and covariance $\Sigma_k$.

**Initial State Distribution:**

$$
p(x_1 = k) = \pi_k^0
$$

**Transition Probabilities:**

$$
p(x_t = j \mid x_{t-1} = i) = T_{ij}
$$

---

## 3. Inference

Inference in HMMs involves computing the posterior distributions of the hidden states given the observations. This allows us to make predictions, detect patterns, and learn model parameters.

### Varieties of Inference

Different inference tasks require different distributions:

| Estimator Type | Marginal Distribution | Joint Distribution        |
| -------------- | --------------------- | ------------------------- |
| Filter         | $p(x_t \mid y_{1:t})$ | $p(x_{1:t} \mid y_{1:t})$ |
| Smoother       | $p(x_t \mid y_{1:T})$ | $p(x_{1:T} \mid y_{1:T})$ |

- **Filtering**: Computing the distribution over the hidden state at time $t$ given observations up to time $t$.
- **Smoothing**: Computing the distribution over the hidden state at time $t$ given all observations from time 1 to $T$.

#### Point Estimates

- **Most Probable State:**

$$
x_t^* = \arg \max_{x_t} p(x_t \mid y_{1:T})
$$

- **Expected State:**

$$
x_t^* = \mathbb{E}[x_t \mid y_{1:T}]
$$

**Questions to Consider:**

- Are these estimates the same as in the Linear Gaussian State Space Model (LGSSM)?
- How do these concepts apply to discrete hidden state HMMs?

### Kalman Filter

The Kalman filter is an optimal recursive algorithm for estimating the state of a linear dynamic system in the presence of Gaussian noise.

#### Recursive Update Formula

The posterior distribution of the state $x_t$ given observations up to time $t$ is:

$$
p(x_t \mid y_{1:t}) = \int p(x_t \mid x_{t-1}) p(x_{t-1} \mid y_{1:t-1}) dx_{t-1} 
$$

**Derivation for LGSSM**  
Given:

- **State Dynamics:**

$$
x_t = A x_{t-1} + w_t, \quad w_t \sim N(0, Q)
$$

- **Observation Model:**

$$
y_t = C x_t + v_t, \quad v_t \sim N(0, R)
$$

**Steps:**

1. **Prediction Step**  
   Compute the prior $p(x_t \mid y_{1:t-1})$ using the state dynamics.

   The mean and covariance are:

   $$
   \hat{x}_t^- = A \hat{x}_{t-1}
   $$

   $$
   P_t^- = A P_{t-1} A^\top + Q
   $$

2. **Update Step**  
   Incorporate the new observation $y_t$.

   Compute the Kalman Gain $K_t$:

   $$
   K_t = P_t^- C^\top (C P_t^- C^\top + R)^{-1}
   $$

   Update the state estimate and covariance:

   $$
   \hat{x}_t = \hat{x}_t^- + K_t (y_t - C \hat{x}_t^-)
   $$

   $$
   P_t = (I - K_t C) P_t^-
   $$

   - **Kalman Gain** $K_t$: Determines the weight given to the new observation versus the prior estimate.

### Forward Algorithm

In discrete HMMs, the Forward Algorithm efficiently computes the filtering distribution $p(x_t \mid y_{1:t})$.

#### Forward Probabilities

Define the forward variable:

$$
\alpha_t(k) = p(y_{1:t}, x_t = k)
$$

**Recursion**

1. **Initialization:**

   $$
   \alpha_1(k) = p(y_1 \mid x_1 = k) \pi_k^0
   $$

2. **Induction:**

   $$
   \alpha_t(k) = p(y_t \mid x_t = k) \sum_{l=1}^{K} \alpha_{t-1}(l) T_{lk}
   $$

**Explanation:** The probability of being in state $k$ at time $t$ is the sum over all possible previous states $l$ of the probability of being in state $l$ at time $t-1$ (captured by $\alpha_{t-1}(l)$) multiplied by the transition probability from $l$ to $k$, and then multiplied by the emission probability $p(y_t \mid x_t = k)$.

#### Normalization

To obtain the filtering distribution $p(x_t = k \mid y_{1:t})$:

$$
p(x_t = k \mid y_{1:t}) = \frac{\alpha_t(k)}{\sum_{k=1}^{K} \alpha_t(k)}
$$

### Computational Complexity

The algorithm runs in $O(TK^2)$ time, which is linear in the sequence length $T$ and quadratic in the number of states $K$. This is significantly more efficient than enumerating all possible state sequences, which would be computationally infeasible for large $T$ or $K$.

---

## 4. Maximum Likelihood Estimation of HMMs

Learning the parameters $\theta$ of an HMM involves maximizing the likelihood of the observed data.

### Objective Function

The log-likelihood of the observations given the parameters is:

$$
\log p(y_{1:T} \mid \theta) = \log \int p(y_{1:T}, x_{1:T} \mid \theta) \, dx_{1:T}
$$

### Gradient Computation

The gradient of the log-likelihood with respect to $\theta$ is:

$$
\frac{\partial}{\partial \theta} \log p(y_{1:T} \mid \theta) = \frac{1}{p(y_{1:T} \mid \theta)} \int \frac{\partial}{\partial \theta} p(y_{1:T}, x_{1:T} \mid \theta) \, dx_{1:T}
$$

Using the posterior distribution $p(x_{1:T} \mid y_{1:T}, \theta)$:

$$
\frac{\partial}{\partial \theta} \log p(y_{1:T} \mid \theta) = \int p(x_{1:T} \mid y_{1:T}, \theta) \frac{\partial}{\partial \theta} \log p(y_{1:T}, x_{1:T} \mid \theta) \, dx_{1:T}
$$

Interpretation: The gradient is the expected value of the gradient of the complete-data log-likelihood, weighted by the posterior distribution over the hidden states.

### For Linear Gaussian State Space Models (LGSSMs)

The expected complete-data log-likelihood is:

$$
E(\theta; x_{1:T}, y_{1:T}) = \sum_{t=1}^{T} \left[ \log p(y_t \mid x_t, \theta) + \log p(x_t \mid x_{t-1}, \theta) \right]
$$

#### Requirements for Gradient Computation

- **Posterior Moments**: To compute the gradient, we need the marginal distributions $p(x_t \mid y_{1:T})$ and the pairwise marginals $p(x_t, x_{t-1} \mid y_{1:T})$.
- **Expectation-Maximization (EM) Algorithm**: Often used to iteratively estimate the parameters by alternating between computing the expected sufficient statistics (E-step) and maximizing the expected log-likelihood (M-step).

---

## Summary and Key Takeaways

- **Markov Models**: Fundamental for modeling sequences with dependencies; higher-order models capture more context but require more parameters.
- **Hidden Markov Models**: Introduce hidden states to model complex dependencies; widely used in speech recognition, bioinformatics, and more.
- **Inference Techniques**: Filtering and smoothing are crucial for estimating hidden states; algorithms like the Kalman filter and Forward algorithm provide efficient solutions.
- **Parameter Estimation**: Maximum likelihood estimation requires computing gradients of the log-likelihood; the EM algorithm is often used for HMMs.
