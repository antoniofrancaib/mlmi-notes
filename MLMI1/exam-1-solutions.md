### Solutions to the Exam

---

### Question 1

#### (a) Conjugate Prior Definition and Example

**Definition:**

A conjugate prior is a prior distribution that, when combined with a particular likelihood function through Bayes' theorem, results in a posterior distribution that is in the same family as the prior. This property greatly simplifies Bayesian inference because it allows for analytical computation of the posterior distribution.

**Example:**

Consider the case of observing data from a Bernoulli process, where each observation $x_n$ is either 0 or 1, with a probability $\theta$ of observing 1.

**Likelihood Function:**

The likelihood of observing $x$ successes in $N$ trials is given by the Binomial distribution:

$$
p(x \mid \theta) = \theta^x (1 - \theta)^{N - x}
$$

**Conjugate Prior:**

The conjugate prior for the Bernoulli likelihood is the Beta distribution:

$$
p(\theta) = \text{Beta}(\theta; \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} \theta^{\alpha - 1} (1 - \theta)^{\beta - 1}
$$

**Posterior Distribution:**

Combining the likelihood and prior, the posterior is:

$$
p(\theta \mid x) \propto \theta^{\alpha - 1 + x} (1 - \theta)^{\beta - 1 + N - x}
$$

This is also a Beta distribution with updated parameters $\alpha' = \alpha + x$ and $\beta' = \beta + N - x$.

**Why It Simplifies Computation:**

Using a conjugate prior allows us to update the posterior distribution simply by updating the parameters of the prior with sufficient statistics from the data. This avoids complex integrations and makes analytical solutions feasible.

---

#### (b)(i) Deriving the Posterior Distribution

**Given:**

Data: $x = \sum_{n=1}^N x_n$ successes in $N$ Bernoulli trials.

Prior: $\theta \sim \text{Beta}(\alpha, \beta)$.

**Posterior Derivation:**

**Likelihood Function:**

$$
p(x \mid \theta) = \theta^x (1 - \theta)^{N - x}
$$

**Prior Distribution:**

$$
p(\theta) = \frac{\theta^{\alpha - 1} (1 - \theta)^{\beta - 1}}{B(\alpha, \beta)}
$$

where $B(\alpha, \beta)$ is the Beta function (normalizing constant).

**Unnormalized Posterior:**

$$
p(\theta \mid x) \propto p(x \mid \theta) p(\theta) \propto \theta^x (1 - \theta)^{N - x} \theta^{\alpha - 1} (1 - \theta)^{\beta - 1}
$$

$$
p(\theta \mid x) \propto \theta^{\alpha + x - 1} (1 - \theta)^{\beta + N - x - 1}
$$

**Normalization:**

Since the posterior is proportional to a Beta distribution, it is:

$$
p(\theta \mid x) = \text{Beta}(\theta; \alpha', \beta')
$$

where:

$$
\alpha' = \alpha + x
$$

$$
\beta' = \beta + N - x
$$

---

#### (b)(ii) Computing the Predictive Distribution

We want to compute:

$$
p(x_{N+1} = 1 \mid \text{data})
$$

**Predictive Distribution:**

$$
p(x_{N+1} = 1 \mid \text{data}) = \int_0^1 p(x_{N+1} = 1 \mid \theta) p(\theta \mid \text{data}) d\theta
$$

Since $x_{N+1}$ is Bernoulli with parameter $\theta$:

$$
p(x_{N+1} = 1 \mid \theta) = \theta
$$

Therefore:

$$
p(x_{N+1} = 1 \mid \text{data}) = \int_0^1 \theta \cdot \text{Beta}(\theta; \alpha', \beta') d\theta
$$

**Using the Mean of the Beta Distribution:**

The mean of a Beta distribution is:

$$
E[\theta] = \frac{\alpha'}{\alpha' + \beta'}
$$

Thus:

$$
p(x_{N+1} = 1 \mid \text{data}) = \frac{\alpha + x}{\alpha + \beta + N}
$$

---

#### (c) Hierarchical Bayesian Model with Hyperpriors

**Effect on Inference:**

- Introducing hyperpriors on $\alpha$ and $\beta$ makes the model hierarchical.
- The prior parameters $\alpha$ and $\beta$ are now random variables, allowing for greater flexibility in modeling uncertainty.
- This approach can capture variability in the prior beliefs and can lead to a more robust posterior distribution.

**Computational Challenges:**

- **Non-Conjugacy:** The Beta prior is no longer conjugate when $\alpha$ and $\beta$ are random, complicating the computation of the posterior.
- **Intractable Integrals:** The posterior distribution $p(\theta, \alpha, \beta \mid \text{data})$ involves integrating over $\alpha$ and $\beta$, which may not have a closed-form solution.
- **Need for Approximate Methods:** Techniques such as Markov Chain Monte Carlo (MCMC) methods (e.g., Gibbs sampling) or variational inference are required to perform inference.
- **Increased Computational Cost:** Sampling methods can be computationally intensive and may require careful tuning to ensure convergence.

---

### Question 2

#### (a) Hierarchical Probabilistic Model

**Model Specification:**

**Observation Model:**

For each $n=1, \dots, N$:

$$
x_n \mid \mu, \sigma_n^2 \sim N(x_n; \mu, \sigma_n^2)
$$

**Prior on Variances:**

$$
\sigma_n^2 \sim \text{Inv-Gamma}(\sigma_n^2; \alpha, \beta)
$$

**Prior on Mean:**

Assume a prior on $\mu$, e.g., a Gaussian prior:

$$
\mu \sim N(\mu; \mu_0, \tau^2)
$$

**Joint Distribution:**

$$
p(\{x_n\}, \{\sigma_n^2\}, \mu) = p(\mu) \prod_{n=1}^N p(\sigma_n^2) p(x_n \mid \mu, \sigma_n^2)
$$

where:

$$
p(\sigma_n^2) = \text{Inv-Gamma}(\sigma_n^2; \alpha, \beta)
$$

$$
p(x_n \mid \mu, \sigma_n^2) = N(x_n; \mu, \sigma_n^2)
$$

$$
p(\mu) = N(\mu; \mu_0, \tau^2)
$$

---

#### (b) Conditional Posterior Distribution of $\mu$

**Goal:**

Compute $p(\mu \mid \{x_n\}, \{\sigma_n^2\})$.

**Derivation:**

**Joint Distribution (up to proportionality):**

$$
p(\mu \mid \{x_n\}, \{\sigma_n^2\}) \propto p(\mu) \prod_{n=1}^N p(x_n \mid \mu, \sigma_n^2)
$$

**Substitute the Distributions:**

$$
p(\mu \mid \cdot) \propto \exp \left( -\frac{1}{2 \tau^2} (\mu - \mu_0)^2 \right) \prod_{n=1}^N \exp \left( -\frac{1}{2 \sigma_n^2} (x_n - \mu)^2 \right)
$$

**Combine Exponents:**

$$
\ln p(\mu \mid \cdot) = -\frac{1}{2 \tau^2} (\mu - \mu_0)^2 - \sum_{n=1}^N \frac{1}{2 \sigma_n^2} (x_n - \mu)^2 + \text{const}
$$

**Complete the Square:**

Collect terms involving $\mu$:

$$
\ln p(\mu \mid \cdot) = -\frac{1}{2} \left( \left( \frac{1}{\tau^2} + \sum_{n=1}^N \frac{1}{\sigma_n^2} \right) \mu^2 - 2 \left( \frac{\mu_0}{\tau^2} + \sum_{n=1}^N \frac{x_n}{\sigma_n^2} \right) \mu \right) + \text{const}
$$

**Identify Gaussian Form:**

The posterior is Gaussian with:

**Mean:**

$$
\mu_{\text{post}} = \sigma_{\text{post}}^2 \left( \frac{\mu_0}{\tau^2} + \sum_{n=1}^N \frac{x_n}{\sigma_n^2} \right)
$$

**Variance:**

$$
\sigma_{\text{post}}^2 = \left( \frac{1}{\tau^2} + \sum_{n=1}^N \frac{1}{\sigma_n^2} \right)^{-1}
$$

---

#### (c) Gibbs Sampling Algorithm

**Algorithm Steps:**

1. **Initialization:**

   Set initial values for $\mu^{(0)}$ and $\{\sigma_n^2^{(0)}\}$.

2. **Iterative Sampling:**

   For iteration $t=1, \dots, T$:

   - a. **Sample $\mu^{(t)}$ from $p(\mu \mid \{x_n\}, \{\sigma_n^2^{(t-1)}\})$:**

     Use the Gaussian posterior derived in part (b).

   - b. **For each $n$, sample $\sigma_n^2^{(t)}$ from $p(\sigma_n^2 \mid x_n, \mu^{(t)})$:**

     **Conditional Posterior of $\sigma_n^2$:**

     $$
     p(\sigma_n^2 \mid x_n, \mu) \propto p(\sigma_n^2) p(x_n \mid \mu, \sigma_n^2)
     $$

     **Substitute the Distributions:**

     $$
     p(\sigma_n^2 \mid \cdot) \propto (\sigma_n^2)^{-\alpha - 1} \exp \left( -\frac{\beta}{\sigma_n^2} \right) \cdot (\sigma_n^2)^{-1/2} \exp \left( -\frac{(x_n - \mu)^2}{2 \sigma_n^2} \right)
     $$

     **Simplify:**

     $$
     p(\sigma_n^2 \mid \cdot) \propto (\sigma_n^2)^{-(\alpha + 3/2)} \exp \left( -\frac{\beta + (x_n - \mu)^2 / 2}{\sigma_n^2} \right)
     $$

     This is the kernel of an inverse gamma distribution with parameters:

     $$
     \alpha_n' = \alpha + \frac{1}{2}
     $$

     $$
     \beta_n' = \beta + \frac{(x_n - \mu)^2}{2}
     $$

     Sample $\sigma_n^2$ from $\text{Inv-Gamma}(\alpha_n', \beta_n')$.

3. **Repeat Steps a and b until convergence.**

**Conditional Distributions Needed:**

- $p(\mu \mid \{x_n\}, \{\sigma_n^2\})$: Gaussian.
- $p(\sigma_n^2 \mid x_n, \mu)$: Inverse Gamma.

---

### Question 3

#### (a) Advantages and Drawbacks of a Shared Covariance Matrix

**Advantages:**

1. **Parameter Reduction:**

   In high-dimensional spaces, estimating a full covariance matrix for each component is computationally intensive and requires a large number of parameters $\frac{D(D+1)}{2}$ per component. Using a shared covariance matrix reduces the number of parameters from $K \times \frac{D(D+1)}{2}$ to $\frac{D(D+1)}{2}$, mitigating overfitting.

2. **Computational Efficiency:**

   Simplifies the M-step in the EM algorithm, as the covariance matrix needs to be computed only once per iteration.

3. **Numerical Stability:**

   In high dimensions, individual covariance estimates can become singular or ill-conditioned due to insufficient data. Sharing the covariance matrix can improve numerical stability.

**Drawbacks:**

1. **Less Flexibility:**

   Assumes all clusters have the same shape and orientation, which may not be appropriate if the true clusters have different covariance structures.

2. **Potential Misclassification:**

   May lead to poor clustering performance if clusters differ significantly in variance or covariance patterns.

3. **Bias Toward Spherical Clusters:**

   Encourages clusters to be similar in spread, potentially oversimplifying the data structure.

---

#### (b) Complete Data Log-Likelihood

Let $z_n$ be the latent variable indicating the component membership of $x_n$, where $z_n \in \{1, \dots, K\}$.

**Complete Data Log-Likelihood:**

$$
\ln p(\{x_n, z_n\} \mid \{\pi_k, \mu_k, \Sigma\}) = \sum_{n=1}^N \left( \ln \pi_{z_n} + \ln N(x_n; \mu_{z_n}, \Sigma) \right)
$$

**Explicitly:**

$$
\ln p(\{x_n, z_n\} \mid \cdot) = \sum_{n=1}^N \left( \ln \pi_{z_n} - \frac{D}{2} \ln (2 \pi) - \frac{1}{2} \ln \det \Sigma - \frac{1}{2} (x_n - \mu_{z_n})^\top \Sigma^{-1} (x_n - \mu_{z_n}) \right)
$$

---

#### (c) EM Algorithm Updates

**E-Step:** Compute Responsibilities

$$
\gamma_{nk} = p(z_n = k \mid x_n) = \frac{\pi_k N(x_n; \mu_k, \Sigma)}{\sum_{j=1}^K \pi_j N(x_n; \mu_j, \Sigma)}
$$

**M-Step:** Update Parameters

1. **Mixing Proportions $\pi_k$:**

   $$
   \pi_k^{\text{new}} = \frac{N_k}{N}
   $$

   $$
   N_k = \sum_{n=1}^N \gamma_{nk}
   $$

2. **Means $\mu_k$:**

   $$
   \mu_k^{\text{new}} = \frac{1}{N_k} \sum_{n=1}^N \gamma_{nk} x_n
   $$

3. **Shared Covariance Matrix $\Sigma$:**

   $$
   \Sigma^{\text{new}} = \frac{1}{N} \sum_{k=1}^K \sum_{n=1}^N \gamma_{nk} (x_n - \mu_k^{\text{new}}) (x_n - \mu_k^{\text{new}})^\top
   $$

**Derivation Steps:**

- Maximize Expected Complete Log-Likelihood with respect to $\pi_k$:

  Subject to $\sum_{k=1}^K \pi_k = 1$.

- Maximize with respect to $\mu_k$:

  Set derivative to zero:

  $$
  \frac{\partial}{\partial \mu_k} \sum_{n=1}^N \gamma_{nk} \left( -\frac{1}{2} (x_n - \mu_k)^\top \Sigma^{-1} (x_n - \mu_k) \right) = 0
  $$

  Solving yields the update for $\mu_k$.

- Maximize with respect to $\Sigma$:

  Since $\Sigma$ is shared, combine contributions from all components.

---

### Question 4

#### (a) Joint Probability Distribution

**Components:**

1. **Initial State Distribution:**

   $$
   p(s_1)
   $$

2. **Transition Probabilities:**

   $$
   p(s_t \mid s_{t-1})
   $$

3. **Emission Probabilities:**

   $$
   p(y_t \mid s_t) = N(y_t; w_{s_t}^\top x_t, \sigma^2)
   $$

**Joint Distribution:**

$$
p(y_{1:T}, s_{1:T}) = p(s_1) p(y_1 \mid s_1) \prod_{t=2}^T p(s_t \mid s_{t-1}) p(y_t \mid s_t)
$$

---

#### (b) Forward-Backward Algorithm

**Purpose:**

Compute $p(s_t \mid y_{1:T})$ for all $t$.

**Steps:**

1. **Forward Pass:**

   Compute $\alpha_t(s_t) = p(y_{1:t}, s_t)$.

   - **Initialization:**

     $$
     \alpha_1(s_1) = p(s_1) p(y_1 \mid s_1)
     $$

   - **Recursion:**

     $$
     \alpha_t(s_t) = \left( \sum_{s_{t-1}} \alpha_{t-1}(s_{t-1}) p(s_t \mid s_{t-1}) \right) p(y_t \mid s_t)
     $$

2. **Backward Pass:**

   Compute $\beta_t(s_t) = p(y_{t+1:T} \mid s_t)$.

   - **Initialization:**

     $$
     \beta_T(s_T) = 1
     $$

   - **Recursion:**

     $$
     \beta_t(s_t) = \sum_{s_{t+1}} p(s_{t+1} \mid s_t) p(y_{t+1} \mid s_{t+1}) \beta_{t+1}(s_{t+1})
     $$

3. **Compute Marginal Posteriors:**

   $$
   p(s_t \mid y_{1:T}) = \frac{\alpha_t(s_t) \beta_t(s_t)}{\sum_{s_t} \alpha_t(s_t) \beta_t(s_t)}
   $$

---

#### (c) EM Algorithm for HMM Parameter Estimation

**E-Step:**

1. Compute $\gamma_t(s_t) = p(s_t \mid y_{1:T})$.

2. Compute $\xi_t(s_{t-1}, s_t) = p(s_{t-1}, s_t \mid y_{1:T})$.

**M-Step:**

1. **Update Transition Probabilities:**

   $$
   p(s_t \mid s_{t-1}) = \frac{\sum_{t=2}^T \xi_t(s_{t-1}, s_t)}{\sum_{t=2}^T \gamma_{t-1}(s_{t-1})}
   $$

2. **Update Emission Parameters $w_k$ and $\sigma^2$:**

   - For $w_k$:

     $$
     w_k = \left( \sum_{t=1}^T \gamma_t(k) x_t x_t^\top \right)^{-1} \left( \sum_{t=1}^T \gamma_t(k) x_t y_t \right)
     $$

   - For $\sigma^2$:

     $$
     \sigma^2 = \frac{\sum_{t=1}^T \sum_{k=1}^K \gamma_t(k) (y_t - w_k^\top x_t)^2}{\sum_{t=1}^T \sum_{k=1}^K \gamma_t(k)}
     $$

---

