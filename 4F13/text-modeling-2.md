- [36-Document-Models ](#36-Document-Models)
- [37-Expectation-Maximization-Algorithm ](#37-Expectation-Maximization-Algorithm)

# 36-Document-Models 

## A Simple Document Model

In the simplest form, we can model a collection of documents by considering each word in the documents as being drawn independently from a fixed vocabulary according to some probability distribution.

### Notation and Definitions
- **Vocabulary Size ($M$):** The number of unique words in our vocabulary.
- **Number of Documents ($D$):** The total number of documents in our collection.
- **Number of Words in Document $d$ ($N_d$):** Each document may contain a different number of words.
- **Word Index:** Each word in the vocabulary is assigned an index from $1$ to $M$.
- **Word in Document $d$ at Position $n$ ($w_{dn}$):** The $n$-th word in document $d$, where $w_{dn} \in \{1, 2, ..., M\}$.
- **Word Distribution ($\beta$):** A vector of probabilities $\beta = [\beta_1, \beta_2, ..., \beta_M]^T$, where $\beta_m$ is the probability of word $m$ being selected.
![[text-simpl.png]]

### The Categorical Distribution
The categorical distribution is a discrete probability distribution that describes the probability of a single trial resulting in one of $M$ possible outcomes.

**Probability Mass Function (PMF):**
$$P(w_{dn} = m \mid \beta) = \beta_m$$

**Properties:**
1. $\beta_m \geq 0$ for all $m$.
2. $\sum_{m=1}^M \beta_m = 1$.

### Generative Process
Under this model, each word $w_{dn}$ in document $d$ is generated independently by:
1. **Sampling a Word:** For each position $n$ in document $d$, sample $w_{dn}$ from the categorical distribution $\text{Cat}(\beta)$.

### Maximum Likelihood Estimation (MLE)
To estimate the word distribution $\beta$ from data, we use Maximum Likelihood Estimation.

#### Likelihood Function
The likelihood of observing the entire collection of documents $W = \{w_{dn}\}$ given $\beta$ is:
$$L(\beta; W) = \prod_{d=1}^D \prod_{n=1}^{N_d} P(w_{dn} \mid \beta) = \prod_{d=1}^D \prod_{n=1}^{N_d} \beta_{w_{dn}}$$
We can fit $\beta$ by maximising the likelihood: 
$$\hat{\beta} = \arg\max_{\beta} \prod_{d=1}^{D} \prod_{n=1}^{N_d} \text{Cat}(w_{nd}|\beta) = \arg\max_{\beta} \text{Mult}(c_1, \dots, c_M | \beta, N)$$

#### Log-Likelihood
Taking the logarithm simplifies the product into a sum:
$$\log L(\beta; W) = \sum_{d=1}^D \sum_{n=1}^{N_d} \log \beta_{w_{dn}}$$

#### Sufficient Statistics
Define $c_m$ as the total count of word $m$ across all documents:
$$c_m = \sum_{d=1}^D \sum_{n=1}^{N_d} \delta(w_{dn}, m)$$
where $\delta(a, b)$ is the Kronecker delta function, which is $1$ if $a = b$ and $0$ otherwise.

Then, the log-likelihood becomes:
$$\log L(\beta; W) = \sum_{m=1}^M c_m \log \beta_m$$

#### Constraint
Since $\beta$ is a probability distribution, the parameters must satisfy:
$$\sum_{m=1}^M \beta_m = 1$$

#### Optimization Using Lagrange Multipliers
To maximize the log-likelihood under the constraint, we set up the Lagrangian $F$:
$$F(\beta, \lambda) = \sum_{m=1}^M c_m \log \beta_m + \lambda \left(1 - \sum_{m=1}^M \beta_m\right)$$

Taking the derivative of $F$ with respect to $\beta_m$ and setting it to zero:
$$\frac{\partial F}{\partial \beta_m} = \frac{c_m}{\beta_m} - \lambda = 0 \implies \beta_m = \frac{c_m}{\lambda}$$

Using the constraint $\sum_{m=1}^M \beta_m = 1$:
$$\lambda = \sum_{m=1}^M c_m = N$$
where $N$ is the total number of words across all documents.

Therefore, the Maximum Likelihood Estimate (MLE) of $\beta_m$ is:
$$\hat{\beta}_m = \frac{c_m}{N}$$

#### Interpretation
The estimated probability $\hat{\beta}_m$ is the relative frequency of word $m$ in the entire collection.

### Limitations of the Simple Model
- **Lack of Specialization:** The model does not account for differences between documents. All documents are assumed to have the same distribution over words.
- **No Topic Modeling:** There is no mechanism to capture different topics or categories that might be present in the document collection.
- **Assumption of Independence:** Words are assumed to be independent, which ignores syntactic and semantic relationships.

## Mixture Models for Documents

### Motivation
To overcome the limitations of the simple model, we introduce a mixture model that allows for documents to belong to different categories or topics, each with its own word distribution.

### Generative Process
1. **Document Category Assignment:**  
   For each document $d$, assign it to a category $z_d$ by sampling from a categorical distribution:
   $$z_d \sim \text{Cat}(\pi)$$
   where $\pi = [\pi_1, \pi_2, ..., \pi_K]^T$ and $\pi_k = P(z_d = k)$ is the probability of a document belonging to category $k$.

2. **Word Generation:**  
   For each word position $n$ in document $d$, sample $w_{dn}$ from the categorical distribution corresponding to category $z_d$:
   $$w_{dn} \sim \text{Cat}(\beta_{z_d})$$
   where $\beta_{z_d}$ is the word distribution for category $z_d$.

![[word-doc.png]]

### Model Parameters
- **Category Probabilities ($\pi$):** Parameters of the categorical distribution over document categories.
- **Word Distributions ($\beta_k$):** For each category $k$, $\beta_k = [\beta_{k1}, \beta_{k2}, ..., \beta_{kM}]^T$ represents the probability distribution over words.

### Latent Variables
- **Document Category ($z_d$):** A hidden variable indicating the category of document $d$.  
  **Purpose:** Capturing the idea that different documents may discuss different topics, each with its own characteristic word distribution.

### Likelihood Function
The likelihood of observing the entire collection $W$ given the parameters $\pi$ and $\{\beta_k\}$ is:
$$P(W \mid \pi, \{\beta_k\}) = \prod_{d=1}^D P(w_d \mid \pi, \{\beta_k\}) = \prod_{d=1}^D \sum_{k=1}^K P(z_d = k \mid \pi) P(w_d \mid z_d = k, \beta_k)$$
where:
$$P(w_d \mid z_d = k, \beta_k) = \prod_{n=1}^{N_d} P(w_{dn} \mid \beta_k) = \prod_{n=1}^{N_d} \beta_{k w_{dn}}$$

### Challenges
- **Latent Variables:** The document categories $\{z_d\}$ are not observed.
- **Nonlinear Optimization:** The presence of the sum over $k$ inside the product over $d$ complicates direct maximization of the likelihood.

## Fitting the Mixture Model with the EM Algorithm

The Expectation-Maximization (EM) algorithm is an iterative method used for finding Maximum Likelihood Estimates (MLE) in models with latent variables.

1. **E-Step (Expectation):** Estimate the posterior distribution of the latent variables given the current parameters.
2. **M-Step (Maximization):** Maximize the expected log-likelihood with respect to the parameters, using the estimated posterior from the E-Step.

### Applying EM to the Mixture Model

#### E-Step
For each document $d$, compute the posterior probability (responsibility) that it belongs to category $k$:
$$r_{kd} = P(z_d = k \mid w_d, \pi, \{\beta_k\}) = \frac{\pi_k P(w_d \mid \beta_k)}{\sum_{k'=1}^K \pi_{k'} P(w_d \mid \beta_{k'})}$$
where:
$$P(w_d \mid \beta_k) = \prod_{n=1}^{N_d} \beta_{k w_{dn}} = \text{Mult}(\{c_{md}\} \mid \beta_k, N_d)$$
and $c_{md}$ is the count of word $m$ in document $d$.  
$\text{Mult}(\{c_{md}\} \mid \beta_k, N_d)$ is the multinomial probability of the word counts in document $d$ under category $k$.

#### M-Step
Update the parameters $\pi$ and $\{\beta_k\}$ by maximizing the expected log-likelihood:
$$Q(\pi, \{\beta_k\}) = \sum_{d=1}^D \sum_{k=1}^K r_{kd} \left( \log \pi_k + \sum_{m=1}^M c_{md} \log \beta_{km} \right)$$
**Subject to the constraints:**
$$\sum_{k=1}^K \pi_k = 1, \quad \sum_{m=1}^M \beta_{km} = 1 \quad \forall k$$

**Updating $\pi_k$:**
$$\pi_k^{\text{new}} = \frac{\sum_{d=1}^D r_{kd}}{D}$$

**Updating $\beta_{km}$:**
$$\beta_{km}^{\text{new}} = \frac{\sum_{d=1}^D r_{kd} c_{md}}{\sum_{d=1}^D r_{kd} N_d}$$

#### Derivation Using Lagrange Multipliers
To maximize $Q$ under the constraints, we set up Lagrangian functions.

For $\pi$:
$$L(\pi, \lambda) = \sum_{k=1}^K \left( \sum_{d=1}^D r_{kd} \log \pi_k \right) + \lambda \left( 1 - \sum_{k=1}^K \pi_k \right)$$

Taking derivatives and setting to zero:
$$\frac{\partial L}{\partial \pi_k} = \frac{\sum_{d=1}^D r_{kd}}{\pi_k} - \lambda = 0 \implies \pi_k = \frac{\sum_{d=1}^D r_{kd}}{\lambda}$$

Using the constraint $\sum_{k=1}^K \pi_k = 1$:
$$\lambda = \sum_{k=1}^K \sum_{d=1}^D r_{kd} = D$$
Thus:
$$\pi_k^{\text{new}} = \frac{\sum_{d=1}^D r_{kd}}{D}$$

For $\beta_k$:
$$L(\beta_k, \mu_k) = \sum_{m=1}^M \left( \sum_{d=1}^D r_{kd} c_{md} \log \beta_{km} \right) + \mu_k \left( 1 - \sum_{m=1}^M \beta_{km} \right)$$

Taking derivatives and setting to zero:
$$\frac{\partial L}{\partial \beta_{km}} = \frac{\sum_{d=1}^D r_{kd} c_{md}}{\beta_{km}} - \mu_k = 0 \implies \beta_{km} = \frac{\sum_{d=1}^D r_{kd} c_{md}}{\mu_k}$$

Using the constraint $\sum_{m=1}^M \beta_{km} = 1$:
$$\mu_k = \sum_{m=1}^M \sum_{d=1}^D r_{kd} c_{md} = \sum_{d=1}^D r_{kd} N_d$$
Thus:
$$\beta_{km}^{\text{new}} = \frac{\sum_{d=1}^D r_{kd} c_{md}}{\sum_{d=1}^D r_{kd} N_d}$$

---

### 3.3 Interpretation
- **E-Step:** Calculates the expected assignment of documents to categories based on current parameter estimates.
- **M-Step:** Updates the parameter estimates to maximize the likelihood, weighted by the expected assignments.

### 3.4 Convergence
- The EM algorithm is guaranteed to not decrease the likelihood at each iteration.
- It converges to a local maximum of the likelihood function.

## 4. Bayesian Mixture of Categoricals Model

### 4.1 Motivation
The EM algorithm provides point estimates of the parameters $\pi$ and $\{\beta_k\}$. A Bayesian approach introduces prior distributions over the parameters, allowing us to incorporate prior beliefs and quantify uncertainty.

### 4.2 Priors
- **Dirichlet Prior for $\pi$:**
  $$\pi \sim \text{Dir}(\alpha)$$
  where $\alpha = [\alpha_1, \alpha_2, ..., \alpha_K]^T$ are the concentration parameters.

- **Dirichlet Prior for $\beta_k$:**
  $$\beta_k \sim \text{Dir}(\gamma)$$
  where $\gamma = [\gamma_1, \gamma_2, ..., \gamma_M]^T$.

### 4.3 Bayesian Inference
- **Posterior Distributions:** Instead of point estimates, we aim to compute the posterior distributions:
  $$P(\pi, \{\beta_k\} \mid W)$$
- **Inference Methods:** Exact inference is often intractable; we may use approximate methods such as Variational Inference or Markov Chain Monte Carlo (MCMC).

### 4.4 Benefits
- **Uncertainty Quantification:** Provides a measure of confidence in the parameter estimates.
- **Regularization:** The priors can prevent overfitting, especially in cases with limited data.
- **Incorporation of Prior Knowledge:** Prior beliefs about the distribution of topics or word frequencies can be included.

---

## 37-Expectation-Maximization-Algorithm

### 5.1 Overview
The EM algorithm is a general technique for finding maximum likelihood estimates in models with latent variables.

### 5.2 Notation
- **Observed Data ($y$):** The data we can observe directly.
- **Latent Variables ($z$):** Hidden variables that are not directly observed.
- **Parameters ($\theta$):** The parameters of the model we wish to estimate.
- **Complete Data:** The combination of observed data and latent variables $(y, z)$.
- **Likelihood:**
  $$p(y \mid \theta) = \int p(y, z \mid \theta) \, dz$$

### 5.3 The EM Algorithm Steps

#### 5.3.1 E-Step (Expectation Step)
Compute the expected value of the log-likelihood with respect to the current estimate of the distribution over the latent variables:
$$Q(\theta \mid \theta^{(t)}) = \mathbb{E}_{z \mid y, \theta^{(t)}}[\log p(y, z \mid \theta)]$$
where $\theta^{(t)}$ is the estimate of the parameters at iteration $t$, and the expectation is taken over the posterior distribution of $z$ given $y$ and $\theta^{(t)}$.

#### 5.3.2 M-Step (Maximization Step)
Maximize $Q(\theta \mid \theta^{(t)})$ with respect to $\theta$:
$$\theta^{(t+1)} = \arg \max_\theta Q(\theta \mid \theta^{(t)})$$

### 5.4 Derivation Using the Lower Bound

#### 5.4.1 Jensen's Inequality
The log-likelihood can be decomposed as:
$$\log p(y \mid \theta) = F(q, \theta) + \text{KL}(q(z) \parallel p(z \mid y, \theta))$$
where:
- $$F(q, \theta) = \int q(z) \log \left( \frac{p(y, z \mid \theta)}{q(z)} \right) \, dz$$
- $\text{KL}(q(z) \parallel p(z \mid y, \theta))$ is the Kullback-Leibler divergence, which is non-negative.

#### 5.4.2 Lower Bound Maximization
- **E-Step:** Choose $q(z) = p(z \mid y, \theta^{(t)})$ to minimize the KL divergence.
- **M-Step:** Maximize $F(q, \theta)$ with respect to $\theta$.

### 5.5 EM as Coordinate Ascent
The EM algorithm can be viewed as coordinate ascent in the space of $q(z)$ and $\theta$, iteratively improving the lower bound $F(q, \theta)$.

---

## 6. Example: Gaussian Mixture Model (GMM)

### 6.1 Model Definition
- **Observations ($y_i$):** Real-valued data points.
- **Latent Variables ($z_i$):** Indicate which Gaussian component generated $y_i$, where $z_i \in \{1, 2, ..., K\}$.

**Parameters:**
- Mixing coefficients: $\pi_j = P(z_i = j)$
- Means: $\mu_j$
- Variances: $\sigma_j^2$

### 6.2 Generative Process
For each data point $i$:
1. Sample $z_i \sim \text{Cat}(\pi)$.
2. Sample $y_i \sim \mathcal{N}(\mu_{z_i}, \sigma_{z_i}^2)$.

### 6.3 Applying EM to GMM

#### 6.3.1 E-Step
Compute the responsibilities $r_{ij}$:
$$r_{ij} = P(z_i = j \mid y_i, \theta^{(t)}) = \frac{\pi_j^{(t)} \mathcal{N}(y_i \mid \mu_j^{(t)}, \sigma_j^{2(t)})}{\sum_{k=1}^K \pi_k^{(t)} \mathcal{N}(y_i \mid \mu_k^{(t)}, \sigma_k^{2(t)})}$$

#### 6.3.2 M-Step
Update the parameters using the responsibilities:

- **Update Mixing Coefficients:**
  $$\pi_j^{(t+1)} = \frac{1}{N} \sum_{i=1}^N r_{ij}$$

- **Update Means:**
  $$\mu_j^{(t+1)} = \frac{\sum_{i=1}^N r_{ij} y_i}{\sum_{i=1}^N r_{ij}}$$

- **Update Variances:**
  $$\sigma_j^{2(t+1)} = \frac{\sum_{i=1}^N r_{ij} (y_i - \mu_j^{(t+1)})^2}{\sum_{i=1}^N r_{ij}}$$

### 6.4 Interpretation
- **Responsibilities:** Measure the probability that data point $i$ was generated by component $j$.
- **Parameter Updates:** Weighted averages using the responsibilities as weights.

---

## 7. Appendix: Kullback-Leibler (KL) Divergence

### 7.1 Definition
The KL divergence between two probability distributions $q(x)$ and $p(x)$ is defined as:
$$\text{KL}(q(x) \parallel p(x)) = \int q(x) \log \left( \frac{q(x)}{p(x)} \right) \, dx$$

### 7.2 Properties
1. **Non-negativity:** $\text{KL}(q(x) \parallel p(x)) \geq 0$.
2. **Zero Minimum:** $\text{KL}(q(x) \parallel p(x)) = 0$ if and only if $q(x) = p(x)$ almost everywhere.
3. **Asymmetry:** $\text{KL}(q(x) \parallel p(x)) \neq \text{KL}(p(x) \parallel q(x))$ in general.

### 7.3 Role in EM Algorithm
The KL divergence measures how close the approximate distribution $q(z)$ is to the true posterior $p(z \mid y, \theta)$. Minimizing the KL divergence in the E-Step ensures that $q(z)$ is the best approximation to the true posterior given the current parameters.

---

## Conclusion
Understanding document models and the EM algorithm provides a foundation for advanced topics in machine learning and NLP, such as topic modeling (e.g., Latent Dirichlet Allocation), clustering, and hidden Markov models. By modeling documents with mixture models and applying the EM algorithm, we can uncover latent structures within data, leading to insights and improvements in various applications.

### Key Takeaways
1. **Simple Document Models:** Provide a baseline but lack the ability to capture document-specific topics.
2. **Mixture Models:** Introduce latent variables to model documents belonging to different categories, each with its own word distribution.
3. **EM Algorithm:** An iterative method to estimate parameters in models with latent variables by alternating between inferring the latent variables (E-Step) and optimizing the parameters (M-Step).
4. **Bayesian Approach:** Incorporates prior knowledge and uncertainty by placing priors over parameters and inferring posterior distributions.

