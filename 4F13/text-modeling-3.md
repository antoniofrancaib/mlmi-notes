# Gibbs Sampling for Bayesian Mixture Models and Latent Dirichlet Allocation (LDA) for Topic Modeling

## Introduction
In the realm of probabilistic modeling, understanding how to infer latent structures in data is crucial. Mixture models and topic models like Latent Dirichlet Allocation (LDA) are powerful tools for uncovering hidden patterns in complex datasets, such as collections of documents. This masterclass will delve into the Bayesian mixture model for documents, the derivation of Gibbs sampling for parameter estimation, and the extension to LDA for topic modeling. We will explore the limitations of simple mixture models, motivate the need for more sophisticated approaches like LDA, and thoroughly explain the concepts and mathematical foundations underlying these models.

## Part 1: Bayesian Mixture Models for Documents

### 1.1 Overview of the Bayesian Document Mixture Model

#### Model Components
- **Observations ($w_d$):** The words in document $d$, where $d=1,2,\dots,D$.
- **Parameters ($\beta_k$ and $\pi$):**
  - $\beta_k$: The parameters of the categorical distribution over words for component $k$, with prior $p(\beta_k)$.
  - $\pi$: The mixing proportions (mixture weights) for the components, with prior $p(\pi)$.
- **Latent Variables ($z_d$):** The component assignments for each document $d$, where $z_d \in \{1,2,\dots,K\}$.

#### Generative Process
1. **Mixing Proportions:**
   Draw mixing proportions $\pi$ from a Dirichlet prior:
   $$\pi \sim \text{Dirichlet}(\alpha).$$
2. **Component Parameters:**
   For each component $k=1,2,\dots,K$, draw word distribution parameters $\beta_k$ from a Dirichlet prior:
   $$\beta_k \sim \text{Dirichlet}(\eta).$$
3. **Document Assignments:**
   For each document $d$, assign it to a component $z_d$ by sampling from the categorical distribution:
   $$z_d \sim \text{Categorical}(\pi).$$
4. **Word Generation:**
   For each word position $n$ in document $d$, sample word $w_{dn}$ from the categorical distribution corresponding to component $z_d$:
   $$w_{dn} \sim \text{Categorical}(\beta_{z_d}).$$

#### Model Representation
The model can be visually represented with a graphical model where:
- **Nodes:** Random variables (observed and latent).
- **Edges:** Dependencies between variables.
- **Plates:** Repetition over indices (documents, words, components).

### 1.2 Gibbs Sampling for Bayesian Mixture Models
Gibbs sampling is a Markov Chain Monte Carlo (MCMC) method used to approximate the posterior distribution of complex models by iteratively sampling from conditional distributions.

#### Conditional Distributions
1. **Sampling Component Parameters ($\beta_k$):**
   Given the component assignments $z$ and word counts $c_{mk}$ for component $k$:
   $$p(\beta_k \mid w, z) \propto p(\beta_k) \prod_{d: z_d=k} p(w_d \mid \beta_k).$$
   Since the prior $p(\beta_k)$ is Dirichlet and the likelihood is multinomial, the posterior is also Dirichlet:
   $$\beta_k \mid w, z \sim \text{Dirichlet}(\eta + c_{mk}),$$
   where $c_{mk}$ is the count of word $m$ in component $k$.

2. **Sampling Mixing Proportions ($\pi$):**
   Given the component assignments $z$ and counts $c_k$:
   $$p(\pi \mid z, \alpha) \propto p(\pi \mid \alpha) p(z \mid \pi).$$
   The posterior is:
   $$\pi \mid z, \alpha \sim \text{Dirichlet}(\alpha + c_k),$$
   where $c_k$ is the number of documents assigned to component $k$.

3. **Sampling Component Assignments ($z_d$):**
   For each document $d$, sample $z_d$ from:
   $$p(z_d = k \mid w_d, \pi, \beta) \propto \pi_k p(w_d \mid \beta_k).$$

#### Gibbs Sampling Algorithm
Iteratively perform the following steps:
1. **Sample $\beta_k$** for each component $k$:
   Use the posterior Dirichlet distribution with updated counts.
2. **Sample $\pi$:**
   Use the posterior Dirichlet distribution with updated counts of component assignments.
3. **Sample $z_d$** for each document $d$:
   Calculate the probabilities proportional to $\pi_k p(w_d \mid \beta_k)$ and sample $z_d$ from the categorical distribution.

### 1.3 Collapsed Gibbs Sampling
#### Motivation
- **Collapsing:** Integrate out the mixing proportions $\pi$ to reduce the number of parameters and dependencies, leading to faster convergence and reduced variance in the estimates.
- **Result:** The component assignments $z_d$ become dependent, but the sampling can be more efficient.

#### Derivation
1. **Marginalize Over $\pi$:**
   Compute the integrated probability:
   $$p(z_d = k \mid z_{-d}, \alpha) = \int p(z_d = k \mid \pi) p(\pi \mid z_{-d}, \alpha) d\pi.$$
2. **Posterior of $\pi$:**
   Given $z_{-d}$ (all component assignments except for document $d$), the posterior over $\pi$ is:
   $$\pi \mid z_{-d}, \alpha \sim \text{Dirichlet}(\alpha + c_{-d}),$$
   where $c_{-d}$ is the count vector excluding document $d$.

3. **Computing the Integrated Probability:**
   Using properties of the Dirichlet distribution:
   $$p(z_d = k \mid z_{-d}, \alpha) = \frac{\alpha_k + c_{-d,k}}{\sum_{j=1}^K \alpha_j + c_{-d,j}}.$$

#### Collapsed Gibbs Sampling Steps
1. **Sample $z_d$** for each document $d$:
   Compute:
   $$p(z_d = k \mid w_d, z_{-d}, \beta, \alpha) \propto (\alpha_k + c_{-d,k}) p(w_d \mid \beta_k).$$
2. **Sample $\beta_k$** for each component $k$:
   Use the posterior Dirichlet distribution with updated counts.

### 1.4 Properties of the Gibbs Sampler
1. **Dependency:** In the collapsed Gibbs sampler, the component assignments $z_d$ become dependent because they share the common mixing proportions $\pi$ that have been marginalized out.
2. **Rich Get Richer:** Components with more assignments tend to attract more assignments due to the term $\alpha_k + c_{-d,k}$ in the sampling probabilities.
3. **Convergence:** Collapsed Gibbs sampling often leads to faster convergence and more accurate estimates compared to the standard Gibbs sampler with explicit mixing proportions.

---

## Part 2: Latent Dirichlet Allocation (LDA) for Topic Modeling

### 2.1 Limitations of the Mixture of Categoricals Model

#### Model Overview
- In the mixture of categoricals model:
  - Each document $d$ is assigned to a single topic $z_d$.
  - All words $w_{dn}$ in document $d$ are drawn from the word distribution $\beta_{z_d}$ of topic $z_d$.

#### Limitations
1. **Single Topic Assumption:** Assumes each document is exclusively about one topic, which is unrealistic for documents covering multiple topics.
2. **Blurred Topics:** When documents span multiple topics, the model tends to learn topics that are a blend of multiple true topics, reducing interpretability.

### 2.2 Motivation for LDA
Latent Dirichlet Allocation (LDA) addresses the limitations by allowing documents to exhibit multiple topics in varying proportions.
- **Flexibility:** Documents can be composed of multiple topics, with each word potentially drawn from a different topic.
- **Interpretability:** Topics are more coherent and distinct, improving the quality of topic modeling.

### 2.3 Intuition Behind LDA
1. **Documents as Mixtures:** Each document is represented as a mixture of topics, characterized by a topic proportion vector $\theta_d$.
2. **Words from Topics:** Each word in a document is generated by first selecting a topic according to $\theta_d$, then sampling a word from the corresponding topic's word distribution $\beta_k$.

### 2.4 Generative Model for LDA

#### Generative Process
1. **Global Parameters:**
   - For each topic $k$, draw word distribution $\beta_k$ from a Dirichlet prior:
     $$\beta_k \sim \text{Dirichlet}(\eta).$$
2. **Document-Level Parameters:**
   - For each document $d$, draw topic proportions $\theta_d$ from a Dirichlet prior:
     $$\theta_d \sim \text{Dirichlet}(\alpha).$$
3. **Word-Level Generation:**
   - For each word position $n$ in document $d$:
     - Draw a topic assignment $z_{dn}$ from the categorical distribution:
       $$z_{dn} \sim \text{Categorical}(\theta_d).$$
     - Draw a word $w_{dn}$ from the topic's word distribution:
       $$w_{dn} \sim \text{Categorical}(\beta_{z_{dn}}).$$

### 2.5 LDA Graphical Model
In the graphical model:
- **Nodes:**
  - $\beta_k$: Word distributions for topics (parameters).
  - $\theta_d$: Topic proportions for documents (latent variables).
  - $z_{dn}$: Topic assignments for words (latent variables).
  - $w_{dn}$: Observed words.
- **Edges:** Dependencies between variables.
- **Plates:** Indicate replication over documents $d$, words $n$, and topics $k$.

### 2.6 Differences Between Mixture of Categoricals and LDA
1. **Per-Document Topic Proportions:**
   - Mixture Model: Single topic per document ($\theta_d$ is a one-hot vector).
   - LDA: Multiple topics per document ($\theta_d$ is a probability distribution over topics).
2. **Word Assignments:**
   - Mixture Model: All words in a document are from the same topic.
   - LDA: Each word can be assigned to different topics within the same document.

### 2.7 The Intractability of Exact Inference in LDA

#### Computational Challenge
- **Posterior Distribution:** Calculating $p(\beta, \theta, z \mid w)$ requires integrating over all possible configurations of $z$ and $\theta$.
- **Exponential Complexity:** The number of possible topic assignments $z$ grows exponentially with the number of words, making exact inference computationally infeasible.

### 2.8 Gibbs Sampling for LDA

#### Collapsed Gibbs Sampling
To perform approximate inference, we can use Gibbs sampling by integrating out $\beta$ and $\theta$.

1. **Marginalize Over $\beta$ and $\theta$:**
   - The integration over Dirichlet-distributed variables leads to simplified expressions due to conjugacy with the categorical distributions.

2. **Sampling Topic Assignments ($z_{dn}$):**
   - For each word $w_{dn}$, sample $z_{dn}$ from:
     $$p(z_{dn} = k \mid z_{-dn}, w, \alpha, \eta) \propto (\alpha_k + c_{d,k}^{-dn}) \frac{\eta_{w_{dn}} + c_{k, w_{dn}}^{-dn}}{\sum_w (\eta_w + c_{k,w}^{-dn})},$$
     where:
     - $c_{d,k}^{-dn}$: Count of words in document $d$ assigned to topic $k$, excluding position $n$.
     - $c_{k,w_{dn}}^{-dn}$: Count of word $w_{dn}$ assigned to topic $k$, excluding position $n$.
     - $z_{-dn}$: All topic assignments excluding $z_{dn}$.

#### Derivation Steps
1. **Compute Conditional Distributions:**
   - Use the properties of the Dirichlet and multinomial distributions to derive the conditional probabilities.
2. **Update Counts:**
   - Update the counts $c_{d,k}$ and $c_{k,w}$ after each sampling step.

### 2.9 Per-Word Perplexity

#### Definition
Perplexity is a measure of how well a probabilistic model predicts a sample. In the context of language models:
$$\text{Perplexity} = \exp\left(-\frac{1}{N} \sum_{d=1}^D \log p(w_d)\right),$$
where $N$ is the total number of words.

#### Interpretation
1. **Lower Perplexity:** Indicates a better fit to the data (model is more certain about word predictions).
2. **Perplexity of $g$:** Equivalent to the uncertainty associated with a $g$-sided die generating each word.

#### Example Calculation
- Suppose we have a uniform distribution over 6 words (like a fair die):
  - The probability of any word is $\frac{1}{6}$.
  - For a sequence of 4 words, the joint probability is $\left(\frac{1}{6}\right)^4$.
  - The per-word log probability is $-\log 6$.
  - Perplexity is $\exp\left(-\frac{1}{4} \times 4 \times \log 6\right) = 6$.

---

## Conclusion
Understanding Gibbs sampling and its application to Bayesian mixture models and LDA is essential for advanced probabilistic modeling. By carefully deriving the conditional distributions and recognizing the role of conjugate priors, we can implement efficient sampling algorithms for complex models. LDA extends mixture models by allowing documents to exhibit multiple topics, providing a more flexible and realistic representation of textual data. Despite the computational challenges, techniques like collapsed Gibbs sampling enable us to perform approximate inference and uncover latent structures in large datasets.

### Key Takeaways
1. **Bayesian Mixture Models:** Utilize prior distributions and latent variables to model data with underlying group structures.
2. **Gibbs Sampling:** An iterative method for sampling from high-dimensional probability distributions when direct sampling is challenging.
3. **Collapsed Gibbs Sampling:** Improves efficiency by integrating out certain parameters, reducing dependencies and variance.
4. **Latent Dirichlet Allocation:** A probabilistic model for topic modeling that accounts for multiple topics per document.
5. **Inference Challenges:** Exact inference in models like LDA is intractable due to the combinatorial explosion of latent variable configurations.
6. **Perplexity:** A metric for evaluating language models, reflecting the model's uncertainty in predicting the data.

By mastering these concepts, you will be equipped to apply advanced probabilistic models to complex datasets, interpret the results meaningfully, and contribute to the development of sophisticated machine learning algorithms.
