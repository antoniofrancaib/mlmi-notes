# Index
- [Overview](#overview)
- [The K-means Algorithm](#the-k-means-algorithm-deterministic-approach)
- [Mixture of Gaussians and the EM Algorithm](#mixture-of-gaussians-probabilistic-approach-and-the-expectation-maximisation-algorithm)

# Overview

Clustering $\rightarrow$ grouping data points into clusters.  
	a dataset of $D$-dimensional points, $\mathbf{x}_n$, the goal is to assign each point to one of $K$ clusters, denoted by $s_n$ -based on some defined similarity measure-.

**unsupervised learning** task = only the input data ${\mathbf{x}_n}$ is provided. 
Goal: to uncover hidden structure in the data without explicit output labels.

### Examples:

|Application|Data|Clusters|
|---|---|---|
|Genetic analysis|Genetic markers|Ancestral groups|
|Medical analysis|Patient records and data|Disease subtypes|
|Image segmentation|Image pixel values|Distinct image regions|
|Social network analysis|Node connections|Social communities|

---
# The K-means Algorithm (deterministic approach)

Given a dataset $\{\mathbf{x}_n\}_{n=1}^N$ of two-dimensional real-valued data points $\mathbf{x}_n = [x_{1,n}, x_{2,n}]^\top$, we aim to cluster the points into $K$ clusters using the K-means algorithm. The algorithm assigns each datapoint to one of $K$ clusters with centers $\{\mathbf{m}_k\}_{k=1}^K$.

### Assignment Representation
- **Simplified notation:** $s_n = k$, where $n^{\text{th}}$ datapoint belongs to the $k^{\text{th}}$ cluster.
- **Indicator notation:** $s_{n,k} = 1$ if the $n^{\text{th}}$ datapoint belongs to cluster $k$, and $s_{n,k} = 0$ otherwise. This creates an $N \times K$ matrix $S$ representing cluster assignments.

For example, a dataset of four points assigned to three clusters might have the assignment matrix:
$$
S = \left[ \begin{array}{ccc}
0 & 0 & 1\\
1 & 0 & 0\\
0 & 1 & 0\\
0 & 1 & 0 
\end{array} \right]
$$
This shows the first datapoint belongs to cluster 3, the second to cluster 1, and the third and fourth to cluster 2.

### Cost Function
K-means minimizes the following cost function:
$$
\mathcal{C} = \sum_{n=1}^N \sum_{k=1}^K s_{n,k} \lvert \lvert \mathbf{x}_n - \mathbf{m}_k \rvert \rvert^2
$$
The K-means algorithm minimizes an energy function called the **within-cluster sum of squares** (WCSS), also referred to as the **inertia** or the **distortion**.
### Optimization Process
The optimization alternates between two steps:

1. **Cluster Assignment:** For each datapoint $n$, assign it to the cluster with the nearest center:
   $$
   s_{nk} = 1 \text{ for } k = \text{arg} \min_k \lvert \lvert \mathbf{x}_n - \mathbf{m}_k \rvert \rvert^2, \text{ and } s_{nk} = 0 \text{ otherwise.}
   $$

2. **Cluster Center Update:** For each cluster, update its center to the mean of the points assigned to it:$$
   \mathbf{m}_k = \frac{1}{N_k} \sum_{s_{nk}=1} \mathbf{x}_n
   $$
   where $N_k = \sum_n s_{nk}$ is the number of points in cluster $k$.

These steps are repeated until the cluster assignments $\{s_{nk}\}$ no longer change.

### Convergence and Initialization
- **Convergence:** K-means is guaranteed to converge because it performs coordinate descent on the cost function $\mathcal{C}$, which is a Lyapunov function, ensuring that the cost either decreases or remains constant after each step.
- **Local Minima:** K-means may converge to a local minimum rather than a global one, depending on the initial cluster centers. Finding the global optimum is NP-hard.
- **Initialization:** Initial cluster centers are important. The **K-means++** algorithm provides a robust initialization by selecting centers that are spread out, improving the final clustering result.

---
# Mixture of Gaussians (probabilistic approach) and the Expectation Maximisation Algorithm

## Introduction
- **K-means Limitations**:
  - **Anisotropic Clusters**: K-means assumes spherical clusters with equal variance, which fails for elongated or differently scaled clusters.
  - **Hard Assignments**: Each data point is assigned definitively to one cluster, disregarding the uncertainty or probability of belonging to other clusters.
- **Mixture of Gaussians (MoG)**:
  - **Flexible Cluster Shapes**: MoG can model clusters with different shapes and sizes by using individual covariance matrices.
  - **Probabilistic Assignments**: Instead of hard assignments, MoG assigns probabilities to each cluster, reflecting the uncertainty in cluster membership.

## Generative Model for MoG
The Mixture of Gaussians model is a probabilistic model that assumes data is generated from a mixture of several Gaussian distributions.

1. **Cluster Assignment**:
   - Each data point $\mathbf{x}_n$ is associated with a latent variable $s_n$, indicating its cluster membership.
   - There are $K$ clusters, and the membership $s_n = k$ is sampled from a categorical distribution:
     $$
     p(s_n = k \mid \{\pi\}) = \pi_k
     $$
     where $\pi_k$ is the prior probability of cluster $k$, and $\sum_{k=1}^K \pi_k = 1$.

2. **Data Generation**:
   - Given $s_n = k$, the data point $\mathbf{x}_n$ is sampled from a multivariate Gaussian distribution:
     $$
     p(\mathbf{x}_n \mid s_n = k) = \mathcal{N}(\mathbf{x}_n; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
     $$
     where:
     - $\boldsymbol{\mu}_k$ is the mean vector of cluster $k$.
     - $\boldsymbol{\Sigma}_k$ is the covariance matrix of cluster $k$.
     - $\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})$ denotes the multivariate Gaussian distribution:$$
       \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{\sqrt{(2\pi)^D |\boldsymbol{\Sigma}|}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
       $$where $D$ is the dimensionality of the data.

3. **Generative Process**:
   - To generate each data point:
     1. Sample the cluster assignment $s_n$ from the categorical distribution $\{\pi_k\}$.
     2. Given $s_n = k$, sample $\mathbf{x}_n$ from $\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$.
   - The collection of data points $\{\mathbf{x}_n\}$ are the observed variables, while the cluster assignments $\{s_n\}$ are latent (hidden) variables.

## Objective
- **Inference Goal**:
  - Given the observed data $\{\mathbf{x}_n\}$, infer the latent cluster assignments $\{s_n\}$ and estimate the model parameters $\theta = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}$.
- **Maximum Likelihood Estimation (MLE)**:
  - Aim to maximize the likelihood of the observed data:$$
    p(\mathbf{X} \mid \theta) = \prod_{n=1}^N \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}_n; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
    $$
  - However, this optimization is intractable due to the summation inside the product.

## Expectation Maximisation (EM) Algorithm
The EM algorithm provides an iterative approach to find maximum likelihood estimates for models with latent variables, like MoG.

### Overview
1. **Expectation (E) Step**:
   - Compute the posterior probabilities (responsibilities) that each data point belongs to each cluster.
2. **Maximisation (M) Step**:
   - Update the model parameters $\theta$ using the responsibilities computed in the E step.
3. **Convergence**:
   - Repeat E and M steps until convergence criteria are met (e.g., negligible change in log-likelihood).

### Detailed Steps

#### 1. Define Free Energy ($\mathcal{F}$)
- **Free Energy** is a lower bound to the log-likelihood:$$
  \mathcal{F}(q(\mathbf{s}), \theta) = \log p(\mathbf{X} \mid \theta) - KL(q(\mathbf{s}) \parallel p(\mathbf{s} \mid \mathbf{X}, \theta)) \leq \log p(\mathbf{X} \mid \theta)
  $$- **KL Divergence**: measure how much **information is lost** when you approximate one distribution by another distribution$$
  KL(q(\mathbf{s}) \parallel p(\mathbf{s} \mid \mathbf{X}, \theta)) = \sum_{\mathbf{s}} q(\mathbf{s}) \log \frac{p(\mathbf{s} \mid \mathbf{X}, \theta)}{q(\mathbf{s})}
  $$  - **Properties**:
    - **Non-Negative**: $KL \geq 0$
    - **Symmetry**: $KL(p_1 \parallel p_2) \neq KL(p_2 \parallel p_1)$
    - **Zero Condition**: $KL = 0$ if and only if $q(\mathbf{s}) = p(\mathbf{s} \mid \mathbf{X}, \theta)$ for all $\mathbf{s}$.

#### 2. Initialization
- **Initial Parameters**:
  - **Responsibilities**: Set initial responsibilities uniformly:
    $$
    q^{(0)}(s_n = k) = \frac{1}{K}, \quad \forall n, k
    $$
  - **Cluster Probabilities**:
    $$
    \pi_k^{(0)} = \frac{1}{K}, \quad \forall k
    $$
  - **Covariance Matrices**:
    $$
    \boldsymbol{\Sigma}_k^{(0)} = \mathbf{I}, \quad \forall k
    $$
  - **Cluster Means**:
    - Initialize $\boldsymbol{\mu}_k^{(0)}$ based on visual inspection or random selection.

#### 3. E Step (Expectation)
- **Objective**: Maximize $\mathcal{F}$ with respect to $q(\mathbf{s})$ while keeping $\theta$ fixed.
	- Since $\log p(\mathbf{X} \mid \theta)$ is independent of $q(\mathbf{s})$, maximize:$$
    \mathcal{F} = - KL(q(\mathbf{s}) \parallel p(\mathbf{s} \mid \mathbf{X}, \theta)) + \text{constant}
    $$
- **Result**: Set $q(\mathbf{s})$ to the posterior distribution $p(\mathbf{s} \mid \mathbf{X}, \theta)$ (posterior distribution represents our best guess about the hidden variables, given the data and the current parameters).
- **Calculation**:
  - **Posterior Probability**:
    $$
    p(s_n = k \mid \mathbf{x}_n, \theta) = \frac{p(s_n = k \mid \theta) p(\mathbf{x}_n \mid s_n = k, \theta)}{p(\mathbf{x}_n \mid \theta)}
    $$$$
    = \frac{\pi_k \mathcal{N}(\mathbf{x}_n; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_n; \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
    $$
  - **Define $u_{nk}$**:
    $$
    u_{nk} = \pi_k \mathcal{N}(\mathbf{x}_n; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
    $$
  - **Responsibilities**:
    $$
    q(s_n = k) = \frac{u_{nk}}{\sum_{j=1}^K u_{nj}}, \quad \forall n, k
    $$
#### 4. M Step (Maximisation)
- **Objective**: Maximize $\mathcal{F}$ with respect to $\theta$ while keeping $q(\mathbf{s})$ fixed.
- **Maximisation of Free Energy**:
  $$
  \mathcal{F}(q(\mathbf{s}), \theta) = \sum_{\mathbf{s}} q(\mathbf{s}) \log p(\mathbf{s}, \mathbf{X} \mid \theta) - \sum_{\mathbf{s}} q(\mathbf{s}) \log q(\mathbf{s})
  $$
  - Since $q(\mathbf{s})$ is fixed, maximize:
    $$
    \mathcal{Q}(\theta) = \sum_{\mathbf{s}} q(\mathbf{s}) \log p(\mathbf{s}, \mathbf{X} \mid \theta)
    $$
- **Parameter Updates**:
  - **Update $\boldsymbol{\mu}_k$**:
    $$
    \boldsymbol{\mu}_k^{(m+1)} = \frac{1}{N_k} \sum_{n=1}^N q^{(m)}(s_n = k) \mathbf{x}_n
    $$
    where:
    $$
    N_k = \sum_{n=1}^N q^{(m)}(s_n = k)
    $$
  - **Update $\pi_k$**:
    $$
    \pi_k^{(m+1)} = \frac{N_k}{N}
    $$
    where $N = \sum_{k=1}^K N_k = \sum_{n=1}^N \sum_{k=1}^K q^{(m)}(s_n = k)$
  - **Update $\boldsymbol{\Sigma}_k$**:
    $$
    \boldsymbol{\Sigma}_k^{(m+1)} = \frac{1}{N_k} \sum_{n=1}^N q^{(m)}(s_n = k) (\mathbf{x}_n - \boldsymbol{\mu}_k^{(m+1)})(\mathbf{x}_n - \boldsymbol{\mu}_k^{(m+1)})^\top
    $$

#### 5. Convergence Check
- **Criteria**:
  - **Change in Free Energy**:
    $$
    |\mathcal{F}^{(m+1)} - \mathcal{F}^{(m)}| < \epsilon
    $$
    where $\epsilon$ is a small threshold.
  - **Maximum Iterations**: Stop after a predefined number of iterations.
- **Decision**:
  - If converged, terminate the algorithm.
  - Else, return to the E Step with updated parameters.

### Mathematical Foundations

#### Free Energy ($\mathcal{F}$) and KL Divergence
- **Expression**:
  $$
  \mathcal{F}(q(\mathbf{s}), \theta) = \log p(\mathbf{X} \mid \theta) - KL(q(\mathbf{s}) \parallel p(\mathbf{s} \mid \mathbf{X}, \theta))
  $$
- **Implications**:
  - **Maximizing $\mathcal{F}$**: Equivalent to minimizing the KL divergence between $q(\mathbf{s})$ and the true posterior $p(\mathbf{s} \mid \mathbf{X}, \theta)$.
  - **Lower Bound**: $\mathcal{F}$ serves as a lower bound to the log-likelihood, ensuring that each EM iteration improves or maintains the likelihood.

#### Derivation of M Step
- **Maximizing $\mathcal{F}$ w.r.t $\theta$**:
  - Focus on maximizing:
    $$
    \mathcal{Q}(\theta) = \sum_{n=1}^N \sum_{k=1}^K q(s_n = k) \log \left( \pi_k \mathcal{N}(\mathbf{x}_n; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)
    $$
  - **Log-Likelihood Expansion**:
    $$
    \mathcal{Q}(\theta) = \sum_{n=1}^N \sum_{k=1}^K q(s_n = k) \left[ \log \pi_k - \frac{1}{2} \log |\boldsymbol{\Sigma}_k| - \frac{1}{2} (\mathbf{x}_n - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k) \right]
    $$
  - **Taking Derivatives**:
    - **With respect to $\boldsymbol{\mu}_k$**:
      $$
      \frac{\partial \mathcal{Q}}{\partial \boldsymbol{\mu}_k} = \sum_{n=1}^N q(s_n = k) \boldsymbol{\Sigma}_k^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k) = 0
      $$
      Solving:
      $$
      \boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{n=1}^N q(s_n = k) \mathbf{x}_n
      $$
    - **With respect to $\pi_k$**:
      - Subject to $\sum_{k=1}^K \pi_k = 1$ using Lagrange multipliers.
      $$
      \pi_k = \frac{N_k}{N}
      $$
    - **With respect to $\boldsymbol{\Sigma}_k$**:
      $$
      \frac{\partial \mathcal{Q}}{\partial \boldsymbol{\Sigma}_k} = -\frac{1}{2} \sum_{n=1}^N q(s_n = k) \left( \boldsymbol{\Sigma}_k^{-1} - \boldsymbol{\Sigma}_k^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1} \right) = 0
      $$
      Solving:
      $$
      \boldsymbol{\Sigma}_k = \frac{1}{N_k} \sum_{n=1}^N q(s_n = k) (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^\top
      $$

### EM Algorithm Recap for MoG
1. **Initialize**:
   - **Number of Clusters**: Set $K = 3$.
   - **Responsibilities**: Initialize uniformly:
     $$
     q(s_n = k) = \frac{1}{K}, \quad \forall n, k
     $$
   - **Cluster Probabilities**:
     $$
     \pi_k = \frac{1}{K}, \quad \forall k
     $$
   - **Covariance Matrices**:
     $$
     \boldsymbol{\Sigma}_k = \mathbf{I}, \quad \forall k
     $$
   - **Cluster Means**:
     - Set $\boldsymbol{\mu}_k$ based on visual inspection or other heuristic methods.
2. **E Step**:
   - **Compute Responsibilities**:
     $$
     q(s_n = k) = \frac{u_{nk}}{\sum_{j=1}^K u_{nj}}
     $$
     where:
     $$
     u_{nk} = \pi_k \mathcal{N}(\mathbf{x}_n; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
     $$
3. **M Step**:
   - **Update Cluster Means**:
     $$
     \boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{n=1}^N q(s_n = k) \mathbf{x}_n
     $$
   - **Update Cluster Probabilities**:
     $$
     \pi_k = \frac{N_k}{N}
     $$
   - **Update Covariance Matrices**:
     $$
     \boldsymbol{\Sigma}_k = \frac{1}{N_k} \sum_{n=1}^N q(s_n = k) (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^\top
     $$
4. **Check for Convergence**:
   - **Convergence Criteria**:
     - Minimal change in free energy $\mathcal{F}$.
     - Reached maximum number of iterations.
   - **Decision**:
     - If converged, terminate.
     - Else, repeat E and M steps.

## Initialization and Convergence Considerations
- **Parameter Initialization**:
  - **Importance**: Crucial for the convergence and quality of the final solution.
  - **Strategies**:
    - **Random Initialization**: Randomly assign initial cluster means.
    - **K-means Initialization**: Use results from K-means clustering as initial parameters.
    - **Manual Initialization**: Set cluster means based on domain knowledge or visual inspection.
- **Convergence of EM**:
  - **Local Maximum**: EM typically converges to a local maximum of the likelihood function.
  - **Global Maximum**: Not guaranteed; dependent on initial parameter settings.
  - **Possible Issues**:
    - **Poor Initialization**: Can lead to suboptimal clustering results.
    - **Singular Covariance Matrices**: Can occur if clusters collapse, requiring regularization.

## Implementation Notes
- **Function Implementation**:
  - **`mog_EM` Function**:
    - **Inputs**:
      - Data matrix $\mathbf{X}$.
      - Number of clusters $K$.
      - Initial parameters $\theta^{(0)}$.
    - **Outputs**:
      - Estimated parameters $\theta = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}$.
      - Responsibilities $q(s_n = k)$ for each data point.
- **Practical Application**:
  - **Synthetic Data Generation**:
    - Create synthetic datasets using known MoG parameters to test the EM implementation.
  - **Visualization**:
    - Plot clusters and their Gaussian contours to assess the quality of clustering.
- **Handling Numerical Stability**:
  - **Logarithms**: Compute log-likelihoods to prevent underflow in probability computations.
  - **Regularization**: Add small values to diagonal of covariance matrices to ensure they remain positive definite.

## Practical Example: Applying EM to K-means Dataset
- **Dataset**: The first clustering dataset used in K-means.
- **Initialization**:
  - **Number of Clusters**: $K = 3$.
  - **Responsibilities**:
    $$
    q(s_n = k) = \frac{1}{K}, \quad \forall n, k
    $$
  - **Cluster Probabilities**:
    $$
    \pi_k = \frac{1}{K}, \quad \forall k
    $$
  - **Covariance Matrices**:
    $$
    \boldsymbol{\Sigma}_k = \mathbf{I}, \quad \forall k
    $$
  - **Cluster Means**:
    - Set manually based on data distribution.

- **EM Iterations**:
  1. **E Step**:
     - Compute responsibilities using current $\theta$.
  2. **M Step**:
     - Update $\boldsymbol{\mu}_k$, $\pi_k$, and $\boldsymbol{\Sigma}_k$ using responsibilities.
  3. **Convergence Check**:
     - Assess if free energy has stabilized or if maximum iterations are reached.

- **Outcome**:
  - **Cluster Assignments**: Soft assignments reflecting the probability of each data point belonging to each cluster.
  - **Cluster Parameters**: Means, covariances, and mixing coefficients refined to best fit the data.

## Conclusion
- **MoG vs. K-means**:
  - **Flexibility**: MoG can model more complex cluster shapes and handle overlapping clusters.
  - **Probabilistic Framework**: Provides a probabilistic interpretation of cluster assignments.
- **EM Algorithm**:
  - **Powerful Tool**: Efficiently estimates parameters in the presence of latent variables.
  - **Limitations**: Sensitive to initialization and may converge to local maxima.
- **Applications**:
  - Widely used in pattern recognition, computer vision, and machine learning for clustering and density estimation tasks.

