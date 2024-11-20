# Introduction to Clustering

Clustering $\rightarrow$ grouping a set of **unlabeled inputs** $\{x_n\}_{n=1}^N$ into clusters based on similarity, without prior knowledge of class labels.

**Clustering Goal**: To find a function $f: \mathbb{R}^D \rightarrow \{1, 2, \dots, K\}$ that assigns each input $x_n$ to one of $K$ clusters. The goal is to group similar inputs together in such a way that:

---
# The K-means Algorithm 

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

2. **Cluster Center Update:** For each cluster, update its center to the mean of the points assigned to it:
$$ \mathbf{m}_k = \frac{1}{N_k} \sum_{s_{nk}=1} \mathbf{x}_n $$
   where $N_k = \sum_n s_{nk}$ is the number of points in cluster $k$.

These steps are repeated until the cluster assignments $\{s_{nk}\}$ no longer change.

![[Pasted image 20241107181335.png]]

### Convergence and Initialization
- **Convergence:** K-means is guaranteed to converge because it performs coordinate descent on the cost function $\mathcal{C}$, which is a Lyapunov function, ensuring that the cost either decreases or remains constant after each step.
- **Local Minima:** K-means may converge to a local minimum rather than a global one, depending on the initial cluster centers. Finding the global optimum is NP-hard.
- **Initialization:** Initial cluster centers are important. The **K-means++** algorithm provides a robust initialization by selecting centers that are spread out, improving the final clustering result.

## **K-means Limitations**:
  - **Anisotropic Clusters**: K-means assumes spherical clusters with equal variance, which fails for elongated or differently scaled clusters.
  - **Hard Assignments**: Each data point is assigned definitively to one cluster, disregarding the uncertainty or probability of belonging to other clusters.

![[Pasted image 20241107181435.png]]

---

# Expectation Maximisation (EM) Algorithm for MoG

## Introduction
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
     - $\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})$ denotes the multivariate Gaussian distribution:
$$ \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{\sqrt{(2\pi)^D |\boldsymbol{\Sigma}|}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)$$
where $D$ is the dimensionality of the data.

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

### Detailed Steps

#### 1. Define Free Energy ($\mathcal{F}$)

- **Free Energy** is a lower bound to the log-likelihood, and makes the optimization problem tractable when dealing with latent variables:

$$ \mathcal{F}(q(\mathbf{s}), \theta) = \log p(\mathbf{X} \mid \theta) - KL(q(\mathbf{s}) \parallel p(\mathbf{s} \mid \mathbf{X}, \theta)) \leq \log p(\mathbf{X} \mid \theta) $$
  
  - **KL Divergence**: Measures how much **information is lost** when approximating one distribution by another. It can be thought of as the **extra number of bits** required to encode samples from the distribution $p(\mathbf{s})$ using the distribution $q(\mathbf{s})$.
  $$ KL(q(\mathbf{s}) \parallel p(\mathbf{s} \mid \mathbf{X}, \theta)) = \sum_{\mathbf{s}} q(\mathbf{s}) \log \frac{p(\mathbf{s} \mid \mathbf{X}, \theta)}{q(\mathbf{s})}$$  
  - **Properties**:
    - **Non-Negative**: $KL \geq 0$
    - **Symmetry**: $KL(p_1 \parallel p_2) \neq KL(p_2 \parallel p_1)$
    - **Zero Condition**: $KL = 0$ if and only if $q(\mathbf{s}) = p(\mathbf{s} \mid \mathbf{X}, \theta)$ for all $\mathbf{s}$.

#### 2. Initialization

**Parameter Initialization**:
  - **Importance**: Crucial for the convergence and quality of the final solution.
  - **Strategies**:
    - **Random Initialization**: Randomly assign initial cluster means.
    - **K-means Initialization**: Use results from K-means clustering as initial parameters.
    - **Manual Initialization**: Set cluster means based on domain knowledge or visual inspection.

- **Initial Parameters**:
  - **Responsibilities**: Set initial responsibilities uniformly:

$$q^{(0)}(s_n = k) = \frac{1}{K}, \quad \forall n, k$$
  
  - **Cluster Probabilities**:
  $$
    \pi_k^{(0)} = \frac{1}{K}, \quad \forall k
    $$
  
  - **Covariance Matrices**:
  - 
$$\boldsymbol{\Sigma}_k^{(0)} = \mathbf{I}, \quad \forall k
    $$
  - **Cluster Means**:
    - Initialize $\boldsymbol{\mu}_k^{(0)}$ based on visual inspection or random selection.

#### 3. E Step (Expectation)

$\rightarrow$ Compute the posterior probabilities (responsibilities) that each data point belongs to each cluster.

- **Objective**: Maximize $\mathcal{F}$ with respect to $q(\mathbf{s})$ while keeping $\theta$ fixed.
	- Since $\log p(\mathbf{X} \mid \theta)$ is independent of $q(\mathbf{s})$, maximize:

$$ \mathcal{F} = - KL(q(\mathbf{s}) \parallel p(\mathbf{s} \mid \mathbf{X}, \theta)) + \text{constant} $$

- **Result**: Set $q(\mathbf{s})$ to the posterior distribution $p(\mathbf{s} \mid \mathbf{X}, \theta)$ (posterior distribution represents our best guess about the hidden variables, given the data and the current parameters).

- **Calculating the Posterior Probability**:
$$ p(s_n = k \mid \mathbf{x}_n, \theta) = \frac{p(s_n = k \mid \theta) p(\mathbf{x}_n \mid s_n = k, \theta)}{p(\mathbf{x}_n \mid \theta)} $$$$
    = \frac{\pi_k \mathcal{N}(\mathbf{x}_n; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_n; \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
    $$
  - **Define $u_{nk}$**:$$
    u_{nk} = \pi_k \mathcal{N}(\mathbf{x}_n; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
    $$
  - **Responsibilities**:$$
    q(s_n = k) = \frac{u_{nk}}{\sum_{j=1}^K u_{nj}}, \quad \forall n, k
    $$
#### 4. M Step (Maximisation)

$\rightarrow$ Update the model parameters $\theta$ using the responsibilities computed in the E step.

- **Objective**: Maximize $\mathcal{F}$ with respect to $\theta$ while keeping $q(\mathbf{s})$ fixed.

- **Maximisation of Free Energy**:

$$ \mathcal{F}(q(\mathbf{s}), \theta) = \sum_{\mathbf{s}} q(\mathbf{s}) \log p(\mathbf{s}, \mathbf{X} \mid \theta) - \sum_{\mathbf{s}} q(\mathbf{s}) \log q(\mathbf{s})$$
  - Since $q(\mathbf{s})$ is fixed, maximize:

$$\mathcal{Q}(\theta) = \sum_{\mathbf{s}} q(\mathbf{s}) \log p(\mathbf{s}, \mathbf{X} \mid \theta) = \sum_{n=1}^N \sum_{k=1}^K q(s_n = k) \log \left( \pi_k \mathcal{N}(\mathbf{x}_n; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)$$   
     - **Log-Likelihood Expansion**:

$$ \mathcal{Q}(\theta) = \sum_{n=1}^N \sum_{k=1}^K q(s_n = k) \left[ \log \pi_k - \frac{1}{2} \log |\boldsymbol{\Sigma}_k| - \frac{1}{2} (\mathbf{x}_n - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k) \right] $$

- **Parameter Updates Taking Derivatives**:

**With respect to $\boldsymbol{\mu}_k$**:$$
      \frac{\partial \mathcal{Q}}{\partial \boldsymbol{\mu}_k} = \sum_{n=1}^N q(s_n = k) \boldsymbol{\Sigma}_k^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k) = 0
      $$      Solving:$$
    \boldsymbol{\mu}_k^{(m+1)} = \frac{1}{N_k} \sum_{n=1}^N q^{(m)}(s_n = k) \mathbf{x}_n
      $$
**With respect to $\pi_k$**:
      - Subject to $\sum_{k=1}^K \pi_k = 1$ using Lagrange multipliers.$$
    \pi_k^{(m+1)} = \frac{N_k}{N}
      $$ where $N = \sum_{k=1}^K N_k = \sum_{n=1}^N \sum_{k=1}^K q^{(m)}(s_n = k)$ 


**With respect to $\boldsymbol{\Sigma}_k$**:$$
      \frac{\partial \mathcal{Q}}{\partial \boldsymbol{\Sigma}_k} = -\frac{1}{2} \sum_{n=1}^N q(s_n = k) \left( \boldsymbol{\Sigma}_k^{-1} - \boldsymbol{\Sigma}_k^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1} \right) = 0
      $$      Solving:$$
    \boldsymbol{\Sigma}_k^{(m+1)} = \frac{1}{N_k} \sum_{n=1}^N q^{(m)}(s_n = k) (\mathbf{x}_n - \boldsymbol{\mu}_k^{(m+1)})(\mathbf{x}_n - \boldsymbol{\mu}_k^{(m+1)})^\top
    $$
#### 5. Convergence Check

- **Criteria**: Repeat E and M steps until convergence criteria are met (e.g., negligible change in log-likelihood).

  - **Change in Free Energy**:
$$ \mathcal{F}^{(m+1)} - \mathcal{F}^{(m)}| < \epsilon $$
    where $\epsilon$ is a small threshold.
    
  - **Maximum Iterations**: Stop after a predefined number of iterations.

- **Decision**:
  - If converged, terminate the algorithm.
  - Else, return to the E Step with updated parameters.

 **Convergence of EM**:
  - **Local Maximum**: EM typically converges to a local maximum of the likelihood function.
  - **Global Maximum**: Not guaranteed; dependent on initial parameter settings.
  - **Possible Issues**:
    - **Poor Initialization**: Can lead to suboptimal clustering results.
    - **Singular Covariance Matrices**: Can occur if clusters collapse, requiring regularization.

## Limitations

1. **Convergence to Local Optima**: EM is a **greedy algorithm** and only guarantees convergence to a local optimum, which might not be the global optimum. 
2. **Slow Convergence**: The algorithm can converge slowly, especially if the likelihood surface is flat or has long, narrow peaks. 
3. **Sensitive to Initialization**: Poor initialization can lead to convergence at an inferior local maximum or slow down the algorithm significantly.
4. **Requires Knowing the Number of Components**: If this number is incorrect, it can lead to poor results.
5. **Assumes Independence Among Latent Variables**: In its basic form, EM assumes that latent variables are independent, which may not be realistic for all datasets. 



