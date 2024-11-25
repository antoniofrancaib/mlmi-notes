# Index

- [9-Gaussian-Processes*](#9-gaussian-processes)
- [10-Gaussian-Processes-and-Data*](#10-gaussian-processes-and-data)
- [11-Gaussian-Process-Marginal-Likelihood-and-Hyperparameters*](#11-gaussian-process-marginal-likelihood-and-hyperparameters)
- [12-Correspondence-Between-Linear-Models-and-Gaussian-Processes*](#12-correspondence-between-linear-models-and-gaussian-processes)
- [13-Covariance-Functions](#13-covariance-functions)
- [14-Finite-and-Infinite-Basis-GPs](#14-finite-and-infinite-basis-gps)

---
## 9-Gaussian-Processes

### From Scalar Gaussians to Multivariate Gaussians to Gaussian Processes

1. **Scalar Gaussian**: A single random variable $x$ with distribution $N(\mu, \sigma^2)$.

2. **Multivariate Gaussian**: A vector $x = [x_1, x_2, \dots, x_N]^T$ with joint Gaussian distribution:
   
   $$p(x \mid \mu, \Sigma) = \frac{1}{(2 \pi)^{N/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)$$

3. **Gaussian Process (GP)**: An extension to infinitely many variables.

   - **Definition**: A collection of random variables, any finite number of which have a joint Gaussian distribution.
   - **Intuition**: Think of functions as infinitely long vectors.

### Gaussian Process Definition
A GP is fully specified by:

- **Mean function** $m(x) = E[f(x)]$
- **Covariance function** $k(x, x') = E[(f(x) - m(x))(f(x') - m(x'))]$

**Notation**:

$$
f(x) \sim GP(m(x), k(x, x'))
$$

### Marginal and Conditional Gaussians
Key properties:

- **Marginalization**: The marginal distribution over any subset of variables is Gaussian.
- **Conditioning**: The conditional distribution given some variables is also Gaussian.

The marginalization property simplifies Gaussian processes (GPs) by leveraging their unique characteristics. The marginalization property allows you to work with finite-dimensional slices of the GP. Specifically: 

$$p(\mathbf{x}) = \int p(\mathbf{x}, \mathbf{y}) \, d\mathbf{y}.$$ 

For a multivariate Gaussian: 

$$\begin{bmatrix} \mathbf{x} \\ \mathbf{y} \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \mathbf{a} \\ \mathbf{b} \end{bmatrix}, \begin{bmatrix} A & B^\top \\ B & C \end{bmatrix} \right) \quad \Rightarrow \quad p(\mathbf{x}) \sim \mathcal{N}(\mathbf{a}, A).$$ 

In Gaussian processes, this property enables predictions based only on finite-dimensional covariance matrices without handling infinite-dimensional computations. 

### GP as a distribution over functions
A GP defines a distribution over functions. Each finite collection of function values follows a multivariate Gaussian distribution. 

**Example**: $$ p(f) \sim \mathcal{N}(m, k), \quad m(x) = 0, \quad k(x, x') = \exp\left(-\frac{1}{2}(x - x')^2\right). $$ 
For a finite set of points $\{x_1, x_2, ..., x_N\}$, the function values $\{f_1, f_2, ..., f_N\}$ are jointly Gaussian: 

$$ f \sim \mathcal{N}(0, \Sigma), \quad \Sigma_{ij} = k(x_i, x_j). $$

To visualize a GP, draw samples from this multivariate Gaussian and plot them as functions.

### Sampling from a Gaussian Process (GP)

- **Goal**: Generate samples from a joint Gaussian distribution with mean $\mathbf{m}$ and covariance $\mathbf{K}$.


The following two methods are **conceptually the same** in the sense that they both generate samples from the same joint Gaussian prior defined by the GP. However, they differ in how they approach the computation:

- **Direct Sampling**: All samples are generated simultaneously using the full covariance matrix $\mathbf{K}$.
- **Sequential Sampling**: Samples are generated one by one using conditional distributions, which can be derived from the same $\mathbf{K}$.

### **1. Direct Sampling Using Cholesky Decomposition**
This method generates samples by directly leveraging the multivariate Gaussian distribution:

- **Steps**:
  1. **Select Inputs**: Choose $N$ input points $\{x_i\}_{i=1}^{N}$.
  2. **Covariance Matrix**: Compute the covariance matrix $\mathbf{K}$ for all chosen input points $\{x_i\}_{i=1}^{N}$ using the kernel $k(x_i, x_j)$.
  3. **Sampling $\mathbf{f}$ from a Gaussian Process**. Both methods are equivalent: 

- **Sample $\mathbf{z}$ and Transform**:
   - Draw $\mathbf{z} \sim \mathcal{N}(0, I)$, where $\mathbf{z}$ is a vector of independent standard normal samples.
   - Transform $\mathbf{z}$ to match the desired covariance $\mathbf{K}$:
 $$
	     \mathbf{f} = \text{chol}(\mathbf{K})^\top \mathbf{z} + \mathbf{m},
	     $$
 where $\mathbf{m}$ is the mean vector.

- **Direct Sampling**:
   - Directly sample $\mathbf{f} \sim \mathcal{N}(\mathbf{m}, \mathbf{K})$ using computational libraries.

- **Purpose**: The Cholesky decomposition ensures that the resulting samples have the correct covariance $\mathbf{K}$ and mean $\mathbf{m}$.

### **2. Sequential Sampling Using Conditional Gaussians**
This method generates samples by iteratively sampling one point at a time, conditioning on previously sampled points:

- **Steps**:

1. **Factorization**: Use the chain rule for multivariate Gaussians to factorize the joint distribution: 
  $$
  p(f_1, ..., f_N \mid x_1, ..., x_N) = \prod_{n=1}^N p(f_n \mid f_{<n}, x_{<n}).
  $$

  2. **The joint prior**:
  
$$
    p(f_n, f_{<n}) = \mathcal{N} \left( \begin{bmatrix} \mathbf{a} \\ \mathbf{b} \end{bmatrix}, \begin{bmatrix} A & B^\top \\ B & C \end{bmatrix} \right).
    $$

  3. **Conditional distribution**:
  
$$
    p(f_n \mid f_{<n}) = \mathcal{N} \left(\mathbf{a} + BC^{-1}(\mathbf{f}_{<n} - \mathbf{b}), A - BC^{-1}B^\top \right).
    $$

- **Purpose**: This approach samples points sequentially, conditioning on previously sampled values.

- **Illustration**:
  - The shaded regions and lines in the plots show how the GP updates its predictions as new data points are added.

![[gp-samp.png]]

#### **Which Method to Use?**

- **Small to Moderate Number of Input Points**: Use direct sampling (Cholesky decomposition) for simplicity and efficiency.
- **Large Number of Input Points or Online Sampling**: Use sequential sampling, especially if you need to incorporate new input points dynamically without recomputing the entire covariance matrix.

---
## 10-Gaussian-Processes-and-Data

### Conditioning on Observations
Given observed data $D = \{(x_i, y_i)\}_{i=1}^N$, we want to predict $f_*$ at new inputs $x_*$.

Assumption: Observations $y_i$ are noisy versions of the true function $f(x_i)$:

$$
y_i = f(x_i) + \epsilon_i, \quad \epsilon_i \sim N(0, \sigma_n^2)
$$
### Non-parametric Gaussian Process Models

In our non-parametric model, the "parameters" are the function itself!

The joint distribution:
$$
p(\mathbf{f}, \mathbf{y}) = p(\mathbf{f}) p(\mathbf{y} \mid \mathbf{f}) = p(\mathbf{y}) p(\mathbf{f} \mid \mathbf{y})
\quad \Rightarrow \quad \mathcal{N}(\mathbf{f} \mid \mathbf{m}, \mathbf{k}) \mathcal{N}(\mathbf{y} \mid \mathbf{f}) = Z_{\mid \mathbf{y}} \mathcal{N}(\mathbf{f} \mid \mathbf{m}_{\mid \mathbf{y}}, \mathbf{k}_{\mid \mathbf{y}}).
$$
where $Z_{\mid \mathbf{y}}$ is the **normalization constant**. 

- **Gaussian process prior** with zero mean and covariance function $\mathbf{k}$:
  $$
  p(\mathbf{f} \mid \mathcal{M}_i) \sim \mathcal{N}(\mathbf{f} \mid \mathbf{m} \equiv 0, \mathbf{k}),
  $$

- **Gaussian likelihood**, with noise variance $\sigma^2_{\text{noise}}$:
  $$
  p(\mathbf{y} \mid \mathbf{f}, \mathcal{M}_i) \sim \mathcal{N}(\mathbf{f}, \sigma^2_{\text{noise}} \mathbf{I}),
  $$

leads to a **Gaussian process posterior**:
$$
p(\mathbf{f} \mid \mathbf{y}, \mathcal{M}_i) \sim \mathcal{N}(\mathbf{f} \mid \mathbf{m}_{\mid \mathbf{y}}, \mathbf{k}_{\mid \mathbf{y}}),
$$
where:
$$
\begin{aligned}
\mathbf{m}_{\mid \mathbf{y}}(x) &= \mathbf{k}(x, \mathbf{x}) [\mathbf{K}(\mathbf{x}, \mathbf{x}) + \sigma^2_\text{noise} \mathbf{I}]^{-1} \mathbf{y}, \\
\mathbf{k}_{\mid \mathbf{y}}(x, x') &= k(x, x') - \mathbf{k}(x, \mathbf{x}) [\mathbf{K}(\mathbf{x}, \mathbf{x}) + \sigma^2_\text{noise} \mathbf{I}]^{-1} \mathbf{k}(\mathbf{x}, x').
\end{aligned}
$$

**Intuition**: 
*Posterior mean:* 
	- $\mathbf{k}(x, \mathbf{x})$: Correlation between the test point $x$ and the training points $\mathbf{x}$. This is a **row vector** (size $1 \times N$) of kernel values between the test point $x$ and the $N$ training points $\mathbf{x}$.
	- $\mathbf{K}(\mathbf{x}, \mathbf{x})$: Encod es correlations among training points . This is an **$N \times N$ matrix**, the inverse of the covariance matrix for the training points (with noise added).
	- $\left[ \mathbf{K} + \sigma^2_\text{noise} \mathbf{I} \right]^{-1} \mathbf{y}$: Scales the influence of observed data $\mathbf{y}$ based on their uncertainty. This is a **column vector** (size $N \times 1$) of the observed outputs corresponding to the $N$ training points.

*Posterior covariance:* 
	- The first term $k(x, x')$: Encodes the prior uncertainty between test points. 
	- The second term subtracts the reduction in uncertainty due to conditioning on the observations $\mathbf{y}$.

**Visualization**:
- **Prior Samples**: Functions drawn from the GP prior.
- **Posterior Samples**: Functions drawn from the GP posterior, which now pass through (or near) the observed data points.

![[gp-data.png]]

### Predictive Distribution
The predictive distribution in Gaussian Processes (GPs) is essentially the posterior distribution over the function values at a new input point $x_*$. This is because GPs are non-parametric models, and we **do not need to integrate over explicit parameters** like in parametric Bayesian models. The predictive distribution for a new input $x_*$ is given by:

$$
p(y_* \mid x_*, \mathbf{x}, \mathbf{y}) \sim \mathcal{N} \left( \mathbf{k}(x_*, \mathbf{x})^\top \left[\mathbf{K} + \sigma_{\text{noise}}^2 \mathbf{I} \right]^{-1} \mathbf{y}, \, \mathbf{k}(x_*, x_*) + \sigma_{\text{noise}}^2 - \mathbf{k}(x_*, \mathbf{x})^\top \left[\mathbf{K} + \sigma_{\text{noise}}^2 \mathbf{I} \right]^{-1} \mathbf{k}(x_*, \mathbf{x}) \right).
$$

- **Mean**: Describes the predicted value at $x_*$.
- **Variance**: Quantifies uncertainty at $x_*$.

### Interpretation of the Predictive Mean and Variance

#### Predictive Mean:
The predictive mean formula: 
$$ \mu(x_*) = \mathbf{k}(x_*, \mathbf{x}) \left[\mathbf{K} + \sigma_\text{noise}^2 \mathbf{I} \right]^{-1} \mathbf{y}, $$ can be rewritten as: 
$$ \mu(x_*) = \sum_{n=1}^N \beta_n y_n = \sum_{n=1}^N \alpha_n k(x_*, x_n), $$ providing significant intuition about how Gaussian Processes make predictions by weighting observations using the kernel.

##### **1. Weighted Sum of Observations** 
- The formula $\sum_{n=1}^N \beta_n y_n$ expresses the predictive mean $\mu(x_*)$ as a **weighted sum of the observed outputs** $y_n$, where the weights $\beta_n = \sum_{m=1}^N \left[\mathbf{K} + \sigma_\text{noise}^2 \mathbf{I}\right]^{-1}_{n,m} k(x_*, x_m)$. Alternatively: 
$$ \boldsymbol{\beta} = \left[\mathbf{K} + \sigma_\text{noise}^2 \mathbf{I}\right]^{-1} \mathbf{k}(\mathbf{x}, x_*), $$

**Intuition**:
- **$\beta_n$ as Influence Weights**: The coefficients $\beta_n$ represent the influence of each training data point's similarity to the new input $x_*$ on the predictive mean $\mu(x_*)$. They quantify how much each observed output $y_n$ contributes to the prediction, weighted by the covariance between $x_n$ and $x_*$. The closer the test point $x_*$ is to a training point $x_n$ in the input space (as measured by the kernel), the larger the corresponding weight $\beta_n$, and vice versa. 

- **Role in Prediction**: In the expression $\mu(x_*) = \sum_{n=1}^N \beta_n y_n$, each $\beta_n$ scales the observed output $y_n$. This means that the prediction at $x_*$ is a weighted sum of the training outputs, where the weights $\beta_n$ depend on both the covariance structure of the data and the similarity between $x_n$ and $x_*$.

- **Effect of Noise and Correlations**: The inversion of $\mathbf{K} + \sigma_\text{noise}^2 \mathbf{I}$ adjusts the weights $\beta_n$ based on noise and correlations in the data. Observations that are more relevant (e.g., closer to $x_*$ or less noisy) will have larger $\beta_n$ values, contributing more to the prediction.

##### **2. Kernel Dependence**
- The formula expresses the predictive mean $\mu(x_*)$ as a **weighted sum of kernel values** between the test point $x_*$ and the training points $x_n$, where the weights $\alpha_n = \sum_{m=1}^N \left[\mathbf{K} + \sigma_\text{noise}^2 \mathbf{I}\right]^{-1}_{n,m} y_m$. Alternatively: 
$$ \boldsymbol{\alpha} = \left[\mathbf{K} + \sigma_\text{noise}^2 \mathbf{I} \right]^{-1} \mathbf{y} $$
 **Intuition**: 
 - **$\alpha_n$ as Influence Weights**:  The coefficients $\alpha_n$ represent the influence of each training data point $(x_n, y_n)$ on the prediction at a new input $x_*$. They quantify how much each observed output $y_n$ contributes to the predictive mean $\mu(x_*)$, after accounting for the correlations (captured by the kernel matrix $\mathbf{K}$) and the noise in the observations. 
 
 - **Role in Prediction**: In the expression $\mu(x_*) = \sum_{n=1}^N \alpha_n k(x_*, x_n),$ each $\alpha_n$ scales the similarity between the new input $x_*$ and the training input $x_n$ (measured by $k(x_*, x_n)$). This means the prediction at $x_*$ is a weighted sum of the similarities to all training points, where the weights $\alpha_n$ are determined by both the training outputs and the covariance structure of the data. 
 
 - **Effect of Noise and Correlations**: The inversion of $\mathbf{K} + \sigma_\text{noise}^2 \mathbf{I}$ adjusts the weights $\alpha_n$ based on how noise and correlations between data points affect the reliability of each observation. Data points that are less affected by noise or are more informative (due to higher correlations) will generally have larger $\alpha_n$ values. 

  Conclude that **weights** ($\beta_n, \alpha_n$): Depend on the covariance structure and how $x_*$ relates to the training points $x_n$.

#### Predictive Variance:
$$
\sigma^2(x_*) = \mathbf{k}(x_*, x_*) - \mathbf{k}(x_*, \mathbf{x}) \left[\mathbf{K(x, x)} + \sigma_{\text{noise}}^2 \mathbf{I} \right]^{-1} \mathbf{k}(\mathbf{x}, x_*).
$$

**Intuition**: 
- The posterior nvariance has two terms:
  1. **Prior Variance** ($\mathbf{k}(x_*, x_*)$): The initial uncertainty in the prior.
  2. **Subtract Information Gained from Data**: Subtracted based on how well the observed data explains $x_*$.
	   - **Similarity to Training Data**:  
	     If $x_*$ is similar to training points, $\mathbf{k}(x_*, \mathbf{x})$ will have larger values.
	   - **Adjust for Noise and Correlation**:  
	     The inverse term adjusts the influence of each training point based on noise and how the points correlate with each other.
	   - **Overall Reduction**:  
	     The product $\mathbf{k}(x_*, \mathbf{x}) \left[\mathbf{K} + \sigma_\text{noise}^2 \mathbf{I}\right]^{-1} \mathbf{k}(x_*, \mathbf{x})^\top$ quantifies the total reduction in uncertainty at $x_*$ due to the observed data.

---

## 11-Gaussian-Process-Marginal-Likelihood-and-Hyperparameters

### The GP Marginal Likelihood
The marginal likelihood (or evidence) is the probability of the observed data under the GP model:

$$
p(y \mid x) = \int p(y \mid f) p(f) \, df
$$

Given $y = f + \epsilon$  where:  
- $f \sim N(m, K)$ 
- $\epsilon \sim N(0, \sigma_n^2 I)$  

Since $f$ and $\epsilon$ are independent, the sum $y$ is also Gaussian. The marginal distribution is:  

$$p(y \mid x) = N(y ; m, K + \sigma_n^2 I)$$  
Taking the natural logarithm of both sides gives the **log marginal likelihood**:  

$$
\log p(y \mid x) = -\frac{1}{2} (y - m)^T (K + \sigma_n^2 I)^{-1} (y - m) - \frac{1}{2} \log |K + \sigma_n^2 I| - \frac{N}{2} \log 2\pi
$$  
where the matrix $K$ represents the **kernel (or covariance) matrix**.  


**Interpretation**:
 **1. First Term: Data Fit**
- The expression $(y - m)^T (K + \sigma_n^2 I)^{-1} (y - m)$ represents the squared **Mahalanobis distance** of $y$ from the mean. Unlike Euclidean distance, the Mahalanobis distance accounts for the covariance structure of the data, effectively scaling the dimensions according to their variances and covariances. 
- During model training (e.g., hyperparameter optimization), the objective is to **minimize** this term. Minimizing the discrepancy ensures that the GP model's predictions are as close as possible to the observed data, considering the uncertainty captured by the covariance matrix.

**2. Second Term: Model Complexity (Occam's Razor)**
- The determinant $|K + \sigma_n^2 I|$ represents the **volume** of the uncertainty captured by the covariance matrix $\Sigma = K + \sigma_n^2 I$ . A larger determinant indicates a more **spread-out** distribution, implying greater uncertainty.
- This term acts as a **penalty for model complexity**. A more complex model (with a covariance matrix that allows for greater variability) will generally have a larger determinant, leading to a higher penalty. Conversely, a simpler model will have a smaller determinant and thus a smaller penalty.
- This embodies the principle of **Occam's Razor**, which favors simpler models when possible.

Third term is normalizing constant, i.e. irrelevant when it comes to optimization. 

### Hyperparameters and Model Selection

- **Hyperparameters** $\theta$: Parameters of the covariance function (e.g., length-scale $\ell$, signal variance $\sigma_f^2$, noise variance $\sigma_n^2$).
- **Optimizing Hyperparameters**:

   Find $\theta$ that maximize the marginal likelihood:
   $$
   \theta^* = \arg \max_\theta \log p(y \mid x, \theta)
   $$
   
   This is a form of model selection.

**Example**:
- **Squared Exponential Covariance Function**:

  $$
  k(x, x') = \sigma_f^2 \exp \left( -\frac{(x - x')^2}{2 \ell^2} \right)
  $$

  By adjusting $\ell$ and $\sigma_f^2$, we can control the smoothness and amplitude of the functions.

### Occam's Razor
The marginal likelihood balances data fit and model complexity:

- Simple models with fewer hyperparameters may not fit the data well but are preferred if they explain the data sufficiently.
- Complex models may overfit the data but are penalized in the marginal likelihood due to increased complexity.

![[Pasted image 20241124175426.png]]

The mean posterior predictive function is plotted for 3 different length scales (the blue curve corresponds to optimizing the marginal likelihood). Notice, that an almost exact fit to the data can be achieved by reducing the length scale – but the marginal likelihood does not favour this!

Bayes' rule helps identify the right model complexity by leveraging the marginal likelihood, which balances goodness-of-fit with model simplicity. Overly simple models (highly peaked marginal likelihood) fail to capture data variability, while overly complex models (broad marginal likelihood) risk overfitting. The optimal model, guided by Occam's Razor, maximizes the marginal likelihood by being complex enough to explain the data but simple enough to generalize well, inherently penalizing unnecessary complexity. This balance ensures a principled trade-off between model flexibility and parsimony.

![[occam.png]]

**An illustrative analogous example**: 

***Recall***: The formula for the log-likelihood is:

$$
\log p(y \mid \mu, \sigma^2) = -\frac{1}{2} y^\top I_y y / \sigma^2 - \frac{1}{2} \log |\sigma^2| - \frac{n}{2} \log (2\pi)
$$

This example demonstrates how fitting the variance $\sigma^2$ of a zero-mean Gaussian distribution affects the likelihood of the observed data. The formula highlights how the log-likelihood balances the goodness-of-fit term $-\frac{1}{2} y^\top I_y y / \sigma^2$ with complexity penalties, such as $-\frac{1}{2} \log |\sigma^2|$ and the constant term $-\frac{n}{2} \log (2\pi)$. The visualizations show how different variances $\sigma^2$ impact the Gaussian’s shape, emphasizing the trade-off between fitting the data well and avoiding overfitting. This optimization ensures the model captures the data's structure effectively.


![[variance-estim.png]]


---

## 12-Correspondence-Between-Linear-Models-and-Gaussian-Processes
### From Linear Models to GPs
Consider a linear model with Gaussian priors:

$$
f(x) = \sum_{m=1}^M w_m \phi_m(x) = \mathbf{w}^\top \boldsymbol{\phi}(x), \quad w_m \sim N(0, \sigma_w^2)
$$

that is,  $p(\mathbf{w}) = \mathcal{N}(\mathbf{w}; \mathbf{0}, \mathbf{A}),$

- **Mean Function**: 
$$m(x) = \mathbb{E}_{\mathbf{w}}(f(x)) = \int \left( \sum_{m=1}^M w_m \phi_m(x) \right) p(\mathbf{w}) d\mathbf{w} = \sum_{m=1}^M \phi_m(x) \int w_m p(w_m) dw_m = 0$$

- **Covariance Function**:

$$
k(x_i, x_j) = \text{Cov}_{\mathbf{w}}(f(x_i), f(x_j)) = \mathbb{E}_{\mathbf{w}}(f(x_i)f(x_j)) = \int \cdots \int \left( \sum_{k=1}^M \sum_{l=1}^M w_k w_l \phi_k(x_i) \phi_l(x_j) \right) p(\mathbf{w}) d\mathbf{w}.
$$

where the integration symbol $\int \cdots \int$ is shorthand for integrating over all $M$ dimensions of $\mathbf{w}$. 

This simplifies to:

$$
k(x_i, x_j) = \sum_{k=1}^M \sum_{l=1}^M \phi_k(x_i) \phi_l(x_j) \int \int w_k w_l p(w_k, w_l) dw_k dw_l = \sum_{k=1}^M \sum_{l=1}^M A_{kl} \phi_k(x_i) \phi_l(x_j).
$$

Finally, this can be written compactly as:

$$
k(x_i, x_j) = \boldsymbol{\phi}(x_i)^\top \mathbf{A} \boldsymbol{\phi}(x_j).
$$

#### Special Case:
If $\mathbf{A} = \sigma_w^2 \mathbf{I}$, then:

$$
k(x_i, x_j) = \sigma_w^2 \sum_{k=1}^M \phi_k(x_i) \phi_k(x_j) = \sigma_w^2 \boldsymbol{\phi}(x_i)^\top \boldsymbol{\phi}(x_j).
$$

The inner product $\boldsymbol{\phi}(x_i)^\top \boldsymbol{\phi}(x_j)$ measures the **similarity** between the feature vectors $\phi(x_i)$ and $\phi(x_j)$. If the two inputs $x_i$ and $x_j$ are very similar, their feature vectors will also be similar, resulting in a large inner product. This means a high covariance, and viceversa. 


### From GPs to Linear Models
Conversely, any GP with covariance function $k(x, x') = \phi(x)^T A \phi(x')$ can be represented as a linear model with basis functions $\phi(x)$ and weight covariance $A$.

- **Mercer's Theorem**: Some covariance functions correspond to infinite-dimensional feature spaces.

## Computational Considerations

- **Gaussian Processes**: Complexity is $O(N^3)$ due to inversion of the $N \times N$ covariance matrix. Feasible for small to medium-sized datasets.
- **Linear Models**: Complexity is $O(N M^2)$, where $M$ is the number of basis functions. Can be more efficient when $M$ is small.

---

## 13-Covariance-Functions

### Key Concepts

1. **Covariance Functions and Hyperparameters**
   - Covariance functions define the structure of relationships in Gaussian Processes (GPs).
   - Hyperparameters control the behavior of covariance functions and are set using marginal likelihood.
   - Choosing the right covariance function and hyperparameters can aid in model selection and data interpretation.

2. **Common Covariance Functions**
   - **Stationary Covariance Functions**: Squared exponential, rational quadratic, and Matérn.
   - **Special Cases**: Radial Basis Function (RBF) networks, splines, large neural networks.
   - Covariance functions can be combined into more complex forms for better flexibility.

---

### Model Selection and Hyperparameters

1. **Hierarchical Model and ARD**
   - Hyperparameters of the covariance function are critical for model selection.
   - Automatic Relevance Determination (ARD) is useful for feature selection. For instance:
     $$
     k(x, x') = v_0^2 \exp\left(-\sum_{d=1}^D \frac{(x_d - x'_d)^2}{2v_d^2}\right),
     $$
     where hyperparameters $\theta = (v_0, v_1, \dots, v_D, \sigma_n^2)$.

2. **Interpretation**
   - Hyperparameters $v_d$ scale the importance of input dimensions $d$.
   - ARD enables automatic selection of relevant features in the data.

![[gp-2d.png]]

---

### Rational Quadratic Covariance Function

1. **Definition**
   - The rational quadratic (RQ) covariance function:
     $$
     k_{RQ}(r) = \left(1 + \frac{r^2}{2\alpha \ell^2}\right)^{-\alpha},
     $$
     where $\alpha > 0$ and $\ell$ is the characteristic length-scale.

2. **Interpretation**
   - RQ can be seen as a scale mixture (an infinite sum) of squared exponential (SE) covariance functions with varying length-scales.
   - In the limit $\alpha \to \infty$, the RQ covariance function becomes the SE covariance function:
     $$
     k_{SE}(r) = \exp\left(-\frac{r^2}{2\ell^2}\right).
     $$
     
![[gp-covariance.png]]

### Matérn Covariance Functions

1. **Definition**
   - The Matérn covariance function is given by:
     $$
     k_{\nu}(x, x') = \frac{1}{\Gamma(\nu) 2^{\nu-1}} \left( \sqrt{2\nu} \frac{\|x - x'\|}{\ell} \right)^\nu K_\nu \left( \sqrt{2\nu} \frac{\|x - x'\|}{\ell} \right),
     $$
     where $K_\nu$ is the modified Bessel function of the second kind and $\ell$ is the characteristic length-scale.

2. **Special Cases**
   - $\nu = \frac{1}{2}$: Exponential covariance function (Ornstein-Uhlenbeck process).
     $$
     k(r) = \exp\left(-\frac{r}{\ell}\right).
     $$
   - $\nu = \frac{3}{2}$: Once-differentiable function.
     $$
     k(r) = \left(1 + \sqrt{3} \frac{r}{\ell}\right) \exp\left(-\sqrt{3} \frac{r}{\ell}\right).
     $$
   - $\nu = \frac{5}{2}$: Twice-differentiable function.
     $$
     k(r) = \left(1 + \sqrt{5} \frac{r}{\ell} + \frac{5r^2}{3\ell^2}\right) \exp\left(-\sqrt{5} \frac{r}{\ell}\right).
     $$
   - $\nu \to \infty$: Equivalent to the SE covariance function.

3. **Intuition**
   - The hyperparameter $\nu$ controls the smoothness of the sampled functions. Larger $\nu$ implies smoother functions.

![[gp-cov.png]]

---

### Periodic Covariance Functions

1. **Definition**
   - Periodic covariance functions model periodic data:
     $$
     k_{periodic}(x, x') = \exp\left(-\frac{2 \sin^2(\pi |x - x'| / p)}{\ell^2}\right),
     $$
     where $p$ is the period and $\ell$ is the characteristic length-scale.

2. **Intuition**
   - By transforming the inputs into $u = (\sin(x), \cos(x))^\top$, the covariance measures periodic distances in this transformed space.

![[gp-covar.png]]

Three functions drawn at random; left $> 1$, and right $< 1$.

### Splines and Gaussian Processes

1. **Cubic Splines**
   - The solution to the minimization problem:
     $$
     \sum_{i=1}^n (f(x^{(i)}) - y^{(i)})^2 + \lambda \int (f''(x))^2 dx
     $$
     is the natural cubic spline.

2. **GP Interpretation**
   - The same function can be derived as the posterior mean of a GP with a specific covariance function:
     $$
     k(x, x') = \sigma^2 + xx'\sigma^2 + \lambda \int_0^1 \min(x, x')^3 dx.
     $$

![[gp-vis.png]]
### Neural Networks and GPs

1. **Large Neural Networks**
   - As the number of hidden units in a neural network grows, the output becomes equivalent to a GP with a specific covariance function:
     $$
     k(x, x') = \sigma^2 \arcsin\left(\frac{2x^\top \Sigma x'}{\sqrt{(1 + x^\top \Sigma x)(1 + x'^\top \Sigma x')}}\right).
     $$

2. **Intuition**
   - The prior distribution over neural network weights induces a prior over functions, which resembles a GP.

### Composite Covariance Functions

Covariance functions have to be possitive definite.

1. **Combining Covariance Functions**
   - Covariance functions can be combined to form new ones:
     - **Sum**: $k(x, x') = k_1(x, x') + k_2(x, x')$
     - **Product**: $k(x, x') = k_1(x, x') \cdot k_2(x, x')$
     - **Other**: $k(x, x') = g(x) k(x, x') g(x')$

2. **Applications**
   - Composite covariance functions allow for greater modeling flexibility, tailoring the GP to specific data structures.

---

## 14-Finite-and-Infinite-Basis-GPs

1. **Finite vs. Infinite Models**  
   - A central question in modeling is whether finite or infinite models should be preferred.
   - **Finite Models**: Involve fixed parameters and limited basis functions. These make much stronger assumptions about the data and can lack flexibility.
   - **Infinite Models (Gaussian Processes)**: Allow a theoretically infinite number of basis functions, offering more flexibility. Gaussian Processes (GPs) serve as a formalism to define such infinite models.

2. **Gaussian Processes as Infinite Models**  
   - GPs represent a fancy yet practical way to implement infinite models. But, the key question is:
     - *Do infinite models make a difference in practice?*
   - Yes, because they avoid overfitting and ensure generalization by accounting for all possible functions consistent with the data.

3. **Illustrative Example**  
   - A GP with a squared exponential covariance function corresponds to an infinite linear model with Gaussian basis functions **placed everywhere in the input space**, not just at training points. This results in smoother, more realistic models.
![[gp-finite.png]]
### Dangers of Finite Basis Functions

1. **Finite Linear Models with Localized Basis Functions**
   - Example: A model with only **five basis functions** is constrained to represent limited patterns.
   - **Visualization**:
     - Finite models show high variance and poor uncertainty estimation in regions without training data.
     - As more data is added, the performance improves, but the limited number of basis functions prevents robust generalization.

2. **Gaussian Processes with Infinite Basis Functions**
   - In contrast, a GP:
     - Uses infinitely many basis functions.
     - Ensures smooth predictions and uncertainty estimates across the input space.
   - **Key Difference**: GPs generalize even in regions far from training points by leveraging the covariance function.

### From Infinite Linear Models to Gaussian Processes

1. **Infinite Basis Expansion**  
   The GP framework arises naturally by considering a sum of Gaussian basis functions:
   $$
   f(x) = \lim_{N \to \infty} \frac{1}{N} \sum_{n=-N/2}^{N/2} \gamma_n \exp\left(-\left(x - \frac{n}{\sqrt{N}}\right)^2\right),
   $$
   where $\gamma_n \sim \mathcal{N}(0, 1)$.

   - **Interpretation**: As $N \to \infty$, this sum transitions from a finite representation to a continuous integral:
     $$
     f(x) = \int_{-\infty}^{\infty} \gamma(u) \exp(-(x - u)^2) \, du,
     $$
     with $\gamma(u) \sim \mathcal{N}(0, 1)$.


2. **GP Foundations**  
   - **Mean Function**:

$$
\mu(x) = \mathbb{E}[f(x)] = \int_{-\infty}^\infty \exp(-(x - u)^2) \int_{-\infty}^\infty \gamma(u)p(\gamma(u)) d\gamma(u) \, du = 0,
$$

     assuming zero-mean priors for $\gamma(u)$.

   - **Covariance Function**:

$$
\mathbb{E}[f(x)f(x')] = \int_{-\infty}^\infty \exp\left(-(x - u)^2 - (x' - u)^2\right) du
$$

$$
= \int \exp\left(-2\left(u - \frac{x + x'}{2}\right)^2 + \frac{(x + x')^2}{2} - x^2 - x'^2\right) du \propto \exp\left(-\frac{(x - x')^2}{2}\right).
$$

     - **Key Insight**: The squared exponential covariance function encapsulates an infinite number of Gaussian-shaped basis functions.

3. **Practical Implication**  
   The GP enables regression over the entire input space, avoiding the overfitting often seen in finite models.

### Practical Takeaways

1. **When to Choose GPs**: 
   - When uncertainty matters (e.g., in scientific predictions or safety-critical systems).
   - When flexibility is essential due to limited training data.
2. **Limitations of GPs**:
   - Computational cost grows cubically with the number of data points, making scalability a challenge.
   - Solutions: Sparse approximations or variational inference.


