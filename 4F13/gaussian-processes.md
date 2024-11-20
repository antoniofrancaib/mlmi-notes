## 9. Gaussian Processes

### From Scalar Gaussians to Multivariate Gaussians to Gaussian Processes

1. **Scalar Gaussian**: A single random variable $x$ with distribution $N(\mu, \sigma^2)$.

2. **Multivariate Gaussian**: A vector $x = [x_1, x_2, \dots, x_N]^T$ with joint Gaussian distribution:

   $$
   p(x \mid \mu, \Sigma) = \frac{1}{(2 \pi)^{N/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)
   $$

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

The marginalization property simplifies Gaussian processes (GPs) by leveraging their unique characteristics. The marginalization property allows you to work with finite-dimensional slices of the GP. Specifically: $$ p(\mathbf{x}) = \int p(\mathbf{x}, \mathbf{y}) \, d\mathbf{y}. $$ For a multivariate Gaussian: $$ \begin{bmatrix} \mathbf{x} \\ \mathbf{y} \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \mathbf{a} \\ \mathbf{b} \end{bmatrix}, \begin{bmatrix} A & B^\top \\ B & C \end{bmatrix} \right) \quad \Rightarrow \quad p(\mathbf{x}) \sim \mathcal{N}(\mathbf{a}, A). $$ In Gaussian processes, this property enables predictions based only on finite-dimensional covariance matrices without handling infinite-dimensional computations. 

### GP as a distribution over functions
A GP defines a distribution over functions. Each finite collection of function values follows a multivariate Gaussian distribution. 

**Example**: $$ p(f) \sim \mathcal{N}(m, k), \quad m(x) = 0, \quad k(x, x') = \exp\left(-\frac{1}{2}(x - x')^2\right). $$ For a finite set of points $\{x_1, x_2, ..., x_N\}$, the function values $f(x_1), f(x_2), ..., f(x_N)$ are jointly Gaussian: $$ f \sim \mathcal{N}(0, \Sigma), \quad \Sigma_{ij} = k(x_i, x_j). $$To visualize a GP, draw samples from this multivariate Gaussian and plot them as functions.

### Generating Functions from a GP
- **Goal**: Generate samples from a joint Gaussian distribution with mean $\mathbf{m}$ and covariance $\mathbf{K}$.
Simpler case; assume $m = 0$: 

1. **Select Inputs**: Choose $N$ input points $x_1, x_2, \dots, x_N$.
2. **Compute Covariance Matrix**: $K_{ij} = k(x_i, x_j)$.
3. **Sample Function Values**: Draw $f \sim N(0, K)$.
4. **Plot Function**: Plot $f$ versus $x$.

Similarly, for $m \neq 0$:
  1. Generate random standard normal samples $\mathbf{z} \sim \mathcal{N}(0, I)$.
  2. Compute $\mathbf{y} = \text{chol}(\mathbf{K})^\top \mathbf{z} + \mathbf{m}$,
     where $\text{chol}(\mathbf{K})$ is the Cholesky decomposition of $\mathbf{K}$ such that $\mathbf{R}^\top \mathbf{R} = \mathbf{K}$.

The Cholesky factorization ensures the generated samples have the correct covariance structure $\mathbf{K}$.

#### Sequential Generation
Generate function values one at a time, conditioning on previous values. This uses properties of conditional Gaussians.

- **Factorization**:
  $$
  p(f_1, ..., f_N \mid x_1, ..., x_N) = \prod_{n=1}^N p(f_n \mid f_{<n}, x_{<n}).
  $$

- **Gaussian Process Case**:
  - The joint prior:
    $$
    p(f_n, f_{<n}) = \mathcal{N} \left( \begin{bmatrix} \mathbf{a} \\ \mathbf{b} \end{bmatrix}, \begin{bmatrix} A & B^\top \\ B & C \end{bmatrix} \right).
    $$
  - Conditional distribution:
    $$
    p(f_n \mid f_{<n}) = \mathcal{N} \left(\mathbf{a} + BC^{-1}(\mathbf{f}_{<n} - \mathbf{b}), A - BC^{-1}B^\top \right).
    $$

- **Utility**:
  - Enables sequential sampling of function values from a GP.
  - Sequential updates provide a practical way to incorporate new data points without recomputing the entire covariance matrix.

- **Illustration**:
  - The shaded regions and lines in the plots show how the GP updates its predictions as new data points are added.
  - 
![[Pasted image 20241119181834.png]]

---
## 10. Gaussian Processes and Data

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
\mathbf{m}_{\mid \mathbf{y}}(x) &= \mathbf{k}(x, x)[\mathbf{K}(x, x) + \sigma^2_{\text{noise}} \mathbf{I}]^{-1} \mathbf{y}, \\
\mathbf{k}_{\mid \mathbf{y}}(x, x') &= \mathbf{k}(x, x') - \mathbf{k}(x, x)[\mathbf{K}(x, x) + \sigma^2_{\text{noise}} \mathbf{I}]^{-1} \mathbf{k}(x, x').
\end{aligned}
$$

### Prior and Posterior 
- **Prior**: Represents our beliefs about the function before seeing any data.
- **Posterior**: Updated beliefs after incorporating observed data.

**Visualization**:
- **Prior Samples**: Functions drawn from the GP prior.
- **Posterior Samples**: Functions drawn from the GP posterior, which now pass through (or near) the observed data points.

![[Pasted image 20241119184053.png]]

### Predictive Distribution

The predictive distribution for a new input $x_*$ is given by:

$$
p(y_* \mid x_*, \mathbf{x}, \mathbf{y}) \sim \mathcal{N} \left( \mathbf{k}(x_*, \mathbf{x})^\top \left[\mathbf{K} + \sigma_{\text{noise}}^2 \mathbf{I} \right]^{-1} \mathbf{y}, \, \mathbf{k}(x_*, x_*) + \sigma_{\text{noise}}^2 - \mathbf{k}(x_*, \mathbf{x})^\top \left[\mathbf{K} + \sigma_{\text{noise}}^2 \mathbf{I} \right]^{-1} \mathbf{k}(x_*, \mathbf{x}) \right).
$$

- **Mean**: Describes the predicted value at $x_*$.
- **Variance**: Quantifies uncertainty at $x_*$.

---

### Interpretation of the Predictive Mean and Variance

#### Predictive Mean:
$$
\mu(x_*) = \mathbf{k}(x_*, \mathbf{x}) \left[\mathbf{K} + \sigma_{\text{noise}}^2 \mathbf{I} \right]^{-1} \mathbf{y}.
$$

- This is a weighted sum of the observed data $\mathbf{y}$, where the weights depend on the kernel.

- In kernel literature, this is often expressed as:
  $$
  \mu(x_*) = \sum_{n=1}^N \beta_n y_n = \sum_{n=1}^N \alpha_n k(x_*, x_n).
  $$
  - **Weights** ($\beta_n, \alpha_n$): Depend on the covariance structure and how $x_*$ relates to the training points $x_n$.

---

#### Predictive Variance:
$$
\sigma^2(x_*) = \mathbf{k}(x_*, x_*) - \mathbf{k}(x_*, \mathbf{x}) \left[\mathbf{K} + \sigma_{\text{noise}}^2 \mathbf{I} \right]^{-1} \mathbf{k}(x_*, \mathbf{x}).
$$

- The variance has two terms:
  1. **Prior Variance** ($\mathbf{k}(x_*, x_*)$): The initial uncertainty in the prior.
  2. **Explained Variance**: Subtracted based on how well the observed data explains $x_*$.

#### Key Insight:
1. The variance $\sigma^2(x_*)$ decreases as $x_*$ gets closer to the observed data points, reflecting more confidence in predictions.
2. The variance is **independent of the observed outputs** $\mathbf{y}$, only depending on the input locations $\mathbf{x}$.


---

## 11. Gaussian Process Marginal Likelihood and Hyperparameters

### The GP Marginal Likelihood
The marginal likelihood (or evidence) is the probability of the observed data under the GP model:

$$
p(y \mid x) = \int p(y \mid f) p(f) \, df
$$

For GPs with Gaussian noise, this integral can be computed analytically:

$$
\log p(y \mid x) = -\frac{1}{2} y^T (K + \sigma_n^2 I)^{-1} y - \frac{1}{2} \log |K + \sigma_n^2 I| - \frac{N}{2} \log 2 \pi
$$

**Interpretation**:
- The first term measures how well the model fits the data (data fit).
- The second term penalizes model complexity (complexity penalty).
- Occam's Razor is automatically applied, preferring simpler models that explain the data well.

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

## Correspondence Between Linear Models and Gaussian Processes
<span style="color:red;">LEFT HERE!</span>

### From Linear Models to GPs
Consider a linear model with Gaussian priors:

$$
f(x) = \sum_{m=1}^M w_m \phi_m(x), \quad w_m \sim N(0, \sigma_w^2)
$$

- **Mean Function**: $m(x) = E[f(x)] = 0$
- **Covariance Function**:

  $$
  k(x, x') = E[f(x) f(x')] = \sigma_w^2 \sum_{m=1}^M \phi_m(x) \phi_m(x') = \sigma_w^2 \phi(x)^T \phi(x')
  $$

This shows that the linear model with Gaussian priors corresponds to a GP with covariance function $k(x, x')$.

### From GPs to Linear Models
Conversely, any GP with covariance function $k(x, x') = \phi(x)^T A \phi(x')$ can be represented as a linear model with basis functions $\phi(x)$ and weight covariance $A$.

- **Mercer's Theorem**: Some covariance functions correspond to infinite-dimensional feature spaces.

## Computational Considerations

- **Gaussian Processes**: Complexity is $O(N^3)$ due to inversion of the $N \times N$ covariance matrix. Feasible for small to medium-sized datasets.
- **Linear Models**: Complexity is $O(N M^2)$, where $M$ is the number of basis functions. Can be more efficient when $M$ is small.

## Covariance Functions

### Stationary Covariance Functions
Covariance functions that depend only on $r = |x - x'|$.

1. **Squared Exponential (SE)**

   $$
   k_{\text{SE}}(r) = \sigma_f^2 \exp \left( -\frac{r^2}{2 \ell^2} \right)
   $$

2. **Rational Quadratic (RQ)**

   $$
   k_{\text{RQ}}(r) = \sigma_f^2 \left( 1 + \frac{r^2}{2 \alpha \ell^2} \right)^{-\alpha}
   $$

3. **Matérn**

   $$
   k_{\text{Matérn}}(r) = \sigma_f^2 \frac{2^{1 - \nu}}{\Gamma(\nu)} \left( \frac{\sqrt{2 \nu} r}{\ell} \right)^\nu K_\nu \left( \frac{\sqrt{2 \nu} r}{\ell} \right)
   $$

4. **Periodic Covariance Function**

   $$
   k_{\text{Per}}(x, x') = \sigma_f^2 \exp \left( -\frac{2 \sin^2 \left( \frac{\pi |x - x'|}{p} \right)}{\ell^2} \right)
   $$

5. **Neural Network Covariance Function**

   $$
   k_{\text{NN}}(x, x') = \frac{\sigma_f^2}{\pi} \sin^{-1} \left( \frac{2 x^T \Sigma x'}{\sqrt{(1 + 2 x^T \Sigma x)(1 + 2 x'^T \Sigma x')}} \right)
   $$

### Combining Covariance Functions
- **Addition**: $k(x, x') = k_1(x, x') + k_2(x, x')$
- **Multiplication**: $k(x, x') = k_1(x, x') \cdot k_2(x, x')$
- **Scaling**: $k(x, x') = g(x) k(x, x') g(x')$, where $g(x)$ is a function.

## Conclusion
Gaussian Processes offer a robust, flexible framework for modeling complex datasets without specifying a fixed number of parameters. By defining a prior directly over functions, GPs capture our beliefs about function properties such as smoothness and periodicity. The marginal likelihood provides a principled way to select hyperparameters and models, embodying Occam's Razor by balancing data fit and model complexity. Understanding the relationship between linear models and GPs, as well as the role of covariance functions, is crucial for effectively applying GPs to real-world problems.

