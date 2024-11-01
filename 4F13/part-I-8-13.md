## Distributions Over Parameters and Functions

### Priors on Parameters Induce Priors on Functions
In parametric models, we define a model $f_w(x)$ using parameters $w$:

$$
f_w(x) = \sum_{m=0}^M w_m \phi_m(x)
$$

where $\phi_m(x)$ are basis functions (e.g., polynomial terms $x^m$) and $w_m$ are the weights.

By placing a prior distribution $p(w)$ over the parameters $w$, we implicitly define a prior over functions $f(x)$.

**Example:**
- Choose $M=17$.
- Set $p(w_m) = N(0, \sigma_w^2)$ for all $m$.
- The prior over $w$ induces a distribution over functions $f(x)$.

This means that by sampling from the prior over $w$, we can generate random functions from the prior over $f(x)$.

### Nuisance Parameters and Distributions Over Functions
Parameters $w$ are often nuisance parameters—variables that are not of direct interest but are necessary for the model. In many cases, we care more about the functions $f(x)$ and the predictions they make than about the specific values of $w$.

In Bayesian inference, we marginalize over the nuisance parameters to make predictions:

$$
p(f_* \mid y) = \int p(f_* \mid w) p(w \mid y) \, dw
$$

### Working Directly with Functions
Given that parameters can be a nuisance and that we are primarily interested in functions, a natural question arises: Can we work directly in the space of functions?

**Advantages:**
- **Simpler inference**: Avoids integrating over high-dimensional parameter spaces.
- **Better understanding**: Directly specifies our beliefs about functions.

This leads us to consider models that define priors over functions without explicit parameters, such as Gaussian Processes.

## Gaussian Processes

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

### Generating Functions from a GP
To generate sample functions:

1. **Select Inputs**: Choose $N$ input points $x_1, x_2, \dots, x_N$.
2. **Compute Covariance Matrix**: $K_{ij} = k(x_i, x_j)$.
3. **Sample Function Values**: Draw $f \sim N(0, K)$.
4. **Plot Function**: Plot $f$ versus $x$.

#### Sequential Generation
Generate function values one at a time, conditioning on previous values. This uses properties of conditional Gaussians.

## Gaussian Processes and Data

### Conditioning on Observations
Given observed data $D = \{(x_i, y_i)\}_{i=1}^N$, we want to predict $f_*$ at new inputs $x_*$.

Assumption: Observations $y_i$ are noisy versions of the true function $f(x_i)$:

$$
y_i = f(x_i) + \epsilon_i, \quad \epsilon_i \sim N(0, \sigma_n^2)
$$

### Posterior Gaussian Process
**Posterior Mean and Covariance**:

$$
E[f_* \mid x, y, x_*] = k(x_*, x) [K + \sigma_n^2 I]^{-1} y
$$

$$
\text{Var}[f_* \mid x, y, x_*] = k(x_*, x_*) - k(x_*, x) [K + \sigma_n^2 I]^{-1} k(x, x_*)
$$

- $K$ is the covariance matrix of the training inputs.
- $k(x_*, x)$ is the vector of covariances between the test input $x_*$ and training inputs $x$.

### Prior and Posterior in Pictures
- **Prior**: Represents our beliefs about the function before seeing any data.
- **Posterior**: Updated beliefs after incorporating observed data.

**Visualization**:
- **Prior Samples**: Functions drawn from the GP prior.
- **Posterior Samples**: Functions drawn from the GP posterior, which now pass through (or near) the observed data points.

## Gaussian Process Marginal Likelihood and Hyperparameters

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

## Appendix: Useful Gaussian and Matrix Identities

### Matrix Identities
1. **Matrix Inversion Lemma (Woodbury Identity)**:

   $$
   (A + UCV^T)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V^T A^{-1} U)^{-1} V^T A^{-1}
   $$

2. **Determinant Identity**:

   $$
   |A + UCV^T| = |A| |C| |C^{-1} + V^T A^{-1} U|
   $$

### Gaussian Identities

**Conditional of a Joint Gaussian**:
Given $\begin{bmatrix} x \\ y \end{bmatrix} \sim N \left( \begin{bmatrix} a \\ b \end{bmatrix}, \begin{bmatrix} A & B \\ B^T & C \end{bmatrix} \right)$:

1. **Marginal of $x$**: $p(x) = N(x \mid a, A)$
2. **Conditional of $x$ given $y$**:

   $$
   p(x \mid y) = N(x \mid a + BC^{-1}(y - b), A - BC^{-1}B^T)
   $$
