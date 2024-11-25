# Regression

![Alt text for the image](../assets/1.png)

Regression $\rightarrow$ predicting a **continuous output** $y^\star$ given an input vector $x^\star \in \mathbb{R}^D$

**Regression Goal**: to find a function $f: \mathbb{R}^D \rightarrow \mathbb{R}^{D'}$ to make future predictions $y^\star = f(x^\star)$.

_Regression Jargon_: 
- The inputs $\mathbf{x}_n$ are also known as the features, covariates, or independent variables.
- The outputs $y_n$ are also known as the responses, targets, or dependent variables.

Output space marks the difference between scalar regression or multivariate regression. 
Problems may involve ***interpolation***, where predictions are close to the training data, or ***extrapolation***, where predictions are far from the training data.

![[Pasted image 20241107124021.png]]

---
# Linear-Regression

## Least Squares Fitting

Consider a dataset with $N$ scalar input-output pairs $\{(x_{n}, y_{n})\}_{n=1}^N$. We aim to fit a linear model:
$$f(x) = w_1 x + w_0$$

We estimate $w_0$ and $w_1$ by minimizing the cost function $C_2$, defined as the sum of squared distances between observed outputs and the model predictions:

$$
C_2 = \sum_{n=1}^N [y_n - f(x_n)]^2 = \sum_{n=1}^N [y_n - (w_1 x_n + w_0)]^2 \geq 0
$$

This can be compactly written in matrix notation as:

$$
C_2 = \| \mathbf{y} - \mathbf{X} \mathbf{w} \|^2 = (\mathbf{y} - \mathbf{X} \mathbf{w})^\top (\mathbf{y} - \mathbf{X} \mathbf{w})
$$

To minimize $C_2$, we differentiate and set the derivative to zero, resulting in the normal equation:

$$
\frac{\partial C_2}{\partial \mathbf{w}} = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w}) = 0 \implies \mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}
$$

Solving for $\mathbf{w}$, we get the least squares solution:

$$
\boxed{\mathbf{w} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}}
$$

The Moore-Penrose pseudoinverse generalizes this inverse for non-square matrices.

### Justification for Least Squares

Why minimize $E_2$ instead of a more general error function $E_p = \sum_{n=1}^N | y_n - (w_1 x_n + w_0) |^p$ for some $p > 0$? Two key reasons are:

- **Mathematical Tractability**: Minimizing $E_2$ yields a closed-form solution.
- **Maximum Likelihood Interpretation**: If we assume the outputs $y_n$ are corrupted by Gaussian noise:
$$ y_n = w_1 x_n + w_0 + \epsilon_n, \quad \epsilon_n \sim \mathcal{N}(0, \sigma^2)$$
  then minimizing $E_2$ is equivalent to finding the maximum likelihood estimates of $w_0$ and $w_1$.

![Linear Regression Weight Excursion](reg_lin_weight_excursion.gif)
### Maximum Likelihood Derivation

The likelihood of the data given the parameters is:
$$
p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}, \sigma^2) = \frac{1}{(2\pi \sigma^2)^{N/2}} \exp\left( -\frac{1}{2\sigma^2} (\mathbf{y} - \mathbf{X} \mathbf{w})^\top (\mathbf{y} - \mathbf{X} \mathbf{w}) \right)
$$

Maximizing the likelihood is equivalent to minimizing the negative log-likelihood:

$$
- \mathcal{L}(\mathbf{w}) = - \ln p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}, \sigma^2) = \frac{N}{2} \ln(2\pi \sigma^2) + \frac{1}{2\sigma^2} (\mathbf{y} - \mathbf{X} \mathbf{w})^\top (\mathbf{y} - \mathbf{X} \mathbf{w})
$$

Since the first term is independent of $\mathbf{w}$, minimizing the negative log-likelihood reduces to minimizing $E_2$. Therefore:

$$
\boxed{ \text{Least Squares} \equiv \text{Maximum Likelihood Estimation with Gaussian Noise} }
$$

---
# Non-Linear-Regression 

The non-linear function is modeled as a linear combination of basis functions:

$$
y_n = f(x_n) + \epsilon_n \quad \text{where} \quad \epsilon_n \sim \mathcal{N}(0, \sigma_{y}^2)
$$

$$
f(x_n) = w_0 + w_1 \phi_{1}(x_n) + w_2 \phi_{2}(x_n) + \cdots + w_D \phi_{D}(x_n)
$$

Suitable basis functions include:
- Polynomials: $\phi_{d}(x) = x^d$
- Sinusoids: $\phi_{d}(x) = \cos(\omega_d x + \phi_d)$
- Gaussian bumps: $\phi_d(x) = \exp\left( -\frac{(x - \mu_d)^2}{2\sigma^2} \right)$

![[Pasted image 20241107130144.png]]

## Least Squares and Maximum Likelihood Fitting

The model remains linear in parameters. Define parameter vector $\mathbf{w}$ and basis function vector $\boldsymbol{\phi}(x_n)$:

$$
\mathbf{w} = [w_0, w_1, \ldots, w_D]^\top
$$

$$
\boldsymbol{\phi}(x_n) = [1, \phi_1(x_n), \ldots, \phi_D(x_n)]^\top
$$

So the model becomes:

$$
y_n = \boldsymbol{\phi}(x_n)^\top \mathbf{w} + \epsilon_n
$$

Collect observations into vectors $\mathbf{y}$ and $\boldsymbol{\epsilon}$:

$$
\mathbf{y} = \boldsymbol{\Phi}\mathbf{w} + \boldsymbol{\epsilon}
$$

Where the design matrix $\boldsymbol{\Phi}$ has entries $\phi_d(x_n)$:

$$
\boldsymbol{\Phi} = \begin{pmatrix}
1 & \phi_1(x_1) & \cdots & \phi_D(x_1)\\
1 & \phi_1(x_2) & \cdots & \phi_D(x_2)\\
\vdots & \vdots & \ddots & \vdots \\
1 & \phi_1(x_N) & \cdots & \phi_D(x_N)
\end{pmatrix}
$$

### Least Squares Error

Minimize the squared error:
$$
C_2 = \big|\big| \mathbf{y} - \boldsymbol{\Phi}\mathbf{w} \big|\big|^2 = (\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})^\top (\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})
$$

### Maximum Likelihood

Negative log-likelihood:

$$
- \mathcal{L}(\mathbf{w}) = \frac{N}{2}\log(2\pi \sigma^2) + \frac{1}{2\sigma^2}(\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})^\top (\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})
$$

Both approaches yield the same solution for weights:

$$
\boxed{\mathbf{w} = \big( \boldsymbol{\Phi}^\top \boldsymbol{\Phi} \big)^{-1} \boldsymbol{\Phi}^\top \mathbf{y}}
$$
### Overfitting

Using a large number of basis functions can lead to **overfitting**, where the model fits the training data too closely, capturing noise rather than the underlying relationship. Indicators of overfitting include:

- Small training error but large test error.
- Excessively large weights with alternating signs.
- Highly oscillatory fitted functions.

### Regularization

To combat overfitting, we introduce a **regularization** term to penalize large weights:

$$
E_{\text{reg}} = E_2 + \frac{\lambda}{2} \mathbf{w}^\top \mathbf{w}
$$

Minimizing $E_{\text{reg}}$ leads to the regularized solution:
$$
\boxed{ \mathbf{w} = (\boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \lambda \mathbf{I})^{-1} \boldsymbol{\Phi}^\top \mathbf{y} }
$$

This method, known as **ridge regression** or **$L2$ regularization**, discourages large weights and improves generalization.

## Sensitivity to Hyperparameters

Experiment with:
- Order of the polynomial ($D$)
- Number of Gaussian basis functions ($D$)
- Width of basis functions ($\sigma^2_{\phi}$)

--- 
# Bayesian-Linear-Regression

## Probabilistic Model

In Bayesian linear regression, we model the relationship between inputs $\mathbf{x}$ and outputs $y$ using a linear combination of basis functions with a probabilistic approach.

### Prior Over Weights

We place a Gaussian prior over the weights $\mathbf{w}$:

$$
\mathbf{w} \sim \mathcal{N}\left( \mathbf{w}; \mathbf{0}, \sigma_{\mathbf{w}}^2 \mathbf{I} \right)
$$

where $\mathbf{I}$ is the identity matrix, and $\sigma_{\mathbf{w}}^2$ controls the prior variance.

### Regression Function

The regression function is defined as:

$$
f_{\mathbf{w}}(\mathbf{x}) = \boldsymbol{\phi}(\mathbf{x})^\top \mathbf{w}
$$

where $\boldsymbol{\phi}(\mathbf{x})$ is a vector of basis functions evaluated at input $\mathbf{x}$.

### Observation Noise

Observations are generated by adding Gaussian noise to the regression function:

$$
y_n = f_{\mathbf{w}}(\mathbf{x}_n) + \epsilon_n, \quad \epsilon_n \sim \mathcal{N}\left( \epsilon_n; 0, \sigma_y^2 \right)
$$

This implies:

$$
y_n \mid \mathbf{w}, \mathbf{x}_n \sim \mathcal{N}\left( y_n; f_{\mathbf{w}}(\mathbf{x}_n), \sigma_y^2 \right)
$$

### Likelihood of Observations

Given a set of observations $\mathbf{y} = [y_1, y_2, \dots, y_N]^\top$ and corresponding inputs $\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N]^\top$, the likelihood is:

$$
p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}, \sigma_y^2) = \mathcal{N}\left( \mathbf{y}; \boldsymbol{\Phi} \mathbf{w}, \sigma_y^2 \mathbf{I} \right)
$$

where $\boldsymbol{\Phi}$ is the design matrix with rows $\boldsymbol{\phi}(\mathbf{x}_n)^\top$.

### Full Probabilistic Model

The joint distribution over weights, observations, and inputs is:

$$
p(\mathbf{w}, \mathbf{y}, \mathbf{X}) = p(\mathbf{w}) \prod_{n=1}^{N} p(y_n \mid \mathbf{w}, \mathbf{x}_n)
$$

## Probabilistic Inference for the Weights

Our goal is to compute the posterior distribution of the weights $\mathbf{w}$ given the data $\mathbf{y}$ and inputs $\mathbf{X}$:

$$
p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}) = \frac{p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) p(\mathbf{w})}{p(\mathbf{y} \mid \mathbf{X})}
$$

Since $p(\mathbf{y} \mid \mathbf{X})$ does not depend on $\mathbf{w}$, we have:

$$
p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}) \propto p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) p(\mathbf{w})
$$
**marginal likelihood**, **evidence**, or **normalizing constant** $\rightarrow$ measure of how well a model explains the data, regardless of the specific parameter values $\mathbf{w}$ -- different models can be compared by their marginal likelihoods; models with higher marginal likelihoods are considered to better explain the observed data.
### Prior Distribution

The prior over weights is:

$$
p(\mathbf{w}) = \mathcal{N}\left( \mathbf{w}; \mathbf{0}, \sigma_{\mathbf{w}}^2 \mathbf{I} \right)
$$

### Likelihood Function

The likelihood function is:

$$
p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) = \mathcal{N}\left( \mathbf{y}; \boldsymbol{\Phi} \mathbf{w}, \sigma_y^2 \mathbf{I} \right)
$$

### Combining Prior and Likelihood

The posterior is proportional to:

$$
p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}) \propto \exp\left( -\frac{1}{2\sigma_{\mathbf{w}}^2} \mathbf{w}^\top \mathbf{w} - \frac{1}{2\sigma_y^2} (\mathbf{y} - \boldsymbol{\Phi} \mathbf{w})^\top (\mathbf{y} - \boldsymbol{\Phi} \mathbf{w}) \right)
$$

### Completing the Square

To express the posterior in standard Gaussian form, we complete the square in $\mathbf{w}$.

Let:

$$
\begin{align*}
\mathbf{A} &= \frac{1}{\sigma_{\mathbf{w}}^2} \mathbf{I} + \frac{1}{\sigma_y^2} \boldsymbol{\Phi}^\top \boldsymbol{\Phi} \\
\mathbf{b} &= \frac{1}{\sigma_y^2} \boldsymbol{\Phi}^\top \mathbf{y}
\end{align*}
$$

Then the exponent becomes:

$$
-\frac{1}{2} \left( \mathbf{w}^\top \mathbf{A} \mathbf{w} - 2 \mathbf{b}^\top \mathbf{w} \right)
$$

### Posterior Distribution

The posterior distribution of the weights is Gaussian:

$$
p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}) = \mathcal{N}\left( \mathbf{w}; \boldsymbol{\mu}, \boldsymbol{\Sigma} \right)
$$

where:

$$
\boldsymbol{\Sigma} = \mathbf{A}^{-1} = \left( \frac{1}{\sigma_{\mathbf{w}}^2} \mathbf{I} + \frac{1}{\sigma_y^2} \boldsymbol{\Phi}^\top \boldsymbol{\Phi} \right)^{-1}
$$

and:

$$
\boldsymbol{\mu} = \boldsymbol{\Sigma} \left( \frac{1}{\sigma_y^2} \boldsymbol{\Phi}^\top \mathbf{y} \right)
$$

## Predictive Inference

To make predictions for a new input $\mathbf{x}^\ast$, we compute the predictive distribution of the output $y^\ast$ by integrating over the posterior distribution of $\mathbf{w}$:

$$
p(y^\ast \mid \mathbf{x}^\ast, \mathbf{y}, \mathbf{X}) = \int p(y^\ast \mid \mathbf{x}^\ast, \mathbf{w}) p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}) \, d\mathbf{w}
$$

### Conditional Distribution of $y^\ast$ Given $\mathbf{w}$

The conditional distribution is:

$$
p(y^\ast \mid \mathbf{x}^\ast, \mathbf{w}) = \mathcal{N}\left( y^\ast; \boldsymbol{\phi}(\mathbf{x}^\ast)^\top \mathbf{w}, \sigma_y^2 \right)
$$

### Predictive Distribution

Since both $p(y^\ast \mid \mathbf{x}^\ast, \mathbf{w})$ and $p(\mathbf{w} \mid \mathbf{y}, \mathbf{X})$ are Gaussian, the predictive distribution is also Gaussian:

$$
p(y^\ast \mid \mathbf{x}^\ast, \mathbf{y}, \mathbf{X}) = \mathcal{N}\left( y^\ast; \mu_{y^\ast}, \sigma_{y^\ast}^2 \right)
$$

where:

$$
\begin{align*}
\mu_{y^\ast} &= \boldsymbol{\phi}(\mathbf{x}^\ast)^\top \boldsymbol{\mu} \\
\sigma_{y^\ast}^2 &= \boldsymbol{\phi}(\mathbf{x}^\ast)^\top \boldsymbol{\Sigma} \boldsymbol{\phi}(\mathbf{x}^\ast) + \sigma_y^2
\end{align*}
$$

![[Pasted image 20241107170412.png]]

## Equivalence to Regularization

### Posterior Distribution and MAP Estimation

When the prior over weights is Gaussian with mean $\mathbf{m}_0$ and covariance $\mathbf{S}_0$, the posterior mean is:

$$
\boldsymbol{\mu} = \boldsymbol{\Sigma} \left( \frac{1}{\sigma_y^2} \boldsymbol{\Phi}^\top \mathbf{y} + \mathbf{S}_0^{-1} \mathbf{m}_0 \right)
$$

with:

$$
\boldsymbol{\Sigma} = \left( \frac{1}{\sigma_y^2} \boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \mathbf{S}_0^{-1} \right)^{-1}
$$

### Equivalence to Ridge Regression

Consider the case where $\mathbf{m}_0 = \mathbf{0}$ and $\mathbf{S}_0^{-1} = \lambda \mathbf{I}$. Then the posterior mean simplifies to:

$$
\boldsymbol{\mu} = \left( \boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \lambda \sigma_y^2 \mathbf{I} \right)^{-1} \boldsymbol{\Phi}^\top \mathbf{y}
$$

Assuming $\sigma_y^2 = 1$ for simplicity:

$$
\boldsymbol{\mu} = \left( \boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \lambda \mathbf{I} \right)^{-1} \boldsymbol{\Phi}^\top \mathbf{y}
$$

This is the solution to the ridge regression problem, showing that the Maximum A Posteriori (MAP) estimate in Bayesian linear regression with a Gaussian prior corresponds to regularized least squares.

Thus, introducing a Gaussian prior with zero mean and precision $\lambda \mathbf{I}$ is equivalent to adding an $\ell_2$ regularization term to the least squares objective function:

$$
\text{Regularized Least Squares} \equiv \text{MAP Estimation with Gaussian Prior}
$$

## Visualising Bayesian linear regression: Online Learning

In online learning, a model is trained incrementally as new data arrives, rather than being trained on a fixed dataset from the start.

![[Pasted image 20241107170825.png]]
![[Pasted image 20241107170859.png]]
![[Pasted image 20241107170946.png]]