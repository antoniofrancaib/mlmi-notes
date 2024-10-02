# Regression



### Matrix Formulation

We express $E_2$ in matrix notation for convenience. Let:

$$
\mathbf{y} = \begin{pmatrix}
y_1 \\ y_2 \\ \vdots \\ y_N
\end{pmatrix}, \quad
\mathbf{X} = \begin{pmatrix}
1 & x_1 \\
1 & x_2 \\
\vdots & \vdots \\
1 & x_N
\end{pmatrix}, \quad
\mathbf{w} = \begin{pmatrix}
w_0 \\ w_1
\end{pmatrix}
$$

Then the error becomes:

$$
E_2 = \left| \mathbf{y} - \mathbf{X} \mathbf{w} \right|^2 = (\mathbf{y} - \mathbf{X} \mathbf{w})^\top (\mathbf{y} - \mathbf{X} \mathbf{w})
$$

### Minimization

To find the weights that minimize $E_2$, we take the derivative with respect to $\mathbf{w}$ and set it to zero:

$$
\frac{\partial E_2}{\partial \mathbf{w}} = -2 \mathbf{X}^\top (\mathbf{y} - \mathbf{X} \mathbf{w}) = 0
$$

Solving for $\mathbf{w}$:

$$
\mathbf{X}^\top \mathbf{X} \mathbf{w} = \mathbf{X}^\top \mathbf{y}
$$

$$
\implies \boxed{ \mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y} }
$$

This is the closed-form solution for the weights.

### Justification for Least Squares

Why minimize $E_2$ instead of a more general error function $E_p = \sum_{n=1}^N | y_n - (w_1 x_n + w_0) |^p$ for some $p > 0$? Two key reasons are:

- **Mathematical Tractability**: Minimizing $E_2$ yields a closed-form solution.
- **Maximum Likelihood Interpretation**: If we assume the outputs $y_n$ are corrupted by Gaussian noise:

  $$
  y_n = w_1 x_n + w_0 + \epsilon_n, \quad \epsilon_n \sim \mathcal{N}(0, \sigma^2)
  $$

  then minimizing $E_2$ is equivalent to finding the maximum likelihood estimates of $w_0$ and $w_1$.

### Maximum Likelihood Derivation

The likelihood of the data given the parameters is:

$$
p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}, \sigma^2) = \frac{1}{(2\pi \sigma^2)^{N/2}} \exp\left( -\frac{1}{2\sigma^2} (\mathbf{y} - \mathbf{X} \mathbf{w})^\top (\mathbf{y} - \mathbf{X} \mathbf{w}) \right)
$$

Maximizing the likelihood is equivalent to minimizing the negative log-likelihood:

$$
- \ln p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}, \sigma^2) = \frac{N}{2} \ln(2\pi \sigma^2) + \frac{1}{2\sigma^2} (\mathbf{y} - \mathbf{X} \mathbf{w})^\top (\mathbf{y} - \mathbf{X} \mathbf{w})
$$

Since the first term is independent of $\mathbf{w}$, minimizing the negative log-likelihood reduces to minimizing $E_2$. Therefore:

$$
\boxed{ \text{Least Squares} \equiv \text{Maximum Likelihood Estimation with Gaussian Noise} }
$$

## 2.2 Non-linear Basis Regression

To model non-linear relationships, we extend linear regression using **basis functions** $\phi_d(x)$:

$$
y_n = w_0 + w_1 \phi_1(x_n) + w_2 \phi_2(x_n) + \dots + w_D \phi_D(x_n) + \epsilon_n = \boldsymbol{\phi}(x_n)^\top \mathbf{w} + \epsilon_n
$$

Here, $\boldsymbol{\phi}(x_n) = \begin{pmatrix} 1 \\ \phi_1(x_n) \\ \vdots \\ \phi_D(x_n) \end{pmatrix}$, and $\epsilon_n \sim \mathcal{N}(0, \sigma^2)$.

The design matrix $\boldsymbol{\Phi}$ is:

$$
\boldsymbol{\Phi} = \begin{pmatrix}
1 & \phi_1(x_1) & \cdots & \phi_D(x_1) \\
1 & \phi_1(x_2) & \cdots & \phi_D(x_2) \\
\vdots & \vdots & \ddots & \vdots \\
1 & \phi_1(x_N) & \cdots & \phi_D(x_N)
\end{pmatrix}
$$

### Minimization

The error function becomes:

$$
E_2 = \left| \mathbf{y} - \boldsymbol{\Phi} \mathbf{w} \right|^2 = (\mathbf{y} - \boldsymbol{\Phi} \mathbf{w})^\top (\mathbf{y} - \boldsymbol{\Phi} \mathbf{w})
$$

Minimizing $E_2$ yields the solution:

$$
\boxed{ \mathbf{w} = (\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{y} }
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

## 2.3 Bayesian Linear Regression

In Bayesian regression, we treat the weights $\mathbf{w}$ as random variables with a prior distribution. Assuming a Gaussian prior:

$$
p(\mathbf{w}) = \mathcal{N}(\mathbf{w} \mid \mathbf{m}_0, \mathbf{S}_0)
$$

and a Gaussian likelihood:

$$
p(\mathbf{y} \mid \boldsymbol{\Phi}, \mathbf{w}, \sigma^2) = \mathcal{N}(\mathbf{y} \mid \boldsymbol{\Phi} \mathbf{w}, \sigma^2 \mathbf{I})
$$

### Posterior Distribution

Using Bayes' theorem:

$$
p(\mathbf{w} \mid \mathbf{y}, \boldsymbol{\Phi}, \sigma^2) \propto p(\mathbf{y} \mid \boldsymbol{\Phi}, \mathbf{w}, \sigma^2) p(\mathbf{w})
$$

The posterior is also Gaussian with:

$$
\mathbf{S}^{-1} = \sigma^{-2} \boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \mathbf{S}_0^{-1}
$$

$$
\boldsymbol{\mu} = \mathbf{S} \left( \sigma^{-2} \boldsymbol{\Phi}^\top \mathbf{y} + \mathbf{S}_0^{-1} \mathbf{m}_0 \right)
$$

### Equivalence to Regularization

For a zero-mean prior $\mathbf{m}_0 = \mathbf{0}$ and $\mathbf{S}_0^{-1} = \lambda \mathbf{I}$:

$$
\boldsymbol{\mu} = (\boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \lambda \mathbf{I})^{-1} \boldsymbol{\Phi}^\top \mathbf{y}
$$

Thus, the MAP estimate in Bayesian regression with a Gaussian prior is equivalent to ridge regression:

$$
\boxed{ \text{Regularized Least Squares} \equiv \text{MAP Estimation with Gaussian Prior} }
$$

### Predictive Distribution

The predictive distribution for a new input $x^\star$ is:

$$
p(y^\star \mid x^\star, \mathbf{y}, \boldsymbol{\Phi}, \sigma^2) = \int p(y^\star \mid x^\star, \mathbf{w}, \sigma^2) p(\mathbf{w} \mid \mathbf{y}, \boldsymbol{\Phi}, \sigma^2) \, d\mathbf{w}
$$

Since both the likelihood and posterior are Gaussian, the predictive distribution is Gaussian with mean and variance:

$$
\mathbb{E}[y^\star] = \boldsymbol{\phi}(x^\star)^\top \boldsymbol{\mu}
$$

$$
\operatorname{Var}(y^\star) = \boldsymbol{\phi}(x^\star)^\top \mathbf{S} \boldsymbol{\phi}(x^\star) + \sigma^2
$$

This provides not only a point estimate but also a measure of uncertainty.

### Online Learning and Posterior Updating

Bayesian regression naturally accommodates **online learning**. When new data $(x_n, y_n)$ arrives, we update the posterior:

$$
p(\mathbf{w} \mid \mathbf{y}_{1:n}, \boldsymbol{\Phi}_{1:n}, \sigma^2) \propto p(y_n \mid x_n, \mathbf{w}, \sigma^2) p(\mathbf{w} \mid \mathbf{y}_{1:n-1}, \boldsymbol{\Phi}_{1:n-1}, \sigma^2)
$$

The updated posterior becomes the prior for future updates, allowing us to incrementally refine our estimates as more data becomes available.
