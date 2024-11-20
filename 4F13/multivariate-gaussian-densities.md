## Background: Gaussian Probability Densities
### Multivariate Gaussian Distribution

#### Density Function
$$
p(x \mid \mu, \Sigma) = \frac{1}{(2 \pi)^{D/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)
$$

- **Properties**: Mean $\mu$, covariance $\Sigma$.
- **Marginals and conditionals** of a joint Gaussian are Gaussian.

### Marginal and Conditional Distributions
Given:

$$
\begin{bmatrix} x \\ y \end{bmatrix} \sim N\left( \begin{bmatrix} \mu_x \\ \mu_y \end{bmatrix}, \begin{bmatrix} \Sigma_{xx} & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_{yy} \end{bmatrix} \right)
$$

- **Marginal of $x$**: $x \sim N(\mu_x, \Sigma_{xx})$.
- **Conditional of $x \mid y$**:

$$
p(x \mid y) = N(x \mid \mu_x + \Sigma_{xy} \Sigma_{yy}^{-1} (y - \mu_y), \Sigma_{xx} - \Sigma_{xy} \Sigma_{yy}^{-1} \Sigma_{yx})
$$

### Gaussian Identities

#### Linear Transformations
If $z = Ax + b$ and $x \sim N(\mu, \Sigma)$, then:

$$
z \sim N(A \mu + b, A \Sigma A^T)
$$

#### Product of Gaussians
The product of two Gaussian densities is proportional to another Gaussian density.

#### Matrix Inversion Lemma
Useful for simplifying expressions involving matrix inverses.

## Appendix: Useful Gaussian and Matrix Identities

### Matrix Identities

#### Matrix Inversion Lemma (Woodbury Identity)
$$
(Z + UWV^T)^{-1} = Z^{-1} - Z^{-1} U (W^{-1} + V^T Z^{-1} U)^{-1} V^T Z^{-1}
$$

#### Determinant Identity
$$
|Z + UWV^T| = |Z| \, |W| \, |W^{-1} + V^T Z^{-1} U|
$$

### Gaussian Identities

#### Expectation and Variance
If $x \sim N(\mu, \Sigma)$, then:

- $\mathbb{E}[x] = \mu$
- $\text{Var}[x] = \Sigma$

#### Linear Transformation
For $z = Ax + b$:

$$
z \sim N(A \mu + b, A \Sigma A^T)
$$

#### Product of Gaussians
The product $N(x \mid \mu_1, \Sigma_1) N(x \mid \mu_2, \Sigma_2)$ is proportional to $N(x \mid \mu_*, \Sigma_*)$, where:

- $\Sigma_* = (\Sigma_1^{-1} + \Sigma_2^{-1})^{-1}$
- $\mu_* = \Sigma_* (\Sigma_1^{-1} \mu_1 + \Sigma_2^{-1} \mu_2)$

#### Kullback-Leibler Divergence between Gaussians
For $N_0 = N(\mu_0, \Sigma_0)$ and $N_1 = N(\mu_1, \Sigma_1)$:

$$
\text{KL}(N_0 \parallel N_1) = \frac{1}{2} \left( \log \frac{|\Sigma_1|}{|\Sigma_0|} - D + \text{tr}(\Sigma_1^{-1} \Sigma_0) + (\mu_1 - \mu_0)^T \Sigma_1^{-1} (\mu_1 - \mu_0) \right)
$$
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
