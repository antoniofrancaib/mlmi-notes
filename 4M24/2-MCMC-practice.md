## Overview

In this lecture, we will delve deeper into the practical aspects of the Monte Carlo method, a fundamental tool in computational statistics and machine learning. We will cover the following topics:

- **Monte Carlo in Practice**: How to implement Monte Carlo simulations with practical considerations.
- **Monte Carlo Error**: Understanding and quantifying the error inherent in Monte Carlo estimates.
- **Importance Sampling**: A technique to reduce variance and improve estimates by changing the sampling distribution.
- **Control Variates**: An advanced variance reduction method using additional information to enhance estimates.

---

### 1. Monte Carlo in Practice

#### 1.1 Generating Random Variables

To perform Monte Carlo simulations, we require methods to simulate random variables exactly from the desired probability distributions. This involves generating samples $x_n$ from a probability density function $p(x)$.

- **Random Number Generators**: Modern software libraries provide comprehensive tools for generating random numbers from various distributions (e.g., normal, uniform, exponential).
- **Assumption**: We will assume that these tools are available and reliable for our purposes.

#### 1.2 Practical Considerations

- **Availability of $p(x)$**: For the methods discussed in this lecture, we assume that:
  - We can evaluate $p(x)$ for any given $x$.
  - We can draw samples $x_n$ from $p(x)$.
- **Complex Distributions**: In some cases, the probability density $p(x)$ may not be fully known or may be difficult to sample from directly.
- **Future Topics**: We will later explore techniques like Markov Chain Monte Carlo (MCMC) that address sampling from such distributions.

#### 1.3 Vanilla Monte Carlo Method

The basic Monte Carlo method involves:

- **Sampling**: Draw $N$ independent and identically distributed (i.i.d.) samples $x_1, x_2, \dots, x_N$ from $p(x)$.
- **Estimation**: Compute the estimate of the expectation $\mathbb{E}_p[f(X)]$ using:

  $$
  \hat{\mu}_f = \frac{1}{N} \sum_{n=1}^N f(x_n)
  $$

---

### 2. Monte Carlo Error

#### 2.1 Central Limit Theorem (CLT) for Monte Carlo Estimates

The Central Limit Theorem provides a foundation for understanding the behavior of the Monte Carlo estimator:

- **CLT Statement**:

  $$
  \hat{\mu}_f = \frac{1}{N} \sum_{n=1}^N f(x_n) \sim N\left(\mathbb{E}_p[f(X)], \frac{\sigma_f^2}{N}\right)
  $$

  where:
  - $\mathbb{E}_p[f(X)]$ is the true expected value.
  - $\sigma_f^2 = \text{Var}_p[f(X)] = \mathbb{E}_p[f^2(X)] - (\mathbb{E}_p[f(X)])^2$ is the variance of $f(X)$ under $p(x)$.
  - The notation $\sim N(\mu, \sigma^2)$ denotes that the estimator is approximately normally distributed with mean $\mu$ and variance $\sigma^2$.

- **Implications**:
  - **Unbiasedness**: The Monte Carlo estimate $\hat{\mu}_f$ is an unbiased estimator of $\mathbb{E}_p[f(X)]$.
  - **Error Bars**: The standard error (standard deviation of the estimator) is $\sigma_f / \sqrt{N}$.
  - **Convergence Rate**: The error decreases at a rate proportional to $1 / \sqrt{N}$.

#### 2.2 Independence from Dimensionality

- **Dimensionality Benefit**: The error's dependence on $\sigma_f^2$ and $N$ means that, regardless of the dimensionality of the integral, the convergence rate remains $O(1 / \sqrt{N})$.
- **Contrast with Deterministic Methods**: In deterministic numerical integration methods, the number of required evaluation points increases exponentially with the number of dimensions (the "curse of dimensionality").

#### 2.3 Components of Monte Carlo Error

The variance of the estimator can be dissected into two components:

- **Fixed Component**: $(\mathbb{E}_p[f(X)])^2$, a constant proportion of the squared integral value.
- **Variable Component**: $\mathbb{E}_p[f^2(X)]$, dependent on both the choice of the function $f$ and the probability density $p(x)$.

#### 2.4 Impact of $\sigma_f^2$ on Error Reduction

- **Significance of $\sigma_f^2$**: While increasing $N$ reduces the variance, the magnitude of $\sigma_f^2$ significantly affects the rate of error reduction.
- **Variance Reduction**: Strategies to reduce $\sigma_f^2$ can lead to more efficient estimators.

#### 2.5 Example: Estimating $\mathbb{E}[X^{10}]$ with Uniform Distribution

Suppose we wish to estimate:

$$
I = \mathbb{E}[X^{10}], \text{ with } X \sim \text{Uniform}(0,1)
$$

- **Observation**:
  - The function $f(X) = X^{10}$ is very small for most values of $X$ in $[0,1]$.
  - Only values of $X$ close to 1 contribute significantly to the integral.
- **Consequence**:
  - **High Variance**: Most samples will be "uninformative," resulting in a high variance estimator.
  - **Inefficient Estimation**: A large $N$ is required to achieve a desired level of accuracy.

---

### 3. Importance Sampling

#### 3.1 Motivation

- **Problem with Standard Monte Carlo**: When $p(x)$ assigns low probability to regions where $f(x)$ is significant, the variance of the estimator becomes large.
- **Goal**: Reduce variance by focusing sampling efforts on the "important" regions of the integration domain.

#### 3.2 Concept of Importance Sampling

- **Idea**: Introduce an alternative probability density function $q(x)$, called the importance distribution, that:
  - Is non-zero wherever $p(x)$ is non-zero.
  - Emphasizes regions where $f(x) / p(x)$ is large.
- **Adjustment**: Rewrite the expectation with respect to $q(x)$:

  $$
  I = \mathbb{E}_p[f(X)] = \int f(x) \, p(x) \, dx = \int f(x) \, \frac{p(x)}{q(x)} \, q(x) \, dx = \mathbb{E}_q\left[ \frac{f(X) \, p(X)}{q(X)} \right]
  $$

#### 3.3 Importance Sampling Estimator

- **Estimator**:

  $$
  \hat{I} = \frac{1}{N} \sum_{n=1}^N f(x_n) \frac{p(x_n)}{q(x_n)}, \quad \text{with } x_n \sim q(x)
  $$

- **Unbiasedness**:

  - **Expectation**:

    $$
    \mathbb{E}_q[\hat{I}] = \mathbb{E}_q\left[ \frac{f(X) \, p(X)}{q(X)} \right] = I
    $$

  - **Conclusion**: The importance sampling estimator is unbiased.

#### 3.4 Variance Reduction through Choice of $q(x)$

- **Variance of the Estimator**:

  $$
  \tilde{\sigma}_f^2 = \text{Var}_q\left[\frac{f(X) \, p(X)}{q(X)}\right] = \mathbb{E}_q\left[\left(\frac{f(X) \, p(X)}{q(X)}\right)^2\right] - I^2
  $$

- **Optimal Choice of $q(x)$**:
  - To minimize $\tilde{\sigma}_f^2$, set:

    $$
    q_{\text{opt}}(x) = \frac{|f(x)| \, p(x)}{\int |f(x)| \, p(x) \, dx}
    $$

  - **Result**: When using $q_{\text{opt}}(x)$, the variance $\tilde{\sigma}_f^2$ becomes zero.

- **Practical Limitations**:
  - **Infeasibility**: The optimal $q_{\text{opt}}(x)$ depends on the integral we are trying to compute, which is generally unknown.
  - **Approximations**: In practice, we choose $q(x)$ to approximate $q_{\text{opt}}(x)$, balancing variance reduction with computational tractability.

#### 3.5 Intuitive Explanation

- **Alignment of $q(x)$ with $f(x) / p(x)$**:
  - **High $f(x) / p(x)$**: In regions where $f(x) / p(x)$ is large, $q(x)$ should be large to increase the chance of sampling important values.
  - **Low $f(x) / p(x)$**: In regions where $f(x) / p(x)$ is negligible, $q(x)$ should be small to avoid wasting samples.
  - **Goal**: Concentrate sampling efforts where they will contribute most to the estimate.

#### 3.6 Example: Estimating Tail Probabilities

- **3.6.1 Problem Statement**: Let $T_{\text{life}}$ be the lifetime of a system component with an exponential distribution:

  $$
  T_{\text{life}} \sim p(x) = \begin{cases} e^{-x}, & x \geq 0 \\ 0, & x < 0 \end{cases}
  $$

  - **Objective**: Estimate the probability that the component lasts at least 25 years:

    $$
    I = P(T_{\text{life}} \geq 25) = \mathbb{E}_p[I\{x \geq 25\}] = \int_0^{\infty} I\{x \geq 25\} e^{-x} \, dx
    $$

#### 3.7 Practical Considerations in Importance Sampling

- **Weight Variability**:
  - If the weights $w_n = \frac{p(x_n)}{q(x_n)}$ vary greatly, the variance can still be high.
  - **Aim**: Choose $q(x)$ such that the weights are stable.

- **Normalization**:
  - In some cases, $p(x)$ and $q(x)$ may not integrate to 1 over the same domain.
  - Ensure that $q(x)$ is properly normalized over the support of $p(x)$.

- **Implementation**:
  - **Sampling**: Efficiently sample from $q(x)$.
  - **Weight Calculation**: Accurately compute $\frac{p(x_n)}{q(x_n)}$.

---

### 4. Control Variates

#### 4.1 Motivation

- **Objective**: Reduce the variance of Monte Carlo estimators by leveraging known information about related functions.

#### 4.2 Concept of Control Variates

- **Idea**: Use a function $g(x)$ for which we know $\mathbb{E}_p[g(X)]$, to adjust our estimate of $\mathbb{E}_p[f(X)]$.

- **Adjusted Estimator**:

  $$
  \hat{\mu}_c = \hat{\mu}_f + c \left(\mathbb{E}_p[g(X)] - \hat{\mu}_g\right)
  $$

  where:
  - $\hat{\mu}_f = \frac{1}{N} \sum_{n=1}^N f(x_n)$
  - $\hat{\mu}_g = \frac{1}{N} \sum_{n=1}^N g(x_n)$
  - $c$ is a constant to be determined.

#### 4.3 Variance of the Control Variate Estimator

- **Variance Calculation**:

  $$
  \text{Var}(\hat{\mu}_c) = \text{Var}(\hat{\mu}_f) + c^2 \text{Var}(\hat{\mu}_g) - 2c \, \text{Cov}(\hat{\mu}_f, \hat{\mu}_g)
  $$

- **Goal**: Find the value of $c$ that minimizes $\text{Var}(\hat{\mu}_c)$.

#### 4.4 Optimal Choice of $c$

- **Minimization**:

  - Take the derivative of $\text{Var}(\hat{\mu}_c)$ with respect to $c$ and set it to zero:

    $$
    \frac{d}{dc} \text{Var}(\hat{\mu}_c) = 2c \, \text{Var}(\hat{\mu}_g) - 2 \, \text{Cov}(\hat{\mu}_f, \hat{\mu}_g) = 0
    $$

  - **Solution**:

    $$
    c_{\text{opt}} = \frac{\text{Cov}(\hat{\mu}_f, \hat{\mu}_g)}{\text{Var}(\hat{\mu}_g)}
    $$

#### 4.5 Interpretation

- **Variance Reduction**:
  - The variance of the adjusted estimator $\hat{\mu}_c$ is lower than the variance of the original estimator $\hat{\mu}_f$ as long as $\text{Cov}(\hat{\mu}_f, \hat{\mu}_g) \neq 0$.
  - **Maximum Reduction**: Achieved when the covariance is large in magnitude.

---

## Conclusion

- **Monte Carlo in Practice**: We have explored practical aspects of implementing Monte Carlo simulations, assuming access to reliable random number generators.
- **Monte Carlo Error**: Understanding the components of error in Monte Carlo estimates is crucial for evaluating estimator performance.
- **Importance Sampling**: By changing the sampling distribution to focus on important regions, we can significantly reduce variance and improve efficiency.
- **Control Variates**: Utilizing known expectations of related functions allows us to adjust estimates and achieve variance reduction.

By thoroughly understanding these advanced concepts, you can enhance the efficiency and accuracy of Monte Carlo simulations, making them powerful tools for tackling complex computational problems in statistics and machine learning.
