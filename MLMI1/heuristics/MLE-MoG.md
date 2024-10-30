To derive the likelihood of the data point $x_n$ sampled from a mixture of Gaussians and obtain the maximum likelihood estimates (MLE) for the parameters $\pi_k$, $\mu_k$, and $\sigma_k^2$ without using the Expectation-Maximization (EM) algorithm, we'll proceed step by step.

1. **Define the Mixture Model**  
   Suppose $x_n$ for $n=1, \dots, N$ are independent and identically distributed (i.i.d.) samples from a mixture of $K$ Gaussian distributions. The probability density function (pdf) of the mixture model is:
   $$p(x_n) = \sum_{k=1}^K \pi_k \cdot N(x_n \mid \mu_k, \sigma_k^2)$$
   where:
   - $\pi_k$ is the mixing coefficient for component $k$, with $\sum_{k=1}^K \pi_k = 1$ and $\pi_k \geq 0$.
   - $N(x_n \mid \mu_k, \sigma_k^2)$ is the Gaussian pdf with mean $\mu_k$ and variance $\sigma_k^2$:
     $$N(x_n \mid \mu_k, \sigma_k^2) = \frac{1}{\sqrt{2 \pi \sigma_k^2}} \exp\left(-\frac{(x_n - \mu_k)^2}{2 \sigma_k^2}\right)$$

2. **Write the Likelihood Function**  
   Since the data points are i.i.d., the likelihood $L$ of the entire dataset is:
   $$L = \prod_{n=1}^N p(x_n) = \prod_{n=1}^N \left[\sum_{k=1}^K \pi_k \cdot N(x_n \mid \mu_k, \sigma_k^2)\right]$$

   Taking the natural logarithm to simplify differentiation:
   $$\ln L = \sum_{n=1}^N \ln \left[\sum_{k=1}^K \pi_k \cdot N(x_n \mid \mu_k, \sigma_k^2)\right]$$

3. **Derive the Maximum Likelihood Estimates**  
   We aim to find the parameter values that maximize $\ln L$. To do this, we'll compute the partial derivatives of $\ln L$ with respect to $\pi_k$, $\mu_k$, and $\sigma_k^2$, set them to zero, and solve the resulting equations.

   **a. Derivative with Respect to $\mu_k$**  
   Compute $\frac{\partial \ln L}{\partial \mu_k}$:
   $$\frac{\partial \ln L}{\partial \mu_k} = \sum_{n=1}^N \frac{\pi_k \cdot N(x_n \mid \mu_k, \sigma_k^2)}{p(x_n)} \cdot \frac{x_n - \mu_k}{\sigma_k^2}$$

   Set the derivative to zero:
   $$\sum_{n=1}^N \gamma_{nk} (x_n - \mu_k) = 0 \Rightarrow \mu_k = \frac{\sum_{n=1}^N \gamma_{nk} x_n}{\sum_{n=1}^N \gamma_{nk}}$$

   where $\gamma_{nk}$ is defined as the "responsibility" of component $k$ for data point $x_n$:
   $$\gamma_{nk} = \frac{\pi_k \cdot N(x_n \mid \mu_k, \sigma_k^2)}{p(x_n)}$$

   **b. Derivative with Respect to $\sigma_k^2$**  
   Compute $\frac{\partial \ln L}{\partial \sigma_k^2}$:
   $$\frac{\partial \ln L}{\partial \sigma_k^2} = \sum_{n=1}^N \gamma_{nk} \left[\frac{(x_n - \mu_k)^2}{\sigma_k^4} - \frac{1}{\sigma_k^2}\right]$$

   Set the derivative to zero and solve:
   $$\sigma_k^2 = \frac{\sum_{n=1}^N \gamma_{nk} (x_n - \mu_k)^2}{\sum_{n=1}^N \gamma_{nk}}$$

   **c. Derivative with Respect to $\pi_k$**  
   Include the constraint $\sum_{k=1}^K \pi_k = 1$ using a Lagrange multiplier $\lambda$. The Lagrangian $\mathcal{L}$ is:

   $$\mathcal{L} = \ln L + \lambda \left(\sum_{k=1}^K \pi_k - 1\right)$$

   Compute $\frac{\partial \mathcal{L}}{\partial \pi_k}$:

   $$\frac{\partial \mathcal{L}}{\partial \pi_k} = \sum_{n=1}^N \gamma_{nk} + \lambda = 0$$

   Sum over $k$ and solve for $\lambda$:

   $$\pi_k = \frac{1}{N} \sum_{n=1}^N \gamma_{nk}$$

4. **Summary of the Maximum Likelihood Estimates**  
   The MLEs for the parameters are given by:

   - **Mixing Coefficients $\pi_k$:**

     $$\pi_k = \frac{1}{N} \sum_{n=1}^N \gamma_{nk}$$

   - **Means $\mu_k$:**

     $$\mu_k = \frac{\sum_{n=1}^N \gamma_{nk} x_n}{\sum_{n=1}^N \gamma_{nk}}$$

   - **Variances $\sigma_k^2$:**

     $$\sigma_k^2 = \frac{\sum_{n=1}^N \gamma_{nk} (x_n - \mu_k)^2}{\sum_{n=1}^N \gamma_{nk}}$$

5. **Interpretation**  
   These equations represent a set of implicit equations because the responsibilities $\gamma_{nk}$ depend on the parameters $\pi_k$, $\mu_k$, and $\sigma_k^2$ themselves:

   $$\gamma_{nk} = \frac{\pi_k \cdot N(x_n \mid \mu_k, \sigma_k^2)}{\sum_{j=1}^K \pi_j \cdot N(x_n \mid \mu_j, \sigma_j^2)}$$

   To solve for the MLEs, one would typically need to use an iterative numerical method, updating the parameters until convergence. However, since we're instructed not to use the EM algorithm, which is designed for this purpose, we acknowledge that a closed-form solution isn't attainable through direct differentiation and setting derivatives to zero.

**Conclusion**:

By differentiating the log-likelihood function with respect to the parameters and setting the derivatives to zero, we've derived equations for the MLEs of $\pi_k$, $\mu_k$, and $\sigma_k^2$. These equations involve the responsibilities $\gamma_{nk}$, which depend on the parameters themselves, leading to an implicit set of equations that generally require iterative methods to solve.

**Reference Equations**:

- **Responsibilities**:

  $$\gamma_{nk} = \frac{\pi_k \cdot N(x_n \mid \mu_k, \sigma_k^2)}{\sum_{j=1}^K \pi_j \cdot N(x_n \mid \mu_j, \sigma_j^2)}$$

- **Maximum Likelihood Estimates**:

  - Mixing Coefficients:

    $$\pi_k = \frac{1}{N} \sum_{n=1}^N \gamma_{nk}$$

  - Means:

    $$\mu_k = \frac{\sum_{n=1}^N \gamma_{nk} x_n}{\sum_{n=1}^N \gamma_{nk}}$$

  - Variances:

    $$\sigma_k^2 = \frac{\sum_{n=1}^N \gamma_{nk} (x_n - \mu_k)^2}{\sum_{n=1}^N \gamma_{nk}}$$

The likelihood of a data point $x_n$ in a Gaussian mixture is $p(x_n) = \sum_{k=1}^K \pi_k \cdot N(x_n \mid \mu_k, \sigma_k^2)$. To find the maximum likelihood estimates (MLEs) for the parameters $\pi_k$, $\mu_k$, and $\sigma_k^2$, we take derivatives of the log-likelihood with respect to each parameter, set them to zero, and solve the resulting equations. This yields:

$$\mu_k = \frac{\sum_{n=1}^N \gamma_{nk} x_n}{\sum_{n=1}^N \gamma_{nk}}$$
$$\sigma_k^2 = \frac{\sum_{n=1}^N \gamma_{nk} (x_n - \mu_k)^2}{\sum_{n=1}^N \gamma_{nk}}$$
$$\pi_k = \frac{1}{N} \sum_{n=1}^N \gamma_{nk}$$

where $\gamma_{nk} = \frac{\pi_k \cdot N(x_n \mid \mu_k, \sigma_k^2)}{p(x_n)}$. These equations must be solved simultaneously, as the responsibilities $\gamma_{nk}$ depend on the parameters themselves.
