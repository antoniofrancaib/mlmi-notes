When multiplying two Gaussian distributions, the result is another Gaussian with a new mean and variance. If we have two Gaussian distributions $\mathcal{N}(\mu_1, \sigma_1^2)$ and $\mathcal{N}(\mu_2, \sigma_2^2)$, the product of these two distributions will be a Gaussian distribution with the following mean and variance:

1. **New Mean:**
   $$
   \mu_{\text{new}} = \frac{\mu_1 \sigma_2^2 + \mu_2 \sigma_1^2}{\sigma_1^2 + \sigma_2^2}
   $$

2. **New Variance:**
   $$
   \sigma_{\text{new}}^2 = \frac{\sigma_1^2 \sigma_2^2}{\sigma_1^2 + \sigma_2^2}
   $$

### Explanation

The formula comes from the fact that multiplying two Gaussians is equivalent to performing a Bayesian update, where the combined distribution reflects a compromise between the two initial distributions. This new Gaussian has a mean that is a weighted average of the original means (weighted by their variances) and a variance that is smaller than either of the original variances, reflecting increased certainty.
