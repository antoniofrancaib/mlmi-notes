# Probabilistic Inference

Probabilistic inference is the process of deducing the probabilities of certain outcomes or parameters given observed data, within the framework of probability theory. 

## Bayesian Inference

Bayesian inference is a statistical method that applies Bayes' theorem to update the probability for a hypothesis as more evidence or information becomes available.

### Learning: Parameter Estimation

In Bayesian learning, we estimate the parameters $\theta$ of a model $m$ given observed data $D$:

$$p(\theta | D, m) = \frac{p(D | \theta, m) \, p(\theta | m)}{p(D | m)}$$

- **Posterior**: $p(\theta \mid D, m)$: Updated belief about parameters after observing data.
- **Likelihood**: $p(D \mid \theta, m)$: Probability of observing data $D$ given parameters $\theta$.
- **Prior**: $p(\theta \mid m)$: Initial belief about parameters before observing data.
- **Evidence**: $p(D \mid m)$: Probability of observing data under model $m$.

### Prediction: Predictive Distribution

Once we have the posterior distribution of the parameters, we can make predictions about new data $x^\ast$:

$$p(x^\ast | D, m) = \int p(x^\ast | \theta, m) \, p(\theta | D, m) \, d\theta$$

- **Predictive Distribution**: $p(x^\ast \mid D, m)$: Probability of future observations given past data.
- **Integrating Over Parameters**: We average all possible predictions $p(x^\ast \mid \theta, m)$, weighting each by how plausible $\theta$ is given the observed data $D$. 

This approach naturally incorporates uncertainty in the parameter estimates into our predictions.

### Model Comparison

In Bayesian inference, we can compare different models (Bayes factor) to see which one explains the data best:

$$p(m | D) = \frac{p(D | m) \, p(m)}{p(D)}$$

### Bayesian Decision Theory

Bayesian decision theory provides a framework for making optimal decisions under uncertainty by maximizing expected utility (or reward), or minimising the expected loss.

#### Expected Reward

The expected reward $R(a)$ for taking action $a$ is calculated as:

$$R(a) = \sum_x R(a, x) \, p(x \mid D)$$
$$R(a) = \int R(a, x) \, p(x \mid D) \, dx$$
- **Reward** $R(a, x)$: Reward for taking action $a$ when the true state of the world is $x$.
- **Posterior Probability** $p(x \mid D)$: Probability of state $x$ given data $D$.



## Flavours of Inference and Decision Problems

1. **Supervised Learning**
   - **Objective**: Learn a mapping from inputs $x$ to outputs $y$ based on observed pairs $(x_i, y_i)$.
   - **Applications**: Regression, classification, time series prediction.

2. **Unsupervised Learning**
   - **Objective**: Model the underlying structure or distribution in data without explicit output labels.
   - **Applications**: Clustering, dimensionality reduction, density estimation.

3. **Reinforcement Learning**
   - **Objective**: Learn to make decisions by performing actions $a_t$ in an environment to maximize cumulative rewards $r_t$.
   - **Applications**: Robotics, game playing, adaptive control systems.

---
## Maximum Likelihood Estimation (MLE)

As an alternative to Bayesian inference (point-estimate method), we can use maximum likelihood estimation to find the value of $\theta$ that maximizes the likelihood of the observed data.

### MLE Formulation

$$\theta_{\text{ML}} = \underset{\lambda}{\text{argmax}} \; p(\{x_n\}_{n=1}^N | \theta)$$

#### Comparison with Bayesian Approach

- **Point Estimate**: MLE provides a single estimate of $\theta$, whereas Bayesian inference provides a full posterior distribution.
- **Uncertainty Quantification**: Bayesian inference naturally accounts for uncertainty in $\theta$, while MLE does not.
- **Prior Information**: Bayesian inference incorporates prior beliefs, which can be beneficial when data are scarce.

The **MAP** and **MLE** estimates are the same when the prior is ***flat*** and non-zero over the relevant parameter space where the likelihood function has support.


