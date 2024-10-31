---
layout: default
title: Inference
---

## Introduction to Probabilistic Inference

Probabilistic inference is the process of deducing the probabilities of certain outcomes or parameters given observed data, within the framework of probability theory. It forms the cornerstone of many fields such as statistics, machine learning, artificial intelligence, and data science. By leveraging probabilistic models, we can make informed decisions, predict future events, and understand underlying patterns in data.

## Importance of Probabilistic Inference

In many real-world scenarios, we are faced with uncertainty due to incomplete or noisy data. Probabilistic inference allows us to quantify this uncertainty and make predictions or decisions accordingly. It provides a principled way to update our beliefs in light of new evidence, ensuring that our conclusions are grounded in both prior knowledge and observed data.

## Applications of Probabilistic Inference

Probabilistic inference is widely used across many real-world problems involve estimating unobserved variables from observed data. Examples include:

<table>
    <tr>
        <th style="width: 30%;">Application</th>
        <th style="width: 30%;">Observed variable</th>
        <th style="width: 40%;">Unobserved variable</th>
    </tr>
    <tr>
        <td>climate science</td>
        <td>earth observations</td>
        <td>climate forecast</td>
    </tr>
    <tr>
        <td>autonomous driving</td>
        <td>image pixel values</td>
        <td>pedestrians and vehicles present </td>
    </tr>
    <tr>
        <td>movie recommendation</td>
        <td>ratings of watched films</td>
        <td>ratings of unwatched films</td>
    </tr>
    <tr>
        <td>medicine</td>
        <td>genome DNA</td>
        <td>susceptibility to genetic diseases</td>
    </tr>
</table>

## Fundamental Concepts

### Sum Rule

The sum rule, also known as the marginalization rule, allows us to compute the marginal probability of a random variable by summing (or integrating) over all possible values of another variable:

For discrete variables:
$$p(x) = \sum_y p(x, y)$$

For continuous variables:
$$p(x) = \int p(x, y) \, dy$$

### Product Rule

The product rule expresses the joint probability of two events as the product of a conditional probability and a marginal probability:

$$p(x, y) = p(x) \, p(y | x) = p(y) \, p(x | y)$$

These two rules form the basis for all probabilistic reasoning and inference.

## Bayes' Theorem

Bayes' theorem is derived from the product rule and provides a way to update our beliefs about the parameters or hypotheses in light of new data:

$$p(\theta | D) = \frac{p(D | \theta) \, p(\theta)}{p(D)}$$

- **Posterior**: $p(\theta \mid D)$: Probability of parameters $\theta$ given data $D$.
- **Likelihood**: $p(D \mid \theta)$: Probability of data $D$ given parameters $\theta$.
- **Prior**: $p(\theta)$: Initial probability of parameters $\theta$.
- **Marginal Likelihood**: $p(D)$: Probability of data $D$.

## Bayesian Inference

Bayesian inference is a statistical method that applies Bayes' theorem to update the probability for a hypothesis as more evidence or information becomes available.

### Learning: Parameter Estimation

In Bayesian learning, we estimate the parameters $\theta$ of a model $m$ given observed data $D$:

$$p(\theta | D, m) = \frac{p(D | \theta, m) \, p(\theta | m)}{p(D | m)}$$

- **Posterior**: $p(\theta \mid D, m)$: Updated belief about parameters after observing data.
- **Likelihood**: $p(D \mid \theta, m)$: Probability of observing data $D$ given parameters $\theta$.
- **Prior**: $p(\theta \mid m)$: Initial belief about parameters before observing data.
- **Evidence**: $p(D \mid m)$: Probability of observing data under model $m$.

#### Explanation

- **Posterior** represents what we know about the parameters after seeing the data.
- **Likelihood** encapsulates what the data tells us about the parameters.
- **Prior** reflects what we knew (or assumed) before observing the data.

### Prediction: Predictive Distribution

Once we have the posterior distribution of the parameters, we can make predictions about new data $x^\ast$:

$$p(x^\ast | D, m) = \int p(x^\ast | \theta, m) \, p(\theta | D, m) \, d\theta$$

- **Predictive Distribution**: $p(x^\ast \mid D, m)$: Probability of future observations given past data.
- **Integrating Over Parameters**: We average over all possible parameter values, weighted by their posterior probability.

#### Interpretation

We average all possible predictions $p(x^\ast \mid \theta, m)$, weighting each by how plausible $\theta$ is given the observed data $D$. This approach naturally incorporates uncertainty in the parameter estimates into our predictions.

### Model Comparison

In Bayesian inference, we can compare different models to see which one explains the data best:

$$p(m | D) = \frac{p(D | m) \, p(m)}{p(D)}$$

- **Posterior Probability of Model** $p(m \mid D)$: Probability that model $m$ is the correct model given the data.
- **Model Evidence** $p(D \mid m)$: Probability of the data under model $m$.
- **Prior Probability of Model** $p(m)$: Initial belief about the plausibility of model $m$.

#### Bayes Factors

The ratio of the posterior probabilities of two models is known as the Bayes factor, which quantifies the evidence in favor of one model over another.

### Bayesian Decision Theory

Bayesian decision theory provides a framework for making optimal decisions under uncertainty by maximizing expected utility (or reward).

#### Expected Reward

The expected reward $R(a)$ for taking action $a$ is calculated as:

$$R(a) = \sum_x R(a, x) \, p(x | D)$$

- **Reward** $R(a, x)$: Reward for taking action $a$ when the true state of the world is $x$.
- **Posterior Probability** $p(x \mid D)$: Probability of state $x$ given data $D$.

#### Explanation

We compute the action $a$ with the highest expected conditional reward, considering all possible states of the world. This approach separates inference and decision-making, allowing us to first infer probabilities and then make decisions based on these probabilities.

## Flavours of Inference and Decision Problems

Machine learning and inference problems can generally be categorized into three main types:

1. **Supervised Learning**
   - **Objective**: Learn a mapping from inputs $x$ to outputs $y$ based on observed pairs $(x_i, y_i)$.
   - **Applications**: Regression, classification, time series prediction.

2. **Unsupervised Learning**
   - **Objective**: Model the underlying structure or distribution in data without explicit output labels.
   - **Applications**: Clustering, dimensionality reduction, density estimation.

3. **Reinforcement Learning**
   - **Objective**: Learn to make decisions by performing actions $a_t$ in an environment to maximize cumulative rewards $r_t$.
   - **Applications**: Robotics, game playing, adaptive control systems.

## Example: The Radioactive Decay Problem

To illustrate the concepts of probabilistic inference, let's consider a classic problem in statistical estimation: estimating the decay constant of a radioactive substance.

### Problem Setup

Unstable particles decay at distances $x$ from a source, following an exponential distribution characterized by a decay constant $\lambda$:

$$p(x | \lambda) = \frac{1}{Z(\lambda)} \exp\left(-\frac{x}{\lambda}\right)$$

- **Normalization Constant** $Z(\lambda)$: Ensures the probability density integrates to one over the observed range.

We observe $N$ decay events within a specific range $(x_{\text{min}}, x_{\text{max}})$. Our goal is to infer the value of $\lambda$ based on these observations.

### Heuristic Approaches

Before delving into Bayesian inference, let's explore two heuristic methods for estimating $\lambda$.

1. **Histogram-Based Approach**
   - **Method**: Bin the observed decay distances into a histogram and perform linear regression on the logarithm of the bin counts.
   - **Assumption**: The logarithm of the counts should decrease linearly with distance for an exponential distribution.
   - **Issues**:
     - **Bin Size Sensitivity**: The choice of bin size can significantly affect the estimate.
     - **Uncertainty Estimation**: Does not provide a measure of uncertainty for $\lambda$.
     - **Justification**: Linear regression may not be the most appropriate method due to the discrete nature of histogram counts.

2. **Statistic-Based Approach**
   - **Method**: Use the sample mean of the observed distances to estimate $\lambda$.

#### Formula

$$\mu = \lambda + \frac{x_{\text{min}} \exp(-x_{\text{min}} / \lambda) - x_{\text{max}} \exp(-x_{\text{max}} / \lambda)}{\exp(-x_{\text{min}} / \lambda) - \exp(-x_{\text{max}} / \lambda)}$$

- **Issues**:
  - **Sample Mean Limitations**: The sample mean might exceed the maximum possible value due to the truncated range.
  - **Arbitrariness**: The choice of using the mean is somewhat arbitrary and may not fully utilize the information in the data.

### Bayesian Inference Approach

A more principled method is to apply Bayesian inference to estimate $\lambda$.

#### Steps in Bayesian Inference

1. **Specify the Likelihood Function**:

   The likelihood of observing the data $\{x_n\}_{n=1}^N$ given $\lambda$ is:

   $$p(\{x_n\}_{n=1}^N | \lambda) = \prod_{n=1}^N p(x_n | \lambda)$$

2. **Choose a Prior Distribution**:

   We select a prior distribution $p(\lambda)$ that reflects our initial beliefs about $\lambda$. For example, a uniform prior over a reasonable range:

   $$p(\lambda) = U(\lambda; \lambda_{\text{min}}, \lambda_{\text{max}})$$

3. **Compute the Posterior Distribution**:

   Applying Bayes' theorem:

   $$p(\lambda | \{x_n\}_{n=1}^N) = \frac{p(\{x_n\}_{n=1}^N | \lambda) \, p(\lambda)}{p(\{x_n\}_{n=1}^N)}$$

   Since $p(\{x_n\}_{n=1}^N)$ does not depend on $\lambda$, we can write:

   $$p(\lambda | \{x_n\}_{n=1}^N) \propto p(\lambda) \prod_{n=1}^N p(x_n | \lambda)$$

4. **Simplify the Posterior Expression**:

   Substituting the exponential likelihood:

   $$p(\lambda | \{x_n\}_{n=1}^N) \propto p(\lambda) \left(\frac{1}{Z(\lambda)}\right)^N \exp\left(-\frac{1}{\lambda} \sum_{n=1}^N x_n\right)$$

The posterior depends on $\lambda$ through the normalization constant $Z(\lambda)$ and the exponential term.

5. **Compute Sufficient Statistics**:

   Note that the data enter the posterior only through the sum $S = \sum_{n=1}^N x_n$ and the number of observations $N$. These are known as sufficient statistics.

#### Understanding the Likelihood

The likelihood function $p(\{x_n\}_{n=1}^N \mid \lambda)$ represents how probable the observed data are for different values of $\lambda$. It typically peaks at the value of $\lambda$ that makes the observed data most probable.

#### Posterior Visualization

By plotting the posterior distribution $p(\lambda \mid \{x_n\}_{n=1}^N)$, we can visualize our updated beliefs about $\lambda$ after observing the data. The shape of the posterior reflects both the data and the prior.

#### Summarizing the Posterior

We can compute summaries of the posterior distribution, such as:

- **Mean**: Expected value of $\lambda$ under the posterior.
- **Variance**: Measures the uncertainty in our estimate of $\lambda$.
- **Credible Intervals**: Ranges within which $\lambda$ lies with a certain probability (e.g., 95% credible interval).

### Predictive Distribution

With the posterior distribution in hand, we can make predictions about future decay events.

#### Computing the Predictive Distribution

The predictive distribution for a new observation $x^\ast$ is:

$$p(x^\ast | \{x_n\}_{n=1}^N) = \int p(x^\ast | \lambda) \, p(\lambda | \{x_n\}_{n=1}^N) \, d\lambda$$

This integral averages over all possible values of $\lambda$, weighted by their posterior probabilities.

#### Interpretation

The predictive distribution incorporates both the uncertainty in $\lambda$ and the inherent randomness of the decay process. It provides a full probabilistic description of where we expect future decay events to occur.

## Maximum Likelihood Estimation (MLE)

As an alternative to Bayesian inference, we can use maximum likelihood estimation to find the value of $\lambda$ that maximizes the likelihood of the observed data.

### MLE Formulation

$$\lambda_{\text{ML}} = \underset{\lambda}{\text{argmax}} \; p(\{x_n\}_{n=1}^N | \lambda)$$

#### Comparison with Bayesian Approach

- **Point Estimate**: MLE provides a single estimate of $\lambda$, whereas Bayesian inference provides a full posterior distribution.
- **Uncertainty Quantification**: Bayesian inference naturally accounts for uncertainty in $\lambda$, while MLE does not.
- **Prior Information**: Bayesian inference incorporates prior beliefs, which can be beneficial when data are scarce.

## Summary of the Radioactive Decay Problem

The Bayesian approach to the radioactive decay problem involves:

1. **Model Specification**: Assuming an exponential decay model $p(x \mid \lambda)$.
2. **Prior Selection**: Choosing a prior $p(\lambda)$ that reflects prior beliefs.
3. **Posterior Computation**: Applying Bayes' theorem to compute $p(\lambda \mid \{x_n\}_{n=1}^N)$.
4. **Prediction**: Calculating the predictive distribution for future observations.

This approach provides a principled and coherent method for parameter estimation and prediction, fully accounting for uncertainty.

## Conclusion

Probabilistic inference is a powerful framework for reasoning under uncertainty. By leveraging the fundamental rules of probability and Bayesian principles, we can:

1. Update our beliefs in light of new data.
2. Make predictions that account for parameter uncertainty.
3. Compare models in a principled way.
4. Make optimal decisions based on expected rewards.
