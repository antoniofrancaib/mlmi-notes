
# 1. An Introduction to Inference

Many real-world problems involve estimating unobserved variables from observed data, which is the essence of **inference**. Examples include predicting future temperatures from climate data, estimating dark matter distribution from galaxy rotation, identifying pedestrians from camera images for autonomous vehicles, recommending movies based on user ratings, and predicting genetic diseases from DNA sequences.

<table>
    <tr>
        <th>Application</th>
        <th>Observed variable</th>
        <th>Unobserved variable</th>
    </tr>
    <tr>
        <td>climate science</td>
        <td>earth observations</td>
        <td>climate forecast</td>
    </tr>
    <tr>
        <td>physics</td>
        <td>observational measurements</td>
        <td>fundamental constants</td>
    </tr>
    <tr>
        <td>autonomous driving</td>
        <td>image pixel values</td>
        <td>pedestrians and vehicles present in image</td>
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

## Inference and Decision Making

In scientific applications, estimating the value of the unobserved variables is often critical. In engineering, business, and medicine, the purpose is not just inference but also decision making, where uncertainty plays a crucial role. An example is an autonomous car in adverse weather where the certainty of the detected environment influences driving decisions.

## Flavours of Inference and Decision Problems

Machine learning problems can generally be categorized as follows:

1. **Supervised Learning**: The machine learns to produce the correct output given new inputs, based on observed pairs $(x_1, y_1), (x_2, y_2), \ldots$.
2. **Unsupervised Learning**: The machine models the data to perform reasoning, decision making, predicting, or communicating.
3. **Reinforcement Learning**: The machine learns to act to maximize rewards based on actions $a_1, a_2, \ldots$ and received rewards $r_1, r_2, \ldots$.

## Radioactive Decay Problem

### Example: Estimating a Radioactivity Decay Constant

Unstable particles decay at distances $x$ from a source, following an exponential distribution with decay constant $\lambda$. The density function is given by:

$$
p(x|\lambda) = \frac{1}{Z(\lambda)} \exp\left(-\frac{x}{\lambda}\right)
$$

where $Z(\lambda)$ is a normalization constant. The dataset consists of measurements within a specific range $(x_{\text{min}}, x_{\text{max}}) = (5, 50)$. The objective is to infer $\lambda$.

### Heuristic Approaches

1. **Histogram-Based Approach**: Create a histogram of decay events and use linear least squares regression on the log of histogram counts to estimate $\lambda$.
   
   Issues:
   - Bin size affects the estimate.
   - No uncertainty estimate for $\lambda$.
   - Justification for least squares regression is questionable.

2. **Statistic-Based Approach**: Use the mean of the observations to infer $\lambda$. The mean of the exponential distribution is:

$$
\mu = \lambda + \frac{x_{\text{min}} \exp(-x_{\text{min}}/\lambda) - x_{\text{max}} \exp(-x_{\text{max}}/\lambda)}{\exp(-x_{\text{min}}/\lambda) - \exp(-x_{\text{max}}/\lambda)}
$$

   Issues:
   - Sample mean might exceed maximum possible value.
   - The choice of mean is somewhat arbitrary.

### Probabilistic Inference

A more principled approach is Bayesian inference:

1. Use Bayes' rule to compute the posterior distribution of $\lambda$ given the observed data $\{ x_n \}_{n=1}^N$:

$$
p(\lambda | \{ x_n \}_{n=1}^N) = \frac{p(\lambda) \prod_{n=1}^N p(x_n|\lambda)}{p(\{ x_n \}_{n=1}^N)}
$$

   Simplifying, we get:

$$
p(\lambda | \{ x_n \}_{n=1}^N) \propto p(\lambda) \prod_{n=1}^N p(x_n|\lambda)
$$

2. Substitute the exponential distribution into Bayes' rule:

$$
p(\lambda | \{ x_n \}_{n=1}^N) \propto p(\lambda) \frac{1}{Z(\lambda)^N} \exp\left(-\frac{1}{\lambda} \sum_{n=1}^N x_n \right)
$$

   The posterior is influenced by the prior $p(\lambda)$ and the likelihood $\prod_{n=1}^N p(x_n|\lambda)$.

3. **Sufficient Statistics**: Only the sum $\sum_{n=1}^N x_n$ and the number of observations $N$ are needed.

### Understanding the Likelihood

The likelihood as a function of $\lambda$ peaks at a value where the decay events are most consistent with the observed data. Consider a uniform prior distribution:

$$
p(\lambda) = \mathcal{U}(\lambda; 0, 100)
$$

Visualize the posterior distribution and compute summaries such as the mean and standard deviation.

### Predictive Distribution

The predictive distribution of a future decay event $x^\star$ given the observed data is:

$$
p(x^\star \lvert \{ x_n \}_{n=1}^N) = \int p(x^\star \lvert  \lambda) p(\lambda | \{ x_n \}_{n=1}^N) \text{d} \lambda
$$

### Maximum Likelihood Estimation (MLE)

An alternative to Bayesian inference is MLE, which finds $\lambda$ that maximizes the likelihood of the observed data:

$$
\lambda_{\text{ML}} = \underset{\lambda}{\mathrm{arg\max}} \;\; p(\{ x_n \}_{n=1}^N | \lambda )
$$

### Summary

Bayesian inference provides a principled way to estimate parameters by considering the whole distribution of the variable given the data. Steps include:

1. Assumes a model $p(x_n|\lambda)$.
2. Specify a prior $p(\lambda)$.
3. Apply Bayes' rule to find the posterior $p(\lambda|\{x_n\}_{n=1}^N)$.
4. Calculate the predictive distribution $p(x^\star |\{x_{n}\}_{n=1}^N)$.

The Bayesian approach retains probability distributions over parameters, unlike point estimate methods like MLE and MAP.

# Inference and Decision Making: A Medical Example

### Part 1: Medical Diagnosis

Alice has a disease test. Let \( a = 1 \) indicate disease presence, and \( b = 1 \) a positive test result. The test's reliability is 95%, and 5% of people like Alice have the disease. If Alice tests positive, compute the probability she has the disease.

Use Bayes' rule:

$$
p(a=1|b=1) = \frac{p(b=1|a=1) p(a=1)}{p(b=1)}
$$

### Part 2: Treatment Decision

The disease has a treatment that affects quality of life. Treatment decisions involve:

$$
R(a, t) = \left[ 
\begin{array}{cc}
R(a=0, t=0) & R(a=0, t=1) \\
R(a=1, t=0) & R(a=1, t=1) \\
\end{array}
\right] 
= 
\left [ \begin{array}{cc}
10 & 7 \\
3 & 5 \\
\end{array}
\right]
$$

Using Bayesian decision theory, calculate expected reward for both treatment and non-treatment actions and choose the one with the highest expected reward.
