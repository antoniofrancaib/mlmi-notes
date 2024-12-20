## Index

- [1. Modelling Data](#1-modelling-data)
- [2. Linear in the Parameters Regression](#2-linear-in-the-parameters-regression)
- [4. Probability Fundamentals](#4-probability-fundamentals)
- [5. Bayesian Inference and Prediction with Finite Regression Models](#5-bayesian-inference-and-prediction-with-finite-regression-models)
- [7. Marginal Likelihood](#7-marginal-likelihood)
- [8. Distributions Over Parameters and Functions](#8-distributions-over-parameters-and-functions)

---
## 1-Modelling-Data

### Purpose of Models
Mathematical models are useful for many purposes, including

- **Making Predictions**: In time series models, we predict future values based on past observations. Predictions are inherently uncertain, and *probabilistic model*s express this uncertainty through probabilities.
- **Generalization**: interpolate and extrapolate from training data to new, unseen test cases.
- **Understanding Relationships**: They enable us to uncover and interpret statistical relationships within data.
- **Hypothesis Evaluation**: Models assess the relative probabilities of different hypotheses explaining the data.
- **Data Compression**: They summarize or compress data by capturing essential patterns.
- **Data Generation**: Models can generate new data that follows the same statistical distribution as the training set.

Different tasks require different models. Effective models focus on key aspects while neglecting others to balance **accuracy with simplicity** and interpretability.

### Origin of Models
Models can originate from:

- **First Principles**: Derived from fundamental theories (e.g., Newtonian mechanics for planetary motion).
- **Data Observations**: Empirical models built from observed data (e.g., modeling timber production based on environmental factors).

Most practical models combine both first principles and data. Machine learning emphasizes models that significantly rely on data to learn patterns and make predictions.

### Knowledge, Assumptions, and Simplifying Assumptions
Every model is built upon explicit or implicit assumptions:

- **Knowledge**: Facts we know to be true (e.g., distances are non-negative).
- **Assumptions**: Hypotheses we accept as true for modeling purposes, which may or may not hold in reality (e.g., income is independent of gender given age and profession).
- **Simplifying Assumptions**: Approximations made to simplify models (e.g., assuming a linear relationship between drug dosage and response within a certain range).

Probabilistic models use priors to express knowledge or beliefs about model components. Simplifying assumptions make models tractable but limit their expressiveness. Thus, practical modeling is a trade-off between model ***complexity*** and ***computational*** feasibility.

### Observations, Parameters, and Latent Variables
Consider a time series model:
![[Pasted image 20241119102548.png]]

- **Observations** $(y)$: Measured data points (shaded nodes).
- **Latent Variables** $(x)$: Unobserved or hidden variables that influence observations.
- **Parameters** $(A \text{ and } C)$: Constants defining transitions and emissions in the model.

In modeling, we must decide how to handle unobserved quantities—through inference, estimation, sampling, or marginalization. The key difference between parameters and latent variables is that the number of latent variables grows with data size, while parameters remain fixed.

### Practical Modelling Tasks
When specifying a model, we need to:

1. **Handle Unobserved Quantities**:
    - Estimate latent variables and parameters.
    - Possibly infer aspects of the model structure.
2. **Make Predictions**: Apply the model to new, unseen data.
3. **Interpret the Model**: Extract insights and understand underlying mechanisms.
4. **Evaluate Accuracy**: Assess model performance on training and test data.
5. **Model Selection and Criticism**: Compare different models or variants and identify limitations.

These tasks must be addressed within computational and memory constraints, often requiring approximate solutions.

### A Common Misunderstanding
A model's role is to make predictions and provide insights into certain data aspects, not to fully represent all data intricacies. Thus, notions like "true" or "correct" models are meaningless in machine learning.

> "Essentially, all models are wrong, but some are useful."
> — George E. P. Box

---
## 2-Linear-in-the-Parameters-Regression

### The Regression Task
Given a dataset of $N$ pairs $\{(x_i, y_i)\}_{i=1}^N$:

- **Goal**: Predict the target $y^*$ for a new input $x^*$.

This is a regression problem, where inputs can be scalar or vector-valued.

### Model Selection
We need to choose:

- **Model Structure**: What form does the function $f(x)$ take?
- **Model Parameters**: For a given structure, what are the best parameter values?

### Polynomial Models
Consider modeling $f(x)$ as an $M$-degree polynomial:

$$
f_w(x) = w_0 + w_1 x + w_2 x^2 + \dots + w_M x^M
$$

- **Parameters**: Coefficients $w = [w_0, w_1, \dots, w_M]^T$.
- **Basis Functions**: $\phi_j(x) = x^j$ for $j=0, 1, \dots, M$.

![[pols2.png]]

### Fitting the Model: Least Squares Approach
We aim to find $w$ that minimizes the sum of squared errors:

$$
E(w) = \sum_{i=1}^N e_i^2 = \sum_{i=1}^N (y_i - f_w(x_i))^2
$$

This is a **convex optimization** problem.

The solution is given by the Normal Equations:

$$
\hat{w} = ( \Phi^T \Phi )^{-1} \Phi^T y
$$

- **Design Matrix** $\Phi$: A matrix where $\Phi_{ij} = \phi_j(x_i)$.

### Overfitting and Model Complexity
- **Underfitting**: Model is too simple to capture data patterns (e.g., low-degree polynomial).
- **Overfitting**: Model is too complex, capturing noise instead of underlying patterns (e.g., high-degree polynomial).

Overfitting leads to poor generalization on new data. Solution: Introduce assumptions or regularization to constrain the model.

### Open Questions: 

- Do we think that all models are equally probable... before we see any data? What does the probability of a model even mean? 
- Do we need to choose a single “best” model or can we consider several? We need a framework to answer such questions. 
- Perhaps our training targets are contaminated with noise. What to do? This question is a bit easier, we will start here.

---
## Likelihood and the Concept of Noise

#### Observation Noise
Assuming that data is generated by:

$$
y_n = f(x_n) + \epsilon_n
$$

- **Noise Term** $\epsilon_n$: Represents measurement errors or inherent randomness.
- **Assumption**: $\epsilon_n$ are independent and identically distributed (i.i.d.) random variables.

#### Gaussian Noise Model
- **Gaussian Distribution**: Common choice for $\epsilon_n$:

$$
p(\epsilon_n) = \frac{1}{\sqrt{2 \pi \sigma_{\text{noise}}^2}} \exp \left( - \frac{\epsilon_n^2}{2 \sigma_{\text{noise}}^2} \right)
$$

- **Likelihood of Observations**:

$$
p(y \mid f, \sigma_{\text{noise}}^2) = \prod_{n=1}^N p(y_n \mid f(x_n), \sigma_{\text{noise}}^2)
$$

- **Equivalence to Least Squares**: Maximizing the likelihood under Gaussian noise is equivalent to minimizing the sum of squared errors.

#### Maximum Likelihood Estimation
- **Objective**: Find $\hat{w}$ that maximizes the likelihood $p(y \mid w, \sigma_{\text{noise}}^2)$.
 $$
\hat{\mathbf{w}} = \arg\max \mathcal{L}(\mathbf{w}) = \arg\max \exp\left(-\frac{E(\mathbf{w})}{2\sigma_{\text{noise}}^2}\right) = \arg\min E(\mathbf{w})
$$
where
$$
E(\mathbf{w}) = \sum_{n=1}^N \left(y_n - f_w(\mathbf{x}_n)\right)^2 = \|\mathbf{y} - \Phi \mathbf{w}\|^2 = \boldsymbol{\epsilon}^\top \boldsymbol{\epsilon} \quad \text{is the sum of squared errors}
$$

- **Solution**: Same as least squares solution due to equivalence.

#### Multiple Explanations and Uncertainty
- We do not believe all models are equally probable to explain the data. 
- We may believe a simpler model is more probable than a complex one. 

Model complexity and uncertainty: 
- We do not know what particular function generated the data. 
- More than one of our models can perfectly fit the data. 
- We believe more than one of our models could have generated the data. 
- We want to reason in terms of a set of possible explanations, not just one.

---
## 4-Probability-Fundamentals

### Basics of Probability
- **Random Variables**: Quantities whose outcomes are uncertain.
- **Probability Distribution**: Describes how probabilities are assigned to different outcomes.
- **Joint Probability**: Probability of two events occurring together.
- **Marginal Probability**: Probability of an event irrespective of other variables.
- **Conditional Probability**: Probability of an event given that another event has occurred.

### The Two Rules of Probability

#### Product Rule
$$
p(A, B) = p(A \mid B) p(B) = p(B \mid A) p(A)
$$

#### Sum Rule (Marginalization)
$$
p(A) = \sum_B p(A, B)
$$

### Bayes' Rule
Derived from the product rule:

$$
p(A \mid B) = \frac{p(B \mid A) p(A)}{p(B)}
$$

- **Interpretation**: How to update the probability of $A$ given new evidence $B$.

### Medical Diagnosis Example
- **Context**: Breast cancer screening.
- Given:
    - **Prevalence of cancer** $p(C) = 1\%$.
    - **True positive rate** $p(M \mid C) = 80\%$.
    - **False positive rate** $p(M \mid \neg C) = 9.6\%$.
- **Question**: What is $p(C \mid M)$, the probability of having cancer given a positive test?

- **Solution Using Bayes' Rule**:

$$
p(C \mid M) = \frac{p(M \mid C) p(C)}{p(M)} = \frac{0.8 \times 0.01}{(0.8 \times 0.01) + (0.096 \times 0.99)} \approx 7.8\%
$$

- **Interpretation**: Despite a positive test, the probability of cancer is still relatively low due to the low prevalence.

---
## 5-Bayesian-Inference-and-Prediction-with-Finite-Regression-Models

### Maximum Likelihood vs. Bayesian Inference

- **Maximum Likelihood (ML)**:
    - Estimates parameters $w_{\text{ML}}$ that maximize $p(y \mid x, w)$.
    - Does not account for parameter uncertainty.
    - Prone to overfitting with complex models.
    - The marginal likelihood prefers an intermediate model size (Occam’s Razor).

- **Bayesian Inference**:
    - Considers the posterior distribution of parameters $p(w \mid x, y)$.
    - Incorporates prior beliefs $p(w)$ and updates them with data via Bayes' Rule:

$$
p(w \mid x, y) = \frac{p(y \mid x, w) p(w)}{p(y \mid x)}
$$

### Gaussian Prior and Likelihood

- **Prior**: Assume $p(w) = N(w \mid 0, \sigma_w^2 I)$.
- **Likelihood**: $p(y \mid x, w) = N(y \mid \Phi w, \sigma_{\text{noise}}^2 I)$, where $\Phi$ is the design matrix.
- **Posterior**: <span style="color:blue;">Prove this: </span>

$$
p(w \mid x, y) = N(w \mid \mu, \Sigma)
$$

- **Posterior Mean**:

$$
\mu = \Sigma \Phi^T y / \sigma_{\text{noise}}^2
$$

- **Posterior Covariance**:

$$
\Sigma = \left( \frac{1}{\sigma_{\text{noise}}^2} \Phi^T \Phi + \frac{1}{\sigma_w^2} I \right)^{-1}
$$

### Predictive Distribution
To predict $y^*$ for a new input $x^*$:

$$
p(y^* \mid x^*, x, y) = \int p(y^* \mid x^*, w) p(w \mid x, y) \, dw
$$

- **Result**: Predictive distribution is also Gaussian: <span style="color:blue;">Prove this: </span>

$$
p(y^* \mid x^*, x, y) = N(y^* \mid \Phi(x^*)^T \mu, \Phi(x^*)^T \Sigma \Phi(x^*) + \sigma_{\text{noise}}^2)
$$

[Some useful Gaussian and Matrix equations](4F13/multivariate-gaussian-densities)

--- 
## 7-Marginal-Likelihood

### Definition: 
The **marginal likelihood**, also known as the **model evidence** is the probability of observing the data given a particular model, integrating out all the model's parameters. 
$$
p(y \mid x, \mathcal{M}) = \int p(\mathbf{w} \mid \mathcal{M}) p(y \mid x, \mathbf{w}, \mathcal{M}) \, d\mathbf{w}.
$$

- **Use in Model Selection**:
    - Penalizes model complexity automatically (i.e. a model is more complex - prior spreads out over a larger parameter space - lower average likelihood - reducing the marginal likelihood unless those parameters are strongly supported by the data)
    - Favors models that balance data fit and simplicity.

### Analytical Solution (for Gaussian prior and likelihood)
$$
p(y \mid x) = N(y \mid 0, \sigma_w^2 \Phi \Phi^T + \sigma_{\text{noise}}^2 I)
$$

### Understanding Marginal Likelihood
The marginal likelihood integrates over all possible parameter values, weighting them by how likely they are under the prior and how well they explain the data.

- **Interpretation**: Models that can explain the data well without requiring improbable parameter values achieve higher marginal likelihood.

### Bayesian Model Selection
1. **Compute Marginal Likelihood** for different models.
2. **Select Model** with the highest marginal likelihood.

- **Occam's Razor**: Bayesian inference inherently prefers simpler models unless complexity is justified by data.

### Computing Intractable Marginal Likelihoods: Monte Carlo Approximation
#### Goal
Estimate $p(y \mid x, M)$ when the integral is intractable.
#### Simple Monte Carlo
1. **Sample** $w^{(s)}$ from the prior $p(w \mid M)$.
2. **Approximate**:

$$ p(y \mid x, M) = \int p(y \mid x, \mathbf{w}) p(\mathbf{w} \mid M) \, d\mathbf{w} \approx \frac{1}{S} \sum_{s=1}^S p(y \mid x, w^{(s)}) $$

#### Acceptance-Rejection Sampling
Accept samples where $p(y \mid x, w^{(s)})$ is non-zero. This provides insight into how the marginal likelihood penalizes overly complex models.

### Bayesian Occam's Razor
The marginal likelihood naturally embodies Occam's Razor by penalizing unnecessary complexity. Complex models have larger parameter spaces but require fine-tuning to fit data, leading to a lower marginal likelihood if the data does not justify the complexity.

## Conclusion
Probability theory provides a robust framework for:

1. **Inference**: Drawing conclusions from data within a model.
2. **Prediction**: Making probabilistic forecasts about unseen data.
3. **Model Comparison**: Evaluating and selecting models based on their marginal likelihood.

By adopting Bayesian methods, we account for parameter uncertainty, avoid overfitting, and make principled decisions about model complexity. Understanding Gaussian processes and probabilistic inference equips us with powerful tools to tackle complex machine learning problems with confidence.

---

## 8-Distributions-Over-Parameters-and-Functions

### Priors on Parameters Induce Priors on Functions
In parametric models, we define a model $f_w(x)$ using parameters $w$:

$$
f_w(x) = \sum_{m=0}^M w_m \phi_m(x)
$$

where $\phi_m(x)$ are basis functions (e.g., polynomial terms $x^m$) and $w_m$ are the weights.

By placing a prior distribution $p(w)$ over the parameters $w$, we implicitly define a prior over functions $f(x)$. This means that by sampling from the prior over $w$, we can generate random functions from the prior over $f(x)$.

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
