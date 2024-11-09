# Supervised Non-Parametric Probabilistic Inference Using Gaussian Processes

## Modelling Data

### Purpose of Models
Mathematical models are essential tools in various fields, serving multiple purposes:

- **Making Predictions**: In time series models, for example, we aim to predict future values based on past observations. Predictions are inherently uncertain, and probabilistic models express this uncertainty through probabilities.
- **Generalization**: Models help us interpolate and extrapolate from training data to new, unseen test cases.
- **Understanding Relationships**: They enable us to uncover and interpret statistical relationships within data.
- **Hypothesis Evaluation**: Models assess the relative probabilities of different hypotheses explaining the data.
- **Data Compression**: They summarize or compress data by capturing essential patterns.
- **Data Generation**: Models can generate new data that follows the same statistical distribution as the training set.

Different tasks necessitate different models. Effective models focus on key aspects while neglecting others to balance accuracy with simplicity and interpretability.

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
## Linear in the Parameters Regression

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

### Fitting the Model: Least Squares Approach
We aim to find $w$ that minimizes the sum of squared errors:

$$
E(w) = \sum_{i=1}^N e_i^2 = \sum_{i=1}^N (y_i - f_w(x_i))^2
$$

This is a convex optimization problem.

The solution is given by the Normal Equations:

$$
\hat{w} = ( \Phi^T \Phi )^{-1} \Phi^T y
$$

- **Design Matrix** $\Phi$: A matrix where $\Phi_{ij} = \phi_j(x_i)$.

### Overfitting and Model Complexity
- **Underfitting**: Model is too simple to capture data patterns (e.g., low-degree polynomial).
- **Overfitting**: Model is too complex, capturing noise instead of underlying patterns (e.g., high-degree polynomial).

Overfitting leads to poor generalization on new data. Solution: Introduce assumptions or regularization to constrain the model.

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
- **Solution**: Same as least squares solution due to equivalence.

#### Multiple Explanations and Uncertainty
- Recognize that multiple models can explain the data equally well. 
- Need to consider model uncertainty and multiple hypotheses. 
- The Bayesian framework provides tools to handle this uncertainty.

---
## Probability Fundamentals

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
## Bayesian Inference and Prediction with Finite Regression Models

### Maximum Likelihood vs. Bayesian Inference

- **Maximum Likelihood (ML)**:
    - Estimates parameters $w_{\text{ML}}$ that maximize $p(y \mid x, w)$.
    - Does not account for parameter uncertainty.
    - Prone to overfitting with complex models.

- **Bayesian Inference**:
    - Considers the posterior distribution of parameters $p(w \mid x, y)$.
    - Incorporates prior beliefs $p(w)$ and updates them with data via Bayes' Rule:

$$
p(w \mid x, y) = \frac{p(y \mid x, w) p(w)}{p(y \mid x)}
$$

### Gaussian Prior and Likelihood

- **Prior**: Assume $p(w) = N(w \mid 0, \sigma_w^2 I)$.
- **Likelihood**: $p(y \mid x, w) = N(y \mid \Phi w, \sigma_{\text{noise}}^2 I)$, where $\Phi$ is the design matrix.
- **Posterior**:

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

- **Result**: Predictive distribution is also Gaussian:

$$
p(y^* \mid x^*, x, y) = N(y^* \mid \Phi(x^*)^T \mu, \Phi(x^*)^T \Sigma \Phi(x^*) + \sigma_{\text{noise}}^2)
$$

--- 
## Marginal Likelihood

### Definition
$$
p(y \mid x) = \int p(y \mid x, w) p(w) \, dw
$$

- **Use in Model Selection**:
    - Penalizes model complexity automatically (i.e. a model is more complex - larger parameter space - lower average likelihood)
    - Favors models that balance data fit and simplicity.

### Analytical Solution (for Gaussian prior and likelihood)
$$
p(y \mid x) = N(y \mid 0, \sigma_w^2 \Phi \Phi^T + \sigma_{\text{noise}}^2 I)
$$

### Bayesian Model Selection
1. **Compute Marginal Likelihood** for different models.
2. **Select Model** with the highest marginal likelihood.

- **Occam's Razor**: Bayesian inference inherently prefers simpler models unless complexity is justified by data.

---
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

---

## Marginal Likelihood and Model Selection

### Understanding Marginal Likelihood
The marginal likelihood integrates over all possible parameter values, weighting them by how likely they are under the prior and how well they explain the data.

- **Interpretation**: Models that can explain the data well without requiring improbable parameter values achieve higher marginal likelihood.

### Monte Carlo Approximation

#### Goal
Estimate $p(y \mid x, M)$ when the integral is intractable.

#### Simple Monte Carlo
1. **Sample** $w^{(s)}$ from the prior $p(w \mid M)$.
2. **Approximate**:

   $$
   p(y \mid x, M) \approx \frac{1}{S} \sum_{s=1}^S p(y \mid x, w^{(s)})
   $$

#### Acceptance-Rejection Sampling
Accept samples where $p(y \mid x, w^{(s)})$ is non-zero. This provides insight into how the marginal likelihood penalizes overly complex models.

### Example: Model Comparison
Given models $M_1$, $M_2$, and $M_3$:

- $M_1$: Simpler model.
- $M_2$: Intermediate complexity.
- $M_3$: More complex model.

#### Observation
The marginal likelihood $p(y \mid x, M)$ may favor $M_2$, balancing fit and simplicity.

### Bayesian Occam's Razor
The marginal likelihood naturally embodies Occam's Razor by penalizing unnecessary complexity. Complex models have larger parameter spaces but require fine-tuning to fit data, leading to a lower marginal likelihood if the data does not justify the complexity.

## Conclusion
Probability theory provides a robust framework for:

1. **Inference**: Drawing conclusions from data within a model.
2. **Prediction**: Making probabilistic forecasts about unseen data.
3. **Model Comparison**: Evaluating and selecting models based on their marginal likelihood.

By adopting Bayesian methods, we account for parameter uncertainty, avoid overfitting, and make principled decisions about model complexity. Understanding Gaussian processes and probabilistic inference equips us with powerful tools to tackle complex machine learning problems with confidence.

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

---

## Distributions Over Parameters and Functions

### Priors on Parameters Induce Priors on Functions
In parametric models, we define a model $f_w(x)$ using parameters $w$:

$$
f_w(x) = \sum_{m=0}^M w_m \phi_m(x)
$$

where $\phi_m(x)$ are basis functions (e.g., polynomial terms $x^m$) and $w_m$ are the weights.

By placing a prior distribution $p(w)$ over the parameters $w$, we implicitly define a prior over functions $f(x)$.

**Example:**
- Choose $M=17$.
- Set $p(w_m) = N(0, \sigma_w^2)$ for all $m$.
- The prior over $w$ induces a distribution over functions $f(x)$.

This means that by sampling from the prior over $w$, we can generate random functions from the prior over $f(x)$.

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

## Gaussian Processes

### From Scalar Gaussians to Multivariate Gaussians to Gaussian Processes

1. **Scalar Gaussian**: A single random variable $x$ with distribution $N(\mu, \sigma^2)$.

2. **Multivariate Gaussian**: A vector $x = [x_1, x_2, \dots, x_N]^T$ with joint Gaussian distribution:

   $$
   p(x \mid \mu, \Sigma) = \frac{1}{(2 \pi)^{N/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)
   $$

3. **Gaussian Process (GP)**: An extension to infinitely many variables.

   - **Definition**: A collection of random variables, any finite number of which have a joint Gaussian distribution.
   - **Intuition**: Think of functions as infinitely long vectors.

### Gaussian Process Definition
A GP is fully specified by:

- **Mean function** $m(x) = E[f(x)]$
- **Covariance function** $k(x, x') = E[(f(x) - m(x))(f(x') - m(x'))]$

**Notation**:

$$
f(x) \sim GP(m(x), k(x, x'))
$$

### Marginal and Conditional Gaussians
Key properties:

- **Marginalization**: The marginal distribution over any subset of variables is Gaussian.
- **Conditioning**: The conditional distribution given some variables is also Gaussian.

### Generating Functions from a GP
To generate sample functions:

1. **Select Inputs**: Choose $N$ input points $x_1, x_2, \dots, x_N$.
2. **Compute Covariance Matrix**: $K_{ij} = k(x_i, x_j)$.
3. **Sample Function Values**: Draw $f \sim N(0, K)$.
4. **Plot Function**: Plot $f$ versus $x$.

#### Sequential Generation
Generate function values one at a time, conditioning on previous values. This uses properties of conditional Gaussians.

## Gaussian Processes and Data

### Conditioning on Observations
Given observed data $D = \{(x_i, y_i)\}_{i=1}^N$, we want to predict $f_*$ at new inputs $x_*$.

Assumption: Observations $y_i$ are noisy versions of the true function $f(x_i)$:

$$
y_i = f(x_i) + \epsilon_i, \quad \epsilon_i \sim N(0, \sigma_n^2)
$$

### Posterior Gaussian Process
**Posterior Mean and Covariance**:

$$
E[f_* \mid x, y, x_*] = k(x_*, x) [K + \sigma_n^2 I]^{-1} y
$$

$$
\text{Var}[f_* \mid x, y, x_*] = k(x_*, x_*) - k(x_*, x) [K + \sigma_n^2 I]^{-1} k(x, x_*)
$$

- $K$ is the covariance matrix of the training inputs.
- $k(x_*, x)$ is the vector of covariances between the test input $x_*$ and training inputs $x$.

### Prior and Posterior in Pictures
- **Prior**: Represents our beliefs about the function before seeing any data.
- **Posterior**: Updated beliefs after incorporating observed data.

**Visualization**:
- **Prior Samples**: Functions drawn from the GP prior.
- **Posterior Samples**: Functions drawn from the GP posterior, which now pass through (or near) the observed data points.

## Gaussian Process Marginal Likelihood and Hyperparameters

### The GP Marginal Likelihood
The marginal likelihood (or evidence) is the probability of the observed data under the GP model:

$$
p(y \mid x) = \int p(y \mid f) p(f) \, df
$$

For GPs with Gaussian noise, this integral can be computed analytically:

$$
\log p(y \mid x) = -\frac{1}{2} y^T (K + \sigma_n^2 I)^{-1} y - \frac{1}{2} \log |K + \sigma_n^2 I| - \frac{N}{2} \log 2 \pi
$$

**Interpretation**:
- The first term measures how well the model fits the data (data fit).
- The second term penalizes model complexity (complexity penalty).
- Occam's Razor is automatically applied, preferring simpler models that explain the data well.

### Hyperparameters and Model Selection

- **Hyperparameters** $\theta$: Parameters of the covariance function (e.g., length-scale $\ell$, signal variance $\sigma_f^2$, noise variance $\sigma_n^2$).
- **Optimizing Hyperparameters**:

   Find $\theta$ that maximize the marginal likelihood:

   $$
   \theta^* = \arg \max_\theta \log p(y \mid x, \theta)
   $$

   This is a form of model selection.

**Example**:
- **Squared Exponential Covariance Function**:

  $$
  k(x, x') = \sigma_f^2 \exp \left( -\frac{(x - x')^2}{2 \ell^2} \right)
  $$

  By adjusting $\ell$ and $\sigma_f^2$, we can control the smoothness and amplitude of the functions.

### Occam's Razor
The marginal likelihood balances data fit and model complexity:

- Simple models with fewer hyperparameters may not fit the data well but are preferred if they explain the data sufficiently.
- Complex models may overfit the data but are penalized in the marginal likelihood due to increased complexity.

## Correspondence Between Linear Models and Gaussian Processes

### From Linear Models to GPs
Consider a linear model with Gaussian priors:

$$
f(x) = \sum_{m=1}^M w_m \phi_m(x), \quad w_m \sim N(0, \sigma_w^2)
$$

- **Mean Function**: $m(x) = E[f(x)] = 0$
- **Covariance Function**:

  $$
  k(x, x') = E[f(x) f(x')] = \sigma_w^2 \sum_{m=1}^M \phi_m(x) \phi_m(x') = \sigma_w^2 \phi(x)^T \phi(x')
  $$

This shows that the linear model with Gaussian priors corresponds to a GP with covariance function $k(x, x')$.

### From GPs to Linear Models
Conversely, any GP with covariance function $k(x, x') = \phi(x)^T A \phi(x')$ can be represented as a linear model with basis functions $\phi(x)$ and weight covariance $A$.

- **Mercer's Theorem**: Some covariance functions correspond to infinite-dimensional feature spaces.

## Computational Considerations

- **Gaussian Processes**: Complexity is $O(N^3)$ due to inversion of the $N \times N$ covariance matrix. Feasible for small to medium-sized datasets.
- **Linear Models**: Complexity is $O(N M^2)$, where $M$ is the number of basis functions. Can be more efficient when $M$ is small.

## Covariance Functions

### Stationary Covariance Functions
Covariance functions that depend only on $r = |x - x'|$.

1. **Squared Exponential (SE)**

   $$
   k_{\text{SE}}(r) = \sigma_f^2 \exp \left( -\frac{r^2}{2 \ell^2} \right)
   $$

2. **Rational Quadratic (RQ)**

   $$
   k_{\text{RQ}}(r) = \sigma_f^2 \left( 1 + \frac{r^2}{2 \alpha \ell^2} \right)^{-\alpha}
   $$

3. **Matérn**

   $$
   k_{\text{Matérn}}(r) = \sigma_f^2 \frac{2^{1 - \nu}}{\Gamma(\nu)} \left( \frac{\sqrt{2 \nu} r}{\ell} \right)^\nu K_\nu \left( \frac{\sqrt{2 \nu} r}{\ell} \right)
   $$

4. **Periodic Covariance Function**

   $$
   k_{\text{Per}}(x, x') = \sigma_f^2 \exp \left( -\frac{2 \sin^2 \left( \frac{\pi |x - x'|}{p} \right)}{\ell^2} \right)
   $$

5. **Neural Network Covariance Function**

   $$
   k_{\text{NN}}(x, x') = \frac{\sigma_f^2}{\pi} \sin^{-1} \left( \frac{2 x^T \Sigma x'}{\sqrt{(1 + 2 x^T \Sigma x)(1 + 2 x'^T \Sigma x')}} \right)
   $$

### Combining Covariance Functions
- **Addition**: $k(x, x') = k_1(x, x') + k_2(x, x')$
- **Multiplication**: $k(x, x') = k_1(x, x') \cdot k_2(x, x')$
- **Scaling**: $k(x, x') = g(x) k(x, x') g(x')$, where $g(x)$ is a function.

## Conclusion
Gaussian Processes offer a robust, flexible framework for modeling complex datasets without specifying a fixed number of parameters. By defining a prior directly over functions, GPs capture our beliefs about function properties such as smoothness and periodicity. The marginal likelihood provides a principled way to select hyperparameters and models, embodying Occam's Razor by balancing data fit and model complexity. Understanding the relationship between linear models and GPs, as well as the role of covariance functions, is crucial for effectively applying GPs to real-world problems.

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
