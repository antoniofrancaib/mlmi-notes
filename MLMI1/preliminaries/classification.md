---
layout: default
title: Classification
---
## Outline of Topics

- **Binary Logistic Classification**
- **Multi-class Softmax Classification**
- **Non-linear Classification**
- **Overfitting in Classification**
- **k-Nearest Neighbours (kNN) Classification**
- **Bayesian Logistic Regression and Laplace Approximation**

## Introduction to Classification

Classification is a fundamental task in supervised machine learning where the goal is to predict a discrete output $y^\ast$ for a given input $x^\ast$, based on a training set of input-output pairs $\{(x_n, y_n)\}_{n=1}^N$. Unlike regression, which predicts continuous outputs, classification focuses on assigning inputs to discrete categories or classes.

## Types of Classification Problems

- **Binary Classification**: Two possible output classes (e.g., spam vs. not spam).
- **Multi-class Classification**: More than two classes (e.g., classifying types of flowers).
- **Multi-label Classification**: Each instance may belong to multiple classes simultaneously.
- **Structured Prediction**: Outputs have interdependent components (e.g., sequence labeling in natural language processing).

## Key Concepts in Classification

- **Generative Models**: Model the joint probability $p(x, y)$.
- **Discriminative Models**: Model the conditional probability $p(y \mid x)$ directly.
- **Maximum Likelihood Estimation (MLE)**: Estimates model parameters by maximizing the likelihood of the observed data.
- **Overfitting**: When a model learns the training data too well, including noise, leading to poor generalization.
- **Probabilistic Inference**: Uses probability distributions to make predictions and quantify uncertainty.

---
# Binary Logistic Classification

Logistic regression models the probability that a given input $x_n$ belongs to class $y_n = 1$ (assuming binary classes $y_n \in \{0, 1\}$).

- **Linear Combination**:
  $$a_n = w^\top x_n = \sum_{d=1}^D w_d x_{n,d}$$
  - $w$: Weight vector.
  - $a_n$: Linear activation for input $x_n$.

- **Sigmoid Function**:
  $$p(y_n = 1 \mid x_n, w) = \sigma(a_n) = \frac{1}{1 + \exp(-a_n)}$$
  - Maps the linear activation $a_n$ to a probability between 0 and 1.

### Decision Boundary

- The decision boundary is defined where $p(y_n = 1 \mid x_n, w) = 0.5$, which corresponds to $a_n = 0$.
- The orientation of the decision boundary is determined by $w$.

### Maximum Likelihood Estimation for Logistic Regression

To find the optimal weights $w$, we use maximum likelihood estimation.

#### Likelihood Function

Given the training data $\{(x_n, y_n)\}_{n=1}^N$, the likelihood is:
$$L(w) = \prod_{n=1}^N p(y_n \mid x_n, w)$$

For binary outputs:
$$p(y_n \mid x_n, w) = \sigma(a_n)^{y_n} [1 - \sigma(a_n)]^{1 - y_n}$$

Thus, the likelihood becomes:
$$L(w) = \prod_{n=1}^N \sigma(a_n)^{y_n} [1 - \sigma(a_n)]^{1 - y_n}$$

#### Log-Likelihood Function

Taking the logarithm simplifies calculations:
$$\ell(w) = \sum_{n=1}^N \left[ y_n \log \sigma(a_n) + (1 - y_n) \log (1 - \sigma(a_n)) \right]$$

#### Gradient of the Log-Likelihood

To maximize $\ell(w)$, compute its gradient:
$$\nabla_w \ell(w) = \sum_{n=1}^N (y_n - \sigma(a_n)) x_n$$

The gradient is a sum over the training examples, weighted by the difference between the actual label $y_n$ and the predicted probability $\sigma(a_n)$.

#### Optimization via Gradient Ascent

Update the weights iteratively:
$$w_{t+1} = w_t + \eta \nabla_w \ell(w_t)$$

- $\eta$: Learning rate.
- Continue until convergence.

### Convergence and Uniqueness

- The log-likelihood function is concave, ensuring a unique global maximum.
- The Hessian matrix is negative definite:
  $$\nabla_w^2 \ell(w) = - \sum_{n=1}^N \sigma(a_n) [1 - \sigma(a_n)] x_n x_n^\top$$

### Decision Boundaries and Thresholds

#### Predicting Class Labels

- Default rule: Predict $y_n = 1$ if $p(y_n = 1 \mid x_n, w) \ge 0.5$.
- Decision boundary is where $a_n = 0$.

#### Adjusting for Misclassification Costs

- In some cases, misclassification costs are asymmetric.
- Adjust the threshold to minimize expected cost:
  $$\text{Threshold} = \frac{L(0, 0) - L(0, 1)}{[L(0, 0) - L(0, 1)] + [L(1, 1) - L(1, 0)]}$$

- $L(y, \hat{y})$: Loss function representing the cost of predicting $\hat{y}$ when the true label is $y$.

---
# Multi-class Softmax Classification

We extend the binary logistic regression model to handle multiple classes by introducing the **softmax classification model**. In this model, each data point consists of an input vector $\mathbf{x}_n$ and an output label $y_n$, where $y_n \in \{1, 2, \dots, K\}$ indicates the class of the $n$-th data point.

## The Softmax Classification Model

The model comprises two stages:

1. **Computing Activations**:

   For each class $k$, compute the activation:

   $$
   a_{n,k} = \mathbf{w}_k^\top \mathbf{x}_n
   $$

   where $\mathbf{w}_k$ is the weight vector associated with class $k$.

2. **Softmax Function**:

   The activations are passed through the softmax function to obtain the probability that data point $\mathbf{x}_n$ belongs to class $k$:

   $$
   p(y_n = k \mid \mathbf{x}_n, \{\mathbf{w}_k\}_{k=1}^K) = \frac{\exp(a_{n,k})}{\sum_{k'=1}^K \exp(a_{n,k'})} = \frac{\exp(\mathbf{w}_k^\top \mathbf{x}_n)}{\sum_{k'=1}^K \exp(\mathbf{w}_{k'}^\top \mathbf{x}_n)}
   $$

By construction, the probabilities are normalized:

$$
\sum_{k=1}^K p(y_n = k \mid \mathbf{x}_n, \{\mathbf{w}_k\}_{k=1}^K) = 1
$$

Thus, the softmax function parameterizes a categorical distribution over the output classes.

## Fitting Using Maximum Likelihood Estimation

To estimate the weight vectors $\{\mathbf{w}_k\}$, we use maximum likelihood estimation (MLE).

### Likelihood Function

We first represent the output labels using one-hot encoding. For each data point $n$, the label $y_n$ is encoded as a vector $\mathbf{y}_n$ of length $K$:

$$
y_{n,k} =
\begin{cases}
1 & \text{if } y_n = k \\
0 & \text{otherwise}
\end{cases}
$$

The likelihood of the parameters given the data is:

$$
p(\{ y_n \}_{n=1}^N \mid \{ \mathbf{x}_n \}_{n=1}^N, \{ \mathbf{w}_k \}_{k=1}^K) = \prod_{n=1}^N \prod_{k=1}^K s_{n,k}^{y_{n,k}}
$$

where:

$$
s_{n,k} = p(y_n = k \mid \mathbf{x}_n, \{\mathbf{w}_k\}_{k=1}^K)
$$

### Log-Likelihood Function

Taking the logarithm of the likelihood simplifies the product into a sum:

$$
\mathcal{L}(\{ \mathbf{w}_k \}_{k=1}^K) = \sum_{n=1}^N \sum_{k=1}^K y_{n,k} \log s_{n,k}
$$

### Gradient of the Log-Likelihood

To maximize the log-likelihood, we compute its gradient with respect to each weight vector $\mathbf{w}_j$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}_j} = \sum_{n=1}^N (y_{n,j} - s_{n,j}) \mathbf{x}_n
$$

#### Derivation of the Gradient

Starting from the log-likelihood:

$$
\mathcal{L} = \sum_{n=1}^N \sum_{k=1}^K y_{n,k} \log s_{n,k}
$$

Compute the derivative with respect to $\mathbf{w}_j$:

1. **Compute $\frac{\partial s_{n,k}}{\partial a_{n,j}}$**:

   The softmax function is:

   $$
   s_{n,k} = \frac{\exp(a_{n,k})}{\sum_{k'=1}^K \exp(a_{n,k'})}
   $$

   The derivative of $s_{n,k}$ with respect to $a_{n,j}$ is:

   $$
   \frac{\partial s_{n,k}}{\partial a_{n,j}} = s_{n,k} (\delta_{k,j} - s_{n,j})
   $$

   where $\delta_{k,j}$ is the Kronecker delta function:

   $$
   \delta_{k,j} =
   \begin{cases}
   1 & \text{if } k = j \\
   0 & \text{if } k \ne j
   \end{cases}
   $$

2. **Compute $\frac{\partial \mathcal{L}}{\partial \mathbf{w}_j}$**:

   $$
   \begin{align*}
   \frac{\partial \mathcal{L}}{\partial \mathbf{w}_j} &= \sum_{n=1}^N \sum_{k=1}^K y_{n,k} \frac{1}{s_{n,k}} \frac{\partial s_{n,k}}{\partial \mathbf{w}_j} \\
   &= \sum_{n=1}^N \sum_{k=1}^K y_{n,k} \frac{1}{s_{n,k}} \left( \frac{\partial s_{n,k}}{\partial a_{n,j}} \frac{\partial a_{n,j}}{\partial \mathbf{w}_j} \right) \\
   &= \sum_{n=1}^N \sum_{k=1}^K y_{n,k} \frac{1}{s_{n,k}} s_{n,k} (\delta_{k,j} - s_{n,j}) \mathbf{x}_n \\
   &= \sum_{n=1}^N \sum_{k=1}^K y_{n,k} (\delta_{k,j} - s_{n,j}) \mathbf{x}_n \\
   &= \sum_{n=1}^N \left( y_{n,j} - s_{n,j} \right) \mathbf{x}_n
   \end{align*}
   $$

This gradient has an intuitive interpretation: for each data point, we compute the difference between the actual label $y_{n,j}$ and the predicted probability $s_{n,j}$ for class $j$, multiplied by the input $\mathbf{x}_n$.

### Optimization via Gradient Ascent

We can use gradient ascent to iteratively update the weights:

$$
\mathbf{w}_j^{(t+1)} = \mathbf{w}_j^{(t)} + \eta \frac{\partial \mathcal{L}}{\partial \mathbf{w}_j}
$$

where $\eta$ is the learning rate.

## Applying Softmax Classification to the Iris Dataset

The softmax classification model can be applied to datasets with multiple classes, such as the Iris dataset, which contains measurements of iris flowers from three species.

- **Features**: Sepal length, sepal width, petal length, petal width.
- **Classes**: Iris setosa, Iris versicolor, Iris virginica.

By using all four features and the three classes, we can train a softmax classifier to predict the species of an iris flower based on its measurements.

**Visualization**: In practice, we can visualize the decision boundaries and probabilities for each class in the feature space to understand how the model is making predictions.

## Summary

- The **softmax classification model** generalizes logistic regression to handle multiple classes by computing a set of linear activations and passing them through the softmax function to obtain class probabilities.
- **Maximum likelihood estimation** is used to fit the model, and the gradient of the log-likelihood has a concise form that facilitates optimization.
- The model can be applied to real-world datasets, and understanding its properties helps in interpreting and improving classification tasks.

---
# Non-linear Classification

Linear classification models are limited to linear decision boundaries, which may not be suitable for complex datasets. To handle non-linear decision boundaries, we introduce **non-linear basis functions**.

## Non-linear Classification through Basis Functions

We enhance the model by transforming the input features into a higher-dimensional space using non-linear basis functions.

### Basis Function Expansion

Define a set of basis functions $\boldsymbol{\Phi}(\mathbf{x}_n) = [\phi_1(\mathbf{x}_n), \phi_2(\mathbf{x}_n), \dots, \phi_D(\mathbf{x}_n)]^\top$.

Compute the activation:

$$
a_n = \mathbf{w}^\top \boldsymbol{\Phi}(\mathbf{x}_n)
$$

The probability is then:

$$
p(y_n = 1 \mid \mathbf{x}_n, \mathbf{w}) = \sigma(a_n) = \frac{1}{1 + \exp(-a_n)}
$$

### Radial Basis Functions (RBFs)

A common choice for basis functions is radial basis functions:

$$
\phi_d(\mathbf{x}) = \exp\left( -\frac{\|\mathbf{x} - \boldsymbol{\mu}_d\|^2}{2l^2} \right)
$$

- $\boldsymbol{\mu}_d$ is the center of the $d$-th basis function.
- $l$ is the length-scale parameter controlling the width.

### Overfitting in Non-linear Classification

Using too many basis functions or a small length-scale can lead to overfitting:

- The model becomes overly complex, fitting the noise in the training data.
- Poor generalization to unseen data.

### Visualizing Predictions

By plotting the probability contours or decision boundaries, we can observe how the non-linear model captures complex patterns in the data.

## Summary

- Introducing non-linear basis functions allows the model to capture complex, non-linear relationships between inputs and outputs.
- Careful selection of the number and type of basis functions is crucial to avoid overfitting.

---
# Overfitting in Classification

Overfitting occurs when a model learns the training data too well, including its noise, resulting in poor performance on unseen data.

## Example 1: Overfitting in Linear Binary Logistic Classification

- When training data are **linearly separable**, the logistic regression model tends to increase the magnitude of the weights indefinitely to maximize the likelihood.
- The decision boundary becomes too sharp, leading to overconfident and incorrect predictions on test data.
- **Symptoms**:
  - Training log-likelihood approaches zero (perfect fit).
  - Test log-likelihood decreases (poor generalization).

## Example 2: Overfitting in Non-linear Binary Logistic Classification

- Using a large number of narrow basis functions (small length-scale) can lead to overfitting.
- The model fits the training data perfectly but fails to generalize.
- **Diagnosing Overfitting**:
  - Plot training and test log-likelihoods against model complexity.
  - Overfitting is indicated by increasing training log-likelihood and decreasing test log-likelihood.

## Consequences of Overfitting

- The model makes overconfident predictions on training data.
- Predictions on test data are poor, with low likelihoods.
- The model fails to capture the underlying distribution.

## Mitigating Overfitting

- **Regularization**: Introduce penalty terms in the loss function to discourage large weights.
- **Cross-Validation**: Use validation sets to monitor performance and select model parameters.
- **Early Stopping**: Halt training when performance on validation data starts to degrade.

## Summary

- Overfitting is a critical issue in classification models, especially when using maximum likelihood estimation without regularization.
- Understanding and diagnosing overfitting helps in developing models that generalize well to new data.

---
# k-Nearest Neighbours (kNN) Classification Algorithm

The **k-nearest neighbours (kNN)** algorithm is a simple, non-parametric method used for classification and regression tasks. It classifies a new data point based on the majority class among its $k$ nearest neighbours in the feature space.

## Nearest Neighbour Classification

For a given unseen point $\mathbf{x}^\ast$, the **nearest neighbour** algorithm assigns it to the class of its closest point in the training set $\mathbf{x}_n$, where the closest point is determined by a distance metric:

$$
i = \arg\min_{n} \, d(\mathbf{x}^\ast, \mathbf{x}_n)
$$

### Distance Metrics

Choosing an appropriate distance metric is crucial, especially in high-dimensional spaces or when features have different units. Common distance metrics include:

1. **Manhattan Distance ($L1$ norm)**:

   $$
   d(\mathbf{x}_1, \mathbf{x}_2) = \sum_{d=1}^D | x_{1,d} - x_{2,d} |
   $$

2. **Euclidean Distance ($L2$ norm)**:

   $$
   d(\mathbf{x}_1, \mathbf{x}_2) = \left( \sum_{d=1}^D | x_{1,d} - x_{2,d} |^2 \right)^{1/2}
   $$

3. **Minkowski Distance ($L_p$ norm)**:

   $$ 
   d(\mathbf{x}_1, \mathbf{x}_2) = \left( \sum_{d=1}^D | x_{1,d} - x_{2,d} |^p \right)^{1/p}
   $$

## k-Nearest Neighbours Algorithm

The kNN algorithm generalizes the nearest neighbour approach by considering the $k$ closest points to $\mathbf{x}^\ast$ and assigning the class by majority vote:

1. **Find the $k$ nearest neighbours** of $\mathbf{x}^\ast$ in the training set using a chosen distance metric.

2. **Assign the class** to $\mathbf{x}^\ast$ based on the most frequent class among its $k$ nearest neighbours.

In case of a tie, the class can be chosen randomly among those with the highest frequency.

### Effect of $k$ on Decision Boundaries

- **Small $k$** (e.g., $k=1$): Decision boundaries can be irregular and sensitive to noise, potentially leading to overfitting.
- **Large $k$**: Decision boundaries become smoother, which may lead to underfitting and loss of important local patterns.

## Choosing the Optimal Value of $k$

To select the optimal $k$, we can use **cross-validation**:

1. **Validation Set Approach**:

   - Split the training data into a smaller training set and a validation set.
   - Train the model on the smaller training set and evaluate its performance on the validation set for different values of $k$.
   - Choose the $k$ that yields the best performance on the validation set.

2. **n-Fold Cross-Validation**:

   - Divide the training data into $n$ equal-sized folds.
   - For each fold:
     - Use the fold as the validation set and the remaining data as the training set.
     - Evaluate the model's performance.
   - Compute the average performance across all folds for each $k$.
   - Select the $k$ with the highest average performance.

## Limitations of the kNN Algorithm

1. **Computational Complexity**:

   - Classification requires comparing the test instance to all training data, which is computationally expensive for large datasets.

2. **Storage Requirements**:

   - The entire training dataset must be stored in memory.

3. **Curse of Dimensionality**:

   - In high-dimensional spaces, distances between points become less meaningful, affecting the algorithm's performance.

4. **No Probabilistic Outputs**:

   - kNN provides hard class assignments without probabilities, making it difficult to gauge uncertainty.

5. **Feature Scaling Sensitivity**:

   - Distance metrics are sensitive to the scale of features; therefore, feature scaling or normalization is necessary.

## Comparison with Logistic Classification

- **Logistic Regression**:

  - A parametric model that provides probabilistic outputs.
  - Captures the relationship between features and the log-odds of the outcome.
  - Computationally efficient during prediction since it requires evaluating a function with learned parameters.

- **kNN Classification**:

  - A non-parametric, instance-based method.
  - Makes no assumptions about the underlying data distribution.
  - Computationally intensive during prediction due to distance calculations.

---
# Bayesian Logistic Regression and Laplace Approximation

Logistic regression models the probability of a binary outcome using the logistic function. In a Bayesian framework, we treat the model parameters $\mathbf{w}$ as random variables with a prior distribution and update this belief using observed data.

## Bayesian Logistic Regression

Given data $\mathcal{D} = \{ (\mathbf{x}_n, y_n) \}_{n=1}^N$, where $y_n \in \{0, 1\}$, the Bayesian approach involves:

1. **Likelihood Function**:

   $$
   p(\mathcal{D} \mid \mathbf{w}) = \prod_{n=1}^N p(y_n \mid \mathbf{x}_n, \mathbf{w}) = \prod_{n=1}^N \sigma(\mathbf{w}^\top \mathbf{x}_n)^{y_n} \left[ 1 - \sigma(\mathbf{w}^\top \mathbf{x}_n) \right]^{1 - y_n}
   $$

   where $\sigma(z) = 1 / (1 + e^{-z})$ is the logistic sigmoid function.

2. **Prior Distribution**:

   - Assume a Gaussian prior over $\mathbf{w}$:

     $$
     p(\mathbf{w}) = \mathcal{N}(\mathbf{w}; \mathbf{m}_0, \boldsymbol{\Sigma}_0)
     $$

3. **Posterior Distribution**:

   - Using Bayes' theorem:

     $$
     p(\mathbf{w} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \mathbf{w}) p(\mathbf{w})}{p(\mathcal{D})}
     $$

   - The posterior does not have a closed-form solution due to the non-conjugate likelihood.

## Laplace Approximation

To approximate the posterior, we use the **Laplace approximation**, which approximates a probability distribution with a Gaussian centered at the mode.

### Approximation Steps

1. **Find the Mode**:

   - Compute the maximum a posteriori (MAP) estimate $\mathbf{w}_{MAP}$ by maximizing $\log p(\mathbf{w} \mid \mathcal{D})$.

     $$
     \log p(\mathbf{w} \mid \mathcal{D}) = \log p(\mathcal{D} \mid \mathbf{w}) + \log p(\mathbf{w}) + \text{const}
     $$

   - The gradient is:

     $$
     \nabla \log p(\mathbf{w} \mid \mathcal{D}) = -\boldsymbol{\Sigma}_0^{-1} (\mathbf{w} - \mathbf{m}_0) + \sum_{n=1}^N \left[ y_n - \sigma(\mathbf{w}^\top \mathbf{x}_n) \right] \mathbf{x}_n
     $$

2. **Compute the Hessian**:

   - The negative Hessian at $\mathbf{w}_{MAP}$ is:

     $$
     \mathbf{H} = -\nabla \nabla \log p(\mathbf{w} \mid \mathcal{D}) = \boldsymbol{\Sigma}_0^{-1} + \sum_{n=1}^N \sigma(\mathbf{w}_{MAP}^\top \mathbf{x}_n) \left[ 1 - \sigma(\mathbf{w}_{MAP}^\top \mathbf{x}_n) \right] \mathbf{x}_n \mathbf{x}_n^\top
     $$

3. **Form the Gaussian Approximation**:

   - The posterior is approximated by:

     $$
     p(\mathbf{w} \mid \mathcal{D}) \approx \mathcal{N}(\mathbf{w}; \mathbf{w}_{MAP}, \boldsymbol{\Sigma})
     $$

     where $\boldsymbol{\Sigma} = \mathbf{H}^{-1}$.

## Predictive Distribution

We want to compute the predictive probability for a new input $\mathbf{x}^\ast$:

$$
p(y^\ast = 1 \mid \mathbf{x}^\ast, \mathcal{D}) = \int \sigma(\mathbf{w}^\top \mathbf{x}^\ast) p(\mathbf{w} \mid \mathcal{D}) d\mathbf{w}
$$

### Approximating the Predictive Distribution

1. **Approximate Integral with Laplace Posterior**:

   $$
   p(y^\ast = 1 \mid \mathbf{x}^\ast, \mathcal{D}) \approx \int \sigma(\mathbf{w}^\top \mathbf{x}^\ast) \mathcal{N}(\mathbf{w}; \mathbf{w}_{MAP}, \boldsymbol{\Sigma}) d\mathbf{w}
   $$

2. **Introduce the Variable $a$**:

   - Let $a = \mathbf{w}^\top \mathbf{x}^\ast$, then $a$ is normally distributed:

     $$
     p(a) = \mathcal{N}(a; \mu_a, \sigma_a^2)
     $$

     where:

     - Mean:

       $$
       \mu_a = \mathbf{w}_{MAP}^\top \mathbf{x}^\ast
       $$

     - Variance:

       $$
       \sigma_a^2 = \mathbf{x}^{\ast\top} \boldsymbol{\Sigma} \mathbf{x}^\ast
       $$

3. **Compute the Integral over $a$**:

   $$
   p(y^\ast = 1 \mid \mathbf{x}^\ast, \mathcal{D}) \approx \int \sigma(a) \mathcal{N}(a; \mu_a, \sigma_a^2) da
   $$

4. **Approximate the Sigmoid Function**:

   - Approximate the sigmoid with a probit function:

     $$
     \sigma(a) \approx \Phi(\kappa a)
     $$

     where:

     - $\Phi(\cdot)$ is the cumulative distribution function (CDF) of the standard normal distribution.
     - $\kappa \approx 0.√(π/8)$ is a scaling factor chosen to match the slope of $\sigma(a)$ at $a=0$.

5. **Evaluate the Integral**:

   $$
   \int \Phi(\kappa a) \mathcal{N}(a; \mu_a, \sigma_a^2) da = \Phi\left( \frac{\mu_a}{\sqrt{\sigma_a^2 + \kappa^{-2}}} \right)
   $$

   Therefore:

   $$
   p(y^\ast = 1 \mid \mathbf{x}^\ast, \mathcal{D}) \approx \Phi\left( \frac{\mu_a}{\sqrt{\sigma_a^2 + \kappa^{-2}}} \right)
   $$

## Interpretation

- The predictive distribution accounts for uncertainty in the parameters $\mathbf{w}$ due to limited data.
- The variance $\sigma_a^2$ captures how uncertainty in $\mathbf{w}$ affects the prediction at $\mathbf{x}^\ast$.
- As the amount of data increases, $\boldsymbol{\Sigma}$ shrinks, reducing $\sigma_a^2$ and making predictions more certain.

## Summary

- **Bayesian Logistic Regression** incorporates prior beliefs and provides a probabilistic framework for classification.
- **Laplace Approximation** allows us to approximate the posterior distribution when a closed-form solution is intractable.
- **Predictive Distribution** can be approximated by integrating over the approximate posterior, providing estimates that account for model uncertainty.

---

## Conclusion

Classification is a vital area of machine learning with applications across numerous fields. Understanding different classification models, their assumptions, and limitations is essential for building robust predictive models. Techniques like logistic regression, softmax regression, and kNN provide foundational methods, while Bayesian approaches offer ways to incorporate prior knowledge and quantify uncertainty. Recognizing and addressing overfitting ensures that models generalize well to unseen data.



# Questions

### 1. Why the Name 'Softmax' and Relation to Logistic Classification

**Question**: What happens to the softmax function as the magnitude of the weights tends to infinity? For $K=2$ classes, how does the softmax classification model compare to binary logistic classification? Is the softmax function identifiable?

**Answer**:

- The softmax function approaches a **hard max** function. The class with the highest activation $a_{n,k}$ will have a probability tending to 1, while all other classes will have probabilities tending to 0. This is because the exponential function amplifies differences in the activations, and the normalization ensures that the highest value dominates.
- For $K=2$ classes, the softmax function reduces to the logistic (sigmoid) function:

  $$
  p(y_n = 1 \mid \mathbf{x}_n) = \frac{\exp(\mathbf{w}_1^\top \mathbf{x}_n)}{\exp(\mathbf{w}_1^\top \mathbf{x}_n) + \exp(\mathbf{w}_2^\top \mathbf{x}_n)}
  $$

  By setting $\mathbf{w} = \mathbf{w}_1 - \mathbf{w}_2$, we recover the logistic regression model.

- The softmax function is **not identifiable** because adding the same vector $\mathbf{b}$ to all weight vectors does not change the probabilities:

  $$
  \frac{\exp(\mathbf{w}_k^\top \mathbf{x}_n + \mathbf{b}^\top \mathbf{x}_n)}{\sum_{k'=1}^K \exp(\mathbf{w}_{k'}^\top \mathbf{x}_n + \mathbf{b}^\top \mathbf{x}_n)} = \frac{\exp(\mathbf{w}_k^\top \mathbf{x}_n)}{\sum_{k'=1}^K \exp(\mathbf{w}_{k'}^\top \mathbf{x}_n)}
  $$

  Therefore, the model parameters are not uniquely determined.

### 2. Making Multi-Class Classifiers from Binary Classifiers

**Question**: How can a set of binary classifiers be used to solve a multi-class problem? Compare and contrast these approaches to softmax classification.

**Answer**:

- **One-vs-Rest (OvR)**:
  - Train $K$ binary classifiers, each distinguishing one class from all others.
  - For prediction, the classifier with the highest confidence score determines the class.
  - *Comparison*: OvR may not produce well-calibrated probabilities, and the classifiers are trained independently.

- **One-vs-One (OvO)**:
  - Train a binary classifier for each pair of classes, resulting in $K(K-1)/2$ classifiers.
  - For prediction, use a voting scheme among classifiers.
  - *Comparison*: OvO can be computationally intensive and may suffer from inconsistencies in voting.

- **Hierarchical Classification**:
  - Build a tree structure where at each node, a binary classifier splits the data.
  - *Comparison*: The tree structure can introduce biases based on the hierarchy chosen.

- **Softmax Classification**:
  - Models all classes simultaneously with a unified framework.
  - Ensures that probabilities across classes sum to one.
  - Parameters are learned jointly, capturing relationships between classes.

*Conclusion*: While binary decomposition methods can be useful, softmax classification provides a more coherent probabilistic model for multi-class problems.

### 3. Extrapolation in Radial Basis Function Logistic Classification

**Question**:

a) Estimate the prediction $p(y_n^\ast = 1 \mid \mathbf{w}, \mathbf{x}_n^\ast)$ at a test point $\mathbf{x}_n^\ast$ located five length-scales away from the nearest training point.

b) Would this estimate change if basis functions were placed in this region?

c) What implications does this have for the ability of this model to generalize away from the training data?

**Answer**:

a) Since radial basis functions (RBFs) decay exponentially with distance, at five length-scales away, the activation from the nearest basis function is negligible:

$$
\phi_d(\mathbf{x}_n^\ast) = \exp\left( -\frac{\|\mathbf{x}_n^\ast - \boldsymbol{\mu}_d\|^2}{2l^2} \right) \approx \exp(-12.5) \approx 3 \times 10^{-6}
$$

Therefore, the model's output defaults to the bias term, resulting in a prediction close to $p(y_n^\ast = 1) = 0.5$ (assuming zero bias).

b) Placing basis functions in this region would not change the prediction unless they are associated with training data. Since the weights for these basis functions are learned from the data, and there is no data in this region, their weights remain at initial values (typically zero).

c) The model cannot generalize to regions of the input space not covered by training data. Predictions in these regions are unreliable, and the model effectively outputs an uninformed prior. This highlights a limitation of models relying on localized basis functions without global components.
















