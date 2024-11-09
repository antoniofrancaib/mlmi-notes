## Introduction

Deep learning has revolutionized fields like computer vision, natural language processing, and bioinformatics by enabling machines to learn complex patterns from large datasets. Deep learning involves training artificial neural networks with multiple layers to approximate complex functions mapping inputs to outputs.

In this masterclass, we will explore foundational concepts of deep learning, starting with basic regression and advancing to optimization techniques used in training deep neural networks. Each concept will be explained in detail to ensure a deep understanding and strong intuition around these ideas.

---

## Table of Contents

1. Fundamentals of Regression
2. Neural Networks and Activation Functions
3. Deep Neural Networks
4. Loss Functions
    - 4.1 Regression Loss Functions
    - 4.2 Classification Loss Functions
5. Gradient Descent Optimization
6. Computing Gradients in Neural Networks
7. Backpropagation and Automatic Differentiation
8. Advanced Optimization Techniques
    - 8.1 Stochastic Gradient Descent (SGD)
    - 8.2 Momentum
    - 8.3 Adaptive Moment Estimation (Adam)
9. Practical Considerations in Training
10. Summary

---

## 1. Fundamentals of Regression

### 1.1 Problem Setup

Regression involves modeling the relationship between input variables (features) and continuous output variables (targets). Given a dataset:

$$
D = \{ (x^i, y^i) \}_{i=1}^n
$$

where:

- $x^i \in \mathbb{R}^d$: Input feature vector for the $i^{\text{th}}$ sample.
- $y^i \in \mathbb{R}$: Corresponding continuous target value.

Our goal is to find a function $f(x; \theta)$ parameterized by $\theta$ that can predict $y$ given $x$.

### 1.2 Loss Function

To quantify the error between predictions and actual targets, we define a loss function $L(y, \hat{y})$. For regression, a common choice is the mean squared error (MSE) loss:

$$
L(y, \hat{y}) = \| y - \hat{y} \|^2 = (y - \hat{y})^2
$$

This penalizes large deviations between the predicted value $\hat{y}$ and the true value $y$.

### 1.3 Objective Function

The objective is to find parameters $\theta$ that minimize the total loss over the dataset:

$$
L(\theta; D) = \sum_{i=1}^n L(f(x^i; \theta), y^i)
$$

We seek to solve:

$$
\theta^* = \arg \min_{\theta} L(\theta; D)
$$

---

## 2. Neural Networks and Activation Functions

### 2.1 Neural Networks as Function Approximators

A neural network is a computational model of interconnected nodes (neurons) organized in layers. Each neuron performs a linear transformation followed by a nonlinear activation function.

For example, a neural network function can be defined as:

$$
f(x; \theta) = \phi(Ax + b)
$$

where:

- $A$: Weight matrix.
- $b$: Bias vector.
- $\phi$: Activation function.

### 2.2 Activation Functions

Activation functions introduce nonlinearity into neural networks, allowing them to learn complex patterns. Common activation functions include:

#### 2.2.1 Rectified Linear Unit (ReLU)

$$
\phi(t) = \max(0, t)
$$

Properties:

- Simple and efficient to compute.
- Helps mitigate the vanishing gradient problem.
- Introduces sparsity by zeroing out negative inputs.

#### 2.2.2 Sigmoid Function

$$
\phi(t) = \frac{1}{1 + \exp(-t)}
$$

Properties:

- Outputs values between 0 and 1.
- Historically used in early neural networks.
- Prone to vanishing gradients for large inputs.

#### 2.2.3 Hyperbolic Tangent (Tanh)

$$
\phi(t) = \tanh(t) = \frac{\exp(t) - \exp(-t)}{\exp(t) + \exp(-t)}
$$

Properties:

- Outputs values between -1 and 1.
- Zero-centered, which can aid optimization.

---

## 3. Deep Neural Networks

### 3.1 Network Architecture

A deep neural network consists of multiple layers of neurons, enabling the modeling of hierarchical representations. A deep network can be defined recursively:

$$
\begin{align*}
h_1 &= \phi(A_1 x + b_1) \\
h_2 &= \phi(A_2 h_1 + b_2) \\
h_3 &= \phi(A_3 h_2 + b_3) \\
&\vdots \\
y &= \phi'(A_L h_{L-1} + b_L)
\end{align*}
$$

where:

- $L$: Number of layers.
- $A_l, b_l$: Weights and biases for layer $l$.
- $\phi, \phi'$: Activation functions.

---

## 4. Loss Functions

### 4.1 Regression Loss Functions

For regression tasks with continuous outputs, we commonly use the least squares loss:

$$
L(y, \hat{y}) = \| \hat{y} - y \|^2 = (\hat{y} - y)^2
$$

This penalizes the squared differences between predicted and true values.

### 4.2 Classification Loss Functions

In classification tasks, the outputs are discrete labels. The loss function needs to handle categorical outputs.

#### 4.2.1 Softmax Function

To convert raw outputs into probabilities, we use the softmax function:

$$
z_k = \frac{\exp(y_k)}{\sum_{k'} \exp(y_{k'})}
$$

where:

- $y_k$: Raw output (logit) for class $k$.
- $z_k$: Probability of class $k$.

#### 4.2.2 Cross-Entropy Loss

The cross-entropy loss measures the dissimilarity between predicted probabilities and actual distribution (one-hot encoded labels):

$$
L(y, \hat{y} = l) = -\log(z_l) = -y_l + \log\left(\sum_{k'} \exp(y_{k'})\right)
$$

where $l$ is the true class label.

---

## 5. Gradient Descent Optimization

### 5.1 Optimization Objective

We aim to minimize the loss over the dataset:

$$
L(\theta; D) = \sum_{i=1}^n L(f(x^i; \theta), y^i)
$$

### 5.2 Gradient Descent Algorithm

1. **Initialization**: Start with an initial guess $\theta_0$.
2. **Iterative Update**:

   - Compute the Gradient:

     $$
     g = \frac{\partial L}{\partial \theta} (\theta_i; D)
     $$

   - Update Parameters:

     $$
     \theta_{i+1} = \theta_i - \lambda g
     $$

3. **Repeat** until convergence or for a predefined number of iterations.

---

## 6. Computing Gradients in Neural Networks

### 6.1 Gradient Computation

To perform gradient descent, we need to compute $\frac{\partial L}{\partial \theta}$. For neural networks, this involves computing gradients with respect to weights and biases across all layers.

### 6.2 Chain Rule and Backpropagation

The chain rule allows us to compute derivatives of composite functions:

$$
\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial \theta}
$$

---

## 7. Backpropagation and Automatic Differentiation

### 7.1 Backpropagation Algorithm

Backpropagation efficiently computes gradients in neural networks by propagating errors backward from the output layer to the input layer.

1. **Forward Pass**: Compute outputs $y$ and store intermediate activations.
2. **Backward Pass**: Compute the gradient of the loss with respect to output and recursively compute gradients for each layer.

### 7.2 Automatic Differentiation

Libraries like TensorFlow, PyTorch, and JAX provide automatic differentiation, automatically computing gradients given a computational graph.

---

## 8. Advanced Optimization Techniques

### 8.1 Stochastic Gradient Descent (SGD)

SGD updates parameters using a subset (mini-batch) of the data. At each iteration:

$$
\theta_{i+1} = \theta_i - \lambda \cdot \frac{1}{|B|} \sum_{i \in B} \frac{\partial L_i}{\partial \theta}
$$

### 8.2 Momentum

Momentum accumulates past gradients to smooth updates, defined by:

$$
v = \beta v + g \quad \text{and} \quad \theta_{i+1} = \theta_i - \lambda v
$$

### 8.3 Adaptive Moment Estimation (Adam)

Adam combines benefits of RMSProp and momentum:

$$
\theta_{t+1} = \theta_t - \lambda \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

---

## 9. Practical Considerations in Training

- **Initialization**: He and Xavier initializations.
- **Regularization**: L2 Regularization and Dropout.
- **Normalization**: Batch Normalization.
- **Learning Rate Scheduling**: Step decay, exponential decay, cosine annealing.

---

## 10. Summary

- **Deep Learning Foundations**: Neurons, activation functions, and loss functions.
- **Optimization Techniques**: Mastery of gradient descent, SGD, Momentum, Adam.
- **Gradient Computation**: Backpropagation for efficient gradient computation.
- **Training Strategies**: Proper initialization, regularization, learning rate strategies.
- **Automatic Differentiation**: Leveraging frameworks like TensorFlow and PyTorch.

Understanding these concepts and techniques is essential for designing, training, and optimizing deep neural networks across complex tasks. Continuous learning and experimentation are key in the evolving field of deep learning.
