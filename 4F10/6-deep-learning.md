## Introduction to Deep Learning

Deep learning is a subset of machine learning that focuses on neural networks with multiple layers (depth). It has revolutionized fields such as computer vision, natural language processing, and speech recognition. At its core, deep learning models are designed to learn hierarchical representations of data through compositions of nonlinear transformations.

This lecture aims to provide a deep understanding of the foundational concepts in deep learning, starting from basic regression and progressing to the architecture and functioning of deep neural networks. We will explore the mathematical formulations, intuitive explanations, and graphical representations to build strong intuitions around complex ideas.

---

### 1. Fundamentals of Regression

#### 1.1 Problem Setup

In regression tasks, the goal is to model the relationship between input variables (features) and continuous output variables (targets). Given a set of training data $D = \{ (x^i, y^i) \}_{i=1}^n$, where $x^i$ represents the input data and $y^i$ represents the corresponding output, we aim to find a function $f(x; \theta)$ parameterized by $\theta$ that can predict $y$ given $x$.

#### 1.2 Loss Function

To measure the discrepancy between the predicted outputs and the actual outputs, we define a loss function $L(y, \hat{y})$. A common choice for regression is the squared error loss:

$$
L(y, \hat{y}) = (y - \hat{y})^2
$$

This loss quantifies the squared difference between the predicted value $\hat{y}$ and the true value $y$.

#### 1.3 Objective Function

The objective is to find the parameters $\theta$ that minimize the total loss over the training dataset:

$$
L(\theta; D) = \sum_{i=1}^n L(f(x^i; \theta), y^i)
$$

Our aim is to solve the optimization problem:

$$
\theta^* = \arg \min_{\theta} L(\theta; D)
$$

---

### 2. Choosing the Function Family $f(x; \theta)$

#### 2.1 Linear Models

A straightforward choice is a linear function:

$$
f(x; \theta) = \theta_0 + \theta_1 x
$$

This represents a straight line in 1D regression. While simple, linear models are limited in their capacity to capture complex patterns in the data.

#### 2.2 Polynomial Models

To capture nonlinear relationships, we can extend the model to include polynomial terms:

$$
f(x; \theta) = \theta_0 + \theta_1 x + \theta_2 x^2 + \dots + \theta_k x^k
$$

By increasing the degree $k$, the model becomes capable of fitting more complex curves.

#### 2.3 Basis Function Models

Instead of using monomials $x^k$, we can use a set of basis functions $\{\phi_k(x)\}$:

$$
f(x; \theta) = \sum_k \theta_k \phi_k(x)
$$

Examples of basis functions:

- **Polynomial Basis Functions**: $\phi_k(x) = x^k$
- **Sinusoidal Basis Functions**: $\phi_k(x) = \sin(kx)$
- **Gaussian Basis Functions**: $\phi_k(x) = \exp \left( -\frac{(x - \mu_k)^2}{2\sigma_k^2} \right)$

---

### 3. Translated Basis Functions

#### 3.1 Concept

Translated basis functions involve shifting a basic function $\phi$ by different amounts along the input axis:

$$
f(x; \theta) = \sum_k \theta_k \phi(x - b_k)
$$

Here, $b_k$ represents the translation (shift) of the basis function $\phi$.

#### 3.2 Example Functions

- **Gaussian Functions**: $\phi(t) = \exp(-t^2)$
- **Sigmoid Functions**: $\phi(t) = \frac{1}{1 + \exp(-t)}$
- **Rectified Linear Unit (ReLU)**: $\phi(t) = \max(0, t)$

---

### 4. Activation Functions

#### 4.1 Importance

Activation functions introduce nonlinearity into neural networks, enabling them to model complex patterns.

#### 4.2 Common Activation Functions

- **Gaussian Activation**: Smooth and localized, useful for radial basis function networks.
- **Sigmoid Activation**: $\phi(t) = \frac{1}{1 + \exp(-t)}$, smooth and bounded between 0 and 1.
- **ReLU Activation**: $\phi(t) = \max(0, t)$, introduces sparsity and mitigates the vanishing gradient problem.

---

### 5. Neural Networks as Function Approximators

#### 5.1 Graphical Representation

Neural networks can be visualized as computational graphs where inputs are transformed through layers to produce outputs.

#### 5.2 Mathematical Formulation

For a single hidden layer network:

1. **Linear Transformation**: $h_k = a_k x + b_k$
2. **Activation Function**: $h_k = \phi(h_k)$
3. **Output Layer**: $y = \sum_k c_k h_k + c_0$

---

### 6. Universal Approximation Theorem

A feedforward neural network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of $\mathbb{R}^n$, under mild assumptions on the activation function.

---

### 7. Piecewise Linear Functions and ReLU Networks

#### 7.1 ReLU Networks

Using ReLU activation functions, neural networks partition the input space into regions where the function is linear.

---

### 8. Network Capacity and Complexity

#### 8.1 Parameters and Regions

- **Number of Parameters**: Determined by the number of neurons and layers.
- **Number of Regions**: Complexity of the function increases with the number of regions defined by activation patterns.

#### 8.2 Zaslavsky's Theorem

The maximum number of regions $R$ formed by $H$ hyperplanes in $D$ dimensions is:

$$
R = \sum_{i=0}^D \binom{H}{i}
$$

---

### 9. Network Nomenclature

#### 9.1 Terminology

- **Weights**: Parameters that scale inputs or activations.
- **Biases**: Parameters that shift the activation function.
- **Fully Connected Layer**: Every neuron in one layer is connected to every neuron in the next layer.
- **Depth**: Number of layers in the network.
- **Width**: Number of neurons in a layer.

---

### 10. Deep Neural Networks

#### 10.1 Composing Functions

Deep learning leverages the composition of multiple layers to learn hierarchical representations:

$$
f(x; \theta) = f^{(L)}(f^{(L-1)}(\dots f^{(1)}(x)))
$$

---

### 11. Composition of Linear Functions

Composing linear functions results in another linear function. Nonlinear activation functions are necessary to increase the expressive power of the network.

---

### 12. Networks as Composing Functions

A deep neural network can be represented as nested functions:

$$
f(x; \theta) = f^{(L)}(f^{(L-1)}(\dots f^{(1)}(x)))
$$

---

### 13. Combining Networks

#### 13.1 Composing Two Networks

Given two networks:

- **Network 1**: Maps input $x$ to hidden representation $h$.
- **Network 2**: Maps $h$ to output $y$.

Composing them results in a deeper network.

---

### 14. Hyperparameters

#### 14.1 Definition

Hyperparameters are settings that define the network architecture and training process, such as depth, width, activation functions, and learning rate.

#### 14.2 Selection and Optimization

Hyperparameter tuning involves selecting the best hyperparameters, often through grid search or random search, using a validation set for evaluation.

---

### 15. Shallow vs. Deep Networks

#### 15.1 Capacity and Efficiency

- **Shallow Networks**: Can approximate any function given enough hidden units, but may require many units for complex functions.
- **Deep Networks**: Can represent complex functions more efficiently due to hierarchical feature extraction.

---

### 16. Summary

- **Regression**: Modeling relationships between inputs and continuous outputs.
- **Loss Function**: Quantifies discrepancy between predictions and true values.
- **Function Families**: Choice of $f(x; \theta)$ affects model capacity.
- **Activation Functions**: Introduce nonlinearity, critical for network expressiveness.
- **Neural Networks**: Structured as layers applying transformations to inputs.
- **Universal Approximation**: Networks can approximate any function under certain conditions.
- **Hyperparameters**: Define the architecture and are tuned for optimal performance.
- **Shallow vs. Deep**: Depth enables efficient representation of complex functions.

---

### 17. Conclusion

Understanding the foundational concepts of deep learning is crucial for advancing in the field. By dissecting the components of neural networks, exploring the mathematical underpinnings, and visualizing their operations, we develop strong intuitions about how these models learn from data. This knowledge serves as a solid foundation for further exploration into advanced topics such as optimization algorithms, regularization techniques, and specialized architectures.

---

This comprehensive overview provides insights into the structure and functioning of neural networks, forming the basis for a masterclass in deep learning.
