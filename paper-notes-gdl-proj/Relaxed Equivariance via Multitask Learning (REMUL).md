## Introduction

Equivariant machine learning models incorporate symmetries of data into the architecture design as an inductive bias. This approach has achieved notable success in various domains such as computer vision, dynamical systems, chemistry, and structural biology. However, enforcing strict equivariance often leads to highly constrained architectures and increased computational complexity.

For example, modeling molecular structures requires handling symmetries under rotations and translations. Equivariant models ensure that the output transforms appropriately when the input is rotated or translated. Despite their benefits, these models can be computationally expensive and may not generalize well to tasks that do not exhibit full equivariance.

## Equivariance in Machine Learning

### Symmetry Groups and Equivariant Functions

A **symmetry group** $G$ of a set $X$ is a group of bijective functions from $X$ to itself, where the group operation is function composition. In machine learning, we often deal with functions that are equivariant with respect to such groups.

**Definition:** A function $f: X \rightarrow Y$ is **equivariant** with respect to a group $G$ if, for any transformation $g \in G$ and any input $x \in X$,

$$
f\left( \phi(g)(x) \right) = \rho(g) \left( f(x) \right),
$$

where $\phi$ and $\rho$ are representations of $G$ acting on $X$ and $Y$, respectively.

A special case is **invariance**, where $\rho(g)$ is the identity transformation, so the output remains unchanged under the group action on the input.

### Equivariance as a Constrained Optimization Problem

Training an equivariant model involves solving the following constrained optimization problem:

$$
\begin{align*}
\text{Minimize over } \theta:\quad & \mathbb{E}_{(x, y) \sim q} \left[ L\left( f_\theta(x), y \right) \right] \\
\text{Subject to:}\quad & f_\theta \left( \phi(g)(x) \right) = \rho(g) \left( f_\theta(x) \right), \quad \forall g \in G, \forall x \in X,
\end{align*}
$$

where $L$ is a loss function quantifying the discrepancy between predictions and true labels, and $f_\theta$ is parameterized by $\theta$.

This constraint ensures that the model's output transforms appropriately under the group action, but it can lead to complex and computationally intensive architectures.

## Relaxed Equivariance via Multitask Learning (REMUL)

### Reformulating Equivariance as a Learning Objective

To address the challenges of strict equivariance constraints, REMUL proposes treating equivariance as an additional loss term in a multitask learning framework. The constrained optimization is relaxed to an unconstrained one by introducing an **equivariance loss**.

The new optimization problem becomes:

$$
\text{Minimize over } \theta:\quad \mathbb{E}_{(x, y) \sim q} \left[ \alpha L\left( f_\theta(x), y \right) + \beta F_{X, G}(f_\theta) \right],
$$

where:

- $L$ is the original loss function.
- $F_{X, G}(f_\theta)$ is a functional measuring the deviation from equivariance.
- $\alpha$ and $\beta$ are weighting coefficients controlling the trade-off between the primary task and the equivariance constraint.

### Equivariance Loss Function

For a finite dataset $\{ (x_i, y_i) \}_{i=1}^n$, the equivariance loss is defined as:

$$
L_{\text{equi}}(f_\theta, X, Y, G) = \sum_{i=1}^n \ell \left( f_\theta \left( \phi(g_i)(x_i) \right), \rho(g_i)(y_i) \right),
$$

where:

- $\ell$ is a metric function, such as the $L_1$ or $L_2$ norm.
- $g_i$ are randomly sampled group elements from $G$ for each data point.

The total loss function is then:

$$
L_{\text{total}}(f_\theta, X, Y, G) = \alpha L_{\text{obj}}(f_\theta, X, Y) + \beta L_{\text{equi}}(f_\theta, X, Y, G),
$$

where $L_{\text{obj}}$ is the original loss over the dataset.

### Controlling the Degree of Equivariance

By adjusting $\alpha$ and $\beta$, we can control the model's adherence to equivariance:

- A higher $\beta$ encourages the model to be more equivariant.
- A lower $\beta$ allows the model to prioritize the primary task over strict equivariance.

This flexibility is beneficial for tasks where full equivariance may not be necessary or may even hinder performance.

### Adaptive Weighting of Loss Components

To balance the loss components during training, REMUL can employ adaptive weighting strategies, such as **GradNorm**, which adjusts $\alpha$ and $\beta$ based on the relative magnitudes of the gradients of each loss term.

## Quantifying Learned Equivariance

To measure how closely a model approximates equivariance, we define the **equivariance error** using group averages.

### Equivariance Error Metrics

1. **Average Deviation of Group Averages**:

   $$
   E(f, G) = \frac{1}{|D|} \sum_{x \in D} \left\| \frac{1}{M} \sum_{i=1}^M \rho(g_i) \left( f(x) \right) - \frac{1}{M} \sum_{i=1}^M f \left( \phi(g_i)(x) \right) \right\|,
   $$

   where $D$ is the dataset, and $\{ g_i \}_{i=1}^M$ are samples from $G$.

2. **Average of Individual Deviations**:

   $$
   E'(f, G) = \frac{1}{|D|} \sum_{x \in D} \frac{1}{M} \sum_{i=1}^M \left\| f \left( \phi(g_i)(x) \right) - \rho(g_i) \left( f(x) \right) \right\|.
   $$

These metrics provide practical ways to evaluate the degree of equivariance learned by the model. A lower value indicates closer adherence to equivariance.

## Experiments and Results

### N-Body Dynamical System

The task involves predicting the positions of particles after a series of time steps, which is inherently equivariant under rotations and translations.

- **Models Compared**:

  - Standard Transformer trained with REMUL.
  - Equivariant models such as SE(3)-Transformer and Geometric Algebra Transformer (GATr).

- **Findings**:

  - Increasing $\beta$ in REMUL reduces the equivariance error.
  - The Transformer with high $\beta$ outperforms both data augmentation methods and some equivariant architectures.

### Motion Capture Data

Human motion data may not exhibit full rotational symmetry due to gravity and anatomical constraints.

- **Findings**:

  - Equivariant models performed worse than the standard Transformer.
  - An intermediate value of $\beta$ in REMUL yielded the best performance, suggesting that partial equivariance was optimal.

### Molecular Dynamics (MD17 Dataset)

Predicting molecular properties where some molecules exhibit symmetric structures.

- **Findings**:

  - REMUL-trained models outperformed equivariant models on most molecules.
  - The optimal $\beta$ varied across molecules, indicating different requirements for equivariance.

## Computational Efficiency

- REMUL allows the use of unconstrained models, which are computationally more efficient than their equivariant counterparts.
- Experiments showed up to a $10 \times$ speed-up in inference and $2.5 \times$ speed-up in training.

## Conclusion

REMUL provides a flexible approach to incorporating equivariance into machine learning models by treating it as an additional learning objective. This method allows for control over the degree of equivariance, enabling models to balance performance and symmetry constraints based on task requirements. The approach is both effective and computationally efficient, offering advantages over strictly equivariant architectures.

---

**References**

- Brehmer, J., De Haan, P., Behrends, S., & Cohen, T. (2023). Geometric Algebra Transformer.
- Satorras, V. G., Hoogeboom, E., & Welling, M. (2022). E(n) Equivariant Graph Neural Networks.
- Cohen, T. S., & Welling, M. (2016). Group Equivariant Convolutional Networks.

