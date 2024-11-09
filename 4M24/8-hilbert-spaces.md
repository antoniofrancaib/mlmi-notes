# Lecture 8: Hilbert Spaces  
**Computational Statistics & Machine Learning**

## Lecture Outline
- Subspace of Banach Space – Inner Product
- Pre-Hilbert Space
- Hilbert Space
- $l^2$ and $L^2$ Spaces
- Reproducing Kernel Hilbert Space

## Introduction
In this lecture, we delve into the foundational aspects of Hilbert spaces and their significance in machine learning, particularly in function approximation. We will explore the structure of Banach spaces, the concept of inner products, and how they lead us to the definition of Hilbert spaces. Understanding these spaces is crucial for grasping advanced topics like Reproducing Kernel Hilbert Spaces (RKHS), which play a pivotal role in modern machine learning algorithms.

---

### 1. Subspace of Banach Space – Inner Product

#### Banach Spaces in Infinite Dimensions
A Banach space is a *complete normed vector space*; that is, a vector space $V$ equipped with a norm $\| \cdot \|$ such that every Cauchy sequence in $V$ converges to a limit within $V$. In infinite-dimensional settings, completeness becomes a significant property because it ensures that sequences of vectors behave nicely under limits.

However, not all infinite-dimensional normed spaces are complete. Some may lack limit points within the space for certain Cauchy sequences, making them incomplete.

#### Inner Product Spaces and Their Importance
An inner product on a vector space $V$ is a function $\langle \cdot , \cdot \rangle : V \times V \to \mathbb{R}$ (or $\mathbb{C}$) satisfying:

1. **Linearity in the first argument**: $\langle a u + b v , w \rangle = a \langle u , w \rangle + b \langle v , w \rangle$
2. **Conjugate symmetry**: $\langle u , v \rangle = \overline{\langle v , u \rangle}$
3. **Positive-definiteness**: $\langle v , v \rangle \geq 0$ with equality if and only if $v = 0$

An inner product induces a norm via $\| v \| = \sqrt{\langle v , v \rangle}$.

---

### Why is this important in Machine Learning?
Inner product spaces allow for geometric interpretations of data and functions. They enable concepts like orthogonality, projections, and angles between vectors or functions, which are essential in algorithms like Principal Component Analysis (PCA) and in understanding the geometry of high-dimensional data.

#### Inner Products in $l^p$ and $L^p$ Spaces
The $l^p$ and $L^p$ spaces are sets of sequences or functions where the $p$-th power of the absolute value is integrable or summable. Specifically:

- $l^p$ consists of sequences $\{ a_n \}$ such that $\sum_n | a_n |^p < \infty$.
- $L^p$ consists of measurable functions $f$ such that $\int | f |^p \, dx < \infty$.

Only for $p=2$ do these spaces have an inner product that gives rise to the norm. For $p \neq 2$, the norm cannot be derived from an inner product. This is because the parallelogram law, which relates norms and inner products, holds only when $p=2$.

---

### 2. Pre-Hilbert Space

#### Definition
A pre-Hilbert space is an inner product space that is not complete. While it has an inner product structure, there exist Cauchy sequences that do not converge within the space.

#### Example of an Incomplete Inner Product Space
Consider a sequence of functions $\{ f_n \}$ defined on the interval $[0,1]$:

$$
f_n(x) = 
\begin{cases} 
1 & \text{if } 0 \leq x \leq \frac{1}{2} \\ 
1 - 2n \left( x - \frac{1}{2} \right) & \text{if } \frac{1}{2} \leq x \leq \frac{1}{2} - \frac{1}{2} \left( 1 + \frac{1}{n} \right) \\ 
0 & \text{if } \frac{1}{2} \left( 1 + \frac{1}{n} \right) \leq x \leq 1 
\end{cases}
$$

This sequence converges to a step function with a jump at $x = \frac{1}{2}$:

$$
f(x) = 
\begin{cases} 
1 & \text{if } 0 \leq x \leq \frac{1}{2} \\ 
0 & \text{if } \frac{1}{2} < x \leq 1 
\end{cases}
$$

**Why is this significant?**

The functions $f_n$ are continuous, but their limit $f$ is not. In the space of continuous functions on $[0,1]$, $\{ f_n \}$ is a Cauchy sequence that does not converge within the space, illustrating that the space is incomplete.

#### Demonstrating the Incompleteness
Calculating the $L^2$ norm between $f_n$ and $f_m$:

$$
\| f_n - f_m \| = \sqrt{\int_0^1 (f_n(x) - f_m(x))^2 \, dx} = \left(1 - \frac{n}{m}\right) \sqrt{\frac{1}{6n}} \to 0 \quad \text{as } m, n \to \infty
$$

This shows $\{ f_n \}$ is a Cauchy sequence in $L^2$, but its limit $f$ is not in the space of continuous functions, confirming the space's incompleteness.

---

### 3. Hilbert Space

### Infinite Dimensional Orthonormal Basis
In a Hilbert space $V$, consider an infinite set of orthonormal vectors $\{ e_i \}$. For any function $f \in V$:

- **Coefficients via Inner Product**: $c_i = \langle e_i, f \rangle$
- **Partial Sums**: $f_n = \sum_{i=1}^n c_i e_i$
- **Norm of Partial Sums**: $\| f_n \|^2 = \sum_{i=1}^n c_i^2$
- **Approximation of $f$**: $\| f - f_n \|^2 = \| f \|^2 - \| f_n \|^2 \to 0 \text{ as } n \to \infty$

#### Bessel's Inequality
For any $f \in V$:

$$
\sum_{i=1}^{\infty} c_i^2 \leq \| f \|^2
$$

This inequality assures that the series $\sum_{i=1}^{\infty} c_i e_i$ converges in $V$.

#### Parseval's Identity
If $\{ e_i \}$ is a complete orthonormal set, then:

$$
\| f \|^2 = \sum_{i=1}^{\infty} c_i^2
$$

This identity confirms that $\{ e_i \}$ forms a basis for $V$.

---

### 4. $l^2$ and $L^2$ Spaces

#### $L^2$ Space
Definition: The set of square-integrable functions over an interval $[a,b]$:

$$
L^2([a,b]) = \{ f : \int_a^b | f(x) |^2 \, dx < \infty \}
$$

Importance: Convergence in $L^2$ is essential for function approximation and signal processing.

**Convergence in $L^2$**

A sequence $\{ f_n \}$ converges to $f$ in $L^2$ if:

$$
\lim_{n \to \infty} \| f_n - f \|_{L^2} = 0
$$

This convergence is in the "mean," not necessarily pointwise.

**Almost Everywhere Convergence**

Convergence in $L^2$ does not guarantee pointwise convergence. However, for practical purposes, convergence "almost everywhere" (a.e.) is acceptable, meaning convergence holds except on a set of measure zero.

#### $l^2$ Space
Definition: The set of infinite sequences $\{ c_i \}$ such that:

$$
\sum_{i=1}^{\infty} | c_i |^2 < \infty
$$

**Relation to $L^2$**: The Fourier coefficients $c_i$ of functions in $L^2$ belong to $l^2$.

---

### 5. Reproducing Kernel Hilbert Space

#### Function Approximation in Hilbert Spaces
In machine learning, we often need to approximate functions based on data. Hilbert spaces provide the structure to do this efficiently, especially when dealing with infinite-dimensional spaces.

#### Reproducing Kernel Hilbert Space (RKHS)
An RKHS is a Hilbert space of functions where evaluation at a point is a continuous linear functional. This means:

If $f_n \to f$ in $H$, then $f_n(x) \to f(x)$ for all $x$.

**Reproducing Kernel**

A reproducing kernel $K: X \times X \to \mathbb{R}$ satisfies:

1. **Symmetry**: $K(x, y) = K(y, x)$.
2. **Positive Definiteness**: For any finite set $\{ x_i \}$ and scalars $\{ \alpha_i \}$:

   $$
   \sum_{i,j} \alpha_i \alpha_j K(x_i, x_j) \geq 0
   $$

**Reproducing Property**:

$$
f(x) = \langle f, K(\cdot, x) \rangle_H
$$

#### Moore-Aronszajn Theorem
This theorem states that for every positive-definite kernel $K$, there exists a unique RKHS $H$ such that $K$ is its reproducing kernel.

#### Function Approximation Using RKHS
Given data $\{ (x_i, y_i) \}$, we aim to find $f \in H$ minimizing:

$$
\frac{1}{N} \sum_{n=1}^N | f(x_n) - y_n |^2 + \lambda \| f \|_H^2
$$

**Key Result**:

The minimizer $\hat{f}$ is in the span of $\{ K(\cdot, x_n) \}$:

$$
\hat{f} = \sum_{n=1}^N \alpha_n K(\cdot, x_n)
$$

**Deriving the Coefficients $\alpha$**

To find $\alpha = [ \alpha_1, \dots, \alpha_N ]^T$:

- **Set Up the Linear System**:

  $$
  (K + N \lambda I) \alpha = y
  $$

  where $K$ is the kernel matrix with entries $K_{ij} = K(x_i, x_j)$, $I$ is the identity matrix, and $y$ is the vector of observed values.

- **Solve for $\alpha$**: Use linear algebra techniques to solve the system.

#### Interpretation of Regularization Term
The regularization term $\lambda \| f \|_H^2$ controls the smoothness or complexity of the solution $f$:

- **Smaller $\lambda$**: Allows for a more complex function that fits the data closely.
- **Larger $\lambda$**: Encourages smoother functions, potentially improving generalization.

In the context of $L^2$ norm regularization, we are penalizing the "energy" or "magnitude" of $f$, discouraging overly complex functions that may overfit the data.

---

## Conclusion
Understanding Hilbert spaces and their properties is fundamental in advanced machine learning. Hilbert spaces provide the mathematical framework for dealing with infinite-dimensional function spaces, which are common in kernel methods and other advanced algorithms.

Reproducing Kernel Hilbert Spaces, in particular, offer powerful tools for function approximation, enabling us to work in infinite-dimensional spaces while maintaining computational tractability through the kernel trick and the representer theorem.

By grasping these concepts, we equip ourselves with the theoretical foundation necessary for developing and analyzing sophisticated machine learning models that can effectively learn from data.

**Note**: The above exposition aims to provide a comprehensive understanding of the concepts based on the provided material, explaining nontrivial steps and offering intuitive insights into each topic.
