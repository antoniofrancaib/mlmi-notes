# Lecture 3: Lebesgue Integral and Measure

## Overview

In this lecture, we will delve into the foundations of integration theory, which is crucial for advanced topics in computational statistics and machine learning. We will explore:

- **Riemann Integral**: Formal definition, construction, and limitations.
- **Convergence of Functions**: Understanding different modes of convergence and their implications.
- **Lebesgue Integral**: Introduction to Lebesgue integration and its advantages over the Riemann integral.

Our goal is to develop a deep understanding of how integration works, why certain functions may not be integrable in the Riemann sense, and how the Lebesgue integral addresses these issues.

---

### 1. Riemann Integral

#### 1.1 Introduction

- **Integration**: A fundamental mathematical operation that appears repeatedly in machine learning, statistics, and various fields of engineering.
- **Bernhard Riemann**: Formalized the concept of integration in 1862, leading to the development of the Riemann Integral.

#### 1.2 Formal Definition and Construction

The Riemann integral is constructed by partitioning the domain of a function and summing up "slices" that approximate the area under the curve.

##### 1.2.1 Partitioning the Domain

- **Domain**: Consider a bounded function $f(x)$ defined on the interval $[a, b] \subset \mathbb{R}$.
- **Partition $P$**: Divide the interval $[a, b]$ into $n$ subintervals:

  $$
  a = x_1 < x_2 < x_3 < \dots < x_{n+1} = b
  $$

  - Each subinterval is $\Delta x_k = x_{k+1} - x_k$.

##### 1.2.2 Upper and Lower Sums

- **Supremum and Infimum**:

  - $M_k = \sup_{x \in \Delta x_k} f(x)$: The least upper bound (maximum) of $f(x)$ on $\Delta x_k$.
  - $m_k = \inf_{x \in \Delta x_k} f(x)$: The greatest lower bound (minimum) of $f(x)$ on $\Delta x_k$.

- **Upper Sum $S_P(f)$**:

  $$
  S_P(f) = \sum_{k=1}^n M_k \Delta x_k
  $$

- **Lower Sum $s_P(f)$**:

  $$
  s_P(f) = \sum_{k=1}^n m_k \Delta x_k
  $$

##### 1.2.3 Riemann Integral Definition

The Riemann Integral $I$ exists if the limit of the upper and lower sums converges to the same value as the partitions become finer (i.e., as $n \to \infty$ and $\max \Delta x_k \to 0$):

$$
\lim_{n \to \infty} S_P(f) = \lim_{n \to \infty} s_P(f) = I = \int_a^b f(x) \, dx
$$

- **Interpretation**: The area under the curve $f(x)$ over $[a, b]$ is approximated by summing the areas of rectangles defined by the partitions.

##### 1.2.4 Conditions for Integrability

- **Continuity**: The function $f(x)$ must be continuous almost everywhere on $[a, b]$ for the Riemann integral to exist.
- **Boundedness**: $f(x)$ must be bounded on $[a, b]$.

#### 1.3 Visual Representation

This figure illustrates how the upper and lower sums approximate the area under $f(x)$ as the number of partitions increases.

#### 1.4 Limitations of the Riemann Integral

Despite its foundational importance, the Riemann integral has several limitations, especially in advanced applications:

##### 1.4.1 Partitioning Complex Domains

- **Non-Euclidean Domains**: In machine learning, we often deal with domains that are not subsets of $\mathbb{R}^d$, such as:
  - Spaces of continuous functions (e.g., $C[0,1]$): Infinite-dimensional function spaces.
  - Manifolds or other abstract spaces.
- **Challenge**: Partitioning these domains into subintervals of equal size, as required by the Riemann integral, is not straightforward.

##### 1.4.2 Exchange of Limit Operations

- **Limit and Integration**: In many applications, we need to exchange the order of limits and integrals, such as:

  $$
  \lim_{n \to \infty} \int_{-\infty}^{\infty} f_n(x) \, dx = \int_{-\infty}^{\infty} \lim_{n \to \infty} f_n(x) \, dx
  $$

  - **Problem**: The Riemann integral allows this exchange only under stringent conditions (e.g., uniform convergence), which may not hold in practical scenarios.

##### 1.4.3 Example of Limit Exchange Failure

Consider the sequence of functions:

$$
f_n(x) = 2n^2 x e^{-n^2 x^2} \quad \text{on the interval } [0,1].
$$

- **Integration Before Limit**:

  $$
  \lim_{n \to \infty} \int_0^1 f_n(x) \, dx = \lim_{n \to \infty} (1 - e^{-n^2}) = 1
  $$

- **Limit Before Integration**:

  $$
  \int_0^1 \lim_{n \to \infty} f_n(x) \, dx = \int_0^1 0 \, dx = 0
  $$

  - **Conflict**: The two computations yield different results, indicating that the limit and integral cannot be interchanged under the Riemann framework in this case.

---

### 2. Convergence of Functions

Understanding how sequences of functions converge is essential, especially when dealing with limits and integration.

#### 2.1 Importance in Machine Learning

- **Optimization**: When training models (e.g., neural networks), we often consider sequences of functions that approximate some target function.
- **Integration and Limits**: The ability to interchange limits and integration depends on the mode of convergence.

#### 2.2 Modes of Convergence

We will focus on two primary modes of convergence:

- **Pointwise Convergence**
- **Uniform Convergence**

#### 2.3 Pointwise Convergence

##### 2.3.1 Definition

A sequence of functions $\{f_n(x)\}$ converges pointwise to a function $f(x)$ on a domain $D$ if, for every $x \in D$:

$$
\lim_{n \to \infty} f_n(x) = f(x)
$$

- **Formal Definition**: For every $x \in D$ and $\epsilon > 0$, there exists $N = N(\epsilon, x)$ such that for all $n > N$:

  $$
  |f_n(x) - f(x)| < \epsilon
  $$

##### 2.3.2 Characteristics

- **Dependence on $x$**: The rate of convergence may vary with $x$; convergence is evaluated at each point individually.
- **Preservation of Continuity**: Pointwise convergence does not guarantee that the limit function $f(x)$ will inherit properties like continuity from the functions $f_n(x)$.

##### 2.3.3 Example

Consider the sequence:

$$
f_n(x) = x^n \quad \text{on } [0,1]
$$

- **Pointwise Limit**:

  $$
  f(x) = \begin{cases} 0, & \text{if } x \in [0,1) \\ 1, & \text{if } x = 1 \end{cases}
  $$

- **Observation**:
  - Each $f_n(x)$ is continuous on $[0,1]$.
  - The limit function $f(x)$ is discontinuous at $x = 1$.
  - **Conclusion**: Pointwise convergence does not preserve continuity.

#### 2.4 Uniform Convergence

##### 2.4.1 Definition

A sequence of functions $\{f_n(x)\}$ converges uniformly to a function $f(x)$ on a domain $D$ if:

$$
\lim_{n \to \infty} \sup_{x \in D} |f_n(x) - f(x)| = 0
$$

- **Formal Definition**: For every $\epsilon > 0$, there exists $N = N(\epsilon)$ (independent of $x$) such that for all $n > N$ and all $x \in D$:

  $$
  |f_n(x) - f(x)| < \epsilon
  $$

##### 2.4.2 Characteristics

- **Independence from $x$**: The convergence is uniform across the entire domain $D$; the same $N$ works for all $x$.
- **Preservation of Continuity**: Uniform convergence does preserve continuity. If each $f_n(x)$ is continuous and $f_n(x) \to f(x)$ uniformly, then $f(x)$ is continuous.

##### 2.4.3 Example

Consider the sequence:

$$
g_n(x) = \frac{x}{n} \quad \text{on } [0,1]
$$

- **Uniform Limit**:

  $$
  g(x) = 0
  $$

- **Verification**:
  - For all $x \in [0,1]$ and $n \geq 1$:

    $$
    |g_n(x) - g(x)| = \left|\frac{x}{n} - 0\right| \leq \frac{1}{n}
    $$

  - Given $\epsilon > 0$, choose $N \geq \frac{1}{\epsilon}$, so for all $n > N$:

    $$
    |g_n(x) - g(x)| < \epsilon
    $$

  - **Conclusion**: $g_n(x)$ converges uniformly to $g(x) = 0$ on $[0,1]$.

#### 2.5 Importance of Uniform Convergence in Integration

- **Exchange of Limit and Integral**: Uniform convergence allows us to interchange limits and integrals:

  $$
  \lim_{n \to \infty} \int_a^b f_n(x) \, dx = \int_a^b \lim_{n \to \infty} f_n(x) \, dx
  $$

- **Riemann Integral Requirement**: The Riemann integral requires uniform convergence to ensure that the limit function is integrable and that the limit and integral operations can be exchanged.

---

### 3. Lebesgue Integral

#### 3.1 Introduction

- **Henri Lebesgue**: Developed the Lebesgue Integral in the early 20th century.
- **Motivation**: Address the limitations of the Riemann integral, especially for more complex functions and domains encountered in advanced mathematics and machine learning.

#### 3.2 Key Differences from Riemann Integral

- **Domain vs. Range Partitioning**:
  - **Riemann Integral**: Partitions the domain $[a, b]$ into subintervals.
  - **Lebesgue Integral**: Partitions the range (codomain) of the function $f(x)$ into subintervals.
- **Measure Theory**:
  - Lebesgue integration relies on the concept of measure, which generalizes the notion of "size" or "volume" of sets, even in abstract spaces.

#### 3.3 Construction of the Lebesgue Integral

##### 3.3.1 Partitioning the Range

- **Function $f(x)$**: Consider a bounded measurable function $f: X \to \mathbb{R}$, where $X$ is any measurable space.
- **Partition the Range**: Divide the range of $f(x)$ into intervals:

  $$
  f_{\min} = f_1 < f_2 < \dots < f_n = f_{\max}
  $$

- **Sets Corresponding to Range Intervals**: For each $k$, define:

  $$
  E_k = \{ x \in X \mid f_k \leq f(x) < f_{k+1} \}
  $$

##### 3.3.2 Measuring the Sets

- **Measure $\mu$**: Assign a measure to each set $E_k$, denoted $\mu(E_k)$.
- **Interpretation**: $\mu(E_k)$ represents the "size" of the set of $x$ values for which $f(x)$ falls within the interval $[f_k, f_{k+1})$.

##### 3.3.3 Lebesgue Sum

- **Sum Over Range Partitions**:

  $$
  S = \sum_{k=1}^n f_k \mu(E_k)
  $$

- **Limit**: As the partitions of the range become finer (i.e., $\max |f_{k+1} - f_k| \to 0$), the Lebesgue sum converges.

##### 3.3.4 Lebesgue Integral Definition

- **Lebesgue Integral**:

  $$
  \int_X f \, d\mu = \lim_{\max |f_{k+1} - f_k| \to 0} \sum_{k=1}^n f_k \mu(E_k)
  $$

- **Interpretation**: The Lebesgue integral sums the products of function values and the measures of the corresponding sets in the domain.

---

### Summary

- **Riemann Integral**:
  - Based on partitioning the domain of integration.
  - Requires functions to be continuous almost everywhere.
  - Limited in handling complex domains and exchanging limits and integrals.

- **Convergence of Functions**:
  - **Pointwise Convergence**: Convergence at individual points; does not preserve continuity.
  - **Uniform Convergence**: Uniform convergence over the domain; preserves continuity and integrability.

- **Lebesgue Integral**:
  - Based on partitioning the range of the function.
  - Utilizes measure theory to handle more general sets and functions.
  - Allows for integration of a broader class of functions, including those not Riemann integrable.
  - Facilitates the exchange of limits and integrals under more general conditions.

---

By embracing the Lebesgue integral, we equip ourselves with a powerful tool that extends the concept of integration to a wider class of functions and domains. This is particularly important in computational statistics and machine learning, where we often encounter complex functions and need robust mathematical foundations to ensure the validity of our methods.
