# Lecture 4: Lebesgue Integral and Measure

## Overview

In this lecture, we delve into the foundational concepts of measure theory and their pivotal role in probability theory and advanced machine learning. We will explore:

- **Definition of Measure of Sets**: Understanding how measures generalize notions like length, area, and volume.
- **Lebesgue Measure**: Introducing the Lebesgue measure and its properties.
- **Defining Probability Measures**: Establishing the formal measure-theoretic foundations of probability.
- **Random Variables**: Conceptualizing random variables as measurable functions.
- **Radon-Nikodym Theorem**: Exploring the relationship between different measures and introducing the Radon-Nikodym derivative.

Our goal is to develop a deep understanding of these concepts to build a strong foundation for advanced topics in computational statistics and machine learning.

---

### 1. Definition of Measure of Sets

#### 1.1 Introduction to Measure Theory

Measure theory provides a rigorous mathematical framework to generalize concepts such as length, area, and volume to more abstract spaces beyond $\mathbb{R}^d$. It plays a crucial role in modern probability theory and integration.

- **Historical Context**:
  - Henri Lebesgue: Pioneered measure theory and the Lebesgue integral.
  - Emile Borel and Johan Radon: Contributed significantly to the development of measure and integration.
  - Andrei Kolmogorov: Formalized the axiomatic foundations of probability theory using measure theory.

#### 1.2 Sigma-Algebras

To define measures on sets, we first need to understand the structure of the sets we are measuring.

##### 1.2.1 Definition of a Sigma-Algebra ($\sigma$-algebra)

A sigma-algebra $\Sigma$ on a set $X$ is a collection of subsets of $X$ satisfying the following properties:

1. **Contains the Empty Set and the Entire Set**: $\emptyset \in \Sigma$ and $X \in \Sigma$.
2. **Closed Under Complements**: If $A \in \Sigma$, then its complement $A^c = X \setminus A \in \Sigma$.
3. **Closed Under Countable Unions**: If $\{A_n\}_{n=1}^\infty \subset \Sigma$, then $\bigcup_{n=1}^\infty A_n \in \Sigma$.
4. **Closed Under Countable Intersections**: By De Morgan's laws, closure under complements and unions implies closure under countable intersections: $\bigcap_{n=1}^\infty A_n \in \Sigma$.

##### 1.2.2 Importance of Sigma-Algebras

- **Measurable Sets**: Elements of $\Sigma$ are the sets for which we can define a measure.
- **Countability**: The sigma-algebra must be closed under countable operations to handle infinite processes and limits, which are common in probability and analysis.

#### 1.3 Measures

##### 1.3.1 Definition of a Measure

A measure $\mu$ on a sigma-algebra $\Sigma$ over a set $X$ is a function $\mu: \Sigma \to [0, \infty]$ satisfying:

1. **Non-negativity**: For all $E \in \Sigma$, $\mu(E) \geq 0$.
2. **Null Empty Set**: $\mu(\emptyset) = 0$.
3. **Countable Additivity ($\sigma$-additivity)**: For any countable collection of disjoint sets $\{E_n\}_{n=1}^\infty \subset \Sigma$:
   
   $$
   \mu\left(\bigcup_{n=1}^\infty E_n\right) = \sum_{n=1}^\infty \mu(E_n)
   $$

   where the sets are pairwise disjoint ($E_i \cap E_j = \emptyset$ for $i \neq j$).

##### 1.3.2 Properties of Measures

- **Monotonicity**: If $E_1 \subseteq E_2$, then $\mu(E_1) \leq \mu(E_2)$.
- **Subadditivity**: For any countable collection of sets $\{E_n\}$:
  
  $$
  \mu\left(\bigcup_{n=1}^\infty E_n\right) \leq \sum_{n=1}^\infty \mu(E_n)
  $$

#### 1.4 Measurable Spaces and Functions

##### 1.4.1 Measurable Spaces

A measurable space is a pair $(X, \Sigma)$ where:

- $X$ is a set (the space).
- $\Sigma$ is a sigma-algebra of subsets of $X$.

##### 1.4.2 Measurable Functions

Given measurable spaces $(X, \Sigma_X)$ and $(Y, \Sigma_Y)$, a function $f: X \to Y$ is measurable if for every measurable set $B \in \Sigma_Y$, the pre-image $f^{-1}(B) \in \Sigma_X$, where:

$$
f^{-1}(B) = \{x \in X \mid f(x) \in B\}
$$

- **Intuition**: Measurable functions preserve the structure of measurable sets through the mapping.

---

### 2. Lebesgue Measure

#### 2.1 Introduction to Lebesgue Measure

The Lebesgue measure generalizes the notions of length, area, and volume to more complex sets and higher-dimensional spaces.

##### 2.1.1 Construction of the Lebesgue Measure

- **Intervals in $\mathbb{R}$**: For an interval $I = [a, b]$, the Lebesgue measure is simply its length:

  $$
  \mu(I) = b - a
  $$

- **Extension to More Complex Sets**: The Lebesgue measure extends to more complicated subsets of $\mathbb{R}$ by considering countable unions and intersections of intervals.

##### 2.2 Properties of the Lebesgue Measure

1. **Measurable Sets**
   - **Countable Sets**: Any countable set, such as the set of rational numbers $\mathbb{Q}$, has Lebesgue measure zero:

     $$
     \mu(\mathbb{Q}) = 0
     $$

2. **Additivity**: If $\{E_n\}$ is a countable collection of disjoint measurable sets, then:

   $$
   \mu\left(\bigcup_{n=1}^\infty E_n\right) = \sum_{n=1}^\infty \mu(E_n)
   $$

3. **Null Sets**: A set $N \subset \mathbb{R}$ is a null set if $\mu(N) = 0$. Functions that differ only on a null set are considered equivalent in measure theory.

---

### 3. Defining Probability Measures

#### 3.1 Probability Spaces

A probability space is a measure space where the total measure is one, formalizing the concept of probability.

##### 3.1.1 Components of a Probability Space

A probability space is a triple $(\Omega, F, P)$, where:

- **Sample Space ($\Omega$)**: The set of all possible outcomes of a random experiment.
- **Sigma-Algebra ($F$)**: A collection of subsets of $\Omega$ (called events), satisfying the properties of a sigma-algebra.
- **Probability Measure ($P$)**: A measure satisfying $P: F \to [0, 1]$ with $P(\Omega) = 1$.

##### 3.1.2 Example: Rolling a Die

- **Sample Space**: $\Omega = \{1, 2, 3, 4, 5, 6\}$
- **Sigma-Algebra ($F$)**: Could be the power set of $\Omega$ or a simpler sigma-algebra like $F = \{\emptyset, \{1, 3, 5\}, \{2, 4, 6\}, \Omega\}$.
- **Probability Measure ($P$)**: Assign probabilities to events: $P(\emptyset) = 0$, $P(\{1, 3, 5\}) = 0.5$, $P(\{2, 4, 6\}) = 0.5$, $P(\Omega) = 1$.

#### 3.2 Constructing Probability Measures

For continuous spaces (e.g., $\mathbb{R}$), constructing probability measures requires more advanced techniques.

- **Lebesgue Measure as a Probability Measure**: The Lebesgue measure on the interval $[0,1]$ can be normalized to define a uniform probability measure.

---

### 4. Random Variables

#### 4.1 Random Variables as Measurable Functions

A random variable is a measurable function from a probability space to a measurable space.

##### 4.1.1 Formal Definition

Given:

- A probability space $(\Omega, F, P)$.
- A measurable space $(E, E)$.
- A function $X: \Omega \to E$ is a random variable if it is measurable, meaning:

  $$
  \forall B \in E, \; X^{-1}(B) = \{\omega \in \Omega \mid X(\omega) \in B\} \in F
  $$

---

### 5. Radon-Nikodym Theorem

#### 5.1 Absolute Continuity of Measures

Given two measures $\mu$ and $\nu$ on the same measurable space $(X, \Sigma)$:

- **Absolute Continuity ($\mu \ll \nu$)**: $\mu$ is absolutely continuous with respect to $\nu$ if for every set $A \in \Sigma$ with $\nu(A) = 0$, it follows that $\mu(A) = 0$.

#### 5.2 Radon-Nikodym Derivative

##### 5.2.1 Statement of the Radon-Nikodym Theorem

If $\mu$ and $\nu$ are sigma-finite measures on $(X, \Sigma)$ with $\mu \ll \nu$, then there exists a measurable function $f: X \to [0, \infty)$ such that for all $A \in \Sigma$:

$$
\mu(A) = \int_A f \, d\nu
$$

---

### Conclusion

- **Measure Theory Foundations**: Understanding measure theory is crucial for advanced topics in probability and machine learning.
- **Lebesgue Integration**: Provides a powerful and general framework for integration, extending beyond the limitations of Riemann integration.
- **Random Variables as Measurable Functions**: Viewing random variables through the lens of measure theory enhances our ability to handle complex stochastic processes.
- **Radon-Nikodym Theorem**: A fundamental result that allows for the comparison and transformation of measures, underpinning many advanced techniques in statistics and machine learning.

By mastering these foundational concepts, we equip ourselves with the mathematical tools necessary to tackle complex problems in computational statistics and machine learning, enabling rigorous analysis and the development of sophisticated models.
