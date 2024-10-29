# Proof of Jensen's Inequality for Convex Functions

Let $f: I \to \mathbb{R}$ be a convex function defined on an interval $I \subseteq \mathbb{R}$. Let $x_1, x_2, \dots, x_n \in I$ and $a_1, a_2, \dots, a_n \geq 0$ be weights such that $\sum_{i=1}^{n} a_i = 1$. Then, Jensen's inequality states:

$$
f\left( \sum_{i=1}^{n} a_i x_i \right) \leq \sum_{i=1}^{n} a_i f(x_i)
$$

### Definition of Convexity:

A function $f: I \to \mathbb{R}$ is convex on interval $I$ if, for all $x, y \in I$ and $\lambda \in [0,1]$:

$$
f(\lambda x + (1 - \lambda) y) \leq \lambda f(x) + (1 - \lambda) f(y)
$$

### Proof by Mathematical Induction:

#### Base Case ($n = 1$):

When $n = 1$, we have:

$$
f(a_1 x_1) = f(x_1) = a_1 f(x_1)
$$

Since $a_1 = 1$, the inequality holds trivially.

#### Base Case ($n = 2$):

For $n = 2$, with $a_1 + a_2 = 1$ and $a_1, a_2 \geq 0$, convexity implies:

$$
f(a_1 x_1 + a_2 x_2) \leq a_1 f(x_1) + a_2 f(x_2)
$$

This directly follows from the definition of convexity.

#### Inductive Step:

Assume that Jensen's inequality holds for some $n = k$. That is, for any $x_1, \dots, x_k \in I$ and weights $a_1, \dots, a_k \geq 0$ with $\sum_{i=1}^{k} a_i = 1$:

$$
f\left( \sum_{i=1}^{k} a_i x_i \right) \leq \sum_{i=1}^{k} a_i f(x_i)
$$

We need to show that the inequality holds for $n = k+1$.

Consider $n = k+1$:

Let $x_1, x_2, \dots, x_{k+1} \in I$ and $a_1, a_2, \dots, a_{k+1} \geq 0$ with $\sum_{i=1}^{k+1} a_i = 1$.

Define:

$$
A = \sum_{i=1}^{k} a_i = 1 - a_{k+1}
$$

If $A = 0$, then $a_{k+1} = 1$, and the inequality holds trivially.

If $A > 0$, define normalized weights $b_i = \frac{a_i}{A}$ for $i = 1, \dots, k$. Note that $\sum_{i=1}^{k} b_i = 1$.

Let:

$$
S = \sum_{i=1}^{k} a_i x_i = A \left( \sum_{i=1}^{k} b_i x_i \right)
$$

By the inductive hypothesis:

$$
f\left( \sum_{i=1}^{k} b_i x_i \right) \leq \sum_{i=1}^{k} b_i f(x_i)
$$

Multiplying both sides by $A$:

$$
A f\left( \sum_{i=1}^{k} b_i x_i \right) \leq A \sum_{i=1}^{k} b_i f(x_i)
$$

Simplify:

$$
A f\left( \frac{S}{A} \right) = f(S) \leq \sum_{i=1}^{k} a_i f(x_i)
$$

### Applying Convexity:

Now, consider the convex combination of $S$ and $x_{k+1}$:

$$
\sum_{i=1}^{k+1} a_i x_i = S + a_{k+1} x_{k+1} = A \left( \frac{S}{A} \right) + a_{k+1} x_{k+1}
$$

Since $A + a_{k+1} = 1$, we can express this as:

$$
\sum_{i=1}^{k+1} a_i x_i = (1 - a_{k+1}) \left( \frac{S}{1 - a_{k+1}} \right) + a_{k+1} x_{k+1}
$$

Applying convexity of $f$:

$$
f\left( \sum_{i=1}^{k+1} a_i x_i \right) \leq (1 - a_{k+1}) f\left( \frac{S}{1 - a_{k+1}} \right) + a_{k+1} f(x_{k+1})
$$

Since $\frac{S}{1 - a_{k+1}} = \sum_{i=1}^{k} b_i x_i$, and from the earlier inequality $f\left( \sum_{i=1}^{k} b_i x_i \right) \leq \sum_{i=1}^{k} b_i f(x_i)$, we have:

$$
f\left( \frac{S}{1 - a_{k+1}} \right) \leq \sum_{i=1}^{k} b_i f(x_i)
$$

Multiply both sides by $1 - a_{k+1}$:

$$
(1 - a_{k+1}) f\left( \frac{S}{1 - a_{k+1}} \right) \leq \sum_{i=1}^{k} a_i f(x_i)
$$

### Combining Inequalities:

Substituting back:

$$
f\left( \sum_{i=1}^{k+1} a_i x_i \right) \leq \sum_{i=1}^{k} a_i f(x_i) + a_{k+1} f(x_{k+1})
$$

### Conclusion:

Thus, Jensen's inequality holds for $n = k+1$. By mathematical induction, it holds for all $n \geq 1$:

$$
f\left( \sum_{i=1}^{n} a_i x_i \right) \leq \sum_{i=1}^{n} a_i f(x_i)
$$

Therefore, the proof is complete.


# Proof of Jensen's Inequality in the General (Continuous) Case

### Setting:

Let $(\Omega, \mathcal{F}, P)$ be a probability space.

Let $X: \Omega \to \mathbb{R}$ be a real-valued random variable such that $E[\lvert X \rvert] < \infty$ (i.e., $X$ is integrable).

Let $f: \mathbb{R} \to \mathbb{R}$ be a convex function such that $E[\lvert f(X) \rvert] < \infty$ (i.e., $f(X)$ is integrable).

### Jensen's Inequality:

Under these conditions, Jensen's inequality states:

$$
f(E[X]) \leq E[f(X)]
$$

### Definitions:

- **Random Variable $X$**: A measurable function from the probability space $(\Omega, \mathcal{F}, P)$ to the real numbers $\mathbb{R}$.
- **Expectation $E[X]$**: The expected value (mean) of $X$, defined as 
$$
E[X] = \int_{\Omega} X(\omega) \, dP(\omega),
$$ 
provided the integral exists (i.e., $X$ is integrable).
- **Convex Function $f$**: A function $f: \mathbb{R} \to \mathbb{R}$ is convex if, for all $x, y \in \mathbb{R}$ and $\lambda \in [0,1]$:
$$
f(\lambda x + (1 - \lambda) y) \leq \lambda f(x) + (1 - \lambda) f(y)
$$

### Goal:

To prove that 
$$
f(E[X]) \leq E[f(X)].
$$

### Proof:

We will use the property of convex functions involving supporting lines (or subgradients) and the linearity of expectations.

#### Step 1: Subgradient of Convex Functions

Since $f$ is convex, for any point $x_0 \in \mathbb{R}$, there exists a subgradient $m \in \mathbb{R}$ such that for all $x \in \mathbb{R}$:

$$
f(x) \geq f(x_0) + m(x - x_0)
$$

This inequality represents the fact that the graph of $f$ lies above the tangent line (or supporting line) at $x_0$.

Note: If $f$ is differentiable at $x_0$, then $m = f'(x_0)$. If $f$ is not differentiable at $x_0$, $m$ can be any value in the subdifferential at $x_0$.

#### Step 2: Applying the Subgradient Inequality to $X$

Let $x_0 = E[X]$. Then, for all $\omega \in \Omega$:

$$
f(X(\omega)) \geq f(E[X]) + m(X(\omega) - E[X])
$$

#### Step 3: Taking Expectations

Since $f(X)$ and $X$ are integrable, we can take expectations on both sides of the inequality:

$$
E[f(X)] \geq f(E[X]) + m E[X - E[X]]
$$

#### Step 4: Simplifying the Right-Hand Side

Notice that 
$$
E[X - E[X]] = E[X] - E[X] = 0.
$$

Therefore, the inequality simplifies to:

$$
E[f(X)] \geq f(E[X]) + m \cdot 0 = f(E[X]).
$$

### Conclusion:

Thus, we have shown that:

$$
f(E[X]) \leq E[f(X)].
$$

This completes the proof of Jensen's inequality in the continuous case.

### Remarks:

- The key idea in the proof is leveraging the convexity of $f$ to establish a global linear lower bound (the supporting line) at the point $E[X]$.
- The expectation operator $E[\cdot]$ is linear, which allows us to move it inside the inequality after applying it to both sides.
- The condition $E[\lvert f(X) \rvert] < \infty$ ensures that $E[f(X)]$ is well-defined.

### Alternative Approach Using Convex Combination:

Alternatively, we can use the definition of convexity involving expectations.

#### Step 1: Definition of Convexity

For any random variable $X$ and any real number $t$, define a random variable $Y$ that takes the constant value $t$.

Convexity of $f$ implies:

$$
f(\lambda X + (1 - \lambda) Y) \leq \lambda f(X) + (1 - \lambda) f(Y)
$$

#### Step 2: Choosing $\lambda$ and $Y$

Set $\lambda = 1$ and $Y = E[X]$. Then:

$$
f(X) \leq f(X)
$$

This is trivial and doesn't help directly.

Instead, consider the random variable $X$ and its expected value $E[X]$. We can think of $X$ as a mixture (convex combination) of its values weighted by the probability distribution.

#### Step 3: Using the Law of Total Expectation

Since the expectation is an average over the distribution of $X$, we can write:

$$
E[X] = \int_{\mathbb{R}} x \, dF_X(x)
$$

Where $F_X$ is the cumulative distribution function of $X$.

Similarly, the expectation of $f(X)$ is:

$$
E[f(X)] = \int_{\mathbb{R}} f(x) \, dF_X(x)
$$

Because $f$ is convex, and $X$ has a probability distribution over $\mathbb{R}$, the inequality $f(E[X]) \leq E[f(X)]$ holds due to the convex combination inherent in the definition of expectation.

However, this approach is less formal without explicitly using the properties of convex functions as in the initial proof.

### Final Note:

The proof provided uses fundamental properties of convex functions and expectations, making it rigorous and self-contained.
