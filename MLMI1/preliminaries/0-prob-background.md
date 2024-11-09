# Random Variable

Formally, given a probability space $(\Omega, \mathcal{F}, P)$:

1. **Definition**: A random variable $X$ is a *measurable* function $X: \Omega \to \mathbb{R}$ that assigns a real number $X(\omega)$ to each outcome $\omega \in \Omega$, where:
    
    - $\Omega$ is the sample space containing all possible outcomes,
    - $\mathcal{F}$ is a $\sigma$-algebra of subsets of $\Omega$, called events, and
    - $P: \mathcal{F} \to [0, 1]$ is a probability measure.
    
2. **Types of Random Variables**:
    
    - **Discrete Random Variable**: A random variable $X$ is discrete if it takes values in a countable subset of $\mathbb{R}$. It is characterized by a **probability mass function** (PMF), $p_X: \mathbb{R} \to [0, 1]$, where $p_X(x) = P(X = x)$.
    - **Continuous Random Variable**: A random variable $X$ is continuous if it can take any value in an interval of $\mathbb{R}$ and is described by a **probability density function** (PDF), $f_X: \mathbb{R} \to [0, \infty)$, where $P(a \leq X \leq b) = \int_a^b f_X(x) , dx$ for any interval $[a, b] \subset \mathbb{R}$.

# Sum Rule

The **sum rule** (or **law of total probability**) allows us to obtain the marginal probability of a random variable by summing (or integrating) over all possible values of another random variable. Given two random variables $X$ and $Y$ defined on the same probability space $(\Omega, \mathcal{F}, P)$, the sum rule is stated as follows:

For discrete random variables $X$ and $Y$:
$$
p_X(x) = \sum_y p_{X, Y}(x, y)
$$
where $p_{X, Y}(x, y) = P(X = x, Y = y)$ is the joint probability mass function of $X$ and $Y$.

For continuous random variables $X$ and $Y$:
$$
f_X(x) = \int_{-\infty}^{\infty} f_{X, Y}(x, y) \, dy
$$
where $f_{X, Y}(x, y)$ is the joint probability density function of $X$ and $Y$, and $f_X(x)$ is the marginal probability density function of $X$.

---

# Product Rule

The **product rule** (or **chain rule of probability**) allows us to express the joint probability of two random variables in terms of a conditional probability and a marginal probability. For two random variables $X$ and $Y$ defined on the same probability space $(\Omega, \mathcal{F}, P)$:

For discrete random variables:
$$
p_{X, Y}(x, y) = p_X(x) \, p_{Y|X}(y | x) = p_Y(y) \, p_{X|Y}(x | y)
$$
where $p_{X, Y}(x, y)$ is the joint probability of $X = x$ and $Y = y$, $p_{Y|X}(y | x)$ is the conditional probability of $Y = y$ given $X = x$, and $p_X(x)$ and $p_Y(y)$ are the marginal probabilities of $X$ and $Y$ respectively.

For continuous random variables:
$$
f_{X, Y}(x, y) = f_X(x) \, f_{Y|X}(y | x) = f_Y(y) \, f_{X|Y}(x | y)
$$
where $f_{X, Y}(x, y)$ is the joint density function, $f_{Y|X}(y | x)$ is the conditional density of $Y$ given $X = x$, and $f_X(x)$ and $f_Y(y)$ are the marginal densities of $X$ and $Y$.

---

# Bayes' Theorem

**Bayes' theorem** is a result derived from the product rule and allows us to update the probability of a hypothesis or parameter given new data. Formally, for two events $A$ and $B$ with $P(B) > 0$, Bayes' theorem states:
$$
P(A | B) = \frac{P(B | A) \, P(A)}{P(B)}
$$

For random variables, Bayes' theorem can be stated as:
$$
p(\theta | D) = \frac{p(D | \theta) \, p(\theta)}{p(D)}
$$
where:
- $p(\theta | D)$ is the **posterior probability**: the probability of the parameter $\theta$ given data $D$.
- $p(D | \theta)$ is the **likelihood**: the probability of data $D$ given parameter $\theta$.
- $p(\theta)$ is the **prior probability**: the initial probability of parameter $\theta$ before observing data $D$.
- $p(D)$ is the **marginal likelihood** or **evidence**: the total probability of data $D$, computed as:
  $$
  p(D) = \int p(D | \theta) \, p(\theta) \, d\theta
  $$
  for continuous parameters or as a sum for discrete parameters.


