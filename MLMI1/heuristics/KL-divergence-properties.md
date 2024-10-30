To formally prove the properties of the Kullback-Leibler (KL) divergence between two probability distributions $q$ and $p$, we'll address each property step by step:

### Definition of KL Divergence
For continuous probability distributions $q(x)$ and $p(x)$ defined over a common support $X$, the KL divergence from $q$ to $p$ is defined as:

$$\text{KL}(q \parallel p) = \int_X q(x) \ln \frac{q(x)}{p(x)} \, dx$$

For discrete distributions, the integral is replaced with a summation:

$$\text{KL}(q \parallel p) = \sum_{x \in X} q(x) \ln \frac{q(x)}{p(x)}$$

### Property 1: Non-Negativity
**Statement:**

$$\text{KL}(q \parallel p) \geq 0$$

with equality if and only if $q(x) = p(x)$ almost everywhere.

**Proof:**

1. **Convexity of the Negative Log Function:**  
   The function $f(t) = -\ln t$ is convex for $t > 0$ because its second derivative $f''(t) = \frac{1}{t^2}$ is positive.

2. **Applying Jensen's Inequality:**  
   For a convex function $f$ and a random variable $T$ under probability measure $q$:

   $$f\left(\int T \, dq\right) \leq \int f(T) \, dq$$

3. **Normalization of Probability Densities:**  
   Since $q(x)$ is a probability density function (pdf):

   $$\int_X q(x) \, dx = 1$$

4. **Applying to KL Divergence:**  
   Let $T(x) = \frac{q(x)}{p(x)}$, which is the Radon-Nikodym derivative when $p$ is absolutely continuous with respect to $q$.

   Applying Jensen's inequality:

   $$- \ln \left( \int_X T(x) q(x) \, dx \right) \leq \int_X -\ln T(x) q(x) \, dx$$

5. **Simplify the Left Side:**  
   $$- \ln \left( \int_X \frac{q(x)}{p(x)} p(x) \, dx \right) = -\ln \left( \int_X p(x) \, dx \right) = -\ln 1 = 0$$

6. **Resulting Inequality:**  
   $$0 \leq \int_X -\ln \left( \frac{q(x)}{p(x)} \right) q(x) \, dx = \int_X q(x) \ln \frac{q(x)}{p(x)} \, dx = \text{KL}(q \parallel p)$$

**Equality Condition:**  
   Equality holds if and only if $T(x)$ is constant $q$-almost everywhere, which means $\frac{q(x)}{p(x)} = c$. Since both $p$ and $q$ integrate to 1, $c=1$, implying $q(x) = p(x)$ almost everywhere.

**Conclusion:**  
   $$\text{KL}(q \parallel p) \geq 0$$  
   with equality if and only if $q(x) = p(x)$ almost everywhere.

### Property 2: Non-Symmetry
**Statement:**  
   $$\text{KL}(q \parallel p) \neq \text{KL}(p \parallel q) \quad \text{in general}$$

**Proof:**

1. **Definition of Symmetry:**  
   If KL divergence were symmetric, we would have:

   $$\text{KL}(q \parallel p) = \text{KL}(p \parallel q)$$

2. **Counterexample:**  
   Consider a discrete space $X = \{0, 1\}$ with the following probability distributions:

   - $q(0) = 0.9$, $q(1) = 0.1$
   - $p(0) = 0.1$, $p(1) = 0.9$

3. **Compute $\text{KL}(q \parallel p)$:**  
   $$\text{KL}(q \parallel p) = 0.9 \ln \frac{0.9}{0.1} + 0.1 \ln \frac{0.1}{0.9} = 0.8 \ln 9$$

4. **Compute $\text{KL}(p \parallel q)$:**  
   $$\text{KL}(p \parallel q) = 0.1 \ln \frac{0.1}{0.9} + 0.9 \ln \frac{0.9}{0.1} = 0.8 \ln 9$$

In this example, the values for $\text{KL}(q \parallel p)$ and $\text{KL}(p \parallel q)$ suggest symmetry, but this result does not generalize to all distributions.

5. **Alternative Counterexample:**  
   Consider:

   - $q(0) = 1$, $q(1) = 0$
   - $p(0) = 0.5$, $p(1) = 0.5$

6. **Compute $\text{KL}(q \parallel p)$ and $\text{KL}(p \parallel q)$:**  
   - $$\text{KL}(q \parallel p) = \ln 2$$
   - $$\text{KL}(p \parallel q) = -0.5 \ln 2 - \infty$$

This example shows that $\text{KL}(q \parallel p)$ and $\text{KL}(p \parallel q)$ can have different values, demonstrating that KL divergence is not symmetric.

### Property 3: Minimum KL Divergence if and Only if $q = p$
**Statement:**  
   $$\text{KL}(q \parallel p) = 0 \quad \text{if and only if} \quad q(x) = p(x) \, \text{almost everywhere}$$

**Proof:**

1. **From Non-Negativity:**  
   We have already established that $\text{KL}(q \parallel p) \geq 0$.

2. **Equality Condition:**  
   The inequality $\text{KL}(q \parallel p) \geq 0$ becomes an equality only if the integrand is zero wherever $q(x) > 0$:

   $$q(x) \ln \frac{q(x)}{p(x)} = 0 \Rightarrow \ln \frac{q(x)}{p(x)} = 0 \Rightarrow \frac{q(x)}{p(x)} = 1 \Rightarrow q(x) = p(x)$$

**Conclusion:**  
   Therefore, $\text{KL}(q \parallel p) = 0$ if and only if $q(x) = p(x)$ almost everywhere (except possibly on a set of measure zero).

### Summary
- **Non-Negativity:** The KL divergence is always non-negative and equals zero if and only if the two distributions are identical almost everywhere.

  $$\text{KL}(q \parallel p) \geq 0, \quad \text{with equality if and only if} \quad q(x) = p(x)$$

- **Non-Symmetry:** KL divergence is not symmetric; swapping $q$ and $p$ generally yields different values.

  $$\text{KL}(q \parallel p) \neq \text{KL}(p \parallel q)$$

- **Uniqueness of Minimum:** The KL divergence attains its minimum value (zero) uniquely when $q$ and $p$ are the same distribution.

**Conclusion:**  
The KL divergence possesses the properties of non-negativity, non-symmetry, and attains its minimum value of zero if and only if the two distributions are identical, which are fundamental characteristics in information theory and statistical inference.

**References:**

- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory.* Wiley-Interscience.
- Csiszár, I., & Shields, P. C. (2004). *Information theory and statistics: A tutorial.*
