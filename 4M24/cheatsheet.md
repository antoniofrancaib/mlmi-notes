

### Vector Space
A **vector space** (or **linear space**) over a field $\mathbb{F}$ (typically $\mathbb{R}$ for real numbers or $\mathbb{C}$ for complex numbers) is a set $V$ of elements, called **vectors**, with two operations:
- **Vector addition**: $+: V \times V \rightarrow V$
- **Scalar multiplication**: $\cdot: \mathbb{F} \times V \rightarrow V$

These operations satisfy the following properties for all vectors $\vec{u}, \vec{v}, \vec{w} \in V$ and all scalars $a, b \in \mathbb{F}$:
1. **Closure of addition**: $\vec{u} + \vec{v} \in V$
2. **Commutativity of addition**: $\vec{u} + \vec{v} = \vec{v} + \vec{u}$
3. **Associativity of addition**: $(\vec{u} + \vec{v}) + \vec{w} = \vec{u} + (\vec{v} + \vec{w})$
4. **Existence of an additive identity**: There exists $\vec{0} \in V$ such that $\vec{u} + \vec{0} = \vec{u}$ for all $\vec{u} \in V$.
5. **Existence of additive inverses**: For each $\vec{u} \in V$, there exists $-\vec{u} \in V$ such that $\vec{u} + (-\vec{u}) = \vec{0}$.
6. **Closure of scalar multiplication**: $a \cdot \vec{u} \in V$
7. **Distributivity of scalar multiplication with respect to vector addition**: $a \cdot (\vec{u} + \vec{v}) = a \cdot \vec{u} + a \cdot \vec{v}$
8. **Distributivity of scalar multiplication with respect to scalar addition**: $(a + b) \cdot \vec{u} = a \cdot \vec{u} + b \cdot \vec{u}$
9. **Associativity of scalar multiplication**: $a \cdot (b \cdot \vec{u}) = (a b) \cdot \vec{u}$
10. **Existence of a multiplicative identity**: There exists an element $1 \in \mathbb{F}$ such that $1 \cdot \vec{u} = \vec{u}$ for all $\vec{u} \in V$.

---

### Completeness
A **normed vector space** $V$ is said to be **complete** if every Cauchy sequence in $V$ converges to a limit that is also within $V$. In other words, for every Cauchy sequence $\{ x_n \}$ in $V$, there exists a point $x \in V$ such that:
$$
\lim_{n \to \infty} x_n = x
$$

---

### Cauchy Sequence
A **Cauchy sequence** in a normed vector space $V$ is a sequence $\{ x_n \}$ such that for any $\epsilon > 0$, there exists an integer $N$ such that for all $m, n \geq N$,
$$
\| x_n - x_m \| < \epsilon
$$
This means that the elements of the sequence get arbitrarily close to each other as $n$ and $m$ increase.

---

### Norm
A **norm** on a vector space $V$ (over the field $\mathbb{R}$ or $\mathbb{C}$) is a function $\| \cdot \| : V \rightarrow \mathbb{R}$ that satisfies the following properties for all vectors $u, v \in V$ and all scalars $\alpha$:
1. **Non-negativity**: $\| u \| \geq 0$ and $\| u \| = 0$ if and only if $u = 0$
2. **Scalar multiplication**: $\| \alpha u \| = |\alpha| \| u \|$
3. **Triangle inequality**: $\| u + v \| \leq \| u \| + \| v \|$

---

### Hilbert Space
A **Hilbert space** is a complete inner product space. That is, it is a vector space $V$ with an inner product $\langle \cdot, \cdot \rangle$ such that:
1. $V$ is complete with respect to the norm induced by the inner product, where $\| u \| = \sqrt{\langle u, u \rangle}$.
2. Every Cauchy sequence in $V$ converges to a point in $V$.
