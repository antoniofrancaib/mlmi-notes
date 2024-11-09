# Conditions for the Existence of an Invariant Distribution

An invariant distribution (or stationary distribution) $\pi$ is a probability vector that remains unchanged under the transition dynamics of the Markov chain. Mathematically, it satisfies:

$$\pi = T \pi$$

### Conditions Required:

- **Finite State Space**: The Markov chain has a finite number of states.

- **Irreducibility**: The chain is irreducible if it's possible to reach any state from any other state in a finite number of steps. This ensures that the chain doesn't decompose into smaller, isolated subsets of states.

- **Aperiodicity**: The chain is aperiodic if it doesn't cycle in fixed-length periods. This means there's no integer $k > 1$ such that all state transitions occur in multiples of $k$.

### Implications:

- **Existence**: For any finite Markov chain with a stochastic matrix (columns summing to 1), at least one invariant distribution exists.

- **Uniqueness**: If the chain is both irreducible and aperiodic (i.e., ergodic), the invariant distribution is unique.

- **Convergence**: Under these conditions, the distribution of the chain after $k$ steps converges to the invariant distribution regardless of the initial state.

## 2. Calculating the Invariant Distribution Using Eigenvalues

Since the invariant distribution $\pi$ satisfies $\pi = T \pi$, it is a right eigenvector of the transition matrix $T$ corresponding to the eigenvalue 1.

### Steps to Calculate $\pi$:

1. **Solve the Eigenvector Equation**:

   $$(T - I) \pi = 0$$

   where $I$ is the identity matrix.

2. **Normalize**:

   Ensure that the components of $\pi$ sum to 1:

   $$\sum_i \pi_i = 1$$

3. **Non-negativity**:

   All components of $\pi$ must be non-negative:

   $$\pi_i \geq 0 \text{ for all } i$$

### Note on Eigenvalues:

The eigenvalue 1 is always present for stochastic matrices (columns summing to 1).

Other eigenvalues' magnitudes determine the rate of convergence to the invariant distribution.

## 3. Interpretation of $T^k$

The matrix $T^k$ represents the $k$-step transition probabilities of the Markov chain.

### Meaning:

Entries of $T^k$:

$$(T^k)_{ij} = P(X_{n+k} = i \mid X_n = j)$$

This is the probability of transitioning from state $j$ to state $i$ in exactly $k$ steps.

### Convergence:

As $k \to \infty$, $T^k$ tends to a matrix where each column is the invariant distribution $\pi$ (assuming the chain is ergodic):

$$\lim_{k \to \infty} T^k = \pi 1^T$$

where $1$ is a column vector of ones.

## Summary

- **Invariant Distribution**: Exists when the chain is finite, and it's unique and convergent under irreducibility and aperiodicity.
- **Calculation via Eigenvalues**: Solve $(T - I) \pi = 0$ with normalization and non-negativity constraints to find $\pi$.
- **$T^k$ Interpretation**: Provides the probabilities of transitioning between states over $k$ steps and illustrates the chain's long-term behavior.
