
### Scalars, Vectors, and Matrices

- **Scalar**: A single number, denoted by lowercase letters (e.g., $a, b, c$).
- **Vector**: A column vector, denoted by bold lowercase letters (e.g., $\mathbf{a}, \mathbf{b}$).
- **Matrix**: A two-dimensional array of numbers, denoted by uppercase letters (e.g., $A, B$).

### Derivatives with Respect to Vectors and Matrices

- **Gradient**: The vector of partial derivatives of a scalar function $f$ with respect to a vector $\mathbf{x}$:
  $$
  \nabla_{\mathbf{x}} f = \begin{bmatrix} \dfrac{\partial f}{\partial x_1} \\ \vdots \\ \dfrac{\partial f}{\partial x_n} \end{bmatrix}
  $$
- **Jacobian Matrix**: The matrix of all first-order partial derivatives of a vector-valued function $\mathbf{f}$ with respect to a vector $\mathbf{x}$:
  $$
  J_{\mathbf{x}}(\mathbf{f}) = \begin{bmatrix} \dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \dfrac{\partial f_m}{\partial x_1} & \cdots & \dfrac{\partial f_m}{\partial x_n} \end{bmatrix}
  $$

## Basic Rules

### Scalar Functions of Vectors

For a scalar function $f: \mathbb{R}^n \rightarrow \mathbb{R}$:

- **Gradient**: The derivative of $f$ with respect to $\mathbf{x}$ is the gradient vector $\nabla_{\mathbf{x}} f$.
- **Directional Derivative**: In the direction of a unit vector $\mathbf{u}$:
  $$
  D_{\mathbf{u}} f = \nabla_{\mathbf{x}} f^\top \mathbf{u}
  $$

### Hessian Matrix

The Hessian matrix is the matrix of second-order partial derivatives of a scalar function $f$ with respect to vector $\mathbf{x}$:
$$
H_{\mathbf{x}}(f) = \begin{bmatrix} \dfrac{\partial^2 f}{\partial x_1^2} & \cdots & \dfrac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \dfrac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2} \end{bmatrix}
$$

## Matrix Derivative Rules

### Derivative of Linear Functions

For a linear function $\mathbf{y} = A\mathbf{x}$, where $A$ is a constant matrix:

- The derivative of $\mathbf{y}$ with respect to $\mathbf{x}$ is the matrix $A$:
  $$
  \dfrac{\partial \mathbf{y}}{\partial \mathbf{x}} = A
  $$

### Product Rule

For differentiable functions $f$ and $g$:

- **Product Rule**:
  $$
  \dfrac{\partial}{\partial \mathbf{x}} [f(\mathbf{x}) g(\mathbf{x})] = \dfrac{\partial f}{\partial \mathbf{x}} g(\mathbf{x}) + f(\mathbf{x}) \dfrac{\partial g}{\partial \mathbf{x}}
  $$

### Chain Rule

For functions $f$ and $g$ where $f$ is a function of $g(\mathbf{x})$:

- **Chain Rule**:
  $$
  \dfrac{\partial f}{\partial \mathbf{x}} = \dfrac{\partial f}{\partial g} \dfrac{\partial g}{\partial \mathbf{x}}
  $$

## Common Derivatives

### Derivative of a Quadratic Form

Consider the quadratic form $f(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x}$, where $A$ is a symmetric matrix.

- **Gradient**:
  $$
  \nabla_{\mathbf{x}} f = (A + A^\top) \mathbf{x} = 2A \mathbf{x}
  $$
  (Since $A$ is symmetric, $A = A^\top$.)

- **Hessian**:
  $$
  H_{\mathbf{x}}(f) = A + A^\top = 2A
  $$

### Derivative of the Determinant

For a square matrix $A(\theta)$ dependent on parameter $\theta$:

- **Derivative of Determinant**:
  $$
  \dfrac{\partial}{\partial \theta} \det A = \det A \cdot \operatorname{tr}\left( A^{-1} \dfrac{\partial A}{\partial \theta} \right)
  $$

### Derivative of the Inverse Matrix

For an invertible matrix $A(\theta)$:

- **Derivative of Inverse**:
  $$
  \dfrac{\partial A^{-1}}{\partial \theta} = -A^{-1} \dfrac{\partial A}{\partial \theta} A^{-1}
  $$

### Derivative of the Trace

For matrices $A$ and $B$ where $A$ depends on $\theta$:

- **Trace Derivative**:
  $$
  \dfrac{\partial}{\partial \theta} \operatorname{tr}(AB) = \operatorname{tr}\left( A \dfrac{\partial B}{\partial \theta} \right) + \operatorname{tr}\left( \dfrac{\partial A}{\partial \theta} B \right)
  $$

## Eigenvalues and Eigenvectors

Differentiating eigenvalues and eigenvectors is more intricate.

### Derivative of Eigenvalues

Given $A(\theta) \mathbf{v} = \lambda \mathbf{v}$, where $\lambda$ is an eigenvalue of $A$:

- **Derivative of Eigenvalue**:
  $$
  \dfrac{\partial \lambda}{\partial \theta} = \mathbf{v}^\top \dfrac{\partial A}{\partial \theta} \mathbf{v}
  $$

### Derivative of Eigenvectors

Differentiating eigenvectors requires careful handling due to their directionality.

## Applications

### Optimization Problems

Matrix calculus is essential in solving optimization problems involving multivariate functions.

- **First-Order Condition**: Setting the gradient to zero:
  $$
  \nabla_{\mathbf{x}} f = \mathbf{0}
  $$
- **Second-Order Condition**: Checking the Hessian for positive definiteness.

### Least Squares

In linear regression, we minimize the function:
$$
f(\mathbf{\beta}) = \|\mathbf{y} - X\mathbf{\beta}\|^2
$$

- **Gradient**:
  $$
  \nabla_{\mathbf{\beta}} f = -2 X^\top (\mathbf{y} - X\mathbf{\beta})
  $$
- **Optimal Solution**:
  $$
  \mathbf{\beta} = (X^\top X)^{-1} X^\top \mathbf{y}
  $$

## Conclusion

Understanding the rules of matrix calculus is fundamental for advanced topics in applied mathematics and engineering. Mastery of these concepts allows for the efficient manipulation and optimization of complex systems involving vectors and matrices.
