### Exercise:

Consider a linear Gaussian state-space model where the hidden state is a two-dimensional vector that evolves through rotation and additive Gaussian noise. Let $\theta \in [0, 2\pi)$ be a fixed angle, and define the rotation matrix:

$$
R(\theta) = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix}.
$$

The state dynamics are given by:

$$
z_t = R(\theta) z_{t-1} + w_t,
$$

where $w_t$ is zero-mean Gaussian noise with covariance matrix $Q = \sigma_w^2 I$:

$$
p(w_t) = N(w_t; 0, \sigma_w^2 I).
$$

The observations are scalar and obtained by projecting the state onto a fixed vector $h$ and adding Gaussian noise:

$$
y_t = h^\top z_t + v_t,
$$

where $v_t$ is zero-mean Gaussian noise with variance $\sigma_v^2$:

$$
p(v_t) = N(v_t; 0, \sigma_v^2).
$$

Assume that $h = \begin{bmatrix} 1 & 0 \end{bmatrix}$.

### Questions:

1. **(a)** Show that the model is stationary under appropriate conditions on $\theta$ and $\sigma_w^2$. Determine the conditions under which the state covariance converges to a stationary distribution.



2. **(b)** Compute the stationary state covariance matrix $P$, i.e., find $P$ satisfying:
   $$
   P = R(\theta) P R(\theta)^\top + Q.
   $$


3. **(c)** Suppose that $\theta = \frac{\pi}{4}$ (a rotation of 45 degrees). Compute the stationary state covariance matrix $P$ in terms of $\sigma_w^2$.




4. **(d)** Derive the autocorrelation function of the observations $y_t$, i.e., compute $E[y_t y_{t-k}]$ for $k \geq 0$, in terms of $\sigma_w^2$, $\sigma_v^2$, $\theta$, and $h$.



5. **(e)** Discuss how the choice of $\theta$ affects the dynamics of the system. Specifically, consider the cases where $\theta = 0$, $\theta = \frac{\pi}{2}$, $\theta = \pi$, and $\theta$ is an irrational multiple of $\pi$. How does the rotation angle $\theta$ influence the properties of the model?



6. **(f)** You have access to an off-the-shelf implementation of the Kalman filter. Explain how you would use it to perform state estimation in this model. Describe any modifications or considerations needed due to the properties of the rotation matrix.





---
### Solution to the Exercise:

#### (a) Show that the model is stationary under appropriate conditions on $\theta$ and $\sigma_w^2$. Determine the conditions under which the state covariance converges to a stationary distribution.

**Answer:**

To determine if the model is stationary, we need to analyze whether the state covariance matrix $P_t = E[z_t z_t^\top]$ converges to a steady-state value as $t \to \infty$.

**State Dynamics:**

$$z_t = R(\theta) z_{t-1} + w_t,$$

where $R(\theta)$ is the rotation matrix:

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix},$$

and $w_t \sim N(0, \sigma_w^2 I)$.

**Covariance Update Equation:**

The covariance update equation is derived as follows:

$$P_t = E[z_t z_t^\top] = E[(R(\theta) z_{t-1} + w_t)(R(\theta) z_{t-1} + w_t)^\top].$$

Expanding and using the independence and zero-mean properties of $w_t$:

$$P_t = R(\theta) P_{t-1} R(\theta)^\top + Q,$$

where $Q = \sigma_w^2 I$.

**Analyzing Stationarity:**

**Eigenvalues of $R(\theta)$:** The rotation matrix $R(\theta)$ is orthogonal ($R(\theta) R(\theta)^\top = I$), and its eigenvalues are $e^{i\theta}$ and $e^{-i\theta}$, which lie on the unit circle in the complex plane.

**Implications:** Since the eigenvalues have a modulus of 1, the system is marginally stable, and without damping (i.e., without eigenvalues inside the unit circle), the covariance may not converge.

**Covariance Growth Over Time:**

Due to the orthogonality of $R(\theta)$:

$$R(\theta) P_{t-1} R(\theta)^\top = P_{t-1}.$$

Thus, the covariance update simplifies to:

$$P_t = P_{t-1} + Q = P_{t-1} + \sigma_w^2 I.$$

This recursion indicates that:

$$P_t = P_0 + t \sigma_w^2 I,$$

which means that the covariance matrix $P_t$ increases linearly with time.

**Conclusion:**

The state covariance does not converge to a stationary distribution because it grows indefinitely over time.
Therefore, the model is non-stationary unless $\sigma_w^2 = 0$. When $\sigma_w^2 = 0$, the process is deterministic (noise-free rotation), and the covariance remains constant, making the model stationary.

**Conditions for Stationarity:**

- **Stationary Model:** $\sigma_w^2 = 0$.
- **Non-Stationary Model:** $\sigma_w^2 > 0$ (covariance increases without bound).

#### (b) Compute the stationary state covariance matrix $P$, i.e., find $P$ satisfying:

$$P = R(\theta) P R(\theta)^\top + Q.$$

**Answer:**

Given the covariance update equation:

$$P_t = R(\theta) P_{t-1} R(\theta)^\top + Q.$$

**Attempt to Find a Stationary Covariance $P$:**

Assuming a stationary covariance $P$ exists, it must satisfy the discrete-time Lyapunov equation:

$$P = R(\theta) P R(\theta)^\top + Q.$$

However, as shown in part (a):

The covariance $P_t$ increases indefinitely due to the additive noise $Q = \sigma_w^2 I$.
The rotation matrix $R(\theta)$ does not have eigenvalues inside the unit circle, preventing the convergence of $P_t$ to a finite value.

**Conclusion:**

The stationary state covariance matrix $P$ does not exist because the covariance grows without bound over time.
Therefore, there is no finite solution to the equation $P = R(\theta) P R(\theta)^\top + Q$ when $\sigma_w^2 > 0$.

#### (c) Suppose that $\theta = \frac{\pi}{4}$ (a rotation of 45 degrees). Compute the stationary state covariance matrix $P$ in terms of $\sigma_w^2$.

**Answer:**

When $\theta = \frac{\pi}{4}$:

$$\cos\theta = \frac{\sqrt{2}}{2},$$
$$\sin\theta = \frac{\sqrt{2}}{2}.$$

However, the key observations from parts (a) and (b) still apply:

- The rotation matrix $R\left(\frac{\pi}{4}\right)$ remains orthogonal.
- The eigenvalues remain on the unit circle.
- The covariance $P_t$ continues to increase linearly over time:

$$P_t = P_0 + t \sigma_w^2 I.$$

**Conclusion:**

Even with $\theta = \frac{\pi}{4}$, the stationary covariance matrix $P$ does not exist because the covariance grows without bound.
The system remains non-stationary, and the covariance cannot be expressed in terms of $\sigma_w^2$ as a finite, steady-state value.

#### (d) Derive the autocorrelation function of the observations $y_t$, i.e., compute $E[y_t y_{t-k}]$ for $k \geq 0$, in terms of $\sigma_w^2$, $\sigma_v^2$, $\theta$, and $h$.

**Answer:**

**Observations:**

$$y_t = h^\top z_t + v_t,$$

where $h = \begin{bmatrix} 1 & 0 \end{bmatrix}$ and $v_t \sim N(0, \sigma_v^2)$.

**Calculating the Autocorrelation Function:**

For $k \geq 1$:

$$E[y_t y_{t-k}] = E[(z_{1,t} + v_t)(z_{1,t-k} + v_{t-k})] = E[z_{1,t} z_{1,t-k}],$$

since $v_t$ is independent of $z_{1,t-k}$ and $v_{t-k}$.

**Computing $E[z_{1,t} z_{1,t-k}]$:**

The state $z_t$ can be expressed as:

$$z_t = \sum_{n=0}^{t-1} R(\theta)^n w_{t-n} + R(\theta)^t z_0.$$

Assuming $z_0 = 0$ for simplicity:

$$z_{1,t} = \sum_{n=0}^{t-1} e_1^\top R(\theta)^n w_{t-n},$$

where $e_1 = \begin{bmatrix} 1 & 0 \end{bmatrix}$.

**Calculating the Expected Value:**

$$E[z_{1,t} z_{1,t-k}] = \sigma_w^2 \sum_{n=k}^{t-1} e_1^\top R(\theta)^n R(\theta)^{n-k} e_1 = \sigma_w^2 \sum_{n=k}^{t-1} e_1^\top R(\theta)^k e_1.$$

**Simplifying the Expression:**

Since $R(\theta)^k$ represents a rotation by $k \theta$, we have:
$$e_1^\top R(\theta)^k e_1 = \cos(k \theta).$$

The sum over $n$ results in $(t - k)$ terms.

**Final Autocorrelation Function:**

For $k \geq 1$:

$$E[y_t y_{t-k}] = \sigma_w^2 (t - k) \cos(k \theta).$$

For $k = 0$:

$$E[y_t^2] = \sigma_w^2 t + \sigma_v^2.$$

**Interpretation:**

The autocorrelation function depends on time $t$ and increases without bound, reflecting the non-stationarity of the process.
The cosine term $\cos(k \theta)$ introduces an oscillatory component based on the rotation angle $\theta$.

#### (e) Discuss how the choice of $\theta$ affects the dynamics of the system. Specifically, consider the cases where $\theta = 0$, $\theta = \frac{\pi}{2}$, $\theta = \pi$, and $\theta$ is an irrational multiple of $\pi$. How does the rotation angle $\theta$ influence the properties of the model?

**Answer:**

**General Effect of $\theta$:**

The rotation angle $\theta$ determines how the state vector $z_t$ rotates in the plane at each time step.
Different values of $\theta$ lead to different rotational dynamics, affecting the path of $z_t$ but not the rate at which the covariance increases.

**Specific Cases:**

- **$\theta = 0$:**

  - Rotation Matrix: $R(0) = I$ (identity matrix).
  - Dynamics: The state updates as $z_t = z_{t-1} + w_t$.
  - Behavior: The system becomes a 2D random walk with no rotation.
  - Variance Growth: The covariance increases linearly over time due to the accumulation of noise.

- **$\theta = \frac{\pi}{2}$:**

  - Rotation Matrix: $R\left(\frac{\pi}{2}\right) = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$.
  - Dynamics: Each state vector is rotated by 90 degrees at each time step.
  - Behavior: The state vector cycles through four orientations but accumulates noise.
  - Variance Growth: Despite the periodic rotation, the covariance still increases over time.

- **$\theta = \pi$:**

  - Rotation Matrix: $R(\pi) = \begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix}$.
  - Dynamics: The state vector is rotated by 180 degrees (reflected through the origin).
  - Behavior: Alternating sign of the state vector at each step.
  - Variance Growth: The covariance increases over time due to noise.

- **$\theta$ is an Irrational Multiple of $\pi$:**

  - Dynamics: The rotations do not repeat periodically.
  - Behavior: The state vector explores the plane more uniformly without repeating patterns.
  - Variance Growth: The covariance still increases over time, leading to non-stationarity.

**Conclusion:**

- **Influence of $\theta$:** While $\theta$ affects the rotational dynamics and the path of the state vector, it does not prevent the covariance from increasing.
- **Common Outcome:** In all cases, the additive noise causes the covariance to grow without bound, making the process non-stationary.
- **Implication for Modeling:** The choice of $\theta$ should be based on the desired rotational behavior, but additional measures (like damping) are needed to achieve stationarity.

#### (f) You have access to an off-the-shelf implementation of the Kalman filter. Explain how you would use it to perform state estimation in this model. Describe any modifications or considerations needed due to the properties of the rotation matrix.

**Answer:**

**Applying the Kalman Filter:**

Despite the non-stationarity, the Kalman filter can be used for state estimation at each time step.

**Kalman Filter Equations:**

- **Prediction Step:**
  - **State Prediction:**
    $$\hat{z}_{t|t-1} = R(\theta) \hat{z}_{t-1|t-1}.$$
  - **Covariance Prediction:**
    $$P_{t|t-1} = R(\theta) P_{t-1|t-1} R(\theta)^\top + Q.$$

- **Update Step:**
  - **Kalman Gain:**
    $$K_t = P_{t|t-1} h(h^\top P_{t|t-1} h + \sigma_v^2)^{-1}.$$
  - **State Update:**
    $$\hat{z}_{t|t} = \hat{z}_{t|t-1} + K_t(y_t - h^\top \hat{z}_{t|t-1}).$$
  - **Covariance Update:**
    $$P_{t|t} = (I - K_t h^\top) P_{t|t-1}.$$

**Considerations Due to Rotation Matrix Properties:**

- **Orthogonality of $R(\theta)$:** Since $R(\theta)$ is orthogonal, the predicted covariance $P_{t|t-1}$ tends to increase due to the additive noise $Q$.
- **Non-Stationarity:** The covariance matrices $P_{t|t-1}$ and $P_{t|t}$ will grow over time, which can lead to numerical instability.

**Modifications and Practical Steps:**

- **Initialization:**
  - Choose an initial state estimate $\hat{z}_{0|0}$ and covariance $P_{0|0}$ based on prior knowledge.
  - If no prior knowledge is available, initialize with $\hat{z}_{0|0} = 0$ and a large $P_{0|0}$ to reflect uncertainty.

- **Numerical Stability:**
  - **Regularization:** Introduce regularization techniques to prevent $P_{t|t}$ from becoming numerically unstable (e.g., limiting the maximum covariance).
  - **Covariance Resetting:** Periodically reset the covariance matrix to prevent it from growing indefinitely.

- **Implementation Details:**
  - Ensure that matrix operations account for potential numerical issues due to increasing covariance.
  - Use data types with sufficient precision to handle large numbers.

- **Interpretation of Results:**
  - Be aware that the increasing covariance reflects growing uncertainty in the state estimates over time.
  - Interpret the state estimates accordingly, especially in long-term predictions.

**Conclusion:**

The Kalman filter can be effectively applied to estimate the states in this model on a step-by-step basis.
Due to the non-stationary nature of the process, special attention must be paid to numerical stability and interpretation of the increasing uncertainty.
No fundamental modifications to the Kalman filter equations are needed, but practical adjustments are essential to handle the properties of the rotation matrix and the additive noise.

### Summary:

- The model is non-stationary because the state covariance grows without bound due to additive noise and the rotational dynamics.
- The stationary covariance matrix does not exist for $\sigma_w^2 > 0$.
- The autocorrelation function of the observations increases over time, reflecting non-stationarity.
- The rotation angle $\theta$ influences the trajectory of the state vector but does not prevent variance growth.
- The Kalman filter can be used for state estimation with considerations for numerical stability and increasing uncertainty.
