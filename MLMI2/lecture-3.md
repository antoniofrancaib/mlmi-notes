# Lecture 3: HMM-Based Speech Recognition—HMM Training and Gaussian Mixture Models

## Overview
In this lecture, we delve deeper into the training of Hidden Markov Models (HMMs) for automatic speech recognition (ASR). We focus on maximum likelihood techniques to estimate the parameters of HMMs, particularly when using multivariate Gaussian distributions and Gaussian Mixture Models (GMMs) for the output probabilities. We will explore the following key topics:

- Viterbi Training using the most likely state sequence.
- The Forward-Backward Algorithm and Baum-Welch Re-estimation.
- Extending output distributions to Gaussian Mixture Models (GMM-HMMs).
- Estimating parameters for GMM-HMMs.

By the end of this lecture, you should have a comprehensive understanding of how to train HMMs effectively and how to handle the complexities introduced by using GMMs as output distributions.

---

## 1. Introduction to HMM Parameter Estimation

### 1.1 Importance of Parameter Estimation
In HMM-based speech recognition, accurately estimating the model parameters is crucial for the performance of the ASR system. The parameters include:

- **Transition Probabilities** ($a_{ij}$): The probabilities of transitioning from state $i$ to state $j$.
- **Emission Probabilities** ($b_j(o)$): The probabilities of observing a feature vector $o$ given that the system is in state $j$.
- **Mean Vectors** ($\mu_j$) and **Covariance Matrices** ($\Sigma_j$): Parameters of the Gaussian distributions used in the emission probabilities.

Our goal is to find the parameter values that maximize the likelihood of the observed data, a process known as **Maximum Likelihood Estimation (MLE)**.

---

## 2. Viterbi Training

### 2.1 Concept of Viterbi Training
Viterbi training is an iterative procedure that estimates HMM parameters using the most likely state sequence obtained from the Viterbi algorithm.

**Key Idea**: Instead of considering all possible state sequences, we focus on the single most probable sequence $X^*$ that explains the observed data $O$.

**Approximation**:
$$P(O \mid \lambda) \approx P(O, X^* \mid \lambda)$$
This provides a lower bound on the total likelihood.

### 2.2 Steps in Viterbi Training

1. **Initialization**:
   Begin with initial estimates of the HMM parameters $\lambda$.

2. **Viterbi Alignment**:
   Use the Viterbi algorithm to find the most likely state sequence $X^*$ given the current parameters.
   - This sequence aligns each observation $o_t$ to a specific state $x^*(t)$.

3. **Parameter Update**:
   - **Transition Probabilities**:
     $$\hat{a}_{ij} = \frac{\text{Number of transitions from state } i \text{ to state } j}{\text{Total number of transitions from state } i}$$
   - **Emission Probabilities**:
     - Mean:
       $$\hat{\mu}_j = \frac{\sum_t \delta(x^*(t) = j) o_t}{\sum_t \delta(x^*(t) = j)}$$
     - Covariance:
       $$\hat{\Sigma}_j = \frac{\sum_t \delta(x^*(t) = j) (o_t - \hat{\mu}_j)(o_t - \hat{\mu}_j)^\top}{\sum_t \delta(x^*(t) = j)}$$
   - $\delta(\cdot)$ is an indicator function:
     $$\delta(x^*(t) = j) =
     \begin{cases} 
     1 & \text{if } x^*(t) = j \\ 
     0 & \text{otherwise}
     \end{cases}$$

4. **Iteration**:
   Repeat the alignment and parameter update steps until convergence (i.e., parameters no longer change significantly).

---

### 2.3 Convergence and Limitations
- **Convergence**: Each iteration increases the likelihood of the observed data given the model parameters.
- **Local Optima**: Viterbi training may converge to a local maximum rather than the global maximum, depending on the initial parameter estimates.
- **Dependency on Initial Estimates**: Different initializations can lead to different final models.

---

## 3. Baum-Welch Re-estimation

### 3.1 Limitations of Viterbi Training
- **Hard Alignment**: Viterbi training uses a hard decision about the state sequence, which may not capture the uncertainty inherent in the state assignments.
- **Ignores Alternative Paths**: By focusing only on the most probable state sequence, Viterbi training neglects other plausible sequences that could contribute to the likelihood.

### 3.2 Introducing Baum-Welch Algorithm
**Key Idea**: The Baum-Welch algorithm, based on the Expectation-Maximization (EM) framework, computes expected counts over all possible state sequences, weighted by their probabilities.

### 3.3 The Forward-Backward Algorithm

#### 3.3.1 Forward Probabilities ($\alpha_j(t)$)
**Definition**: The probability of observing the partial sequence $o_1, o_2, \dots, o_t$ and being in state $j$ at time $t$:
$$\alpha_j(t) = P(o_1, o_2, \dots, o_t, x(t) = j \mid \lambda)$$

- **Initialization**:
  $$\alpha_1(0) = 1, \quad \alpha_j(0) = 0 \text{ for } j \neq 1$$
- **Recursion**:
  $$\alpha_j(t) = \left(\sum_i \alpha_i(t-1) a_{ij}\right) b_j(o_t)$$

#### 3.3.2 Backward Probabilities ($\beta_j(t)$)
**Definition**: The probability of observing the remaining sequence $o_{t+1}, o_{t+2}, \dots, o_T$ given that the system is in state $j$ at time $t$:
$$\beta_j(t) = P(o_{t+1}, o_{t+2}, \dots, o_T \mid x(t) = j, \lambda)$$

- **Initialization**:
  $$\beta_j(T) = a_{jN} \quad \text{for all } j$$
  where $N$ is the non-emitting exit state.
- **Recursion**:
  $$\beta_j(t) = \sum_k a_{jk} b_k(o_{t+1}) \beta_k(t+1)$$

---

### 3.4 Computing State Occupation Probabilities

#### 3.4.1 State Posterior Probabilities ($\gamma_j(t)$)
**Definition**: The probability of being in state $j$ at time $t$ given the entire observation sequence $O$:
$$\gamma_j(t) = P(x(t) = j \mid O, \lambda) = \frac{\alpha_j(t) \beta_j(t)}{P(O \mid \lambda)}$$

**Interpretation**: Represents a soft alignment, assigning a probability to each state at each time frame.

---

#### 3.4.2 Transition Probabilities ($\xi_{ij}(t)$)
**Definition**: The probability of transitioning from state $i$ to state $j$ at time $t$:
$$\xi_{ij}(t) = P(x(t) = i, x(t+1) = j \mid O, \lambda) = \frac{\alpha_i(t) a_{ij} b_j(o_{t+1}) \beta_j(t+1)}{P(O \mid \lambda)}$$

---

### 3.5 Baum-Welch Re-estimation Formulas

#### 3.5.1 Transition Probabilities ($a_{ij}$)
**Update Rule**:
$$\hat{a}_{ij} = \frac{\sum_{r=1}^R \sum_{t=1}^{T^{(r)} - 1} \xi_{ij}^{(r)}(t)}{\sum_{r=1}^R \sum_{t=1}^{T^{(r)} - 1} \gamma_i^{(r)}(t)}$$
where $R$ is the number of training sequences, and $T^{(r)}$ is the length of the $r$-th sequence.

---

#### 3.5.2 Emission Probabilities (Gaussian Parameters)

- **Mean Vectors ($\mu_j$)**:
  $$\hat{\mu}_j = \frac{\sum_{r=1}^R \sum_{t=1}^{T^{(r)}} \gamma_j^{(r)}(t) o_t^{(r)}}{\sum_{r=1}^R \sum_{t=1}^{T^{(r)}} \gamma_j^{(r)}(t)}$$

- **Covariance Matrices ($\Sigma_j$)**:
  $$\hat{\Sigma}_j = \frac{\sum_{r=1}^R \sum_{t=1}^{T^{(r)}} \gamma_j^{(r)}(t) (o_t^{(r)} - \hat{\mu}_j)(o_t^{(r)} - \hat{\mu}_j)^\top}{\sum_{r=1}^R \sum_{t=1}^{T^{(r)}} \gamma_j^{(r)}(t)}$$

---

### 3.6 EM Algorithm Framework

1. **Expectation Step (E-step)**:
   - Compute $\gamma_j(t)$ and $\xi_{ij}(t)$ using the current model parameters.
   - These are the expected counts of state occupancies and transitions.

2. **Maximization Step (M-step)**:
   - Update the model parameters $\lambda$ using the re-estimation formulas to maximize the expected likelihood.

---

### 3.7 Advantages of Baum-Welch Training
- **Global Optimization**: Considers all possible state sequences, potentially leading to a better maximum likelihood estimate.
- **Convergence Guarantee**: Each iteration is guaranteed to increase (or maintain) the likelihood of the observed data unless a local maximum is reached.

---

## 4. Initialization Strategies

### 4.1 Importance of Initialization
The EM algorithm (Baum-Welch training) can converge to local maxima; therefore, the choice of initial parameters can significantly affect the final model. Good initialization can lead to faster convergence and better models.

---

### 4.2 Common Initialization Methods

#### 4.2.1 Flat Start
**Description**: Initialize all HMM parameters uniformly or with global statistics.

- **Procedure**:
  - Set all transition probabilities $a_{ij}$ to be equal or based on a predefined topology.
  - Initialize mean vectors $\mu_j$ to the global mean of the training data.
  - Initialize covariance matrices $\Sigma_j$ to the global covariance.

- **Advantages**:
  - Simple and does not require labeled data.
  - Useful when no prior models are available.

---

#### 4.2.2 Using Previous Models
**Description**: Use parameters from existing trained models as the starting point.

- **Procedure**:
  - Obtain models trained on similar data or tasks.
  - Adjust parameters if necessary to fit the new data.

- **Advantages**:
  - Leverages existing knowledge.
  - Often leads to better initial models than flat start.

---

#### 4.2.3 Using Labeled Data
**Description**: If phone-level or state-level labels are available, use them to initialize the models.

- **Procedure**:
  - Align the training data using the labels.
  - Compute initial estimates of parameters based on these alignments.

- **Advantages**:
  - Provides accurate initial alignments.
  - Can lead to faster convergence.

- **Limitations**:
  - Requires labeled data, which may not be available.

---

## 5. Handling Computational Issues

### 5.1 Underflow Problems
**Issue**: The forward and backward probabilities involve multiplying many small probabilities, leading to numerical underflow.

---

### 5.2 Solutions

#### 5.2.1 Scaling
- **Method**: Scale $\alpha_j(t)$ and $\beta_j(t)$ at each time step to prevent underflow.

- **Procedure**:
  - At each time $t$, compute a scaling factor $c(t)$ such that:
    $$c(t) = \frac{1}{\sum_j \alpha_j(t)}$$
  - Update the scaled forward variables:
    $$\tilde{\alpha}_j(t) = c(t) \alpha_j(t)$$
  - **Reconstruction**:
    $$P(O \mid \lambda) = \prod_{t=1}^T c(t)^{-1}$$

---

#### 5.2.2 Logarithmic Computation
- **Method**: Perform computations in the logarithmic domain to turn multiplications into additions.

- **Advantages**:
  - Prevents underflow and overflow.
  - The log-sum-exp trick can be used to compute $\log(\sum e^{x_i})$ in a numerically stable way.

---

## 6. Extending to Gaussian Mixture Models (GMMs)

### 6.1 Motivation
**Limitations of Single Gaussians**:
- Real-world data distributions are often multimodal and cannot be accurately modeled by a single Gaussian.
- Speech feature distributions may be skewed or have multiple clusters.

**Solution**: Use Gaussian Mixture Models to model the emission probabilities.

---

### 6.2 Gaussian Mixture Model Definition
**Emission Probability**:
$$b_j(o) = \sum_{m=1}^M c_{jm} N(o; \mu_{jm}, \Sigma_{jm})$$
where:
- $M$: Number of mixture components.
- $c_{jm}$: Mixture weights for state $j$ and component $m$, satisfying $\sum_{m=1}^M c_{jm} = 1$ and $c_{jm} \geq 0$.
- $N(o; \mu_{jm}, \Sigma_{jm})$: Multivariate Gaussian density.


### 6.3 Parameter Estimation for GMM-HMMs

#### 6.3.1 Component Responsibilities
**Definition**: The probability that component $m$ in state $j$ generated observation $o_t$:
$$\gamma_{jm}(t) = \frac{\gamma_j(t) c_{jm} N(o_t; \mu_{jm}, \Sigma_{jm})}{b_j(o_t)}$$

---

#### 6.3.2 Re-estimation Formulas
- **Mixture Weights ($c_{jm}$)**:
  $$\hat{c}_{jm} = \frac{\sum_{t=1}^T \gamma_{jm}(t)}{\sum_{t=1}^T \gamma_j(t)}$$

- **Means ($\mu_{jm}$)**:
  $$\hat{\mu}_{jm} = \frac{\sum_{t=1}^T \gamma_{jm}(t) o_t}{\sum_{t=1}^T \gamma_{jm}(t)}$$

- **Covariances ($\Sigma_{jm}$)**:
  $$\hat{\Sigma}_{jm} = \frac{\sum_{t=1}^T \gamma_{jm}(t) (o_t - \hat{\mu}_{jm})(o_t - \hat{\mu}_{jm})^\top}{\sum_{t=1}^T \gamma_{jm}(t)}$$

---

### 6.4 Training Strategies for GMM-HMMs

#### 6.4.1 Initialization Challenges
**Problem**: Cannot use flat start easily because the mixture components need to be initialized properly.

---

#### 6.4.2 Possible Solutions

##### 6.4.2.1 Clustering
- **Approach**:
  - Gather all observations assigned to a state.
  - Use clustering algorithms (e.g., K-means) to partition the data into $M$ clusters.
  - Initialize each Gaussian component with the mean and covariance of a cluster.

- **Advantages**:
  - Provides a good starting point for the EM algorithm.

---

##### 6.4.2.2 Incremental Mixture Splitting ("Mixing-Up")
- **Approach**:
  1. **Start with Single Gaussians**:
     - Train HMMs with a single Gaussian per state using Baum-Welch re-estimation.
  2. **Split Components**:
     - Select components to split based on some criterion (e.g., high variance or high occupancy).
     - Perturb the means slightly to create two new components.
  3. **Re-estimate Parameters**:
     - Retrain the model using Baum-Welch to adjust the parameters.
  4. **Iterate**:
     - Repeat splitting and retraining until the desired number of mixture components is reached.

- **Advantages**:
  - Gradually increases model complexity.
  - Allows the model to adapt to the data progressively.

---

### 6.5 Model Complexity Considerations

- **Number of Parameters**:
  - **Single Gaussian**:
    - Mean: $d$ parameters.
    - Covariance: $\frac{d(d+1)}{2}$ parameters (full covariance).
  - **GMM with $M$ Components**:
    - Means: $M \times d$ parameters.
    - Covariances: $M \times d$ (assuming diagonal covariance).
    - Weights: $M-1$ parameters (since weights sum to 1).

- **Trade-off**:
  - **Model Capacity vs. Overfitting**: More components increase the model's ability to fit the data but risk overfitting.
  - **Computational Complexity**: More components require more computations during training and recognition.

---

## 7. Summary

- **Viterbi Training**:
  - Uses the most likely state sequence to estimate HMM parameters.
  - Provides a straightforward approach but may miss the contributions of other plausible state sequences.

- **Baum-Welch Re-estimation**:
  - An EM algorithm that considers all possible state sequences weighted by their probabilities.
  - Guarantees non-decreasing likelihood and often achieves better parameter estimates than Viterbi training.

- **Forward-Backward Algorithm**:
  - Efficiently computes the forward ($\alpha$) and backward ($\beta$) probabilities.
  - Essential for calculating the state occupation probabilities ($\gamma_j(t)$).

- **Initialization**:
  - Crucial for the convergence and performance of HMM training.
  - Flat start and using existing models are common strategies.

- **Gaussian Mixture Models**:
  - Extend HMMs to model more complex emission distributions.
  - Training GMM-HMMs requires careful initialization and may involve incremental strategies like mixture splitting.

- **Computational Considerations**:
  - Underflow issues are addressed through scaling or logarithmic computations.
  - Model complexity must be balanced against computational resources and overfitting risks.

---

## 8. Advanced Insights

### 8.1 Relationship to Expectation-Maximization (EM)
- **EM Algorithm**:
  - The Baum-Welch algorithm is a specific instance of the EM algorithm applied to HMMs.
  - The EM framework provides theoretical guarantees about convergence to local maxima.

---

### 8.2 Practical Considerations

- **Data Requirements**:
  - Larger models (e.g., with more mixture components) require more training data to estimate parameters reliably.

- **Regularization**:
  - Techniques like variance flooring can prevent covariance matrices from becoming singular.

- **Cross-Validation**:
  - Using a development set to monitor performance can help determine when to stop training or when to increase model complexity.

---

## 9. Conclusion

Understanding the training of HMMs is fundamental to developing effective ASR systems. Both Viterbi training and Baum-Welch re-estimation have their places, with the latter providing a more robust approach through soft alignments and consideration of all possible state sequences. Extending HMMs with Gaussian Mixture Models enhances their expressive power, allowing them to model complex, real-world data distributions more accurately. However, this comes with increased computational demands and the need for more sophisticated training strategies.

---

### Recommended Next Steps:

1. **Explore Discriminative Training Methods**:
   - Investigate techniques like Maximum Mutual Information (MMI) and Minimum Phone Error (MPE) to improve model performance.

2. **Study Context-Dependent Modeling**:
   - Capture the influence of neighboring phonetic contexts on speech sounds.

3. **Investigate Deep Neural Network Integration**:
   - Study how DNN-HMM hybrids can enhance acoustic modeling.
