# Lecture 2: Introduction to Hidden Markov Models for Speech Recognition

## Overview
In this lecture, we delve into the fundamental role of Hidden Markov Models (HMMs) in speech recognition. We explore the structure and assumptions underlying HMMs, discuss feature extraction techniques crucial for acoustic modeling, and introduce key algorithms such as the Viterbi algorithm used for decoding and training. By the end, you should have a deep understanding of how HMMs model speech and how they are applied in automatic speech recognition (ASR) systems.

## 1. Hidden Markov Models in Speech Recognition

### 1.1 The Generative Model of Speech
An HMM is a statistical model that represents a system assumed to be a Markov process with unobserved (hidden) states. In the context of speech recognition, HMMs are used to model the sequence of speech sounds as a sequence of hidden states that generate observable acoustic features.

- **Observation Sequence ($O$):** A sequence of feature vectors extracted from the speech signal (e.g., Mel-Frequency Cepstral Coefficients (MFCCs)).
- **Hidden States ($X$):** The underlying sequence of phonetic or word units that generated the observations, which are not directly observable.

### 1.2 Structure of an HMM
An HMM is characterized by:
- **States:** A finite set of states $\{1, 2, \dots, N\}$, where each state corresponds to a particular segment of speech.
  - **Emitting States:** States that generate observations; typically $N-2$ states.
  - **Non-Emitting Entry State:** An initial state that does not emit observations, used to model the beginning of the sequence.
  - **Non-Emitting Exit State:** A final state that does not emit observations, used to model the end of the sequence.

- **State Transition Probabilities ($a_{ij}$):**
  - The probability of transitioning from state $i$ to state $j$.
  - Defined by a matrix $A = [a_{ij}]$, where $a_{ij} = P(x_t = j \mid x_{t-1} = i)$.

- **Observation Probability Distributions ($b_j(o)$):**
  - For each emitting state $j$, there is a probability distribution $b_j(o)$ that defines the likelihood of emitting observation $o$ when in state $j$.
  - In speech recognition, $b_j(o)$ is often modeled using Gaussian distributions or mixtures thereof.

- **Initial State Distribution:**
  - The probability of starting in a particular state, often implicitly defined by the non-emitting entry state.

### 1.3 The Hidden Aspect
The term "hidden" in HMM refers to the fact that the sequence of states $X = \{x(1), x(2), \dots, x(T)\}$ is not directly observable. We only observe the sequence of acoustic feature vectors $O = \{o_1, o_2, \dots, o_T\}$, and we infer the most probable state sequence that could have generated these observations.

## 2. HMM Assumptions and Likelihood Calculation

### 2.1 Key Assumptions
HMMs rely on two critical assumptions:
1. **First-Order Markov Assumption:**
   - The probability of transitioning to the next state depends only on the current state and not on any previous states.
   - Mathematically: $P(x_t \mid x_{t-1}, x_{t-2}, \dots, x_1) = P(x_t \mid x_{t-1})$.
   - **Limitation:** In speech, this assumption is not entirely accurate because the production of sounds often depends on a broader context (e.g., co-articulation effects).

2. **State Conditional Independence Assumption:**
   - The observation at time $t$ depends only on the current state and is conditionally independent of past and future observations.
   - Mathematically: $P(o_t \mid x_t, o_{t-1}, o_{t-2}, \dots, o_1) = P(o_t \mid x_t)$.
   - **Limitation:** Speech signals exhibit temporal dependencies; the current sound is influenced by previous and upcoming sounds.

Despite these limitations, HMMs have been widely used in speech recognition due to their mathematical tractability and the availability of efficient algorithms for training and decoding.

### 2.2 Joint Probability and Likelihood Calculation
Given the observations $O$ and a state sequence $X$, the joint probability $P(O, X \mid \lambda)$ (where $\lambda$ represents the model parameters) is calculated as:
$$
P(O, X \mid \lambda) = a_{x(0), x(1)} \left( \prod_{t=1}^T b_{x(t)}(o_t) a_{x(t), x(t+1)} \right)
$$
- $x(0)$: The entry (non-emitting) state.
- $x(T+1)$: The exit (non-emitting) state.

The product runs over all time steps $t$ from $1$ to $T$.

To find the likelihood $P(O \mid \lambda)$ of the observation sequence given the model, we need to sum over all possible state sequences:
$$
P(O \mid \lambda) = \sum_X P(O, X \mid \lambda)
$$
However, directly summing over all possible state sequences is computationally infeasible due to the exponential number of possible sequences.

### 2.3 Efficient Likelihood Computation
To compute $P(O \mid \lambda)$ efficiently, we use algorithms that exploit the independence assumptions of HMMs:
- **Forward Algorithm:** Recursively computes the total likelihood by summing over all possible state sequences.
- **Viterbi Algorithm:** Finds the most probable state sequence, which can be used as an approximation for the likelihood.

---
## 3. Output Probability Distributions

### 3.1 Multivariate Gaussian Distributions
A common choice for the output distribution $b_j(o)$ is the multivariate Gaussian distribution:
$$
b_j(o) = N(o; \mu_j, \Sigma_j) = \frac{1}{(2\pi)^{n/2} |\Sigma_j|^{1/2}} \exp\left(-\frac{1}{2}(o - \mu_j)^T \Sigma_j^{-1} (o - \mu_j)\right)
$$
- $\mu_j$: Mean vector of state $j$.
- $\Sigma_j$: Covariance matrix of state $j$.

### 3.2 Diagonal Covariance Matrices
Using full covariance matrices can be computationally intensive and may require a large amount of training data to estimate reliably. To simplify, we often assume that the features are uncorrelated, allowing us to use diagonal covariance matrices. This reduces the number of parameters and computational complexity.

### 3.3 Limitations with Gaussian Distributions
- **Non-Gaussian Nature of Speech:** Speech spectral features are not strictly Gaussian distributed.
- **Feature Correlations:** Elements of feature vectors are often correlated, which violates the assumption required for diagonal covariance matrices.

To address these limitations, we preprocess the features to reduce correlations and make the distributions more Gaussian-like.

## 4. Feature Extraction: Cepstral Features and Differential Features

### 4.1 Mel-Scale Filterbanks
To process speech signals for HMMs, we extract features that are suitable for modeling. The Mel-scale filterbank reduces frequency resolution and mimics the human ear's spectral resolution.

#### 4.1.1 Mel Scale
- **Definition:** The Mel scale is a perceptual scale of pitches judged by listeners to be equal in distance from one another.
- **Formula:** The Mel frequency $m$ corresponding to a frequency $f$ in Hz is given by:
$$
m = 2595 \log_{10}(1 + \frac{f}{700})
$$
- **Characteristics:** The Mel scale is approximately linear up to 1 kHz and logarithmic thereafter.

#### 4.1.2 Filterbank Processing
Process:
1. Compute the Short-Time Fourier Transform (STFT) of the speech signal to obtain the magnitude spectrum.
2. Apply Mel-Scale Filterbanks: Multiply the magnitude spectrum by a set of triangular filters spaced according to the Mel scale.
3. Compute Log Energies: Sum the energy in each filter and take the logarithm.

### 4.2 Discrete Cosine Transform (DCT)
The cepstral coefficients are derived from the log filterbank energies using the DCT, which serves to decorrelate the features and compact the energy into a small number of coefficients.

- **Formula:**
$$
c_n = \sum_{i=1}^P m_i \cos\left(n\left(i - \frac{1}{2}\right)\frac{\pi}{P}\right)
$$
  - $P$: Number of filterbank channels.
  - $m_i$: Log energy in the $i$-th filterbank channel.
  - $c_n$: The $n$-th cepstral coefficient.

The resulting coefficients are known as Mel-Frequency Cepstral Coefficients (MFCCs).

### 4.3 Advantages of MFCCs
- **Decorrelation:** The DCT reduces the correlation between features, allowing for diagonal covariance matrices in Gaussian models.
- **Dimensionality Reduction:** By selecting a subset of the DCT coefficients (e.g., the first 12), we reduce the number of parameters needed to represent a frame of speech.
- **Perceptual Relevance:** The Mel scale approximates the human auditory system's response, enhancing recognition performance.

### 4.4 Delta and Delta-Delta Features
To capture the temporal dynamics of speech, we include time derivatives of the cepstral coefficients.

- **Delta Features ($\Delta y_t$):** Approximate the first derivative (velocity) of the cepstral coefficients.
- **Delta-Delta Features ($\Delta^2 y_t$):** Approximate the second derivative (acceleration).

#### 4.4.1 Computation
- **Delta Features:**
$$
\Delta y_t = \frac{\sum_{n=1}^N n(y_{t+n} - y_{t-n})}{2\sum_{n=1}^N n^2}
$$
  - $N$: Typically 2 or 3 frames on each side.

- **Delta-Delta Features:**
$$
\Delta^2 y_t = \frac{\sum_{n=1}^N n(\Delta y_{t+n} - \Delta y_{t-n})}{2\sum_{n=1}^N n^2}
$$

### 4.5 Final Feature Vector
Combining the static, delta, and delta-delta features, the final feature vector at time $t$ is:
$$
o_t = \begin{bmatrix} y_t \\ \Delta y_t \\ \Delta^2 y_t \end{bmatrix}
$$
For 12 MFCCs and log-energy, including their delta and delta-delta coefficients, we obtain a 39-dimensional feature vector.

### 4.6 Normalization Techniques
- **Cepstral Mean Normalization (CMN):** Subtracting the mean cepstral vector over an utterance to reduce the effects of static channels or microphones.
- **Energy Normalization:** The log-energy feature is often normalized to account for variations in signal amplitude.

---

## 5. Isolated Word HMM Example: Training

### 5.1 Training HMMs for Isolated Words
In an isolated word recognition system, each word in the vocabulary has a corresponding HMM.

#### 5.1.1 Steps
1. **Data Collection:**
   - Collect multiple spoken examples of each word.
   - Ensure variability in speakers and conditions to improve generalization.

2. **Model Initialization:**
   - **Topology Selection:** Decide on the number of states and the allowable transitions (e.g., left-to-right HMMs for speech).
   - **Parameter Initialization:** Initialize the state transition probabilities and output distributions, possibly using uniform probabilities or heuristics.

3. **Parameter Estimation:**
   - **Maximum Likelihood Estimation:** Adjust the model parameters to maximize the likelihood of the training data.
   - **Algorithms:**
     - **Viterbi Training (Segmental K-Means):** Uses the most likely state sequence for each training example to update parameters.
     - **Baum-Welch Algorithm:** Considers all possible state sequences weighted by their probabilities.

### 5.2 Recognition Process
To recognize an unknown isolated word:
1. **Feature Extraction:** Convert the speech signal into a sequence of feature vectors $O$.
2. **Model Evaluation:**
   - For each word model $M_i$, compute the likelihood $P(O \mid M_i)$.
   - Use the Viterbi algorithm to find the most probable state sequence and associated likelihood.
3. **Decision Making:**
   - **Equal Priors:** If all words are equally likely, select the word with the highest likelihood.
   - **Prior Probabilities:** If some words are more likely than others, incorporate prior probabilities.

## 6. The Viterbi Algorithm: Finding the Best State Sequence

### 6.1 Purpose
The Viterbi algorithm efficiently computes the most probable state sequence $X^*$ and its associated probability $P^* = P(O, X^* \mid \lambda)$ for a given HMM $\lambda$ and observation sequence $O$.

### 6.2 Algorithm Steps
1. **Initialization:**
   - Set initial probabilities:
     $$
     \phi_j(0) = 
     \begin{cases} 
     1 & \text{if } j = \text{entry state} \\ 
     0 & \text{otherwise} 
     \end{cases}
     $$
   - For $t = 1$ and each state $j$:
     $$
     \phi_j(1) = \phi_{\text{entry}}(0) \cdot a_{\text{entry}, j} \cdot b_j(o_1)
     $$

2. **Recursion:**
   - For $t = 2$ to $T$ and each state $j$:
     $$
     \phi_j(t) = \max_i \left[ \phi_i(t-1) \cdot a_{ij} \right] \cdot b_j(o_t)
     $$
   - Store the predecessor state:
     $$
     \text{pred}_j(t) = \arg\max_i \left[ \phi_i(t-1) \cdot a_{ij} \right]
     $$

3. **Termination:**
   - Compute the final probability:
     $$
     P^* = \max_i \left[ \phi_i(T) \cdot a_{i, \text{exit}} \right]
     $$
   - Identify the last state:
     $$
     x_T^* = \arg\max_i \left[ \phi_i(T) \cdot a_{i, \text{exit}} \right]
     $$

4. **Backtracking:**
   - For $t = T-1$ down to $1$:
     $$
     x_t^* = \text{pred}_{x_{t+1}^*}(t+1)
     $$

### 6.3 Visualization: The Trellis
- **Trellis Diagram:** A lattice structure where each column corresponds to a time frame $t$ and each node represents a state $j$ at that time.
- **Paths:** Each path through the trellis represents a possible state sequence.
- **Best Path:** The Viterbi algorithm finds the path with the highest probability.

### 6.4 Computational Considerations
- **Log Domain Computation:** To avoid underflow, calculations are often performed in the log domain.
- **Storage:** Only the predecessor state needs to be stored at each step for backtracking, reducing memory requirements.

## 7. Composite HMMs and Continuous Speech Recognition

### 7.1 Building Composite HMMs
Composite HMMs are constructed by concatenating smaller HMMs (e.g., phoneme or word models) to form larger models capable of recognizing continuous speech.

- **Advantages:**
  - **Modularity:** Reuse smaller units to build larger models.
  - **Flexibility:** Easily incorporate new words by adding their models.

- **Implementation:**
  - Use non-emitting entry and exit states to facilitate seamless concatenation.
  - Define transition probabilities between models to incorporate language modeling.

### 7.2 Recognition with Composite Models
For continuous speech recognition:
1. **Network Construction:**
   - Build a network representing all possible word sequences, guided by a vocabulary and a language model.
   - Each word is represented by its HMM, connected according to possible word sequences.

2. **Viterbi Search:**
   - Apply the Viterbi algorithm over the entire network to find the most probable path corresponding to the recognized word sequence.
   - Incorporate language model probabilities into the transition probabilities between word models.

3. **Output:**
   - The best path provides not only the recognized words but also their time alignments and the most likely state sequences.

### 7.3 Example: Simple Word Loop
- **Structure:**
  - A loop where any word can follow any other word.
  - Each word model is connected to every other word model, possibly with transition probabilities based on bigram statistics.

- **Sub-Word Units:**
  - Instead of modeling words directly, model sub-word units like phonemes.
  - Use a pronunciation dictionary to map words to sequences of sub-word units.

## 8. Viterbi Algorithm Efficiency and Implementation

### 8.1 Computational Efficiency
- **Complexity:**
  - The Viterbi algorithm operates in $O(N^2T)$ time, where $N$ is the number of states and $T$ is the length of the observation sequence.
  - Due to the sparse structure of HMMs (e.g., left-to-right models), the actual computation is often linear in $T$.

- **Optimizations:**
  - **Beam Search Pruning:** Discard paths with probabilities significantly lower than the best path at each time step.
  - **Token Passing:** A method to manage hypotheses efficiently in large networks.

### 8.2 Dealing with Underflow
- **Logarithmic Computation:**
  - Convert probabilities to log-probabilities.
  - Replace multiplication with addition:
    $$
    \log(ab) = \log a + \log b
    $$
  - Replace maximization appropriately:
    $$
    \max(\log a, \log b)
    $$

### 8.3 Search Errors
- **Trade-offs:**
  - Pruning increases efficiency but risks discarding the correct path (search errors).
  - A balance must be struck between computational resources and recognition accuracy.

## 9. Maximum Likelihood Estimation for HMMs

### 9.1 Objective
Estimate the model parameters $\lambda$ that maximize the likelihood of the training data:
$$
\hat{\lambda} = \arg\max_\lambda \prod_{r=1}^R P(O^{(r)} \mid \lambda)
$$
- $R$: Number of training utterances.
- $O^{(r)}$: Observation sequence for the $r$-th utterance.

### 9.2 Iterative Estimation Algorithms
Due to the hidden state sequences, parameter estimation is performed iteratively.

#### 9.2.1 Viterbi Training (Segmental K-Means)
- **Approach:** Use the most likely state sequence $X^*$ obtained from the Viterbi algorithm. Update parameters based on this alignment.
- **Process:**
  1. **Initialization:** Start with initial parameters $\lambda$.
  2. **Alignment:** Use Viterbi decoding to find $X^*$ for each training example.
  3. **Update:** Re-estimate parameters $\lambda$ based on the alignments.
  4. **Iteration:** Repeat until convergence.

- **Limitation:** May converge to local maxima and does not consider all possible state sequences.

#### 9.2.2 Baum-Welch Algorithm (Forward-Backward)
- **Approach:** An Expectation-Maximization (EM) algorithm that considers all possible state sequences weighted by their probabilities.
- **Process:**
  1. **E-Step:** Compute the expected number of times each state is occupied and each transition is used using forward and backward probabilities.
  2. **M-Step:** Update the model parameters $\lambda$ to maximize the expected likelihood.

- **Advantages:** Globally maximizes the likelihood given the model structure and is more robust than Viterbi training.

### 9.3 EM Algorithm Framework
- **E-Step:** Estimate the expected values of the hidden variables (state occupancies and transitions) given the current parameters.
- **M-Step:** Update the parameters to maximize the expected likelihood.

## 10. Summary
HMMs in ASR:
- HMMs are foundational models in speech recognition, modeling the sequential nature of speech and accounting for variability in both time and spectral characteristics.
- **Key Assumptions:** While HMMs make simplifying assumptions (first-order Markov and state conditional independence), they provide a practical framework for speech modeling.
- **Feature Extraction:** Cepstral features, especially MFCCs, and their derivatives are crucial for effective acoustic modeling.
- **Viterbi Algorithm:** An essential tool for decoding and training, providing efficient computation of the most probable state sequences.
- **Composite Models:** Building larger models from smaller units allows for scalable and flexible speech recognition systems.
- **Training Methods:** Both Viterbi training and Baum-Welch algorithms are used for parameter estimation, with trade-offs between computational efficiency and robustness.

By thoroughly understanding these concepts, you are well-equipped to delve deeper into advanced topics in speech recognition, such as Gaussian Mixture Models (GMM-HMMs), context-dependent modeling, and neural network-based acoustic models (DNN-HMMs).
