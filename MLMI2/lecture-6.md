# Lecture 6: Recurrent Neural Networks & Connectionist Temporal Classification in Speech Recognition

## Introduction
Automatic Speech Recognition (ASR) has evolved significantly with the advent of deep learning, particularly with the use of Recurrent Neural Networks (RNNs). RNNs are designed to model sequential data, making them well-suited for processing speech, which is inherently a time-varying signal. 

We will begin by understanding the limitations of traditional feed-forward Deep Neural Networks (DNNs) in capturing temporal dependencies in speech. Then, we will explore how RNNs, especially Long Short-Term Memory (LSTM) networks, address these challenges. We will also discuss various enhancements and alternatives to LSTMs, such as Gated Recurrent Units (GRUs) and higher-order RNNs. Finally, we will examine how CTC enables training of speech recognition models without explicit alignment information, paving the way for end-to-end ASR systems.

---

## 1. Recurrent Neural Networks in Acoustic Modeling

### 1.1. Limitations of Feed-Forward DNNs
Feed-forward DNNs have been widely used in ASR for acoustic modeling. However, they process input data independently, lacking the ability to model ***temporal dependencies*** across time frames. Speech signals are continuous and context-dependent; thus, capturing temporal dynamics is crucial for accurate recognition.

### 1.2. Introduction to Recurrent Neural Networks
Recurrent Neural Networks (RNNs) are a class of neural networks that introduce loops within the network architecture, allowing information to persist over time. This property makes RNNs capable of modeling sequential data and capturing temporal dependencies in speech.

In the context of ASR, RNNs process one input frame at a time while maintaining a hidden state that carries information from previous time steps. This hidden state serves as a form of memory, enabling the network to use historical context when making predictions.

---

## 2. Structure of Basic Recurrent Neural Networks

### 2.1. Architecture of RNNs
An RNN consists of input, hidden, and output layers. Unlike feed-forward networks, the hidden layer in an RNN is connected not only to the next layer but also to itself from the previous time step. Mathematically, the hidden state $h_t$ at time $t$ is computed as:

$$
h_t = f(W_{ih}x_t + W_{hh}h_{t-1} + b_h)
$$

- $x_t$: Input at time $t$
- $h_{t-1}$: Hidden state from previous time step
- $W_{ih}$: Input-to-hidden weight matrix
- $W_{hh}$: Hidden-to-hidden weight matrix (recurrent weights)
- $b_h$: Bias term
- $f$: Activation function (e.g., $tanh$, $ReLU$)

### 2.2. Advantages over Feed-Forward Networks
The recurrent connections in RNNs enable the network to maintain a form of memory, allowing it to model temporal dependencies in the input data. This is particularly beneficial for speech recognition, where context over time significantly influences the interpretation of sounds.

### 2.3. Unfolding RNNs in Time
To understand the training of RNNs, it is helpful to visualize the network as being "unfolded" over time. In this representation, each time step is considered a layer in a deep feed-forward network, with shared weights across these layers. This unfolding allows us to apply backpropagation through time (BPTT) for training.

![[Pasted image 20241120142604.png]]
### 2.4. Bidirectional RNNs
Bidirectional RNNs (BiRNNs) extend the basic RNN architecture by processing the input sequence in both forward and backward directions. This means the network has two hidden states at each time step:

- $h_t^{\rightarrow}$: Forward hidden state processing from $t=1$ to $T$
- $h_t^{\leftarrow}$: Backward hidden state processing from $t=T$ to $1$

The final output at each time step can be a function of both $h_t^{\rightarrow}$ and $h_t^{\leftarrow}$, providing the network with both past and future context.

---

## 3. Training Recurrent Neural Networks

### 3.1. Backpropagation Through Time (BPTT)
Training RNNs involves adjusting the weights to minimize a loss function, similar to other neural networks. However, due to the recurrent connections, the standard backpropagation algorithm must be extended to handle the temporal dependencies. This is achieved through Backpropagation Through Time (BPTT).

BPTT involves unfolding the RNN over time and computing gradients at each time step. The gradients are then propagated backward through the network layers and time steps, updating the weights accordingly.

### 3.2. Challenges in Training RNNs

#### 3.2.1. Vanishing and Exploding Gradients
A significant challenge in training RNNs is the vanishing or exploding gradient problem. During BPTT, gradients can exponentially decay or grow as they are propagated through time steps, especially over long sequences. This makes it difficult for the network to learn long-term dependencies.

- **Vanishing Gradient**: The gradient diminishes to near zero, preventing the network from learning from earlier time steps.
- **Exploding Gradient**: The gradient becomes excessively large, leading to numerical instability.

#### 3.2.2. Mitigation Strategies
To address these issues:
- **Gradient Clipping**: Limits the maximum value of the gradients to prevent exploding gradients.
- **Use of Activation Functions**: Choosing activation functions like $ReLU$ can help mitigate the vanishing gradient problem.
- **Truncated BPTT**: Limits the number of time steps over which gradients are backpropagated, reducing computational complexity and mitigating vanishing gradients.

### 3.3. Truncated Backpropagation Through Time
Truncated BPTT involves unfolding the RNN for a fixed number of time steps (e.g., 20 frames) during training. This allows the network to learn dependencies within this window while keeping the computational requirements manageable.


---

## 4. Long Short-Term Memory (LSTM) Models

### 4.1. Motivation for LSTMs
While basic RNNs struggle with learning long-term dependencies due to the *vanishing gradient problem*, Long Short-Term Memory (LSTM) networks are designed to overcome this limitation. LSTMs introduce a more complex memory cell structure with gating mechanisms that control the flow of information.

### 4.2. LSTM Architecture
An LSTM network replaces the standard RNN hidden unit with a memory cell that maintains its state over time. Each memory cell has the following components:

- **Cell State ($c_t$)**: Represents the internal memory of the cell.
- **Hidden State ($h_t$)**: Output of the LSTM cell at time $t$.
- **Input Gate ($i_t$)**: Controls the extent to which new information flows into the cell.
- **Forget Gate ($f_t$)**: Determines what information to discard from the cell state.
- **Output Gate ($o_t$)**: Controls the output of the cell.

### 4.3. LSTM Equations
The LSTM computations at each time step $t$ are as follows:

- **Input Gate**:
  $$
  i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)
  $$

- **Forget Gate**:
  $$
  f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)
  $$

- **Cell Candidate**:
  $$
  \tilde{c}_t = \tanh(W_c[h_{t-1}, x_t] + b_c)
  $$

- **Cell State Update**:
  $$
  c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
  $$

- **Output Gate**:
  $$
  o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)
  $$

- **Hidden State**:
  $$
  h_t = o_t \odot \tanh(c_t)
  $$

Where:
- $\sigma$: Sigmoid activation function
- $\tanh$: Hyperbolic tangent activation function
- $\odot$: Element-wise multiplication
- $W_i, W_f, W_c, W_o$: Weight matrices
- $b_i, b_f, b_c, b_o$: Bias vectors

![[Pasted image 20241120142835.png]]
### 4.4. Gates and Their Functions
- **Input Gate ($i_t$)**: Determines how much new information from the current input and previous hidden state should be added to the cell state.
- **Forget Gate ($f_t$)**: Decides which information from the previous cell state $c_{t-1}$ should be retained or forgotten.
- **Output Gate ($o_t$)**: Controls the amount of information from the cell state $c_t$ that is exposed to the output $h_t$.

### 4.5. Computation Steps in LSTM
At each time step:
1. Compute the input, forget, and output gates using the current input $x_t$ and previous hidden state $h_{t-1}$.
2. Generate a candidate cell state $\tilde{c}_t$ based on $x_t$ and $h_{t-1}$.
3. Update the cell state $c_t$ by combining the previous cell state $c_{t-1}$ (modulated by the forget gate) and the candidate cell state $\tilde{c}_t$ (modulated by the input gate).
4. Compute the hidden state $h_t$ by applying the output gate to the activated cell state $\tanh(c_t)$.

### 4.6. Advantages of LSTMs
- **Long-Term Dependency Learning**: LSTMs effectively capture long-range dependencies due to their gating mechanisms.
- **Mitigation of Vanishing Gradients**: The cell state provides a path for gradients to flow backward over long sequences without vanishing.
- **Flexibility**: LSTMs can be stacked in multiple layers and combined with other network types (e.g., CNNs).

---

## 5. LSTM Enhancements and Alternatives

### 5.1. Peephole Connections
Peephole connections allow the gates to access the cell state directly, providing additional context. The modified equations for the gates are:

- **Input Gate with Peephole**:
  $$
  i_t = \sigma(W_i[h_{t-1}, x_t] + V_i \odot c_{t-1} + b_i)
  $$

- **Forget Gate with Peephole**:
  $$
  f_t = \sigma(W_f[h_{t-1}, x_t] + V_f \odot c_{t-1} + b_f)
  $$

- **Output Gate with Peephole**:
  $$
  o_t = \sigma(W_o[h_{t-1}, x_t] + V_o \odot c_t + b_o)
  $$

Where:
- $V_i, V_f, V_o$: Peephole weight vectors

### 5.2. Gated Recurrent Units (GRUs)
GRUs simplify the LSTM architecture by combining the input and forget gates into a single update gate and merging the cell and hidden states. The key equations are:

- **Update Gate**:
  $$
  z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
  $$

- **Reset Gate**:
  $$
  r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
  $$

- **Candidate Activation**:
  $$
  \tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)
  $$

- **Hidden State Update**:
  $$
  h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
  $$

### 5.3. LSTM Computation Reduction

#### 5.3.1. Projected LSTM (LSTMP)
LSTMP reduces the computational cost of LSTMs by introducing a projection layer that reduces the dimensionality of the hidden state:

- Compute LSTM as usual to obtain $h_t$.
- Project Hidden State:
  $$
  p_t = W_p h_t
  $$

Where $W_p$ is the projection matrix.

#### 5.3.2. Semi-Tied LSTM Units
Semi-tied units aim to reduce redundancy by sharing certain computations among gates. A common "virtual unit" $e_t$ is computed:

$$
e_t = \sigma(W[h_{t-1}, x_t] + b)
$$

The gates are then computed using parameterized activation functions based on $e_t$.

### 5.4. Higher-Order Recurrent Neural Networks (HORNNs)
HORNNs introduce connections from multiple previous time steps:

$$
h_t = f(Wx_t + U_1 h_{t-1} + U_n h_{t-n} + b)
$$

- **Projected HORNNs (HORNNP)**:
  $$
  h_t = f(Wx_t + U_{p1} P h_{t-1} + U_{pn} P h_{t-n} + b)
  $$

Where $P$ is a projection matrix.

---

## 6. Grid Recurrent Networks

### 6.1. Motivation
Speech signals have temporal and frequency dimensions. Grid recurrent networks aim to model dependencies in both time and frequency.

### 6.2. Time-Frequency RNNs
In a Time-Frequency RNN, the hidden state $h_{t, k}$ at time $t$ and frequency band $k$ is computed as:

$$
h_{t, k} = \sigma(Wx_{t, k} + V_T h_{t-1, k} + V_F h_{t, k-1} + b)
$$

Where:
- $V_T$: Time-direction recurrent weights
- $V_F$: Frequency-direction recurrent weights

### 6.3. Grid LSTMs
Grid LSTMs extend this concept using LSTM units. They maintain separate hidden states for time and frequency dimensions:

- **Time LSTM**:
  $$
  h_{t, k}^T = \text{LSTM}_T(x_{t, k}, h_{t-1, k}^T, h_{t, k-1}^F)
  $$

- **Frequency LSTM**:
  $$
  h_{t, k}^F = \text{LSTM}_F(x_{t, k}, h_{t-1, k}^T, h_{t, k-1}^F)
  $$

Bidirectional versions can process data in both forward and backward directions along time and frequency axes.

---

## 7. Combined Models

### 7.1. CLDNN Architecture
Combining Convolutional Neural Networks (CNNs), LSTMs, and DNNs leverages the strengths of each architecture:

- **CNNs**: Extract local features and reduce frequency variance.
- **LSTMs**: Model temporal dependencies.
- **DNNs**: Perform final classification.

An example is the Convolutional, Long Short-Term Memory, Fully Connected Deep Neural Network (CLDNN):

1. **Input**: Acoustic features with context frames.
2. **CNN Layers**: Process input to capture local spatial features.
3. **Linear Layer**: Reduces dimensionality.
4. **LSTM Layers**: Model temporal dynamics.
5. **Fully Connected Layers**: Transform features for classification.
6. **Output**: Predict phoneme or word probabilities.

![[Pasted image 20241120143221.png]]

### 7.2. Other Combinations
- **Grid LSTM with TDNNs**: Time-Delay Neural Networks (TDNNs) capture temporal context, which can be combined with Grid LSTMs.
- **CNNs with GRUs**: Replace LSTMs with GRUs in CLDNN architectures for computational efficiency.

---

## 8. Connectionist Temporal Classification (CTC)

### 8.1. Motivation for CTC
Traditional ASR systems require frame-level alignments between the input acoustic frames and output labels, typically obtained using Hidden Markov Models (HMMs). CTC allows training models without explicit alignment information by learning to align and transcribe sequences simultaneously.

### 8.2. CTC Modeling Approach
CTC introduces an additional "blank" symbol $\phi$ and defines a mapping from input sequences to output labels by collapsing repeated labels and removing blanks.

For a given input sequence $X = (x_1, x_2, \ldots, x_T)$ and target label sequence $C = (c_1, c_2, \ldots, c_U)$, CTC considers all possible alignments $Y$ that map to $C$ after collapsing.

### 8.3. CTC Loss Function
The CTC loss function aims to maximize the total probability of all valid alignments:

$$
L = -\log P(C \mid X) = -\log \sum_{Y \in Y} P(Y \mid X)
$$

Where:
- $Y$: Set of all valid alignments for $C$.
- $P(Y \mid X)$: Probability of alignment $Y$ given input $X$.

![[Pasted image 20241120143327.png]]
### 8.4. Forward-Backward Algorithm in CTC
To compute $P(C \mid X)$, CTC employs a forward-backward algorithm similar to HMMs:
![[Pasted image 20241120143630.png]]

1. **Expand the Target Sequence**:
   Insert blanks between labels and at the ends. Define the expanded sequence:
   $$
   Z = (\phi, c_1, \phi, c_2, \ldots, c_U, \phi)
   $$

2. **Define the Trellis Diagram**:
   States correspond to positions in $Z$.
   Time steps correspond to input frames $t$.

3. **Forward Variables**:
   $$
   \alpha_i(t)
   $$
   Sum of probabilities of all paths reaching state $i$ at time $t$.

4. **Backward Variables**:
   $$
   \beta_i(t)
   $$
   Sum of probabilities of all paths from state $i$ at time $t$ to the end.

5. **Compute Total Probability**:
   $$
   P(C \mid X) = \alpha_{2U+1}(T) + \alpha_{2U}(T)
   $$
![[Pasted image 20241120143702.png]]

### 8.5. Decoding in CTC
- **Greedy Decoding**: At each time step, choose the label with the highest probability.
- **Beam Search**: Considers multiple hypotheses and is necessary when using language models.

### 8.6. Advantages of CTC
- **Alignment-Free Training**: Does not require pre-aligned data.
- **Sequence-Level Objective**: Optimizes the entire output sequence probability.
- **Monotonic Alignment**: Maintains a sequential ordering of outputs.

### 8.7. Disadvantages of CTC
- **Independence Assumption**: Assumes output labels are conditionally independent given the input.
- **Limited Context Modeling**: Does not inherently model dependencies between output labels.
- **Language Model Integration**: Requires additional components to capture language dependencies.

---

## 9. Low Frame Rate Systems

### 9.1. Motivation
Reducing the frame rate in ASR systems decreases computational requirements and aligns the input sequence length more closely with the output sequence length, which is beneficial for models like CTC.

### 9.2. Frame Rate Reduction Techniques
- **Decimation**: Sampling every $n$-th frame (e.g., every 30 ms instead of 10 ms).
- **Frame Stacking**: Combining multiple adjacent frames into a single input vector.

### 9.3. Application in CTC and Other Models
- **CTC**: Lower frame rates reduce the input sequence length, making alignment and training more efficient.
- **Sequence-to-Sequence Models**: Aligning input and output lengths facilitates training and decoding.

---

## 10. Speech Recognition Practical: TIMIT Speech Recognition Using CTC

### 10.1. Overview of the Practical
- **Dataset**: TIMIT Acoustic-Phonetic Continuous Speech Corpus.
- **Objective**: Train a speech recognition model using CTC without explicit alignments.

#### Tasks:
1. Generate acoustic features (e.g., log-mel filterbank coefficients).
2. Map phoneme labels to a standardized set (e.g., 39 phonemes).
3. Implement and train an LSTM-based acoustic model with CTC loss.
4. Evaluate model performance using Phone Error Rate (PER).

### 10.2. Experimentation
- **Model Variations**:
  - Number of LSTM layers and units.
  - Unidirectional vs. bidirectional LSTMs.
  - Regularization techniques (e.g., dropout).
  - Optimization strategies (e.g., learning rate schedules).

- **Analysis**:
  - Monitor loss curves for training and validation sets.
  - Analyze the impact of different architectures on PER.
  - Discuss trade-offs between model complexity and performance.

---

## 11. Summary
- **RNNs**: Introduced to model temporal dependencies in speech, overcoming limitations of feed-forward networks.
- **Training Challenges**: Vanishing/exploding gradients addressed via BPTT, truncated BPTT, and advanced architectures.
- **LSTMs**: Use gating mechanisms to capture long-term dependencies, mitigating vanishing gradients.
- **Enhancements**: Peephole connections, GRUs, LSTMP, and HORNNs offer alternatives and improvements to standard LSTMs.
- **Grid Networks**: Model dependencies across both time and frequency dimensions, capturing more complex patterns.
- **Combined Models**: Integrate CNNs, RNNs, and DNNs to leverage strengths of each architecture.
- **CTC**: Provides alignment-free training, allowing end-to-end ASR models without explicit HMM structures.
- **Low Frame Rates**: Reduce computational load and align input/output sequence lengths, aiding models like CTC.
- **Practical Application**: Training ASR models on TIMIT using CTC illustrates concepts and challenges discussed.
