## Introduction

An important paradigm in natural language processing (NLP) involves large-scale pre-training on general domain data followed by adaptation to specific tasks or domains. As pre-trained models become larger, full fine-tuning—which updates all model parameters—becomes less feasible due to computational and storage constraints. For instance, deploying independent instances of fine-tuned models like GPT-3 with 175 billion parameters for each downstream task is prohibitively expensive.

**Low-Rank Adaptation (LoRA)** addresses this challenge by:

- Freezing the pre-trained model weights.
- Injecting trainable low-rank decomposition matrices into each layer of the Transformer architecture.
- Greatly reducing the number of trainable parameters and computational overhead.

LoRA leverages the observation that weight updates during adaptation have a low "intrinsic rank," allowing efficient training with significantly fewer parameters.

## Problem Statement

Given a pre-trained language model $P_{\boldsymbol{\beta}}(y \mid x)$ parameterized by $\boldsymbol{\beta}$, the goal is to adapt it to downstream conditional text generation tasks using a dataset $Z = \{ (x_i, y_i) \}_{i=1}^N$, where both $x_i$ and $y_i$ are sequences of tokens.

### Full Fine-Tuning

The traditional fine-tuning approach updates all model parameters to maximize the conditional log-likelihood:

$$
\max_{\boldsymbol{\beta}} \sum_{(x, y) \in Z} \sum_{t=1}^{|y|} \log P_{\boldsymbol{\beta}}(y_t \mid x, y_{<t}).
$$

This requires updating and storing a model with the same number of parameters as the original pre-trained model, which is impractical for very large models.

### Parameter-Efficient Adaptation

To reduce the number of trainable parameters, we represent the parameter updates $\Delta \boldsymbol{\beta}$ using a smaller set of parameters $\boldsymbol{\theta}$:

$$
\Delta \boldsymbol{\beta} = \Delta \boldsymbol{\beta}(\boldsymbol{\theta}), \quad \text{with} \quad |\boldsymbol{\theta}| \ll |\boldsymbol{\beta}|.
$$

The optimization problem becomes:

$$
\max_{\boldsymbol{\theta}} \sum_{(x, y) \in Z} \sum_{t=1}^{|y|} \log P_{\boldsymbol{\beta}_0 + \Delta \boldsymbol{\beta}(\boldsymbol{\theta})}(y_t \mid x, y_{<t}),
$$

where $\boldsymbol{\beta}_0$ are the frozen pre-trained weights.

## Low-Rank Parameterization

LoRA parameterizes the weight updates $\Delta W$ using a low-rank decomposition:

$$
\Delta W = B A,
$$

where:

- $W_0 \in \mathbb{R}^{d \times k}$ is the pre-trained weight matrix.
- $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are the low-rank matrices with rank $r \ll \min(d, k)$.

### Modified Forward Pass

The modified forward pass in the neural network layer becomes:

$$
h = W_0 x + \Delta W x = W_0 x + B A x.
$$

Here:

- $W_0 x$ is the output from the frozen pre-trained weights.
- $\Delta W x = B A x$ represents the task-specific adaptation, parameterized by $A$ and $B$.

### Training Procedure

- **Initialization**: Initialize $A$ randomly (e.g., $A \sim \mathcal{N}(0, \sigma^2)$) and $B = 0$, so that $\Delta W = 0$ at the start.
- **Scaling**: Optionally scale $\Delta W x$ by a factor $\alpha / r$ to control the learning rate effectively.

Only $A$ and $B$ are trained during adaptation, significantly reducing the number of trainable parameters.

### Advantages of LoRA

- **Parameter Efficiency**: Reduces trainable parameters by orders of magnitude.
- **Memory Efficiency**: Requires less GPU memory during training since gradients and optimizer states are only maintained for $A$ and $B$.
- **No Inference Latency**: At deployment, $\Delta W$ can be merged with $W_0$, incurring no additional latency.
- **Efficient Task Switching**: Switch between tasks by swapping $A$ and $B$ without altering $W_0$.

## Application to Transformers

In the Transformer architecture, LoRA can be applied to any subset of weight matrices. Specifically, it targets the projection matrices in the self-attention mechanism:

- **Query Projection**: $W_q \in \mathbb{R}^{d_{\text{model}} \times d_{\text{k}}}$.
- **Key Projection**: $W_k \in \mathbb{R}^{d_{\text{model}} \times d_{\text{k}}}$.
- **Value Projection**: $W_v \in \mathbb{R}^{d_{\text{model}} \times d_{\text{v}}}$.
- **Output Projection**: $W_o \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

### Example: Applying LoRA to $W_q$ and $W_v$

For the query and value projections:

$$
\Delta W_q = B_q A_q, \quad \Delta W_v = B_v A_v,
$$

where:

- $B_q, B_v \in \mathbb{R}^{d_{\text{model}} \times r}$.
- $A_q, A_v \in \mathbb{R}^{r \times d_{\text{k}}}$ (or $d_{\text{v}}$ for $A_v$).

These low-rank updates are added to the corresponding weight matrices:

$$
W_q' = W_q + \Delta W_q, \quad W_v' = W_v + \Delta W_v.
$$

## Theoretical Motivation

LoRA is motivated by the following observations:

- **Intrinsic Dimension**: Pre-trained models have a low intrinsic dimension (Aghajanyan et al., 2020), meaning they can be effectively fine-tuned in a low-dimensional subspace.
- **Low-Rank Updates**: The updates to the weights during adaptation lie in a low-dimensional (low-rank) subspace.
- **Efficiency**: By constraining updates to be low-rank, we capture the most significant directions for adaptation without overparameterization.

## Empirical Results

### Reduction in Trainable Parameters

- On GPT-3 175B, LoRA reduces trainable parameters by up to 10,000 times compared to full fine-tuning.
- For example, with rank $r = 1$ and adapting only $W_q$ and $W_v$, the number of trainable parameters is:

$$
\text{Trainable Parameters} = 2 \times L \times d_{\text{model}} \times r,
$$

where $L$ is the number of Transformer layers.

### Performance on Downstream Tasks

LoRA matches or exceeds the performance of full fine-tuning and other adaptation methods on various models and tasks:

- **RoBERTa** on the GLUE benchmark (NLU tasks).
- **GPT-2** on natural language generation tasks like E2E NLG Challenge.
- **GPT-3** on tasks like WikiSQL (NL to SQL) and SAMSum (dialogue summarization).

### Rank Analysis

- Even with very low ranks (e.g., $r = 1$ or $r = 2$), LoRA achieves competitive performance.
- This suggests that the essential information for adaptation is captured in a low-dimensional subspace.

## Analysis of Rank Deficiency

### Empirical Observation

- The adaptation matrices $\Delta W$ are empirically low-rank.
- Singular Value Decomposition (SVD) of $\Delta W$ shows that most of the energy (variance) is concentrated in the top singular values.

### Subspace Similarity

- The top singular vectors of $\Delta W$ learned from different random initializations overlap significantly.
- For adaptation matrices $\Delta W_r$ with rank $r$, we can compute the subspace similarity:

$$
\kappa(\Delta W_{r_1}, \Delta W_{r_2}, i, j) = \frac{\| U_{i}^{\top} V_{j} \|_F^2}{\min(i, j)},
$$

where:

- $U_{i}$ contains the top $i$ left singular vectors of $\Delta W_{r_1}$.
- $V_{j}$ contains the top $j$ left singular vectors of $\Delta W_{r_2}$.
- $\| \cdot \|_F$ denotes the Frobenius norm.

- High $\kappa$ indicates significant overlap in the subspaces spanned by the singular vectors, implying that the low-rank updates capture consistent adaptation directions.

### Relationship with Pre-Trained Weights

- The adaptation matrices $\Delta W$ amplify certain directions not emphasized in the pre-trained weights $W_0$.
- Projection of $W_0$ onto the subspace spanned by $\Delta W$ shows that $\Delta W$ does not simply replicate the top singular directions of $W_0$.

#### Example: Frobenius Norm Comparison

Compute the projection of $W_0$ onto the subspace of $\Delta W$:

$$
\| U^{\top} W_0 V \|_F,
$$

where $U$ and $V$ are from the SVD of $\Delta W$.

- A higher norm indicates greater correlation.
- The adaptation matrices are found to be amplifying components orthogonal to the dominant components of $W_0$.

## Conclusion

LoRA offers an effective and efficient method for adapting large pre-trained language models to downstream tasks by:

- Utilizing low-rank updates to significantly reduce the number of trainable parameters.
- Maintaining or improving performance compared to full fine-tuning.
- Eliminating additional inference latency by merging adaptation matrices with pre-trained weights.
- Providing insights into the nature of weight updates during adaptation, suggesting that these updates lie in a low-dimensional subspace.

The success of LoRA demonstrates that large language models can be adapted efficiently without full fine-tuning, paving the way for more accessible and scalable deployment of NLP models.
