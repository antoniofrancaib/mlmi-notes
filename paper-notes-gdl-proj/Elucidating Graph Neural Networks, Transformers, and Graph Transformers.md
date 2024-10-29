## Introduction

Transformers have become the predominant deep neural network architecture, initially designed for Natural Language Processing (NLP) tasks and later adapted to other domains like Computer Vision (CV) with models such as Vision Transformers (ViT) and Diffusion Transformers (DiTs). Despite these adaptations, the core components of Transformers remain largely unchanged. Concurrently, Transformers are entering the domain of graph representation learning, traditionally dominated by Graph Neural Networks (GNNs) and the message-passing framework.

This summary aims to provide an overview of graph representation learning, delve into traditional GNNs, revisit the Transformer architecture, and explore the adaptation of Transformers for graphs. Additionally, it examines the relationship between Graph Transformers and traditional GNNs, discussing these topics from a unified perspective.

## Graph Representation Learning

Graph representation learning involves developing algorithms and models designed to process and extract insights from data pertaining to nodes, edges, and substructures within a graph. The goal is to model functions over graphs.

### Mathematical Definition of Graphs

A **graph** is defined as an ordered tuple:

$$G = (V, E),$$

where:

- $V$ is a set of **nodes** or **vertices**.
- $E \subseteq (V \times V)$ is a set of **edges**.

If  $v_i$ and $v_j$ are nodes in $G$, their relation is represented by $(v_i, v_j) \in E$ if there is an edge from $v_i$ to $v_j$.

**Adjacency Matrix**:

The graph can be represented using an adjacency matrix $A \in \mathbb{R}^{N \times N}$, where $N = |V|$. For unweighted graphs:

$$
A_{ij} =
\begin{cases}
1 & \text{if } (v_i, v_j) \in E, \\
0 & \text{otherwise}.
\end{cases}
$$

**Degree Matrix**:

The diagonal degree matrix $D \in \mathbb{R}^{N \times N}$ is defined as:

$$D_{ii} = \sum_{j} A_{ij}.$$

**Graph Laplacian**:

The graph Laplacian $L$ is given by:

$$L = D - A.$$

It can also be expressed as:

$$
L_{ij} =
\begin{cases}
\deg(v_i) & \text{if } i = j, \\
-1 & \text{if } i \neq j \text{ and } v_i \text{ is adjacent to } v_j, \\
0 & \text{otherwise}.
\end{cases}
$$

**Node Features**:

Each node $v_i$ has an associated feature vector $\mathbf{x}_i \in \mathbb{R}^{1 \times D}$. The feature matrix $X \in \mathbb{R}^{N \times D}$ stacks the feature vectors for all nodes:

$$
X = \begin{bmatrix}
\mathbf{x}_1 \\
\mathbf{x}_2 \\
\vdots \\
\mathbf{x}_N
\end{bmatrix} = \begin{bmatrix}
x_{11} & x_{12} & \dots & x_{1D} \\
x_{21} & x_{22} & \dots & x_{2D} \\
\vdots & \vdots & \ddots & \vdots \\
x_{N1} & x_{N2} & \dots & x_{ND}
\end{bmatrix}.
$$

## Graph Neural Networks and Message Passing

Graph Neural Networks (GNNs) are deep learning models designed for inference on graph-structured data. They leverage the message-passing framework, where nodes exchange messages with their neighbors to update their feature representations.

### Message Passing Framework

A typical message-passing layer $l$ in a GNN is defined as:

1. **Message Computation**:
$$
   \mathbf{m}_{ij}^{(l)} = \psi\left( \mathbf{x}_i^{(l)}, \mathbf{x}_j^{(l)} \right),
   $$

2. **Message Aggregation**:

   $$
   \mathbf{a}_i^{(l)} = \bigoplus_{j \in \mathcal{N}(v_i)} \mathbf{m}_{ij}^{(l)},
   $$

   where $\bigoplus$ is a permutation-invariant aggregation function (e.g., sum, mean, max).

3. **Node Update**:

   $$
   \mathbf{x}_i^{(l+1)} = \phi\left( \mathbf{x}_i^{(l)}, \mathbf{a}_i^{(l)} \right),
   $$

   where $\psi$ and $\phi$ are learnable functions, often implemented as Multi-Layer Perceptrons (MLPs).

This localized approach enhances scalability and allows the model to capture local structural information.

### Permutation Invariance and Equivariance

- **Permutation Invariance**: A function $f$ is permutation invariant if:

  $$
  f(PX) = f(X),
  $$

  for any permutation matrix $P$.

- **Permutation Equivariance**: A function $F$ is permutation equivariant if:

  $$
  F(PX) = P F(X).
  $$

GNN layers are designed to be permutation equivariant, ensuring that the output is consistent regardless of the ordering of nodes.

### Traditional Graph Neural Network Architectures

#### Graph Convolutional Networks (GCNs)

GCNs update node features using the normalized adjacency matrix:

$$
\mathbf{X}^{(l+1)} = \sigma\left( \tilde{A} \mathbf{X}^{(l)} \mathbf{W}^{(l)} \right),
$$

where:

- $\tilde{A} = \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}$,
- $\hat{A} = A + I$ (adding self-loops),
- $\hat{D}$ is the degree matrix of $\hat{A}$,
- $\sigma$ is an activation function (e.g., ReLU).

Breaking down the operation:

1. **Feature Aggregation**:

   $$
   \tilde{\mathbf{X}}^{(l)} = \tilde{A} \mathbf{X}^{(l)}.
   $$

2. **Linear Transformation**:

   $$
   \mathbf{X}^{(l+1)} = \sigma\left( \tilde{\mathbf{X}}^{(l)} \mathbf{W}^{(l)} \right).
   $$

#### Graph Attention Networks (GATs)

GATs introduce attention mechanisms to weigh the importance of neighboring nodes:

1. **Attention Coefficients**:

   $$
   \alpha_{ij} = \text{softmax}\left( \text{LeakyReLU}\left( \mathbf{a}^\top [\mathbf{W} \mathbf{x}_i \, \| \, \mathbf{W} \mathbf{x}_j] \right) \right),
   $$

   where $\|$ denotes concatenation.

2. **Feature Update**:

   $$
   \mathbf{x}_i^{(l+1)} = \sigma\left( \sum_{j \in \mathcal{N}(v_i)} \alpha_{ij} \mathbf{W} \mathbf{x}_j \right).
   $$

GATs allow the network to focus on the most relevant parts of the graph.

### Limitations of Message Passing GNNs

- **Expressivity**: Limited by the Weisfeiler-Lehman (WL) test; cannot distinguish certain graph structures.
- **Over-smoothing**: Node features become indistinguishable when stacking many layers.
- **Over-squashing**: Difficulty in capturing long-range dependencies due to bottlenecks in message passing.

## Transformers

Transformers are sequence models that have been adapted to various domains. They process sequences of tokens, each represented by a feature vector.

### Transformer Block

The Transformer block consists of:

1. **Multi-Head Self-Attention (MHSA)**:

   $$
   \mathbf{h} = \text{MHSA}(\text{LayerNorm}(\mathbf{x})) + \mathbf{x},
   $$

2. **Position-wise Feed-Forward Network (FFN)**:

   $$
   \mathbf{y} = \text{FFN}(\text{LayerNorm}(\mathbf{h})) + \mathbf{h}.
   $$

3. **Layer Normalization and Residual Connections**.

### Self-Attention Mechanism

The self-attention computes:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} \right) \mathbf{V},
$$

where:

- $\mathbf{Q} = \mathbf{X} \mathbf{W}^Q$,
- $\mathbf{K} = \mathbf{X} \mathbf{W}^K$,
- $\mathbf{V} = \mathbf{X} \mathbf{W}^V$,
- $d_k$ is the dimension of the key vectors.

**Multi-Head Self-Attention** allows the model to focus on different representation subspaces:

$$
\text{MHSA}(\mathbf{X}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O,
$$

with each head computed as:

$$
\text{head}_i = \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i).
$$

### Positional Encodings

To incorporate order information, positional encodings are added to the input embeddings:

$$
\text{PE}_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right),
$$

$$
\text{PE}_{(pos, 2i+1)} = \cos\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right).
$$

These functions inject information about the position of each token in the sequence.

## Graph Transformers

Graph Transformers adapt the Transformer architecture to graph data, combining global attention with message passing.

### Motivation

- **Overcome Limitations of GNNs**: Address over-smoothing, over-squashing, and limited expressivity.
- **Model Long-Range Dependencies**: Allow every node to attend to every other node.

### Graph Transformer Block

The Graph Transformer block includes:

1. **Global Attention**: Allows every node to attend to every other node, capturing long-range dependencies.

2. **Message Passing Neural Network (MPNN)**: Incorporates the local graph structure.

3. **Layer Normalization and Residual Connections**.

The block can be summarized as:

$$
\mathbf{h} = \text{GlobalAttention}(\text{LayerNorm}(\mathbf{x})) + \text{MPNN}(\text{LayerNorm}(\mathbf{x}, A)) + \mathbf{x},
$$

$$
\mathbf{y} = \text{MLP}(\text{LayerNorm}(\mathbf{h})) + \mathbf{h}.
$$

### Positional and Structural Encodings

Graph Transformers use positional and structural encodings to capture the graph topology.

#### Laplacian Positional Encodings

Eigenvectors of the graph Laplacian $L$ are used as positional encodings:

1. **Compute Graph Laplacian**:

   $$L = D - A.$$

2. **Eigenvalue Decomposition**:

   $$L \Phi = \Lambda \Phi,$$

   where $\Lambda$ is a diagonal matrix of eigenvalues, and $\Phi$ contains the corresponding eigenvectors.

3. **Use Eigenvectors as Encodings**:

   The eigenvectors $\Phi$ serve as positional encodings, capturing global structural information.

### Scaling to Large Graphs

- **Computational Challenges**: Global attention has quadratic complexity with respect to the number of nodes.
- **Approximate Attention Mechanisms**:

  - **Sparse Attention**: Limiting attention to a subset of nodes.
  - **Low-Rank Approximations**: Reducing the dimensionality of attention computations.
  - **Randomized Methods**: Sampling techniques to approximate attention.

- **Hybrid Approaches**: Combining local message passing with global attention mechanisms.

### Advantages over Traditional GNNs

- **Expressivity**: Enhanced ability to model complex graph structures.
- **Long-Range Dependencies**: Direct communication between distant nodes.
- **Mitigation of Over-Smoothing**: By combining global and local information.

## Conclusion

Graph Transformers represent a powerful integration of Transformer architectures with graph representation learning, addressing limitations of traditional GNNs and enabling the modeling of complex dependencies in graph-structured data. They leverage both global attention mechanisms and local message passing, augmented with positional and structural encodings derived from graph properties like the Laplacian eigenvectors.

By incorporating these advanced techniques, Graph Transformers offer a promising direction for future research and applications in areas requiring sophisticated analysis of graph-structured data.
