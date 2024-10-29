## Introduction

RNA molecules play crucial roles in various biological processes, and designing RNA sequences that fold into desired 3D structures is a significant challenge in computational biology. Traditional RNA design tasks are often posed as inverse problems, focusing on secondary structure without considering 3D geometry and conformational diversity.

**gRNAde** is a geometric deep learning pipeline that addresses this challenge by performing RNA inverse design conditioned on one or more 3D backbone structures. It utilizes a multi-state Graph Neural Network (GNN) to generate RNA sequences that are compatible with given 3D conformations, accounting explicitly for RNA structure and dynamics.

## The RNA Inverse Folding Problem

The RNA inverse folding problem involves finding an RNA sequence that will fold into a desired 3D structure. Mathematically, given a set of 3D backbone coordinates $\{\mathbf{x}_n\}_{n=1}^N$, we aim to find a sequence $\{\mathbf{s}_n\}_{n=1}^N$ such that the sequence folds into the given structure.

Formally, we seek to solve:
$$
\text{Find} \quad \{\mathbf{s}_n\}_{n=1}^N \quad \text{such that} \quad \mathbf{x}_n = f(\mathbf{s}_n),
$$
where $f$ is a folding function mapping sequences to structures. 
<span style="color:red">how do we know f?</span> $\rightarrow$ external software

## Featurization of RNA Backbone Structures

Each RNA nucleotide is represented as a node in a geometric graph. The featurization involves:

1. **Coarse-Grained Representation**: Using a 3-bead model for each nucleotide, considering the atoms P, C4', and N1 (for pyrimidines) or N9 (for purines).

2. **Node Features**: Each node $i$ is assigned:

   - **Scalar Features** $\mathbf{s}_i$.
   - **Vector Features** $\mathbf{v}_i \in \mathbb{R}^{f' \times 3}$, where $f'$ is the number of vector feature channels.

3. **Edge Features**: For each edge from node $j$ to node $i$:

   - **Unit Vector** $\mathbf{u}_{ij} = \frac{\mathbf{x}_j - \mathbf{x}_i}{\|\mathbf{x}_j - \mathbf{x}_i\|}$.
   - **Distance** $d_{ij} = \|\mathbf{x}_j - \mathbf{x}_i\|$.
   - **Backbone Distance** $|j - i|$.

4. **Adjacency Matrix**: Nodes are connected based on proximity in 3D space, typically connecting each node to its $k$ nearest neighbors.

The input to the GNN is as follows:

- **Node feature matrix** $\mathbf{H} = \{(\mathbf{s}_i, \mathbf{v}_i) \ | \ i \in V\}$:
  - $\mathbf{H}_{\text{scalar}} \in \mathbb{R}^{n \times f}$, where $n$ is the number of nodes (nucleotides) and $f$ is the number of scalar features.
  - $\mathbf{H}_{\text{vector}} \in \mathbb{R}^{n \times f' \times 3}$, where $f'$ is the number of vector features, each represented in 3D.

- **Adjacency matrix** $\mathbf{A} \in \mathbb{R}^{n \times n}$, which defines the connectivity of the graph, where $\mathbf{A}_{ij} = 1$ if nodes $i$ and $j$ are connected, and $0$ otherwise.

- **Edge feature matrix** $\mathbf{E} = \{\mathbf{e}_{ij} \ | \ e_{ij} \in E\}$, a matrix of edge features representing distances, angles, and other geometric relations between connected nucleotides.

## Multi-State Graph Neural Network Encoder

To capture the conformational diversity of RNA, gRNAde employs a multi-state GNN that processes multiple conformations simultaneously.

### Multi-Graph Representation

Given $K$ conformations (realistic assumption, discrete amount of possibilities?), the input is represented as a multi-graph:

- **Node Features**: $\mathbf{S} \in \mathbb{R}^{N \times K \times f}$.
- **Vector Features**: $\mathbf{V} \in \mathbb{R}^{N \times K \times f' \times 3}$.
- **Adjacency Matrices**: $\{\mathbf{A}^{(k)}\}_{k=1}^K$ merged via union into a single adjacency matrix $\mathbf{A}$.

### Message Passing

The GNN updates node features through message passing, maintaining equivariance to rotations (SO(3) symmetry):

1. **Message Computation**:
$$
   \begin{align*}
   \mathbf{m}_i &= \sum_{j \in \mathcal{N}(i)} \text{MSG}(\mathbf{s}_i, \mathbf{v}_i, \mathbf{s}_j, \mathbf{v}_j, \mathbf{e}_{ij}), \\
   \end{align*}
   $$

   where $\mathcal{N}(i)$ denotes the neighbors of node $i$, and $\mathbf{e}_{ij}$ are edge features.

2. **Node Update**:
$$
   \begin{align*}
   \mathbf{s}_i', \mathbf{v}_i' &= \text{UPD}(\mathbf{s}_i, \mathbf{v}_i, \mathbf{m}_i), \\
   \end{align*}
   $$

   where MSG and UPD are functions implemented using Geometric Vector Perceptrons (GVPs), which ensure rotational equivariance.

### Conformation Order-Invariant Pooling

After processing all conformations, features are pooled across the conformations axis to obtain conformation-invariant representations:

$$
\mathbf{S}_i' = \frac{1}{K} \sum_{k=1}^K \mathbf{S}_{i}^{(k)}, \quad \mathbf{V}_i' = \frac{1}{K} \sum_{k=1}^K \mathbf{V}_{i}^{(k)}.
$$

## Sequence Decoding

The pooled node features are used to decode the RNA sequence in an autoregressive manner:

1. **Probability Prediction**:
   For each nucleotide position $i$, predict the probability distribution over the four possible bases (A, G, C, U):$$
   P(s_i \mid s_1, s_2, \dots, s_{i-1}, \mathbf{S}', \mathbf{V}').
   $$

2. **Autoregressive Decoding**:

   The sequence is generated by sampling from the predicted distributions sequentially from $i = 1$ to $N$.

## Loss Function

The model is trained using a cross-entropy loss between the predicted probability distribution and the ground truth sequence:
$$
\mathcal{L} = -\sum_{i=1}^N \sum_{b \in \{\text{A}, \text{G}, \text{C}, \text{U}\}} y_{i,b} \log P(s_i = b),
$$

where $y_{i,b}$ is a one-hot encoding of the ground truth base at position $i$.

## Evaluation Metrics

To assess the quality of the designed sequences, the following metrics are used:

1. **Native Sequence Recovery**:

   The percentage of nucleotides in the designed sequence that match the native sequence:$$
   \text{Recovery} = \frac{1}{N} \sum_{i=1}^N \delta(s_i^{\text{designed}}, s_i^{\text{native}}),
   $$

   where $\delta$ is the Kronecker delta function.

2. **Secondary Structure Self-Consistency**:

   Comparing the predicted secondary structure of the designed sequence to the native secondary structure using Matthews Correlation Coefficient (MCC):
$$
   \text{MCC} = \frac{ TP \times TN - FP \times FN }{ \sqrt{ (TP + FP)(TP + FN)(TN + FP)(TN + FN) } },
   $$

   where $TP$, $TN$, $FP$, and $FN$ are true positives, true negatives, false positives, and false negatives in base pair predictions.

3. **Tertiary Structure Self-Consistency**:

   Measuring the similarity between the predicted 3D structure and the native structure using metrics like Root Mean Square Deviation (RMSD):

   $$
   \text{RMSD} = \sqrt{ \frac{1}{N} \sum_{i=1}^N \| \mathbf{x}_i^{\text{designed}} - \mathbf{x}_i^{\text{native}} \|^2 }.
   $$

4. **Perplexity**:

   The exponentiated negative log-likelihood per nucleotide, indicating the model's confidence:
   $$
   \text{Perplexity} = \exp\left( -\frac{1}{N} \sum_{i=1}^N \log P(s_i) \right).
   $$

## Experiments and Results

### Single-State RNA Design Benchmark

- **Comparison with Rosetta**: gRNAde achieves higher native sequence recovery (56% on average) compared to Rosetta (45% on average).

- **Speed**: gRNAde designs sequences significantly faster than Rosetta due to its deep learning framework.

- **Correlation with Perplexity**: Lower perplexity values correspond to higher native sequence recovery, indicating that perplexity can be used to rank designed sequences.

### Multi-State RNA Design Benchmark

- **Multi-State vs. Single-State Models**: Multi-state gRNAde models show improved sequence recovery over single-state models, especially for nucleotides involved in structural flexibility.

- **Per-Nucleotide Analysis**: Recovery improves for nucleotides with changes in base pairing, higher solvent-accessible surface area, and larger RMSD across conformations.

## Mathematical Formulation of the Multi-State GNN

The multi-state GNN processes conformational ensembles by considering features across multiple states:

- **Input Features**:

  For each node $i$ and state $k$:

  - Scalar features: $\mathbf{s}_i^{(k)}$.
  - Vector features: $\mathbf{v}_i^{(k)} \in \mathbb{R}^{f' \times 3}$.

- **Message Passing Equations**:

  For each node $i$:
$$
  \begin{align*}
  \mathbf{m}_i^{(k)} &= \sum_{j \in \mathcal{N}(i)} \text{MSG}\left( \mathbf{s}_i^{(k)}, \mathbf{v}_i^{(k)}, \mathbf{s}_j^{(k)}, \mathbf{v}_j^{(k)}, \mathbf{e}_{ij}^{(k)} \right), \\
  \mathbf{s}_i^{(k)\prime}, \mathbf{v}_i^{(k)\prime} &= \text{UPD}\left( \mathbf{s}_i^{(k)}, \mathbf{v}_i^{(k)}, \mathbf{m}_i^{(k)} \right).
  \end{align*}
  $$

- **Pooling Across States**:

  After message passing, features are pooled to obtain state-invariant representations:

  $$
  \mathbf{s}_i = \frac{1}{K} \sum_{k=1}^K \mathbf{s}_i^{(k)\prime}, \quad \mathbf{v}_i = \frac{1}{K} \sum_{k=1}^K \mathbf{v}_i^{(k)\prime}.
  $$

## Zero-Shot Ranking of RNA Fitness Landscape

gRNAde's perplexity can be used to rank mutant sequences in terms of their likelihood to fold into a desired structure, which correlates with experimental fitness.

- **Perplexity as Fitness Proxy**:

  Lower perplexity implies higher likelihood, potentially indicating higher fitness.

- **Retrospective Analysis**:

  Applying gRNAde to a dataset of RNA polymerase ribozyme mutants shows that perplexity-based ranking outperforms random selection in identifying high-fitness mutants.

## Conclusion

gRNAde presents a novel approach to RNA inverse design by integrating geometric deep learning with multi-state modeling. The mathematical foundations of the method, including the use of multi-state GNNs and autoregressive decoding, enable efficient and accurate design of RNA sequences compatible with complex 3D structures and conformational dynamics.

The model's success demonstrates the potential of combining geometric representations with deep learning to tackle challenging problems in computational biology, paving the way for future developments in RNA therapeutics and bioengineering.


# Mathematical Summary: 

## Problem Definition

Given RNA backbone 3D coordinates $\{\mathbf{x}_i\}_{i=1}^N$, find an RNA sequence $\{\mathbf{s}_i\}_{i=1}^N$ such that the sequence folds into the given structure.

## Featurization

### Nodes

- Each nucleotide $i$ is a node with:

  - Scalar features: $\mathbf{s}_i \in \mathbb{R}^f$.
  - Vector features: $\mathbf{v}_i \in \mathbb{R}^{f' \times 3}$.

### Edges

- For connected nodes $i$ and $j$:

  - Edge vector: $\mathbf{e}_{ij} = \mathbf{x}_j - \mathbf{x}_i$.
  - Unit vector: $\hat{\mathbf{e}}_{ij} = \frac{\mathbf{e}_{ij}}{\|\mathbf{e}_{ij}\|}$.
  - Distance: $d_{ij} = \|\mathbf{e}_{ij}\|$.
  - Edge features: $\mathbf{e}_{ij}$ (geometric information).

## Multi-State Graph Neural Network

### Input

- For $K$ conformations, each node $i$ has features for each state $k$:

  - Scalar features: $\mathbf{s}_i^{(k)}$.
  - Vector features: $\mathbf{v}_i^{(k)}$.

### Message Passing

For each node $i$ and state $k$:

1. **Message Computation**:

   $$
   \mathbf{m}_i^{(k)} = \sum_{j \in \mathcal{N}(i)} \text{MSG}\left( \mathbf{s}_i^{(k)}, \mathbf{v}_i^{(k)}, \mathbf{s}_j^{(k)}, \mathbf{v}_j^{(k)}, \mathbf{e}_{ij}^{(k)} \right)
   $$

2. **Node Update**:

   $$
   \mathbf{s}_i^{(k)\prime}, \mathbf{v}_i^{(k)\prime} = \text{UPD}\left( \mathbf{s}_i^{(k)}, \mathbf{v}_i^{(k)}, \mathbf{m}_i^{(k)} \right)
   $$

### Pooling Across States

After processing all $K$ states:

$$
\mathbf{s}_i = \frac{1}{K} \sum_{k=1}^K \mathbf{s}_i^{(k)\prime}, \quad \mathbf{v}_i = \frac{1}{K} \sum_{k=1}^K \mathbf{v}_i^{(k)\prime}
$$

## Sequence Decoding

- Predict nucleotide probabilities at each position $i$:

  $$
  P(s_i \mid s_{<i}, \mathbf{s}_i, \mathbf{v}_i)
  $$

- Autoregressive generation from $i = 1$ to $N$ using the predicted probabilities.

## Training Objective

- Minimize the cross-entropy loss:

  $$
  \mathcal{L} = -\sum_{i=1}^N \log P\left( s_i^{\text{true}} \mid s_{<i}^{\text{true}}, \mathbf{s}_i, \mathbf{v}_i \right)
  $$

## Geometric Vector Perceptron (GVP)

- Ensures rotational equivariance in MSG and UPD functions.

### GVP Operations

Given scalar inputs $\mathbf{s}$ and vector inputs $\mathbf{v}$:

1. **Linear Layers**:

   - Scalar: $\mathbf{s}' = W_s \mathbf{s} + b_s$
   - Vector: $\mathbf{v}' = \mathbf{v} W_v$

2. **Nonlinearity**:

   - Scalar activation: $\sigma_s(\mathbf{s}')$
   - Vector gating: $\mathbf{v}'' = \sigma_g\left( \| \mathbf{v}' \| \right) \frac{\mathbf{v}'}{\| \mathbf{v}' \|}$

## Edge Features

- Include geometric information:

  - Distance: $d_{ij}$
  - Direction: $\hat{\mathbf{e}}_{ij}$
  - Backbone distance: $|i - j|$

## Model Flow Summary

1. **Input**: RNA backbone conformations $\{\mathbf{x}_i^{(k)}\}$ for $k = 1, \dots, K$.

2. **Featurization**: Compute node and edge features for each conformation.

3. **GNN Processing**:

   - Perform message passing for each node and state using GVPs.
   - Update node features $\mathbf{s}_i^{(k)\prime}, \mathbf{v}_i^{(k)\prime}$.

4. **Pooling**:

   - Aggregate updated features across states to get $\mathbf{s}_i$, $\mathbf{v}_i$.

5. **Sequence Decoding**:

   - Compute $P(s_i \mid s_{<i}, \mathbf{s}_i, \mathbf{v}_i)$.
   - Generate sequence autoregressively.

6. **Training**:

   - Optimize parameters by minimizing $\mathcal{L}$ over the training data.

