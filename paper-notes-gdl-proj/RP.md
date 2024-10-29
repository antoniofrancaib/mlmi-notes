# Research Proposal

## Title

**Enhancing RNA Sequence Design through Fine-Tuning Pretrained Models and Integrating Graph Transformers with Equivariance Loss**

## Abstract

RNA inverse folding aims to find RNA sequences that fold into desired 3D structures. Traditional methods often overlook 3D geometry and conformational diversity. This proposal seeks to enhance the geometric RNA design pipeline **gRNAde** by integrating **fine-tuned pretrained models** using **Low-Rank Adaptation (LoRA)** and incorporating **Graph Transformers** with **equivariance loss**. By mapping gRNAde's embeddings to the embedding space of pretrained models like **GenerRNA** or **Uni-RNA** and fine-tuning them, we aim to improve autoregressive decoding of RNA sequences. Additionally, integrating self-attention mechanisms in parallel with GNN encoding will capture long-range interactions in RNA structures. We will explore enforcing equivariance loss to study its impact on the geometry of the loss function during optimization, potentially advancing theoretical understanding in this area.

## Introduction

RNA molecules are fundamental to numerous biological processes, acting as catalysts, regulators, and carriers of genetic information. Designing RNA sequences that fold into specific 3D structures is crucial for therapeutic and synthetic biology applications. The RNA inverse folding problem involves finding sequences that adopt a desired structure, which is challenging due to the complexity of RNA folding mechanisms and the influence of 3D geometry.

**gRNAde** is a geometric deep learning pipeline that addresses RNA inverse design by utilizing a multi-state Graph Neural Network (GNN) to generate sequences compatible with given 3D conformations, explicitly accounting for RNA structure and dynamics. Despite its success, there is potential to enhance gRNAde by integrating advanced deep learning techniques.

This proposal aims to:

1. **Fine-tune pretrained models** (GenerRNA or Uni-RNA) using **LoRA**, leveraging gRNAde's embeddings.
2. **Enhance gRNAde's model** by integrating **Graph Transformers** with **equivariance loss**, capturing both local and global dependencies in RNA structures.

## Literature Review

### RNA Inverse Folding and Design

The RNA inverse folding problem is traditionally approached as an inverse of the folding process, focusing on secondary structures [1]. Recent methods incorporate 3D structural information, acknowledging that RNA folding is influenced by its tertiary interactions and conformational flexibility.

### Geometric Deep Learning in RNA Design

**gRNAde** introduced a multi-state GNN that processes multiple conformations simultaneously, accounting for RNA dynamics [2]. The model uses geometric vector perceptrons (GVPs) to ensure rotational equivariance, enabling the capture of intricate geometric relationships.

### Pretrained Models and Fine-Tuning

Pretrained models like **GenerRNA** [3] and **Uni-RNA** [4] have demonstrated success in modeling RNA sequences and structures. **LoRA** [5] provides a parameter-efficient approach to fine-tuning large models by introducing low-rank updates, making it feasible to adapt pretrained models to new tasks without extensive computational resources.

### Graph Transformers and Attention Mechanisms

Graph Transformers integrate self-attention mechanisms into graph neural networks, enabling the modeling of long-range interactions [6]. They have shown promise in various applications, including molecular property prediction and protein structure modeling.

### Equivariance in Neural Networks

Incorporating symmetries through equivariant neural networks ensures that models respect inherent geometric transformations [7]. However, strict equivariance can impose constraints that limit model flexibility. **Relaxed Equivariance via Multitask Learning (REMUL)** [8] proposes treating equivariance as a learning objective rather than a hard constraint, offering a balance between performance and symmetry enforcement.

### Metric Learning in Clifford Group Equivariant Neural Networks

**Metric Learning for Clifford Group Equivariant Neural Networks** [9] extends CGENNs by integrating learnable metrics, allowing networks to adapt internal representations dynamically. This approach leverages eigenvalue decomposition and category theory to ensure mathematical soundness.

## Proposed Methodology

### 1. Fine-Tuning Pretrained Models with LoRA

#### Mapping gRNAde's Embeddings to Pretrained Models

- **gRNAde's Embeddings**: Obtain the scalar and vector embeddings \( \mathbf{S}' \) and \( \mathbf{V}' \) from the multi-state GNN encoder.
- **Embedding Space Mapping**: Develop a mapping function \( \mathcal{M} \) such that:
  \[
  \mathbf{E} = \mathcal{M}(\mathbf{S}', \mathbf{V}')
  \]
  where \( \mathbf{E} \) aligns with the embedding space of the pretrained model (GenerRNA or Uni-RNA).

#### Fine-Tuning with LoRA

- **Low-Rank Adaptation**: Introduce low-rank matrices \( \Delta W \) to the weights \( W \) of the pretrained model:
  \[
  W' = W + \Delta W, \quad \Delta W = A B^\top
  \]
  where \( A \in \mathbb{R}^{n \times r} \) and \( B \in \mathbb{R}^{n \times r} \), with \( r \ll n \).
- **Optimization Objective**: Fine-tune the model to minimize:
  \[
  \mathcal{L} = -\sum_{i=1}^N \log P_{\theta'}(s_i \mid s_{<i}, \mathbf{E})
  \]
  where \( \theta' \) includes the adapted weights \( W' \).

### 2. Enhancing gRNAde with Graph Transformers and Equivariance Loss

#### Integrating Self-Attention Mechanisms

- **Graph Transformer Layers**: Replace or augment GNN message-passing layers with self-attention layers.
- **Attention Computation**: For nodes \( i \) and \( j \):
  \[
  \alpha_{ij} = \frac{\exp\left( \gamma(\mathbf{h}_i, \mathbf{h}_j) \right)}{\sum_{k \in \mathcal{N}(i)} \exp\left( \gamma(\mathbf{h}_i, \mathbf{h}_k) \right)}
  \]
  where \( \gamma \) is a learnable compatibility function, and \( \mathbf{h}_i \) are node features.

- **Updating Node Representations**:
  \[
  \mathbf{h}_i' = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W}_v \mathbf{h}_j
  \]
  where \( \mathbf{W}_v \) is a learnable weight matrix.

#### Hybrid Model Design

- **Local Message Passing**: Maintain GNN's message-passing for short-range interactions.
- **Global Attention**: Incorporate Graph Transformer layers in parallel to capture long-range dependencies.
- **Combined Update**:
  \[
  \mathbf{h}_i^{\text{combined}} = \mathbf{h}_i^{\text{GNN}} + \mathbf{h}_i^{\text{Transformer}}
  \]

#### Enforcing Equivariance Loss

- **Equivariance Loss Function**:
  \[
  \mathcal{L}_{\text{equi}} = \frac{1}{|\mathcal{D}|} \sum_{x \in \mathcal{D}} \left\| f\left( \phi(g)(x) \right) - \rho(g) \left( f(x) \right) \right\|^2
  \]
  where:
  - \( \phi(g) \) is the group action on the input.
  - \( \rho(g) \) is the group action on the output.
  - \( f \) is the neural network.

- **Incorporating into Total Loss**:
  \[
  \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \beta \mathcal{L}_{\text{equi}}
  \]
  where \( \beta \) balances task performance and equivariance.

#### Alternative Approaches

- **Replacing GNN with Standard Transformer**: Investigate replacing the GNN backbone with a transformer and applying equivariance loss.
- **Equivariance in Global Attention**: Apply equivariance loss specifically to the Graph Transformer layers in the hybrid model.

### 3. Exploring the Impact of Equivariance Loss on Optimization Geometry

- **Theoretical Investigation**: Analyze how enforcing equivariance affects the loss landscape.
- **Mathematical Modeling**: Use concepts from differential geometry and optimization theory to model the changes in the loss function's geometry.
- **Expected Outcomes**: Gain insights into convergence properties and potential benefits in training dynamics.

## Mathematical Formulation

### Fine-Tuning with LoRA

- **Low-Rank Updates**:
  \[
  \Delta W = A B^\top, \quad \text{with} \quad A \in \mathbb{R}^{n \times r}, \quad B \in \mathbb{R}^{n \times r}, \quad r \ll n
  \]
- **Adapted Model Output**:
  \[
  y = f_{\theta'}(x) = f_{\theta}(x) + \Delta f(x)
  \]
  where \( \Delta f(x) \) captures the low-rank adaptations.

### Graph Transformer Integration

- **Attention Mechanism**:
  \[
  \mathbf{Q} = \mathbf{H} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{H} \mathbf{W}_V
  \]
  - Compute attention scores:
    \[
    \mathbf{A} = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} \right)
    \]
  - Update node features:
    \[
    \mathbf{H}' = \mathbf{A} \mathbf{V}
    \]

### Equivariance Loss

- **Group Actions**:
  - Input transformation: \( x' = \phi(g)(x) \)
  - Output transformation: \( y' = \rho(g)(y) \)
- **Loss Computation**:
  \[
  \mathcal{L}_{\text{equi}} = \mathbb{E}_{g \sim G} \left[ \left\| f\left( \phi(g)(x) \right) - \rho(g) \left( f(x) \right) \right\|^2 \right]
  \]

## Expected Contributions

- **Improved RNA Sequence Design**: Enhanced decoding accuracy and structural compatibility through fine-tuning and advanced architectures.
- **Novel Hybrid Model**: A model combining local and global encoding capabilities, capturing complex dependencies in RNA structures.
- **Theoretical Insights**: Understanding the impact of equivariance enforcement on optimization landscapes in neural networks.
- **Methodological Advances**: Application of LoRA in the context of RNA design and integration of equivariance loss in Graph Transformers.

## References

1. [Geometric Deep Learning for RNA Design](#)  
2. [gRNAde: Generative RNA Design](#)  
3. [GenerRNA: Pretrained RNA Sequence Models](#)  
4. [Uni-RNA: Universal RNA Modeling](#)  
5. Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, L., & Chen, W. (2021). **LoRA: Low-Rank Adaptation of Large Language Models**. arXiv preprint arXiv:2106.09685.  
6. Dwivedi, V. P., Bresson, X., et al. (2021). **Generalization of Transformer Networks to Graphs**. arXiv preprint arXiv:2012.09699.  
7. Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). **Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges**. arXiv preprint arXiv:2104.13478.  
8. [Relaxed Equivariance via Multitask Learning (REMUL)](#)  
9. [Metric Learning for Clifford Group Equivariant Neural Networks](#)  

(Note: References marked with (#) indicate placeholder citations for provided papers. Actual references should be updated accordingly.)

## Timeline and Milestones

- **Months 1-2**: Literature review and familiarization with gRNAde, GenerRNA, and Uni-RNA models.
- **Months 3-4**: Develop embedding mapping function and implement LoRA fine-tuning.
- **Months 5-6**: Integrate Graph Transformers into gRNAde and implement equivariance loss.
- **Months 7-8**: Experimental evaluation and optimization of models.
- **Months 9-10**: Theoretical analysis of equivariance loss impact on optimization geometry.
- **Months 11-12**: Compilation of results, writing of manuscripts, and preparation for publication.

## Resources Required

- **Computational Resources**: Access to high-performance GPUs for model training.
- **Software**: PyTorch or TensorFlow frameworks for deep learning implementation.
- **Datasets**: RNA structural datasets with 3D conformations and sequences.

---

This proposal outlines a comprehensive plan to enhance RNA sequence design models by leveraging pretrained models, integrating advanced neural network architectures, and exploring theoretical aspects of equivariance in optimization. The expected outcomes have the potential to contribute significantly to computational biology and machine learning fields.

