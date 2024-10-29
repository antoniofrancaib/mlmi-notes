1. **Improving gRNAde’s Sequence Decoding with Pretrained Models & LoRA Fine-Tuning**

- **Fine-tuning a Pretrained Model for Better Decoding**:
    - Leverage **LoRA** (Low-Rank Adaptation) to fine-tune pretrained models, such as **GenerRNA** or **Uni-RNA**, for improved RNA sequence decoding.
    - Use gRNAde's embeddings, **$\mathbf{S}'$** and **$\mathbf{V}'$**, which encode geometric and structural RNA features. These embeddings must be **mapped** to the **embedding space** of the pretrained model. Once mapped, **fine-tune** the pretrained model using LoRA, enabling better autoregressive decoding of RNA sequences while preserving the structural features from the GNN embeddings.

2. **Enhancing gRNAde's Original Model with Graph Transformers**:

- **Integrating Self-Attention Mechanisms in Parallel to GNN Encoding**:
    - **Graph Transformers** could replace or augment the standard **message-passing layers** in the GNN. This allows **nodes** to compute **weighted representations** of other nodes in the graph using **self-attention**, enhancing the model’s capacity to capture complex and **long-range interactions** in RNA structures.
        
    - **Hybrid Model Design**:
        
        - Maintain GNN’s **local message-passing** strength for encoding **short-range, localized dependencies**.
        - In parallel, introduce **global attention mechanisms** through **Graph Transformers**, enabling the model to better capture **global dependencies** and interactions between distant nucleotides in RNA sequences.

By combining the **local precision** of message-passing with **global awareness** from attention mechanisms, gRNAde can significantly improve both **structural understanding** and **sequence decoding**.

2) extenderlo a estructuras 3D para ADN






