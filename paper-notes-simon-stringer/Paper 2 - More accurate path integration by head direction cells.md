# A Speed-Accurate Self-Sustaining Head Direction Cell Path Integration Model Without Recurrent Excitation

## Introduction

Head direction (HD) cells are neurons that represent the animal's head orientation in the environment, acting as an internal compass. They can maintain their activity and update their firing based on internally derived self-motion cues, a process known as **path integration**. A critical question is how the HD system maintains accurate correspondence between true head direction and its internal representation, especially in the absence of visual cues.

Previous models often relied on **continuous attractor neural networks (CANNs)** with recurrent excitatory connections within the HD cell layer to sustain activity. However, such recurrent connections can introduce inaccuracies in path integration due to their inability to adapt to varying head rotation speeds.

This study presents a two-layer HD cell model that achieves highly accurate path integration without recurrent excitatory connections within the HD layer. Instead, it utilizes axonal conduction delays and a combination of HD cells and cells responsive to angular head velocity (AHV) to update the HD representation accurately.

## Model Description

The model consists of three main components:

1. **Head Direction (HD) Cells**: Representing the animal's current head direction.
2. **Angular Head Velocity (AHV) Cells**: Signaling whether the head is rotating or stationary.
3. **Combination (COMB) Cells**: Responding to specific combinations of HD and AHV signals.

### Pre-wired Model

#### Network Architecture

- **HD Cells**: Arranged in a ring topology, each with a preferred firing direction \( x_i \in [0^\circ, 360^\circ) \).
- **AHV Cells**: Two types:
  - **ROT Cells**: Active during head rotation.
  - **NOROT Cells**: Active when the head is stationary.
- **COMB Cells**: Two subpopulations:
  - **ROT-COMB Cells**: Receive inputs from HD cells and ROT cells.
  - **NOROT-COMB Cells**: Receive inputs from HD cells and NOROT cells.

#### Neuron Model Equations

##### HD Cells

The activation of HD cell \( i \) at time \( t \) is given by:

$$
\tau_{\text{HD}} \frac{d h_i^{\text{HD}}(t)}{d t} = - h_i^{\text{HD}}(t) + e_i(t) - \frac{1}{N_{\text{HD}}} \sum_{j} \tilde{w}^{\text{HD}} r_j^{\text{HD}}(t) + \frac{\phi_2}{C_{\text{COMB} \rightarrow \text{HD}}} \sum_{j} w_{ij}^{(2)} r_j^{\text{COMB}}(t - \Delta t)
$$

- \( \tau_{\text{HD}} \): Time constant of HD cells.
- \( h_i^{\text{HD}}(t) \): Activation level.
- \( e_i(t) \): External visual input.
- \( \tilde{w}^{\text{HD}} \): Global inhibitory weight.
- \( r_j^{\text{HD}}(t) \): Firing rate of presynaptic HD cell \( j \).
- \( w_{ij}^{(2)} \): Synaptic weight from COMB cell \( j \) to HD cell \( i \).
- \( \phi_2 \): Scaling factor for COMB to HD connections.
- \( C_{\text{COMB} \rightarrow \text{HD}} \): Number of synapses each HD cell receives from COMB cells.
- \( \Delta t \): Axonal conduction delay.

The firing rate is calculated using a sigmoidal activation function:

$$
r_i^{\text{HD}}(t) = \frac{1}{1 + e^{-2 \beta_{\text{HD}} (h_i^{\text{HD}}(t) - \alpha_{\text{HD}})}}
$$

- \( \alpha_{\text{HD}} \): Threshold.
- \( \beta_{\text{HD}} \): Slope parameter.

##### COMB Cells

For **ROT-COMB** cells:

$$
\tau_{\text{COMB}} \frac{d h_i^{\text{COMB}}(t)}{d t} = - h_i^{\text{COMB}}(t) - \frac{1}{N_{\text{COMB}}} \sum_{j} \tilde{w}^{\text{COMB}} r_j^{\text{COMB}}(t) + \frac{\phi_1}{C_{\text{HD} \rightarrow \text{COMB}}} \sum_{j} w_{ij}^{(1)} r_j^{\text{HD}}(t - \Delta t) + \frac{\phi_3}{C_{\text{ROT} \rightarrow \text{COMB}}} \sum_{j} w_{ij}^{(3)} r_j^{\text{ROT}}(t)
$$

For **NOROT-COMB** cells:

$$
\tau_{\text{COMB}} \frac{d h_i^{\text{COMB}}(t)}{d t} = - h_i^{\text{COMB}}(t) - \frac{1}{N_{\text{COMB}}} \sum_{j} \tilde{w}^{\text{COMB}} r_j^{\text{COMB}}(t) + \frac{\phi_1}{C_{\text{HD} \rightarrow \text{COMB}}} \sum_{j} w_{ij}^{(1)} r_j^{\text{HD}}(t - \Delta t) + \frac{\phi_4}{C_{\text{NOROT} \rightarrow \text{COMB}}} \sum_{j} w_{ij}^{(4)} r_j^{\text{NOROT}}(t)
$$

- \( \tau_{\text{COMB}} \): Time constant of COMB cells.
- \( \tilde{w}^{\text{COMB}} \): Global inhibitory weight.
- \( w_{ij}^{(1)} \): Synaptic weight from HD cell \( j \) to COMB cell \( i \).
- \( w_{ij}^{(3)} \), \( w_{ij}^{(4)} \): Synaptic weights from ROT/NOROT cells to COMB cells.
- \( \phi_1, \phi_3, \phi_4 \): Scaling factors.
- \( C_{\text{HD} \rightarrow \text{COMB}} \), \( C_{\text{ROT} \rightarrow \text{COMB}} \): Number of synapses.

The firing rate of COMB cells is:

$$
r_i^{\text{COMB}}(t) = \frac{1}{1 + e^{-2 \beta_{\text{COMB}} (h_i^{\text{COMB}}(t) - \alpha_{\text{COMB}})}}
$$

#### Synaptic Weights

- **HD to COMB ( \( w_{ij}^{(1)} \) )**:
  - For **NOROT-COMB** cells (non-offset):

    $$
    w_{ij}^{(1)} = e^{- \frac{(s_{ij}^{\text{HD-COMB}})^2}{2 (\sigma_{\text{HD-COMB}})^2}}
    $$

  - For **ROT-COMB** cells (offset by \( O \)):

    $$
    w_{ij}^{(1)} = e^{- \frac{(s_{ij}^{\text{HD-COMB}} + O)^2}{2 (\sigma_{\text{HD-COMB}})^2}}
    $$

  - \( s_{ij}^{\text{HD-COMB}} = \min(|x_i - x_j|, 360^\circ - |x_i - x_j|) \)
  - \( O = V \Delta t \) (offset determined by head rotation speed \( V \) and conduction delay \( \Delta t \)).

- **COMB to HD ( \( w_{ij}^{(2)} \) )**:
  - Similar to \( w_{ij}^{(1)} \), but connections are from COMB to HD cells.

- **ROT/NOROT to COMB ( \( w_{ij}^{(3)} \), \( w_{ij}^{(4)} \) )**:
  - Set to 1 if connected, 0 otherwise.

### Self-Organizing Model

In the self-organizing model, synaptic weights are not pre-set but develop through learning using biologically plausible Hebbian learning rules.

#### Learning Rules

- **HD to COMB Synapses ( \( w_{ij}^{(1)} \) )**:

  $$
  \frac{d w_{ij}^{(1)}(t)}{d t} = k r_i^{\text{COMB}}(t) r_j^{\text{HD}}(t - \Delta t)
  $$

- **COMB to HD Synapses ( \( w_{ij}^{(2)} \) )**:

  $$
  \frac{d w_{ij}^{(2)}(t)}{d t} = k r_i^{\text{HD}}(t) r_j^{\text{COMB}}(t - \Delta t)
  $$

- **ROT to COMB Synapses ( \( w_{ij}^{(3)} \) )**:

  $$
  \frac{d w_{ij}^{(3)}(t)}{d t} = k r_i^{\text{COMB}}(t) r_j^{\text{ROT}}(t)
  $$

- \( k \): Learning rate constant.

**Synaptic Weight Normalization**:

After each update, synaptic weights are normalized to prevent unbounded growth:

$$
\sqrt{ \sum_j (w_{ij}(t))^2 } = 1
$$

## Operating Principles

### Stationary Head (No Rotation)

- **Active Cells**: NOROT-COMB cells.
- **Mechanism**:
  - HD cells project to NOROT-COMB cells corresponding to the current head direction.
  - NOROT-COMB cells project back to the same HD cells.
  - This reciprocal connectivity sustains the HD activity packet at a fixed location.

### Head Rotation

- **Active Cells**: ROT-COMB cells.
- **Mechanism**:
  - HD cells project to ROT-COMB cells with an offset determined by \( V \Delta t \).
  - ROT-COMB cells project back to HD cells with an additional offset, totaling \( 2 V \Delta t \).
  - This shifts the HD activity packet through the HD layer, accurately tracking head direction over time.

## Simulation Protocol

- **Initialization**:
  - All cell activations and firing rates are set to zero.
  - Synaptic weights are initialized (either pre-wired or randomly for self-organizing model).
  - An external visual input \( e_i(t) \) is applied to HD cells to generate an initial activity packet.

- **Simulation Phases**:
  1. **Visual Input Phase**:
     - External input drives the HD activity packet.
     - Used to simulate conditions with visual cues.
  2. **Stationary Phase**:
     - AHV cells signal no head rotation.
     - HD packet remains stationary due to NOROT-COMB cell activity.
  3. **Rotation Phase**:
     - AHV cells signal head rotation at speed \( V \).
     - HD packet shifts through the HD layer due to ROT-COMB cell activity.
  4. **Testing Phase**:
     - Visual input is removed.
     - Observes whether the HD packet maintains accurate path integration in the absence of visual cues.

## Results

### Path Integration Accuracy

- **Pre-Wired Model**:
  - Achieves over 99% accuracy in path integration speed across a range of target velocities (\( 0^\circ/\text{s} \) to \( 360^\circ/\text{s} \)).
  - Demonstrates the model's ability to update the HD representation accurately based on AHV input.

- **Self-Organizing Model**:
  - Achieves approximately 91% of the target velocity.
  - Slight reduction in accuracy due to imperfections in learned synaptic weights and COMB cell specificity.

### Effect of Time Constants

- **Neuronal Time Constant (\( \tau_{\text{HD}}, \tau_{\text{COMB}} \))**:
  - Increasing \( \tau \) leads to increased rise time (\( t_R \)) in neuronal response.
  - Higher \( \tau \) values result in decreased path integration speed due to delayed firing.

- **Simulation Findings**:
  - With \( \tau \) values ranging from \( 0.0001\,\text{s} \) to \( 0.1\,\text{s} \), path integration speed decreases as \( \tau \) increases.

### Effect of Conduction Delays

- **Axonal Conduction Delay (\( \Delta t \))**:
  - Determines the offset in synaptic connectivity and the timing of neuronal interactions.
  - Longer \( \Delta t \) values reduce the relative impact of rise time delays, improving path integration accuracy.

- **Simulation Findings**:
  - Shorter \( \Delta t \) values lead to reduced path integration speed due to the proportionally larger effect of rise time.
  - A distribution of \( \Delta t \) values (rather than a fixed value) can help mitigate periodic behavior and improve continuous packet movement.

### Periodic Behavior

- With a fixed \( \Delta t \), the HD and COMB activity packets exhibit periodic shifts every \( 2 \Delta t \) and \( \Delta t \) respectively.
- Introducing a distribution of \( \Delta t \) values results in smoother, continuous movement of the activity packets.

## Conclusions

- **Absence of Recurrent Excitation**:
  - Removing recurrent excitatory connections within the HD layer eliminates the retardation effect on packet movement.
  - This allows for highly accurate path integration without the need for velocity-specific recurrent connections.

- **Role of Conduction Delays**:
  - Axonal conduction delays serve as a natural timing mechanism, enabling precise updates of the HD representation.
  - The use of delays in both synaptic transmission and learning rules enhances the model's accuracy.

- **Self-Organization**:
  - The model can self-organize using local Hebbian learning rules, developing the necessary synaptic connectivity for accurate path integration.
  - Some reduction in accuracy is observed due to imperfections in learned connections, suggesting areas for further refinement.

- **Biological Plausibility**:
  - The model aligns with observed physiology, where HD cells lack recurrent excitatory connections.
  - It provides a theoretical explanation for how the HD system can maintain accurate path integration through learned reciprocal connections with COMB cells.

## Mathematical Insights

- **Offset Connectivity**:
  - The offset \( O = V \Delta t \) is crucial for aligning synaptic connections with expected future HD activity.
  - This relationship ensures that the HD packet shifts appropriately during rotation.

- **Timing Relationships**:
  - The periodicity of packet shifts is directly related to \( \Delta t \) and \( \tau \).
  - Understanding these relationships allows for precise control of packet movement within the network.

- **Normalization of Synaptic Weights**:
  - Ensures stability and prevents runaway growth in synaptic strengths.
  - Maintains balance in the network's learning dynamics.

