# Path Integration of Head Direction Using Axonal Conduction Delays

## Introduction

Head direction (HD) cells signal the orientation of an animal's head in the horizontal plane. Even without visual input, the HD cell system can accurately represent current head direction through **path integration**, which relies on self-motion (idiothetic) cues. Understanding how the HD system learns to perform accurate path integration is crucial.

## The Model

The proposed model consists of two interconnected neural networks:

1. **Head Direction Cells Network**: Forms a continuous attractor representing all possible head directions. Each cell has a preferred direction and exhibits a Gaussian tuning curve.

2. **Combination Cells Network**: Receives inputs from both the HD cells and rotational velocity (ROT) cells. These cells learn to represent combinations of specific head directions and rotational velocities.

### Rate-Coded Neurons

Each neuron is modeled using rate coding, where the neuron's state is described by its firing rate \( r_i(t) \). The neuron's activation \( h_i(t) \) evolves according to:

$$
\tau \frac{dh_i(t)}{dt} = -h_i(t) + \sum_j w_{ij} r_j(t - \Delta t) + I_{\text{external}}(t)
$$

- \( \tau \): Membrane time constant.
- \( w_{ij} \): Synaptic weight from presynaptic neuron \( j \) to postsynaptic neuron \( i \).
- \( \Delta t \): Axonal conduction delay.
- \( I_{\text{external}}(t) \): External input.

The firing rate \( r_i(t) \) is given by a sigmoid function:

$$
r_i(t) = \frac{1}{1 + e^{-2\beta(h_i(t) - \alpha)}}
$$

- \( \alpha \): Threshold.
- \( \beta \): Slope parameter.

## Axonal Conduction Delays as Timing Mechanism

Axonal conduction delays \( \Delta t \) introduce specific time intervals between presynaptic and postsynaptic activities. This mechanism allows:

- Learning of temporal associations over fixed delays.
- Accurate timing for updating the HD representation during path integration.

## Learning Rules

Synaptic weights are updated using Hebbian-like learning rules that incorporate the conduction delay \( \Delta t \).

### Recurrent Connections in HD Cells (\( w^{(1)} \))

Weights \( w^{(1)}_{ij} \) among HD cells are updated by:

$$
\frac{dw^{(1)}_{ij}(t)}{dt} = k_1 \, r_i^{\text{HD}}(t) \, r_j^{\text{HD}}(t)
$$

- \( k_1 \): Learning rate for recurrent HD connections.

After training, these weights form symmetric profiles, ensuring a stable packet of activity when the agent is stationary.

### Feedforward Connections from HD to Combination Cells (\( w^{(3)} \))

Weights \( w^{(3)}_{ij} \) from HD cells to combination cells:

$$
\frac{dw^{(3)}_{ij}(t)}{dt} = k_3 \, r_i^{\text{COMB}}(t) \, r_j^{\text{HD}}(t - \Delta t)
$$

- \( k_3 \): Learning rate for HD to combination cells.

COMB cells learn to respond to specific combinations of head direction and rotational velocity due to the conjunctive input from HD cells and ROT cells.

### Feedback Connections from Combination to HD Cells (\( w^{(2)} \))

Weights \( w^{(2)}_{ij} \) from combination cells to HD cells:

$$
\frac{dw^{(2)}_{ij}(t)}{dt} = k_2 \, r_i^{\text{HD}}(t) \, r_j^{\text{COMB}}(t - \Delta t)
$$

- \( k_2 \): Learning rate for combination to HD cells.

These weights are asymmetric, reflecting the learned association between previous and future head directions over the conduction delay \( \Delta t \).

### Connections from Rotational Velocity to Combination Cells (\( w^{(4)} \))

Weights \( w^{(4)}_{ij} \) from rotational velocity cells to combination cells:

$$
\frac{dw^{(4)}_{ij}(t)}{dt} = k_4 \, r_i^{\text{COMB}}(t) \, r_j^{\text{ROT}}(t)
$$

- \( k_4 \): Learning rate for rotational velocity connections.

COMB cells receive significant input only from ROT cells signaling either clockwise or counter-clockwise rotation, leading to step-function-like weight profiles.

### Synaptic Weight Normalization

After updates, weights are normalized to prevent unbounded growth:

$$
\sqrt{\sum_j (w_{ij}(t))^2} = 1
$$

## Operation of the Model

### Path Integration Mechanism

During training (with visual input):

1. **Learning Associations**: The model learns associations between head directions at times \( t \), \( t + \Delta t \), and \( t + 2\Delta t \).

2. **Forward Propagation**: HD cells activate combination cells after a delay \( \Delta t \).

3. **Backward Propagation**: Combination cells activate HD cells after another delay \( \Delta t \).

Thus, the model learns to associate a head direction \( h(t) \) with a future head direction \( h(t + 2\Delta t) \), effectively encoding the dynamics of head rotation.

### Updating Neural Activity

In the absence of visual input (testing phase):

- **Rotational Velocity Input**: Active ROT cells indicate head rotation direction and speed.

- **Combination Cell Activation**: COMB cells become active due to inputs from both HD and ROT cells.

- **HD Packet Shift**: The HD activity packet shifts position, updating the internal representation of head direction accurately over time.

## Simulations and Results

### Demonstration of Core Model Performance

#### Simulation Setup

The model simulates an agent rotating at a velocity of \(180^\circ/\text{s}\) during training with visual input. Axonal conduction delays are set to \(\Delta t = 100\,\text{ms}\). The primary goal is to test if the model can update the packet of HD cell activity at the same speed during testing (without visual input) as during training.

#### Synaptic Weights after Training

- **Recurrent Weights within HD Cells (\( w^{(1)} \))**: After training, the weight profiles are symmetric, ensuring stability when stationary.

- **Weights from HD Cells to COMB Cells (\( w^{(3)} \))**: COMB cells receive inputs from specific HD cells, learning to respond to particular head directions and rotational velocities.

- **Weights from COMB Cells to HD Cells (\( w^{(2)} \))**: The weight profiles are asymmetric, allowing the HD packet to shift during rotation.

- **Weights from ROT Cells to COMB Cells (\( w^{(4)} \))**: COMB cells receive significant input only from ROT cells signaling rotation, leading to step-function-like weight profiles.

#### Firing Rates during Training and Testing

- **Training Phase**: Visual input drives HD cells, and the agent's rotation causes the HD activity packet to shift.

- **Testing Phase**: Without visual input:

  - **Stable Packet Maintenance**: When ROT cells are inactive, the HD packet remains stable.

  - **Packet Shifting**: When ROT cells are active, COMB cells stimulate HD cells representing future head directions, causing the packet to shift.

#### Measuring the Speed of Path Integration

The speed at which the HD activity packet moves is calculated using:

$$
\text{speed} = \frac{p_2 - p_1}{t_2 - t_1}
$$

where \( p \) is the position of the packet:

$$
p = \frac{\sum_i r_i h_i}{\sum_i r_i}
$$

- \( r_i \): Firing rate of HD cell \( i \).
- \( h_i \): Preferred head direction of HD cell \( i \).

#### Results

Multiple simulations showed that the mean speed during testing was approximately \(144.1^\circ/\text{s}\) for clockwise rotation and \(132.3^\circ/\text{s}\) for counter-clockwise rotation, compared to the training speed of \(180^\circ/\text{s}\). This indicates the model learned to perform path integration at approximately \(80\%\) of the training speed.

### Effect of Conduction Delays and Rotational Velocities

#### Simulations with Different Parameters

The model was tested with varying conduction delays (\(\Delta t = 50\,\text{ms}, 100\,\text{ms}\)) and rotational velocities (\(180^\circ/\text{s}, 360^\circ/\text{s}\)).

#### Observations

- With \(\Delta t = 100\,\text{ms}\) and training speed \(360^\circ/\text{s}\), the model achieved \(80\%\) of the training speed during testing.

- With \(\Delta t = 50\,\text{ms}\) and training speed \(180^\circ/\text{s}\), the model achieved approximately \(70\%\) of the training speed.

#### Conclusion

The conduction delay \(\Delta t\) is a crucial parameter affecting the accuracy of path integration. Longer delays allowed the model to better approximate the training speed during testing.

### Distribution of Axonal Conduction Delays

The model was tested with a uniform distribution of delays in the range \([1\,\text{ms}, 100\,\text{ms}]\). Despite the variability, the model successfully learned to perform path integration, achieving approximately \(60\%\) of the training speed.

### Conduction Delays in One Set of Synapses

#### Delays in \( w^{(2)} \) Synapses Only

With delays only in the \( w^{(2)} \) synapses, the model still performed path integration, achieving around \(63\%\) of the training speed.

#### Delays in \( w^{(3)} \) Synapses Only

Similarly, with delays only in \( w^{(3)} \) synapses, the model achieved approximately \(66\%\) of the training speed.

#### Conclusion

Conduction delays in either set of synapses (\( w^{(2)} \) or \( w^{(3)} \)) are sufficient for the model to learn path integration, but having delays in both enhances performance.

### Importance of Conduction Delays

When conduction delays were set to zero (\( \Delta t = 0 \)), the model failed to learn path integration. This demonstrates that non-zero conduction delays are vital for the temporal associations required in learning accurate path integration.

### Generalization and Robustness

#### Lack of Generalization

The model does not generalize to unlearned rotational speeds by simply scaling the ROT cell firing rates. This is because the learned associations are specific to the rotational velocities experienced during training.

#### Fault Tolerance

The model exhibits robustness to cell loss. When \(25\%\) of the ROT cells were randomly removed after training, the model's performance in path integration was largely unaffected, demonstrating fault tolerance.

#### Comparison with Previous Models

Unlike models that generalize to different speeds but are sensitive to cell loss, the current model's lack of generalization contributes to its robustness.

### Comparison with Previous Model by Walters and Stringer (2010)

#### Learning Rules

- **Current Model**: Incorporates time-delayed Hebbian learning rules with conduction delays \( \Delta t \) in both the synaptic transmissions and learning updates.

- **Previous Model**: Used standard Hebbian learning without explicit conduction delays in the learning rules.

#### Convergence

The inclusion of \( \Delta t \) in the learning rules allows the current model to converge faster during training, as the temporal associations are consistently learned over the actual delays present in synaptic transmission.

#### Robustness

The current model maintains performance with up to \(40\%\) of ROT cells removed, whereas the previous model showed decreased accuracy with increasing cell loss.

## Implications

- **Natural Timing via Delays**: Axonal conduction delays serve as a biological timing mechanism for learning temporal sequences.

- **Look-Up Table Formation**: The model effectively creates a look-up table for path integration speeds through learned associations.

- **Fault Tolerance**: Learning over fixed delays enhances robustness, as the model does not rely on precise parameter tuning.

## Conclusion

By incorporating axonal conduction delays into the learning rules, the model provides a biologically plausible mechanism for the accurate path integration of head direction. This approach emphasizes the importance of temporal delays in neural computation and learning.

