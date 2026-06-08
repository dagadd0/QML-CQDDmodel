# Conditional Quantum Diffusion Denoising (CQDD)

A quantum machine learning framework for learning and generating distributions of quantum states using diffusion models with conditioning mechanisms.

## Overview

This repository implements a Conditional Quantum Diffusion Denoising (CQDD) model that leverages quantum computing to prepare, characterize, and manipulate quantum states. The model combines:

- **Forward Diffusion Process**: Transforms structured quantum state distributions into Haar-random distributions using quantum scrambling circuits
- **Backward Denoising Process**: Reconstructs the original distributions from noisy states using parametrized quantum circuits (PQC)
- **Conditioning Mechanism**: Enables the generation of multiple distribution classes with a single training process

## Key Features

✨ **Continuous Conditioning**: Generate quantum states for unseen conditioning parameters through learned interpolation

🎯 **Low Generalization Error**: Achieves testing errors below 0.4% for unseen conditioning parameters, significantly outperforming previous approaches

⚡ **JAX + TensorCircuit Integration**: High-performance quantum circuit simulation with automatic differentiation and JIT compilation

📊 **Multiple Distribution Geometries**: Support for:
  - Polar points on the Bloch sphere
  - Parallel ring structures
  - Rotated ring configurations

## Project Structure

```
.
├── Cluster states/          # Experiments on cluster/polar point distributions
│   ├── RESULTS-6mu/        # Results and visualizations
│   ├── cqdd_polarpoints_v6.py
│   ├── Testing_unseen.py
│   └── ploting.py
│
├── Rings/                   # Experiments on ring-shaped distributions
│   ├── RESULTS_paralel-L=25-N=500/
│   ├── RESULTS_ringXYrotated-L=45-N=500/
│   ├── cqdd_rings_v3.py
│   ├── Testing_unseen1D.py
│   ├── Bloch-grid.py
│   └── ploting.py
│
└── README.md
```

## Installation

### Requirements

- Python 3.8+
- TensorCircuit
- JAX
- Optax
- NumPy

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/CQDD.git
cd CQDD

# Install dependencies
pip install tensorcircuit jax jaxlib optax numpy matplotlib

# For GPU acceleration (optional)
pip install jax[cuda12]
```

## Usage

### Training on Polar Points Distribution

```bash
cd "Cluster states"
python cqdd_polarpoints_v6.py
```

This script:
1. Prepares 6 polar point distributions on the Bloch sphere
2. Applies forward scrambling to generate Haar-random distributions
3. Trains the denoising model to reconstruct the original distributions
4. Saves trained parameters and results

### Training on Ring Structures

```bash
cd Rings
python cqdd_rings_v3.py
```

Generates parallel or rotated ring distributions and trains the model accordingly.

### Testing on Unseen Conditioning

```bash
python Testing_unseen.py
python Testing_unseen1D.py
```

Evaluates model generalization on conditioning parameters not seen during training.

### Visualization

```bash
python ploting.py
```

Generates visualizations including:
- Bloch sphere representations
- Loss curves during training
- Heatmaps of generalization error
- Denoising and scrambling process snapshots

## Model Architecture

### Quantum Circuit Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Qubits | 3 | Total qubits (1 data + 2 ancilla) |
| Ancilla | 2 | Measurement ancillas |
| Data Qubits | 1 | Information-carrying qubits |
| Denoising Steps (T) | 20 | Forward/backward diffusion steps |
| Layers (L) | 12 | PQC layers per denoising step |
| States (N) | 500 | Distribution size |

### Conditioning Parameters

The model is conditioned on angles μ = (μ₁, μ₂) ∈ [0,π] × [0,2π] representing:
- μ₁: Polar angle on Bloch sphere
- μ₂: Azimuthal angle on Bloch sphere

## Results

### Polar Points
- **Training Error**: ~10⁻⁵ - 10⁻⁴
- **Testing Error (Unseen)**: ~0.4% (MMD normalized)
- **Improvement over baselines**: ~20x reduction compared to Quinn et al. (0.8% - 6.4%)

### Ring Structures
- Stable generalization across conditioning parameter space
- Error increases at poles due to state collapse
- Smooth interpolation between trained points

## Key Metrics

The model uses **Maximum Mean Discrepancy (MMD)** with fidelity kernel to measure distance between quantum state distributions:

```
MMD(X, Y) = 2⟨K(X,Y)⟩ - ⟨K(X,X)⟩ - ⟨K(Y,Y)⟩
```

where K(ψ,φ) = |⟨ψ|φ⟩|² (fidelity kernel)

## Training Configuration

```python
Nopt_loops = 1000          # Optimization loops per denoising step
learning_rate = 0.01       # Adam optimizer learning rate
optimizer = optax.adam     # Gradient-based optimization
schedule = "quadratic"     # Noise schedule: linear/exponential/quadratic
epsilon = 0.08            # Initial noise level for distributions
```

## Output Files

After training, the following files are saved:

- `params.npy`: Trained PQC parameters for each denoising step
- `bck_distr.npy`: Reconstructed state distributions at each step
- `all_input.npy`: Initial target distributions
- `conditioning_params.npy`: Conditioning parameters used
- `best_losses_t.npy`: Best loss at each denoising step
- `best_losses_target.npy`: Loss relative to target distributions
- `*.png`, `*.pdf`: Visualization outputs (Bloch spheres, loss curves, heatmaps)

## Performance Notes

- **Memory**: ~2-4 GB GPU memory for standard configuration
- **Training Time**: ~4-8 hours per full run (20 denoising steps)
- **JAX Compilation**: First run includes JIT compilation overhead

## Methodology

The CQDD model implements a quantum generative process:

1. **Forward Process**: Apply controlled noise via quantum scrambling
2. **Condition Encoding**: Encode μ in ancilla qubits
3. **PQC Transformation**: Learn to reverse the scrambling process
4. **Measurement**: Measure and discard ancilla qubits
5. **Optimization**: Minimize MMD between generated and target distributions

See the included research paper/thesis for detailed theoretical foundation.

## References

The implementation is based on quantum diffusion models and conditional generation techniques:

- Tensor Network Diffusion for Quantum State Generation
- Quantum Machine Learning: Classical vs Quantum Approaches
- Variational Quantum Algorithms for State Preparation

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest improvements
- Submit pull requests

## Contact

For questions or collaboration inquiries, please reach out via GitHub Issues.

---

**Status**: Active Development
**Last Updated**: June 2026