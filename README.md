# Conditional Quantum Diffusion Denoising (CQDD)

A quantum machine learning framework for learning and generating distributions of quantum states using diffusion models with conditioning mechanisms.

## Overview

This repository implements a Conditional Quantum Diffusion Denoising (CQDD) model that leverages quantum computing to prepare, characterize, and manipulate quantum states. The model combines:

- **Forward Diffusion Process**: Transforms structured quantum state distributions into Haar-random distributions using quantum scrambling circuits
- **Backward Denoising Process**: Reconstructs the original distributions from noisy states using parametrized quantum circuits (PQC)
- **Conditioning Mechanism**: Enables the generation of multiple distribution classes with a single training process

## Key Features

✨ **Continuous Conditioning**: Generate quantum states for unseen conditioning parameters through learned interpolation

🎯 **Low Generalization Error**: Achieves low testing errors for unseen conditioning parameters

⚡ **JAX + TensorCircuit Integration**: High-performance quantum circuit simulation with automatic differentiation and JIT compilation

📊 **Multiple Distribution Geometries**: Support for:
  - Polar points on the Bloch sphere
  - Parallel ring structures
  - Rotated ring configurations

## Web deployment
A trained version of the model with three examples is presented in an interactive web at the repository dagadd0/QML-CQDDapp


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
python cqdd_polarpoints_v6.py
```

This script:
1. Prepares 6 polar point distributions on the Bloch sphere
2. Applies forward scrambling to generate Haar-random distributions
3. Trains the denoising model to reconstruct the original distributions
4. Saves trained parameters and results

### Training on Ring Structures

```bash
python cqdd_rings_v3.py
```

Generates parallel or rotated ring distributions and trains the model accordingly.

### Testing on Unseen Conditioning

```bash
python Testing_unseen.py
python Testing_unseen1D.py
```

Evaluates model generalization on conditioning parameters not seen during training.
- Heatmaps of generalization error

### Visualization

```bash
python ploting.py
```

Generates visualizations including:
- Bloch sphere representations
- Loss curves during training


### Conditioning Parameters

The model is conditioned on angles μ = (μ₁, μ₂) ∈ [0,π] × [0,2π] representing:
- μ₁: Polar angle on Bloch sphere
- μ₂: Azimuthal angle on Bloch sphere

## Output Files

After training, the following files are saved:

- `params.npy`: Trained PQC parameters for each denoising step
- `bck_distr.npy`: Reconstructed state distributions at each step
- `all_input.npy`: Initial target distributions
- `conditioning_params.npy`: Conditioning parameters used
- `best_losses_t.npy`: Best loss at each denoising step
- `best_losses_target.npy`: Loss relative to target distributions
- `*.png`, `*.pdf`: Visualization outputs (Bloch spheres, loss curves, heatmaps)


## Methodology

The CQDD model implements a quantum generative process:

1. **Forward Process**: Apply controlled noise via quantum scrambling
2. **Condition Encoding**: Encode μ in ancilla qubits
3. **PQC Transformation**: Learn to reverse the scrambling process
4. **Measurement**: Measure and discard ancilla qubits
5. **Optimization**: Minimize MMD between generated and target distributions

See the included research paper/thesis for detailed theoretical foundation.


**Status**: Active Development
**Last Updated**: June 2026
