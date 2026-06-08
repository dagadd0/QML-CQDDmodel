# Core libraries
import tensorcircuit as tc
import jax
import jax.numpy as jnp
import optax
import numpy as np
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import time
from pathlib import Path

tc.set_dtype("complex64")
tc.set_backend("jax")
BASE_DIR = Path(__file__).resolve().parent
Nstates = N_points = Npoints= 300    # Number of elements in the distribution
Nqubits=3                   # Number of qubits of each circuit element (including ancilla)
Nancilla=2                  # Number of ancillas
Ndataq = 1                  # Number of data Qubits
dim=2**Nqubits              # Dimension of the state of each circuit
T = 20                      # Steps of Scrambling
L= 12                      # Layers of at each step of denoising
key = jax.random.PRNGKey(0) # Key for deterministic randomness given by (seed)
epsilon = 0.04              # Epsilon for the polar points distribution, determines amount of noise
opt = optax.adam(learning_rate=0.001)
params = jnp.array(np.load("params.npy"))
print(params.shape)

mus = jnp.array(np.load("conditioning_params.npy"))

# DENOISING CIRCUIT
def add_ancilla(state,Nancilla):
    """
    Takes:
        state: data qubits state (2**Ndataq,)
    Returns: 
        states with ancilla (2**Nqubits,)
    """
    ancilla = jnp.zeros((2**Nancilla,),dtype=state.dtype).at[0].set(1)
    return jnp.kron(state, ancilla)
batched_addancilla = jax.vmap(add_ancilla, in_axes=(0,None))  # Takes (Nstates 2**Ndataq)

def measure_ancillas_single(key, state, na, n):
    """
    Takes:
        key: key for randomness
        state: state with ancillas to measure (2**(n+na),)
        ma: number of ancillas
        n: number of data qubits
    Returns:
        post_state: state after ancila measurement (2**n,) 
    """
    # Separar ancilla y sistema
    reshaped = jnp.reshape(state, (2**n, 2**na))
    probs = jnp.sum(jnp.abs(reshaped) ** 2, axis=0)  # suma sobre dato → probs ancilla, shape (4,)
    key, subkey = jax.random.split(key)
    m_res = jax.random.categorical(subkey, jnp.log(probs + 1e-12))
    post_state = reshaped[:, m_res]               # dato post-medición, shape (2,) ✓
    norm = jnp.linalg.norm(post_state)
    return post_state / (norm + 1e-12)

def transform_singlestate(params, mu, current_state, Nancillaa, Ndataqq, Nqubitss, key):
    """
    Takes:
        params: angles for rotations
        mu: conditioning
        current_state: state vector for circuit at step t, with ancila [2**Nqubits]
        key for randomness
    Returns:
        transformed state [2**Nqubits]
    """
    c = tc.Circuit(Nqubits, inputs=current_state)
    for i in range(Nancilla):
        c.ry(Ndataq+i, theta = mu[0])
        c.rz(Ndataq+i, theta = mu[1])
    for l in range(L):
        for i in range(Nqubits):
            c.rx(i, theta=params[l][2*i])
            c.ry(i, theta=params[l][2*i + 1])
        for i in range(Nqubits - 1):
            c.cz(i, i+1)
    full_state = c.state()
    data_state= measure_ancillas_single(key,full_state,Nancilla, Ndataq)
    return data_state # Shape: [2**Nqubits]
batched_transform =  jax.jit(jax.vmap(transform_singlestate, in_axes=(None, None, 0,None,None,None,0)))


def random_bloch_states(key, N):
    """
    Genera N estados aleatorios uniformes en la esfera de Bloch (Haar measure).
    Returns:
        psi: jax.Array complejo de shape (N, 2)
    """
    key_theta, key_phi = jax.random.split(key)
    # theta: distribución correcta para uniformidad en esfera
    u = jax.random.uniform(key_theta, (N,))
    theta = jnp.arccos(1 - 2 * u)
    # phi uniforme
    phi = 2 * jnp.pi * jax.random.uniform(key_phi, (N,))
    # amplitudes del qubit
    alpha = jnp.cos(theta / 2)
    beta = jnp.exp(1j * phi) * jnp.sin(theta / 2)
    psi = jnp.stack([alpha, beta], axis=1)
    return psi


# MMD DISTANCE
@jax.jit
def MMD(X, Y):
    """ Calculate Maximum mean discrepancy between two distributions
    Takes:
        X
        Y
    Returns:
        MMD distance between X and Y using fidelity as kernel
    """
    N = X.shape[0]
    M = Y.shape[0]

    # kernel: fidelity |<psi|phi>|^2 
    overlapXY = jnp.abs(X @ Y.conj().T)**2 
    overlapXX = jnp.abs(X @ X.conj().T)**2
    overlapYY = jnp.abs(Y @ Y.conj().T)**2

    sumXY = 1-jnp.mean(overlapXY)
    sumXX = 1-jnp.sum(overlapXX) / (N * N)
    sumYY = 1-jnp.sum(overlapYY) / (M * M)
    return 2*sumXY - sumYY - sumXX

polarpoint_map = {
    "0":  jnp.array([1,0],dtype=jnp.complex64),
    "1":  jnp.array([0,1],dtype=jnp.complex64),
    "+":  jnp.array([1,1],dtype=jnp.complex64) / jnp.sqrt(2),
    "-":  jnp.array([1,-1],dtype=jnp.complex64) / jnp.sqrt(2),
    "+i": jnp.array([1,1j],dtype=jnp.complex64) / jnp.sqrt(2),
    "-i": jnp.array([1,-1j],dtype=jnp.complex64) / jnp.sqrt(2)
}
polar_points_array = jnp.stack(list(polarpoint_map.values()))
polar_kinds = list(polarpoint_map.keys())
def init_polarpoint(Nstates, Ndata, kind_idx, key, epsilon):
    """ Creates a Polar point distribution
    Takes:
        Nstates: number of states
        Ndata: number of data qubits
        kind: initial state to prepare from polarpoint_map list
        key: determined by seed for reproducible randomness
        epsilon: amount of noise introduced in the distribution
    Returns:
        Polar Point distribution of shape (Nstates, 2**Ndataq) 
    """
    subkey1, subkey2 = jax.random.split(key,2)

    # Set the base of states for selected polar point
    qubit = polar_points_array[kind_idx]         
    base  = jnp.tile(qubit, (Ndata, 1))            

    # Add noise to qubits
    states = base +  (1 * jax.random.normal(subkey1, shape=(Nstates,2**Ndata)) 
        + 1j * jax.random.normal(subkey2, shape=(Nstates,2**Ndata)) ) * epsilon 
    states /= jnp.linalg.norm(states, axis=1, keepdims=True) # Normalise
    return states
batched_init_polarpoint = jax.vmap(init_polarpoint, in_axes=(None,None,0,0,None))




# ─────────────────────────────────────────────────────────────────
# 1. Distribución teórica para cualquier punto de la esfera de Bloch
# ─────────────────────────────────────────────────────────────────
def theoretical_distribution(theta, phi, N, key, epsilon):
    alpha = jnp.cos(theta / 2).astype(jnp.complex64)
    beta  = (jnp.exp(1j * phi) * jnp.sin(theta / 2)).astype(jnp.complex64)
    point = jnp.array([alpha, beta], dtype=jnp.complex64)

    k1, k2 = jax.random.split(key)
    real_part = jax.random.normal(k1, (N, 2)).astype(jnp.complex64)
    imag_part = jax.random.normal(k2, (N, 2)).astype(jnp.complex64)
    noise = (real_part + 1j * imag_part) * jnp.float32(epsilon)

    states = jnp.tile(point, (N, 1)) + noise
    return (states / jnp.linalg.norm(states, axis=1, keepdims=True)).astype(jnp.complex64)

# ─────────────────────────────────────────────────────────────────
# 2. Pasar el circuito completo (T pasos) para un mu dado
# ─────────────────────────────────────────────────────────────────
@jax.jit
def run_full_circuit(mu, initial_states, params, base_key):
    def step_fn(carry, step_params):
        current, key = carry
        keys   = jax.random.split(key, Nstates + 1)
        new_current = batched_transform(
            step_params, mu,
            batched_addancilla(current, Nancilla),
            Nancilla, Ndataq, Nqubits, keys[1:]
        )
        return (new_current, keys[0]), None

    (final_states, _), _ = jax.lax.scan(step_fn, (initial_states, base_key), params)
    return final_states


# ─────────────────────────────────────────────────────────────────
# 3. Calcular la rejilla de MMD
# ─────────────────────────────────────────────────────────────────
N_theta = 150
N_phi   = 300
thetas  = np.linspace(0,        np.pi, N_theta)
phis    = np.linspace(0, 2*np.pi, N_phi, endpoint=True)

mmd_grid = np.zeros((N_theta, N_phi))
key, master_key = jax.random.split(key,2)

total = N_theta * N_phi
start = time.time()
key, subkey= jax.random.split(key,2)
random_distr = random_bloch_states(subkey, Nstates)

master_key = jax.random.PRNGKey(99)
master_key, k_circuit_root, k_theory_root = jax.random.split(master_key, 3)

# Rejilla de mus
all_mus          = jnp.array([[th, ph] for th in thetas for ph in phis])
all_phis_flat    = jnp.array([ph for _ in thetas for ph in phis])
all_thetas_flat  = jnp.array([th for th in thetas for _ in phis])

# Keys
all_keys_circuit = jax.random.split(k_circuit_root, N_theta * N_phi)
all_keys_theory  = jax.random.split(k_theory_root,  N_theta * N_phi)

# Distribuciones teóricas
theo_vmap = jax.jit(jax.vmap(
    lambda th, ph, k: theoretical_distribution(th, ph, Nstates, k, epsilon),
    in_axes=(0, 0, 0)
))

print("Calculando distribuciones teóricas...")
all_theory = theo_vmap(all_thetas_flat, all_phis_flat, all_keys_theory)

print("Ejecutando circuitos (compilando en primer paso)...")
start = time.time()
run_grid = jax.jit(jax.vmap(run_full_circuit, in_axes=(0, None, None, 0)))
all_circuit_states = run_grid(all_mus, random_distr, params, all_keys_circuit)
all_circuit_states.block_until_ready()  # fuerza ejecución para medir tiempo real
print(f"Circuitos: {time.time()-start:.1f}s")

print("Calculando MMD...")
batch_size = 32
mmd_chunks = []
for i in range(0, len(all_circuit_states), batch_size):
    X_batch = all_circuit_states[i:i+batch_size]
    Y_batch = all_theory[i:i+batch_size]
    batch_vals = jax.vmap(MMD)(X_batch, Y_batch)
    mmd_chunks.append(np.array(batch_vals))
mmd_flat = np.concatenate(mmd_chunks)

mmd_grid = np.array(mmd_flat).reshape(N_theta, N_phi)



import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(
    mmd_grid,
    origin="lower", aspect="auto",
    extent=[0, 2*np.pi, 0, np.pi],
    cmap="inferno"
)
# --- marcar puntos mu entrenados ---
mu_theta = np.array(mus)[:, 0]
mu_phi   = np.array(mus)[:, 1]

ax.scatter(
    mu_phi, mu_theta,
    c="cyan",
    s=30,
    marker="o",
    label="Trained conditioning"
)

plt.subplots_adjust(right=0.82)
ax.legend(
    loc="lower left",
    bbox_to_anchor=(1.02, 1.02),
    borderaxespad=0
)


ax.set_xlabel("φ  [0, 2π]")
ax.set_ylabel("θ  [0, π]")
ax.set_title("MMD loss  —  distribución generada vs. teórica")

# ticks legibles
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
ax.set_yticks([0, np.pi/2, np.pi])
ax.set_yticklabels(["0", "π/2", "π"])


plt.colorbar(im, ax=ax, label="MMD")
plt.savefig("mmd_heatmap_2d.png", dpi=150)
plt.show()

import matplotlib.cm as cm

THETA, PHI = np.meshgrid(thetas, phis, indexing="ij")   # (N_theta, N_phi)
X = np.sin(THETA) * np.cos(PHI)
Y = np.sin(THETA) * np.sin(PHI)
Z = np.cos(THETA)

# Normalizar la MMD al rango [0,1] para el colormap
vmin, vmax = mmd_grid.min(), mmd_grid.max()
norm   = plt.Normalize(vmin=vmin, vmax=vmax)
colors = cm.inferno(norm(mmd_grid))        # (N_theta, N_phi, 4)

fig = plt.figure(figsize=(8, 8))
ax  = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    X, Y, Z,
    facecolors=colors,
    rstride=1, cstride=1,
    linewidth=0, antialiased=True, alpha=0.9
)

ax.set_box_aspect([1, 1, 1])


# Añadir colorbar manualmente
sm = cm.ScalarMappable(cmap="inferno", norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.01, label="MMD loss")

# Ejes de referencia
for vec, lbl in zip([[1,0,0],[0,1,0],[0,0,1]], ["X","Y","Z"]):
    ax.quiver(0,0,0,*vec, length=1.3, color="grey", linewidth=0.7, arrow_length_ratio=0.1)
    ax.text(vec[0]*1.4, vec[1]*1.4, vec[2]*1.4, lbl, fontsize=10, color="grey")

ax.set_title("MMD loss pintado sobre la esfera de Bloch")
ax.set_axis_off()
plt.tight_layout()
plt.savefig("mmd_bloch_sphere.png", dpi=150)
plt.show()