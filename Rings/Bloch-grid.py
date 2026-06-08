import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc
import qutip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO

tc.set_dtype("complex64")
tc.set_backend("jax")

# ============================================================
# PARAMS
# ============================================================

Nqubits  = 4
Nancilla = 3
Ndataq   = 1
L        = 25
T        = 20
Nstates  = 1000
key      = jax.random.PRNGKey(42)

# ============================================================
# LOAD DATA
# ============================================================

params_all = jnp.array(np.load("params.npy"))       # (T, L, 2*Nqubits)
bck_distr  = np.load("bck_distr.npy")               # (T+1, n_mus, Nstates, 2)
mus        = np.load("conditioning_params.npy")      # (n_mus, 2)
all_input  = np.load("all_input.npy")                # (n_mus, Nstates, 2)

n_mus = len(mus)

# ============================================================
# CIRCUIT HELPERS
# ============================================================

def add_ancilla(state, Nancilla):
    ancilla = jnp.zeros((2**Nancilla,), dtype=state.dtype).at[0].set(1)
    return jnp.kron(state, ancilla)
batched_addancilla = jax.vmap(add_ancilla, in_axes=(0, None))

def measure_ancillas_single(key, state, na, n):
    reshaped = jnp.reshape(state, (2**n, 2**na))
    probs    = jnp.sum(jnp.abs(reshaped)**2, axis=0)
    key, subkey = jax.random.split(key)
    m_res    = jax.random.categorical(subkey, jnp.log(probs + 1e-12))
    post     = reshaped[:, m_res]
    return post / (jnp.linalg.norm(post) + 1e-12)

def transform_singlestate(params, mu, current_state, key):
    c = tc.Circuit(Nqubits, inputs=current_state)
    for i in range(Nancilla):
        c.ry(Ndataq + i, theta=mu[0])
        c.rz(Ndataq + i, theta=mu[1])
    for l in range(L):
        for i in range(Nqubits):
            c.rx(i, theta=params[l][2*i])
            c.ry(i, theta=params[l][2*i + 1])
        for i in range(Nqubits - 1):
            c.cz(i, i+1)
    full_state = c.state()
    return measure_ancillas_single(key, full_state, Nancilla, Ndataq)

batched_transform = jax.jit(jax.vmap(transform_singlestate, in_axes=(None, None, 0, 0)))

# ============================================================
# GENERATE TEST DISTRIBUTIONS
# ============================================================

def random_bloch_states(key, N):
    key_u, key_phi = jax.random.split(key)
    u     = jax.random.uniform(key_u,   (N,))
    phi   = 2 * jnp.pi * jax.random.uniform(key_phi, (N,))
    theta = jnp.arccos(1 - 2 * u)
    alpha = jnp.cos(theta / 2)
    beta  = jnp.exp(1j * phi) * jnp.sin(theta / 2)
    return jnp.stack([alpha, beta], axis=1)

def apply_denoising(random_states, mu, params_all, key):
    states = random_states
    for step in range(T):
        key, subkey = jax.random.split(key)
        keys   = jax.random.split(subkey, Nstates)
        states = batched_transform(
            params_all[step], mu,
            batched_addancilla(states, Nancilla), keys)
    return states

print("Generating test distributions...")
test_states = []
for i, mu in enumerate(mus):
    key, subkey1, subkey2 = jax.random.split(key, 3)
    random_init = random_bloch_states(subkey1, Nstates)
    result      = apply_denoising(random_init, jnp.array(mu), params_all, subkey2)
    test_states.append(np.array(result))
    print(f"  mu {i+1}/{n_mus} done")

# ============================================================
# BLOCH VECTOR HELPER
# ============================================================

def bloch_vectors(states_array):
    """(Nstates, 2) → (3, Nstates)"""
    points = []
    for sv in states_array:
        ket = qutip.Qobj(np.array(sv, dtype=complex).reshape(2, 1))
        dm  = ket * ket.dag()
        points.append([
            float(np.real((dm * qutip.sigmax()).tr())),
            float(np.real((dm * qutip.sigmay()).tr())),
            float(np.real((dm * qutip.sigmaz()).tr())),
        ])
    return np.array(points).T

# ============================================================
# RENDER ONE BLOCH SPHERE → buffer
# ============================================================

def render_bloch(points, color, point_size=6):
    b = qutip.Bloch()
    b.point_color  = [color]
    b.point_marker = ['o']
    b.point_size   = [point_size]
    b.sphere_alpha = 0.04
    b.frame_alpha  = 0.15
    b.add_points(points, meth='s')
    b.render()
    buf = BytesIO()
    b.fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(b.fig)
    return buf

# ============================================================
# BUILD BUFFERS
# rows = [target, train, test]
# cols = one per mu
# ============================================================

ROW_COLORS = {
    "target" : "#C0392B",
    "train"  : "#2471A3",
    "test"   : "#1E8449",
}
ROW_LABELS = {
    "target" : "Target",
    "train"  : "Training output",
    "test"   : "Testing output",
}

rows    = ["target", "train", "test"]
buffers = {r: [] for r in rows}

print("Rendering Bloch spheres...")
for i in range(n_mus):
    # Target: ideal input states (before scrambling)
    target_pts = bloch_vectors(all_input[i])
    buffers["target"].append(render_bloch(target_pts, ROW_COLORS["target"]))

    # Training: recovered distribution after full denoising
    train_pts = bloch_vectors(bck_distr[0][i])
    buffers["train"].append(render_bloch(train_pts, ROW_COLORS["train"]))

    # Test: random Haar states denoised with trained params
    test_pts = bloch_vectors(test_states[i])
    buffers["test"].append(render_bloch(test_pts, ROW_COLORS["test"]))

    print(f"  mu {i+1}/{n_mus} done")

# ============================================================
# ASSEMBLE FIGURE
# ============================================================

col_labels = [f"({chr(ord('a') + i)})" for i in range(n_mus)]

fig, axes = plt.subplots(
    len(rows), n_mus,
    figsize=(2.2 * n_mus, 2.4 * len(rows)),
)
fig.patch.set_facecolor("white")

# Handle case of single column
if n_mus == 1:
    axes = axes.reshape(-1, 1)

for r_idx, row_key in enumerate(rows):
    for c_idx in range(n_mus):
        ax = axes[r_idx, c_idx]
        ax.imshow(mpimg.imread(buffers[row_key][c_idx]))
        ax.axis('off')

        # Column labels on top row
        if r_idx == 0:
            ax.set_title(col_labels[c_idx], fontsize=10,
                         pad=3, color="#161A26",
                         fontfamily="serif")

    # Row label on left
    axes[r_idx, 0].annotate(
        ROW_LABELS[row_key],
        xy=(0, 0.5), xycoords="axes fraction",
        xytext=(-8, 0), textcoords="offset points",
        ha="right", va="center",
        fontsize=8, color=ROW_COLORS[row_key],
        fontfamily="serif", rotation=90,
    )

plt.subplots_adjust(wspace=0.02, hspace=0.05)
plt.savefig("bloch_grid.pdf", dpi=200, bbox_inches="tight")
plt.savefig("bloch_grid.png", dpi=200, bbox_inches="tight")
print("Saved → bloch_grid.pdf / bloch_grid.png")