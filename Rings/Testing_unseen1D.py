import tensorcircuit as tc
import jax
import jax.numpy as jnp
import numpy as np
import functools
import time
import matplotlib.pyplot as plt

tc.set_dtype("complex64")
tc.set_backend("jax")

Nstates  = 500
Nqubits  = 4
Nancilla = 3
Ndataq   = 1
T        = 20
L        = 25
key      = jax.random.PRNGKey(1)

params = jnp.array(np.load("params.npy"))
mus    = jnp.array(np.load("conditioning_params.npy"))

# Take only mu1, discard mu2
training_mu1 = np.array(mus[:, 0])
mu2_fixed    = 0.0
print(f"Fixed mu2 = {mu2_fixed}")
print(f"Training mu1 values: {training_mu1}")

# ============================================================
# DISTRIBUTION HELPERS
# ============================================================

# RINGS:
def base_ring(phi):
    return jnp.stack([
        jnp.ones_like(phi) / jnp.sqrt(2),
        jnp.exp(-1j * phi) / jnp.sqrt(2)
    ], axis=-1).astype(jnp.complex64)

def Ry(theta):
    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    return jnp.asarray([
        [c, -s],
        [s,  c]
    ], dtype=jnp.complex64)
def Rz(theta):
    phase_minus = jnp.exp(-0.5j * theta)
    phase_plus  = jnp.exp( 0.5j * theta)
    return jnp.asarray([
        [phase_minus, 0.0j],
        [0.0j, phase_plus]
    ], dtype=jnp.complex64)

def tilted_ring(phi, mu1, mu2):
    # estados base
    psi = base_ring(phi)          # (N,2)
    # matriz total
    U = Rz(mu2) @ Ry(mu1)         # (2,2)
    # aplicar rotación
    psi_rot = psi @ U.T
    return psi_rot.astype(jnp.complex64)
def init_ring(Nstates, mu, key):
    mu1 = mu
    mu2=0
    phi = jax.random.uniform(
        key,
        shape=(Nstates,),
        minval=0,
        maxval=2*jnp.pi
    )
    states = tilted_ring(phi, mu1, mu2)
    return states


def base_ring_theta(phi, theta):
    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    return jnp.stack([
        jnp.ones_like(phi) * c,
        jnp.exp(-1j * phi) * s
    ], axis=-1).astype(jnp.complex64)

def tilted_ring_theta(phi, mu1, mu2):
    psi = base_ring_theta(phi, mu1)   # mu1 es theta directamente
    U   = Rz(0) @ Ry(0)
    return (psi @ U.T).astype(jnp.complex64)

def init_ring(Nstates, mu, key):
    mu1 = mu
    mu2=0
    phi = jax.random.uniform(
        key,
        shape=(Nstates,),
        minval=0,
        maxval=2 * jnp.pi
    )
    return tilted_ring_theta(phi, mu1, mu2)

def random_bloch_states_single(key, N):
    key_u, key_phi = jax.random.split(key)
    u     = jax.random.uniform(key_u,   (N,))
    phi   = 2 * jnp.pi * jax.random.uniform(key_phi, (N,))
    theta = jnp.arccos(1 - 2 * u)
    alpha = jnp.cos(theta / 2)
    beta  = jnp.exp(1j * phi) * jnp.sin(theta / 2)
    return jnp.stack([alpha, beta], axis=1).astype(jnp.complex64)

# ============================================================
# CIRCUIT HELPERS
# ============================================================

def add_ancilla_single(state):
    ancilla = jnp.zeros((2**Nancilla,), dtype=state.dtype).at[0].set(1)
    return jnp.kron(state, ancilla)
batched_addancilla = jax.vmap(add_ancilla_single)

def measure_ancillas_single(key, state):
    reshaped  = jnp.reshape(state, (2**Ndataq, 2**Nancilla))
    probs     = jnp.sum(jnp.abs(reshaped)**2, axis=0)
    _, subkey = jax.random.split(key)
    m_res     = jax.random.categorical(subkey, jnp.log(probs + 1e-12))
    post      = reshaped[:, m_res]
    return post / (jnp.linalg.norm(post) + 1e-12)

def transform_single_state(params, mu, state, key):
    c = tc.Circuit(Nqubits, inputs=state)
    for i in range(Nancilla):
        c.ry(Ndataq + i, theta=mu[0])
        c.rz(Ndataq + i, theta=mu[1])
    for l in range(L):
        for i in range(Nqubits):
            c.rx(i, theta=params[l][2*i])
            c.ry(i, theta=params[l][2*i + 1])
        for i in range(Nqubits - 1):
            c.cz(i, i+1)
    return measure_ancillas_single(key, c.state())

transform_batch = jax.jit(jax.vmap(
    transform_single_state, in_axes=(None, None, 0, 0)
))

# ============================================================
# DISTANCE
# ============================================================

@jax.jit
def coulomb_distance(X, Y, delta=0.01):
    N, M = X.shape[0], Y.shape[0]
    fXY  = jnp.abs(X @ Y.conj().T)**2
    fXX  = jnp.abs(X @ X.conj().T)**2
    fYY  = jnp.abs(Y @ Y.conj().T)**2
    kXY  = 1.0 / (1.0 - fXY + delta)
    kXX  = 1.0 / (1.0 - fXX + delta)
    kYY  = 1.0 / (1.0 - fYY + delta)
    return (jnp.sum(kXX) / (N*N + 1e-10)
          + jnp.sum(kYY) / (M*M + 1e-10)
          - 2.0 * jnp.mean(kXY))

# ============================================================
# SINGLE mu1 EVALUATION (mu2 fixed to 0)
# ============================================================

def evaluate_single_mu1(mu1, keys):
    key_target = keys[0]
    key_haar   = keys[1]
    key_init   = keys[2]

    mu_current    = jnp.array([mu1, mu2_fixed])
    target_states = init_ring(Nstates, mu1, key_target)

    haar_states = random_bloch_states_single(key_haar, Nstates)
    haar_loss   = coulomb_distance(target_states, haar_states)

    current_states = random_bloch_states_single(key_init, Nstates)
    current_states = batched_addancilla(current_states)

    def denoise_step(states, step_carry):
        step_idx, step_key = step_carry
        step_keys  = jax.random.split(step_key, Nstates)
        new_states = transform_batch(
            params[step_idx], mu_current, states, step_keys
        )
        return batched_addancilla(new_states), None

    step_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
        keys[3], jnp.arange(T)
    )
    final_states_with_anc, _ = jax.lax.scan(
        denoise_step,
        current_states,
        (jnp.arange(T), step_keys)
    )

    final_states = jax.vmap(
        lambda s: jnp.reshape(s, (2**Ndataq, 2**Nancilla))[:, 0]
    )(final_states_with_anc)
    final_states = final_states / (
        jnp.linalg.norm(final_states, axis=-1, keepdims=True) + 1e-12
    )

    model_loss = coulomb_distance(target_states, final_states)
    return model_loss, haar_loss

# ============================================================
# PARALLEL SWEEP over mu1 ∈ [0, 2π]
# ============================================================

n_mu1      = 200
mu1_values = jnp.linspace(0.0, jnp.pi, n_mu1)

key, sweep_key = jax.random.split(key)
all_keys = jax.vmap(
    lambda k: jax.random.split(k, 3 + T)
)(jax.random.split(sweep_key, n_mu1))

print(f"Compiling and running sweep over mu1 ({n_mu1} points)...")
t0 = time.time()

evaluate_all = jax.jit(jax.vmap(evaluate_single_mu1, in_axes=(0, 0)))
loss_values, haar_losses = evaluate_all(mu1_values, all_keys)

loss_values = np.array(loss_values)
haar_losses = np.array(haar_losses)
mu1_np      = np.array(mu1_values)
print(f"Sweep completed in {time.time()-t0:.2f}s")

# ============================================================
# NORMALISE
# ============================================================

loss_norm          = loss_values / haar_losses
training_indices   = [np.argmin(np.abs(mu1_np - m)) for m in training_mu1]
training_loss_norm = loss_norm[training_indices]

# ============================================================
# PLOT
# ============================================================

plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 13,
    "axes.labelsize"   : 15,
    "axes.linewidth"   : 1.1,
    "xtick.labelsize"  : 12,
    "ytick.labelsize"  : 12,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)

ax.axhline(1.0, color="#999999", linewidth=1.0,
           linestyle=":", label="Haar random level", zorder=1)

ax.plot(mu1_np, loss_norm,
        color="#132D5E", linewidth=2,
        label="Evaluated conditioning", zorder=3)

ax.fill_between(mu1_np, loss_norm, alpha=0.12, color="#132D5E")

ax.scatter(training_mu1, training_loss_norm,
           s=90, color="#B68C32", zorder=5,
           edgecolors="#161A26", linewidths=0.8,
           marker="D", label="Trained conditioning")

for m in training_mu1:
    ax.axvline(m, linestyle="--", linewidth=0.8, alpha=0.4, color="#B68C32")

ax.set_xlabel(r"$\mu_1$", labelpad=8)
ax.set_ylabel(r"$\mathcal{L}/\mathcal{L}_{\mathrm{Haar}}$",
              labelpad=8)
ax.set_xlim(0, np.pi)
ax.set_ylim(0, 1.15)

ax.set_xticks([0.1*np.pi, 0.3*np.pi, 0.5*np.pi, 0.7*np.pi, 0.9*np.pi])
ax.set_xticklabels([r"$0.1\pi$", r"$0.3\pi$", r"$0.5\pi$", r"$0.7\pi$", r"$0.9\pi$"])

ax.legend(frameon=True, fontsize=11, loc="upper right")

plt.tight_layout()
plt.savefig("loss_vs_mu1.pdf", bbox_inches="tight")
plt.savefig("loss_vs_mu1.png", bbox_inches="tight")
plt.show()
print("Saved → loss_vs_mu1.pdf / loss_vs_mu1.png")