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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, PillowWriter

tc.set_dtype("complex64")
tc.set_backend("jax")
BASE_DIR = Path(__file__).resolve().parent
Nstates = N_points = Npoints = 300
Nqubits  = 3
Nancilla = 2
Ndataq   = 1
dim      = 2**Nqubits
T        = 20
L        = 12
key      = jax.random.PRNGKey(0)
epsilon  = 0.04
Nopt_loops = 20000
opt    = optax.adam(learning_rate=0.001)
params = jnp.array(np.load("RESULTS-6mu/params.npy"))
print(params.shape)
mus = jnp.array(np.load("RESULTS-6mu/conditioning_params.npy"))


# ─────────────────────────────────────────────────────────────────
# DENOISING CIRCUIT
# ─────────────────────────────────────────────────────────────────
def add_ancilla(state, Nancilla):
    ancilla = jnp.zeros((2**Nancilla,), dtype=state.dtype).at[0].set(1)
    return jnp.kron(state, ancilla)

batched_addancilla = jax.vmap(add_ancilla, in_axes=(0, None))


def measure_ancillas_single(key, state, na, n):
    reshaped = jnp.reshape(state, (2**n, 2**na))
    probs    = jnp.sum(jnp.abs(reshaped)**2, axis=0)
    key, subkey = jax.random.split(key)
    m_res    = jax.random.categorical(subkey, jnp.log(probs + 1e-12))
    post_state = reshaped[:, m_res]
    norm = jnp.linalg.norm(post_state)
    return post_state / (norm + 1e-12)


def transform_singlestate(params, mu, current_state, Nancillaa, Ndataqq, Nqubitss, key):
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
    data_state = measure_ancillas_single(key, full_state, Nancilla, Ndataq)
    return data_state

batched_transform = jax.jit(jax.vmap(
    transform_singlestate, in_axes=(None, None, 0, None, None, None, 0)
))


def random_bloch_states(key, N):
    key_theta, key_phi = jax.random.split(key)
    u     = jax.random.uniform(key_theta, (N,))
    theta = jnp.arccos(1 - 2*u)
    phi   = 2*jnp.pi * jax.random.uniform(key_phi, (N,))
    alpha = jnp.cos(theta / 2)
    beta  = jnp.exp(1j*phi) * jnp.sin(theta / 2)
    return jnp.stack([alpha, beta], axis=1)


# ─────────────────────────────────────────────────────────────────
# MMD DISTANCE
# ─────────────────────────────────────────────────────────────────
@jax.jit
def MMD(X, Y):
    N = X.shape[0]
    M = Y.shape[0]
    overlapXY = jnp.abs(X @ Y.conj().T)**2
    overlapXX = jnp.abs(X @ X.conj().T)**2
    overlapYY = jnp.abs(Y @ Y.conj().T)**2
    sumXY = 1 - jnp.mean(overlapXY)
    sumXX = 1 - jnp.sum(overlapXX) / (N * N)
    sumYY = 1 - jnp.sum(overlapYY) / (M * M)
    return 2*sumXY - sumYY - sumXX


polarpoint_map = {
    "0":  jnp.array([1, 0],  dtype=jnp.complex64),
    "1":  jnp.array([0, 1],  dtype=jnp.complex64),
    "+":  jnp.array([1, 1],  dtype=jnp.complex64) / jnp.sqrt(2),
    "-":  jnp.array([1, -1], dtype=jnp.complex64) / jnp.sqrt(2),
    "+i": jnp.array([1, 1j], dtype=jnp.complex64) / jnp.sqrt(2),
    "-i": jnp.array([1,-1j], dtype=jnp.complex64) / jnp.sqrt(2),
}
polar_points_array = jnp.stack(list(polarpoint_map.values()))
polar_kinds        = list(polarpoint_map.keys())


def init_polarpoint(Nstates, Ndata, kind_idx, key, epsilon):
    subkey1, subkey2 = jax.random.split(key, 2)
    qubit  = polar_points_array[kind_idx]
    base   = jnp.tile(qubit, (Ndata, 1))
    states = base + (
        1*jax.random.normal(subkey1, shape=(Nstates, 2**Ndata))
      + 1j*jax.random.normal(subkey2, shape=(Nstates, 2**Ndata))
    ) * epsilon
    states /= jnp.linalg.norm(states, axis=1, keepdims=True)
    return states

batched_init_polarpoint = jax.vmap(init_polarpoint, in_axes=(None, None, 0, 0, None))


# ─────────────────────────────────────────────────────────────────
# 1. Distribución teórica
# ─────────────────────────────────────────────────────────────────
def theoretical_distribution(theta, phi, N, key, epsilon):
    alpha = jnp.cos(theta / 2).astype(jnp.complex64)
    beta  = (jnp.exp(1j*phi) * jnp.sin(theta / 2)).astype(jnp.complex64)
    point = jnp.array([alpha, beta], dtype=jnp.complex64)
    k1, k2 = jax.random.split(key)
    noise  = (jax.random.normal(k1, (N, 2)).astype(jnp.complex64)
            + 1j*jax.random.normal(k2, (N, 2)).astype(jnp.complex64)) * jnp.float32(epsilon)
    states = jnp.tile(point, (N, 1)) + noise
    return (states / jnp.linalg.norm(states, axis=1, keepdims=True)).astype(jnp.complex64)


# ─────────────────────────────────────────────────────────────────
# 2. Circuito completo (T pasos)
# ─────────────────────────────────────────────────────────────────
@jax.jit
def run_full_circuit(mu, initial_states, params, base_key):
    def step_fn(carry, step_params):
        current, key = carry
        keys = jax.random.split(key, Nstates + 1)
        new_current = batched_transform(
            step_params, mu,
            batched_addancilla(current, Nancilla),
            Nancilla, Ndataq, Nqubits, keys[1:]
        )
        return (new_current, keys[0]), None
    (final_states, _), _ = jax.lax.scan(step_fn, (initial_states, base_key), params)
    return final_states


# ─────────────────────────────────────────────────────────────────
# 3. Rejilla
# ─────────────────────────────────────────────────────────────────
N_theta = 150
N_phi   = 300
thetas  = np.linspace(0,       np.pi,  N_theta)
phis    = np.linspace(0, 2*np.pi,      N_phi,  endpoint=True)

key, subkey = jax.random.split(key, 2)
random_distr = random_bloch_states(subkey, Nstates)

master_key = jax.random.PRNGKey(99)
master_key, k_circuit_root, k_theory_root = jax.random.split(master_key, 3)

all_mus         = jnp.array([[th, ph] for th in thetas for ph in phis])
all_phis_flat   = jnp.array([ph for _  in thetas for ph in phis])
all_thetas_flat = jnp.array([th for th in thetas for _  in phis])

all_keys_circuit = jax.random.split(k_circuit_root, N_theta * N_phi)
all_keys_theory  = jax.random.split(k_theory_root,  N_theta * N_phi)

theo_vmap = jax.jit(jax.vmap(
    lambda th, ph, k: theoretical_distribution(th, ph, Nstates, k, epsilon),
    in_axes=(0, 0, 0)
))

print("Calculando distribuciones teóricas...")
all_theory = theo_vmap(all_thetas_flat, all_phis_flat, all_keys_theory)

print("Ejecutando circuitos...")
start = time.time()
run_grid = jax.jit(jax.vmap(run_full_circuit, in_axes=(0, None, None, 0)))
all_circuit_states = run_grid(all_mus, random_distr, params, all_keys_circuit)
all_circuit_states.block_until_ready()
print(f"Circuitos: {time.time()-start:.1f}s")

# ── MMD circuito ──────────────────────────────────────────────────
print("Calculando MMD circuito...")
batch_size = 32
mmd_chunks = []
for i in range(0, len(all_circuit_states), batch_size):
    X_batch = all_circuit_states[i:i+batch_size]
    Y_batch = all_theory[i:i+batch_size]
    mmd_chunks.append(np.array(jax.vmap(MMD)(X_batch, Y_batch)))
mmd_flat = np.concatenate(mmd_chunks)

# ── MMD baseline: estado inicial ruidoso vs. teórico ─────────────
# random_distr es Haar-aleatorio (máximo desconocimiento),
# representa el peor caso posible antes de cualquier denoising.
print("Calculando MMD baseline (estado inicial vs. teórico)...")
baseline_chunks = []
# random_distr tiene forma (Nstates, 2); se usa la misma para todos
# los puntos de la rejilla (igual que al inicio del circuito).
random_distr_c64 = random_distr.astype(jnp.complex64)
for i in range(0, len(all_theory), batch_size):
    Y_batch = all_theory[i:i+batch_size]
    # Repetimos random_distr para cada elemento del batch
    X_batch = jnp.broadcast_to(
        random_distr_c64[None],
        (Y_batch.shape[0], Nstates, 2)
    )
    baseline_chunks.append(np.array(jax.vmap(MMD)(X_batch, Y_batch)))
baseline_flat = np.concatenate(baseline_chunks)

# ── Error relativo en % ───────────────────────────────────────────
# relative_error = (MMD_circuito / MMD_baseline) * 100
# 0% → el circuito reproduce perfectamente la distribución teórica
# 100% → el circuito no mejora nada respecto al estado inicial
relative_error_flat = np.clip(mmd_flat / (baseline_flat + 1e-12), 0, None) * 100
error_grid     = relative_error_flat.reshape(N_theta, N_phi)
mmd_grid       = mmd_flat.reshape(N_theta, N_phi)   # guardamos también la MMD cruda


# ─────────────────────────────────────────────────────────────────
# PUBLICATION-QUALITY SETTINGS
# ─────────────────────────────────────────────────────────────────
rcParams.update({
    "text.usetex":        False,
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif"],
    "mathtext.fontset":   "cm",
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.minor.width":  0.5,
    "ytick.minor.width":  0.5,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# Límites del error relativo (sin recortar extremos — los % son interpretables)
vmin = 0
vmax = np.percentile(relative_error_flat, 99)   # evita que outliers compriman la escala


# ─────────────────────────────────────────────────────────────────
# FIGURA 1 — Mapa de calor 2D (error relativo %)
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.0, 3.2))

im = ax.imshow(
    error_grid,
    origin        = "lower",
    aspect        = "auto",
    extent        = [0, 2*np.pi, 0, np.pi],
    cmap          = "inferno",
    vmin          = vmin,
    vmax          = vmax,
    interpolation = "nearest",
)

cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
cbar.set_label(r"error %", labelpad=6)
cbar.ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
cbar.ax.tick_params(width=0.8, direction="in", which="both")

# Formatear ticks de la colorbar con símbolo %
cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))

mu_theta = np.array(mus)[:, 0]
mu_phi   = np.array(mus)[:, 1]
ax.scatter(
    mu_phi, mu_theta,
    c          = "#00CFFF",
    s          = 18,
    marker     = "o",
    linewidths = 0.4,
    edgecolors = "white",
    alpha      = 1.0,
    zorder     = 5,
    label      = r"Training $\mu$",
)

ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
ax.set_yticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
ax.xaxis.set_minor_locator(ticker.MultipleLocator(np.pi/8))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(np.pi/8))
ax.set_xlabel(r"$\mu_2$")
ax.set_ylabel(r"$\mu_1$")

legend = ax.legend(
    loc           = "upper right",
    framealpha    = 0.25,
    edgecolor     = "white",
    handletextpad = 0.4,
    markerscale   = 1.4,
)
legend.get_frame().set_linewidth(0.6)

plt.tight_layout()
plt.savefig("mmd_heatmap_2d.pdf", format="pdf")
plt.savefig("mmd_heatmap_2d.png", format="png")



# ─────────────────────────────────────────────────────────────────
# FIGURA 2 — Esfera de Bloch rotando (GIF, error relativo %)
# ─────────────────────────────────────────────────────────────────
THETA, PHI = np.meshgrid(thetas, phis, indexing="ij")
X = np.sin(THETA) * np.cos(PHI)
Y = np.sin(THETA) * np.sin(PHI)
Z = np.cos(THETA)

norm   = plt.Normalize(vmin=vmin, vmax=vmax)
colors = cm.inferno(norm(error_grid))

fig = plt.figure(figsize=(5, 5))
ax3 = fig.add_subplot(111, projection="3d")

def draw_sphere(ax3):
    ax3.cla()
    ax3.plot_surface(
        X, Y, Z,
        facecolors  = colors,
        rstride     = 1,
        cstride     = 1,
        linewidth   = 0,
        antialiased = True,
        alpha       = 0.95,
    )
    axis_kw = dict(color="#888888", linewidth=0.7, arrow_length_ratio=0.12)
    for vec, lbl in zip([[1,0,0],[0,1,0],[0,0,1]], [r"$x$", r"$y$", r"$z$"]):
        ax3.quiver(0, 0, 0, *vec, length=1.35, **axis_kw)
        ax3.text(vec[0]*1.52, vec[1]*1.52, vec[2]*1.52,
                 lbl, fontsize=9, color="#555555", ha="center", va="center")
    ax3.text( 0,  0,  1.18, r"$|0\rangle$", ha="center", va="bottom", fontsize=9)
    ax3.text( 0,  0, -1.28, r"$|1\rangle$", ha="center", va="top",    fontsize=9)
    ax3.set_box_aspect([1, 1, 1])
    ax3.set_axis_off()

draw_sphere(ax3)
sm = cm.ScalarMappable(cmap="inferno", norm=norm)
sm.set_array([])
cbar3 = fig.colorbar(sm, ax=ax3, shrink=0.42, pad=0.02,
                     label="Relative error (%)")
cbar3.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
cbar3.ax.tick_params(direction="in", width=0.8)

N_frames   = 60
elev_fixed = 20

def update(frame):
    azim = frame * (360 / N_frames)
    ax3.view_init(elev=elev_fixed, azim=azim)
    return []

ani = FuncAnimation(fig, update, frames=N_frames, interval=120, blit=False)

print("Guardando GIF (puede tardar ~30 s)...")
writer = PillowWriter(fps=25)
ani.save("mmd_bloch_sphere.gif", writer=writer, dpi=120)
print("GIF guardado: mmd_bloch_sphere.gif")
plt.close(fig)