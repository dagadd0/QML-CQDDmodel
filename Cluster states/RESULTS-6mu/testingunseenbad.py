# bloch_cases.py
import tensorcircuit as tc
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

tc.set_dtype("complex64")
tc.set_backend("jax")

# ─────────────────────────────────────────────────────────────────
# PUNTOS A REPRESENTAR  ← modifica aquí
# ─────────────────────────────────────────────────────────────────
cases = [
    (3*np.pi/4,   np.pi/4),
    (np.pi/2, 5*np.pi/4),
    (  np.pi/4,          0),
]
col_labels = ["(a)", "(b)", "(c)"]

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
Nstates  = 300
Nqubits  = 3
Nancilla = 2
Ndataq   = 1
L        = 12
epsilon  = 0.04
key      = jax.random.PRNGKey(0)

params = jnp.array(np.load("RESULTS-6mu/params.npy"))


# ─────────────────────────────────────────────────────────────────
# CIRCUITO
# ─────────────────────────────────────────────────────────────────
def add_ancilla(state, Nancilla):
    ancilla = jnp.zeros((2**Nancilla,), dtype=state.dtype).at[0].set(1)
    return jnp.kron(state, ancilla)

batched_addancilla = jax.vmap(add_ancilla, in_axes=(0, None))

def measure_ancillas_single(key, state, na, n):
    reshaped   = jnp.reshape(state, (2**n, 2**na))
    probs      = jnp.sum(jnp.abs(reshaped)**2, axis=0)
    key, subk  = jax.random.split(key)
    m_res      = jax.random.categorical(subk, jnp.log(probs + 1e-12))
    post_state = reshaped[:, m_res]
    return post_state / (jnp.linalg.norm(post_state) + 1e-12)

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
    return measure_ancillas_single(key, c.state(), Nancilla, Ndataq)

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

def theoretical_distribution(theta, phi, N, key, eps):
    alpha = jnp.cos(theta / 2).astype(jnp.complex64)
    beta  = (jnp.exp(1j*phi) * jnp.sin(theta / 2)).astype(jnp.complex64)
    point = jnp.array([alpha, beta], dtype=jnp.complex64)
    k1, k2 = jax.random.split(key)
    noise  = (jax.random.normal(k1, (N, 2)).astype(jnp.complex64)
            + 1j*jax.random.normal(k2, (N, 2)).astype(jnp.complex64)) * jnp.float32(eps)
    states = jnp.tile(point, (N, 1)) + noise
    return (states / jnp.linalg.norm(states, axis=1, keepdims=True)).astype(jnp.complex64)

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
# ESTADOS → COORDENADAS BLOCH
# ─────────────────────────────────────────────────────────────────
def states_to_bloch(states):
    states = np.array(states)
    alpha  = states[:, 0]
    beta   = states[:, 1]
    theta  = 2 * np.arccos(np.clip(np.abs(alpha), 0.0, 1.0))
    phi    = np.angle(beta) - np.angle(alpha)
    return np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)


# ─────────────────────────────────────────────────────────────────
# GENERAR DISTRIBUCIONES
# ─────────────────────────────────────────────────────────────────
key, subkey = jax.random.split(key)
random_distr = random_bloch_states(subkey, Nstates)

target_list  = []
testing_list = []

for i, (th, ph) in enumerate(cases):
    k_th, k_circ = jax.random.split(jax.random.PRNGKey(42 + i), 2)
    target_list.append(np.array(
        theoretical_distribution(jnp.float32(th), jnp.float32(ph), Nstates, k_th, epsilon)
    ))
    testing_list.append(np.array(
        run_full_circuit(jnp.array([th, ph], dtype=jnp.float32), random_distr, params, k_circ)
    ))
    print(f"Caso {col_labels[i]} listo.")


# ─────────────────────────────────────────────────────────────────
# HELPER: esfera de Bloch
# ─────────────────────────────────────────────────────────────────
def draw_bloch(ax, states, color, elev=20, azim=-60):
    # Wireframe
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0,   np.pi, 20)
    ax.plot_wireframe(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
        color="#CCCCCC", alpha=0.35, linewidth=0.4, rstride=2, cstride=2,
    )
    # Círculos principales
    c = np.linspace(0, 2*np.pi, 300)
    for xs, ys, zs in [
        (np.cos(c), np.sin(c),        np.zeros_like(c)),
        (np.cos(c), np.zeros_like(c), np.sin(c)),
        (np.zeros_like(c), np.cos(c), np.sin(c)),
    ]:
        ax.plot(xs, ys, zs, color="#AAAAAA", linewidth=0.5, alpha=0.5)

    # Ejes y etiquetas
    for vec, lbl in zip([[1,0,0],[0,1,0],[0,0,1]], ["x","y","z"]):
        ax.quiver(0,0,0,*vec, length=1.3, color="#999999",
                  linewidth=0.6, arrow_length_ratio=0.1)
        ax.text(vec[0]*1.48, vec[1]*1.48, vec[2]*1.48,
                lbl, fontsize=7, color="#666666", ha="center", va="center")

    ax.text( 0,  0,  1.22, r"$|0\rangle$", ha="center", va="bottom", fontsize=7.5)
    ax.text( 0,  0, -1.32, r"$|1\rangle$", ha="center", va="top",    fontsize=7.5)

    # Puntos
    x, y, z = states_to_bloch(states)
    ax.scatter(x, y, z, c=color, s=4, alpha=0.65, linewidths=0, depthshade=True)

    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.view_init(elev=elev, azim=azim)


# ─────────────────────────────────────────────────────────────────
# FIGURA
# ─────────────────────────────────────────────────────────────────
rcParams.update({
    "text.usetex":        False,
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif"],
    "mathtext.fontset":   "cm",
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

COLOR_TARGET  = "#B03A2E"   # rojo
COLOR_TESTING = "#1A7A4A"   # verde

fig = plt.figure(figsize=(9.6, 6.4))
gs  = gridspec.GridSpec(
    2, 3, figure=fig,
    wspace=0.0, hspace=0.0,
    left=0.08, right=0.98,
    top=0.88,  bottom=0.02,
)

# Etiquetas de columna
for j in range(len(cases)):
    x_pos = gs.left + (j + 0.5) * (gs.right - gs.left) / 3
    fig.text(
        x_pos, 0.93,
        col_labels[j],
        ha="center", va="top", fontsize=9,
    )

# Etiquetas de fila
for i, (lbl, col) in enumerate(zip(["Target", "Testing output"],
                                    [COLOR_TARGET, COLOR_TESTING])):
    y_pos = 1 - (i + 0.5) / 2
    y_pos = gs.bottom + (1 - i/2) * (gs.top - gs.bottom) - (gs.top - gs.bottom)/4
    fig.text(
        0.01, y_pos, lbl,
        ha="left", va="center", fontsize=9,
        color=col, fontweight="bold", rotation=90,
    )

# Dibujar
for j, states in enumerate(target_list):
    ax = fig.add_subplot(gs[0, j], projection="3d")
    draw_bloch(ax, states, color=COLOR_TARGET)

for j, states in enumerate(testing_list):
    ax = fig.add_subplot(gs[1, j], projection="3d")
    draw_bloch(ax, states, color=COLOR_TESTING)

plt.savefig("cases_bloch.pdf", format="pdf")
plt.savefig("cases_bloch.png", format="png")
print("Guardado: cases_bloch.pdf / .png")
plt.show()