# Core libraries
import tensorcircuit as tc
import jax
import jax.numpy as jnp
import optax
import numpy as np
import time

tc.set_dtype("complex64")
tc.set_backend("jax")

Nstates = N_points = Npoints= 500    # Number of elements in the distribution
Nqubits=3                    # Number of qubits of each circuit element (including ancilla)
Nancilla=2                   # Number of ancillas
Ndataq = 1                   # Number of data Qubits
dim=2**Nqubits               # Dimension of the state of each circuit
T = 20                       # Steps of Scrambling
L= 12                        # Layers of at each step of denoising
key = jax.random.PRNGKey(0)  # Key for deterministic randomness given by (seed)
epsilon = 0.08               # Epsilon for the polar points distribution, determines amount of noise
Nopt_loops = 1000           # Number of parameter optimization loops
opt = optax.adam(learning_rate=0.01)   # Lerning rate of optimizer

# Conditioning parameters to train
mus = jnp.array( [ [0,0], 
                [jnp.pi/2, 0], [jnp.pi/2, jnp.pi/2], [jnp.pi/2, jnp.pi], [jnp.pi/2, 3*jnp.pi/2],
                [jnp.pi, 0]
                ])


"""
# Exhaustive training: Prepare conditioning parameters
thetas = jnp.linspace(0, jnp.pi, 5)
phis = jnp.linspace(0,2*jnp.pi, 8, endpoint=False)
mus = []

for theta in thetas:
    if jnp.isclose(theta, 0) or jnp.isclose(theta, jnp.pi):
        mus.append(jnp.array([theta, 0.0]))
    else:
        for phi in phis:
            mus.append(jnp.array([theta, phi]))
mus = jnp.array(mus)"""


# --------   PREPARATION OF INITIAL/TARGET STATES DISTRIBUTION  --------
# POLAR POINTS:
def bloch_point(mu):
    theta = mu[0]
    phi = mu[1]
    state = jnp.array([
        jnp.cos(theta / 2),
        jnp.exp(1j * phi) * jnp.sin(theta / 2)
    ], dtype=jnp.complex64)
    return state / jnp.linalg.norm(state)
polarpoint_map = jax.vmap(bloch_point)(mus)
polar_points_array = jax.vmap(bloch_point)(mus)
polar_kinds = [f"zy_{k}" for k in range(len(mus))]

def init_polarpoint(Nstates, Ndata, kind_idx, key, epsilon):
    """ Creates a Polar point distribution
    Takes:
        Nstates: number of states
        Ndata: number of data qubits
        kind: initial state to prepare
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




# --------   F O R W A R D    S C R A M B L I N G    --------
# SCHEDULE PARAMETERS
schedules_map = {
    "exponential": lambda u, s_in, s_fin, T: s_in + (s_fin - s_in) * (jnp.exp(u) - 1) / (jnp.e - 1),
    "linear": lambda u, s_in, s_fin, T: s_in + (s_fin - s_in) * u,
    "quadratic": lambda u, s_in, s_fin, T: s_in * jnp.arange(1, T + 1) ** 2
    }
def schedule(kind: str, T:int):
    """ Schedule parameters for the distribution of Angles of rotations
    Takes:
        kind: type of increase for the schedule parameter 
        T: number of steps for forward and backward process
    Returns:
        List of schedule parameters for each step, shape (T,)
    """
    s_in, s_fin = 0.01, 5
    u = jnp.linspace(0., 1., T)
    return schedules_map[kind](u, s_in, s_fin, T)

# Rotation of initial state aplying RZ()RY()RZ(), all rotations with a random angle
# selected from a uniform distirbution given the schedule parameter
def scrambling(initial_states, sc_param, Ndata, key): 
    """ Scrambling of initial distribution to Haar random
    Takes:
        initial_states: array of states for initial distribution, shape [Nstates, 2**Ndataq]
        sc_param: list of schedule parameters for each step, shape [T]
        Ndata: number of data qubits
        key: key for randomness
    Returns:
        list of Nstates for each t step of scrambling, shape: (T+1, Nstates, 2**Ndataq)
    """
    T = sc_param.shape[0]
    Nstates = initial_states.shape[0]
    # Generate uniform distribution of angles given the schedule parameter
    angles = jax.random.uniform(
        key,
        shape=(T, Nstates, 3),  # [step, state, angle]
        minval=-jnp.pi/8,
        maxval=jnp.pi/8
    ) * (sc_param[:, jnp.newaxis, jnp.newaxis])

    def rotate_state_i(state,angle_list): 
        c = tc.Circuit(Ndata, inputs=state)
        for q in range(Ndata):
            c.rz(q, theta=angle_list[0])
            c.ry(q, theta=angle_list[1])
            c.rz(q, theta=angle_list[2])
        return c.state()
    def scrambling_step(states,angles_step):
        return jax.vmap(rotate_state_i, in_axes=(0,0))(states,angles_step)
    def scan_fn(carry_states, angles_step):
        next_states= scrambling_step(carry_states,angles_step)
        return next_states, next_states
    
    # Loop for T steps, returns distribution at each step. Shape: (T, Nstates, 2**Ndataq)
    _, states_forward = jax.lax.scan(scan_fn, initial_states, angles) 

    return jnp.concatenate([initial_states[None,:,:],states_forward], axis=0) # [T+1, Nstates, 2**Ndataq]
batched_scrambling = jax.vmap(scrambling, in_axes=(0,None,None,0))




# --------    B A C K W A R D    D E N O S I N G    --------
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


# OPTIMIZATION
def loss(theta, mu, current, target, dist_kind, Nancilla, Ndataq, Nqubits, key):
    """
    Takes:
        theta: parameters that transform state
        mu: conditioning
        current: states that get transformed by theta, #Shape: [Npoints, 2**Ndataq]
        target: target states used for MMD distance, Shape: [Npoints, 2**Ndataq]
        dist_kind: how distance between sets is measured: MMD or WASS
        key
    Returns:
        Distance between states using MMD or Wasserstein
    """
    keys = jax.random.split(key,current.shape[0])
    transformed_states = batched_transform(theta, mu, batched_addancilla(current, Nancilla), Nancilla, Ndataq, Nqubits, keys) # Returns Shape: [Npoints, 2**Nqubits]
    return dist_kind(transformed_states, target)

def train_step(params, mus, opt_state, current_states, target_states, 
               kind, Nancilla, Ndataq, Nqubits, key):
    keys= jax.random.split(key, len(mus))
    def mean_loss_fn(params):
        def single_loss(mu, current_states, target_states, key):
            return loss(params, mu, current_states, target_states,
                        kind, Nancilla, Ndataq, Nqubits, key)
        all_losses = jax.vmap(single_loss, in_axes=(0,0,0,0))(
            mus, current_states, target_states, keys)
        return jnp.mean(all_losses)
    
    mean_loss_val, mean_grads = jax.value_and_grad(mean_loss_fn)(params)
    updates, new_opt_state = opt.update(mean_grads, opt_state, params)
    new_params     = optax.apply_updates(params, updates)
    return new_params, new_opt_state, mean_loss_val
train_step = jax.jit(train_step, static_argnums=(5,6,7,8))

def _eval_loss_target(params, mus, current_states, final_states, kind, Nancilla, Ndataq, Nqubits, key):
    keys = jax.random.split(key, len(mus))
    return jnp.mean(jax.vmap(
        lambda mu, current, target, k: loss(params, mu, current, target, kind, Nancilla, Ndataq, Nqubits, k),
        in_axes=(0, 0, 0, 0)
    )(mus, current_states, final_states, keys))
eval_loss_target = jax.jit(_eval_loss_target, static_argnums=(4, 5, 6, 7))

def denoising_step(params, mus, opt_state, current_states, trg_states, final_states,
                   kind, opt_loops, Nancilla, Ndataq, Nqubits, key):
    n_logs = opt_loops//50
    losses_t, losses_target = [],[]
    best_loss_t   = jnp.inf
    best_params = params

    for step in range(opt_loops):
        key, subkey = jax.random.split(key)
        # Optimization step
        params, opt_state, mean_loss = train_step(
            params, mus, opt_state,
            current_states, trg_states,
            kind, Nancilla, Ndataq, Nqubits,
            subkey)
        # Save losses
        losses_t.append(mean_loss)
        if step % 50 == 0:
            #print(f"step {step} | loss {mean_loss:.4f}")
            key, subkey = jax.random.split(key,2)
            loss_target = eval_loss_target(params, mus, current_states, final_states,
                                  kind, Nancilla, Ndataq, Nqubits, subkey)
            losses_target.append(loss_target)
                    # Keep the best parameters (lower loss)
            if mean_loss < best_loss_t:
                best_loss_t = mean_loss
                best_params = params
    losses_t = jnp.stack(losses_t).tolist()
    losses_target = jnp.stack(losses_target).tolist()

    # Final transform: vmap over all the cases (mu, states, keys)
    final_keys = jax.random.split(key, len(mus) * current_states.shape[1]).reshape(len(mus), -1, 2)
    batched_final_transform = jax.vmap(
        lambda mu, states, keys: batched_transform(
            best_params, mu, batched_addancilla(states, Nancilla),
            Nancilla, Ndataq, Nqubits, keys), in_axes=(0, 0, 0))
    transformed_states = batched_final_transform(mus, current_states, final_keys) # shape: (len(mus), Nstates, 2**Nqubits)
    best_loss_target = float(jnp.mean(jax.vmap(MMD)(transformed_states, final_states)))
    
    return transformed_states, best_params, opt_state, losses_t, losses_target, float(best_loss_t), float(best_loss_target)

def denoising(all_target_distr, mus, dist_kind, opt_loops, 
              Nancilla, Ndataq, L, key):
    """
    Takes:
        target_distr: list of states for each step of scrambling, with shape [T+1, Npoints, 2**Ndata]
        dist_kind: how distance between sets is measured: MMD or WASS
        opt_loops: number of loops for parameter optimization
        Nancilla: number of ancillas
        Ndataq: number of data qubits
        Nqubits: total number of qubits (Nancilla+Ndataq)
        key for deterministic randomness
    Returns:
        list of distribution of states for initial and each denoising step. Shape: [T+1, Npoints, 2**Ndataq]
        list of parameters for each denoising step. Shape: [T,]
    """
    Nqubits = Nancilla+Ndataq
    T = all_target_distr.shape[1]-1

    parameters = []
    all_losses_t, all_losses_target, best_losses_t, best_losses_target = [],[],[], []
    bck_distr = [all_target_distr[:,T]]
    final_states = all_target_distr[:,0]

    for i in range(T):
        key, subkey = jax.random.split(key)
        params    = jax.random.normal(subkey, (L, 2*Nqubits))   # single shared params
        opt_state = opt.init(params)

        key, subkey = jax.random.split(key,2)
        trg_states = all_target_distr[:,T-i-1]    # Shape: [mus, Npoints, 2**Ndataq]
        current_states = bck_distr[i]       # Shape: [mus, Npoints, 2**Ndataq]
        
        temp3=time.time()
        next_states, params, opt_state, losses_t, losses_target, best_loss_t, best_loss_target = denoising_step(
                    params, mus, opt_state,
                    current_states, trg_states, final_states,
                    dist_kind, opt_loops, 
                    Nancilla, Ndataq, Nqubits, subkey
                )
        all_losses_t.append(losses_t)
        all_losses_target.append(losses_target)
        best_losses_t.append(best_loss_t)
        best_losses_target.append(best_loss_target)
        parameters.append(params)
        bck_distr.append(next_states)
        print(f"Denoising step {i+1}, time:  {time.time()-temp3:.4f}")
    bck_distr = bck_distr[::-1]
    return bck_distr, parameters, all_losses_t, all_losses_target, best_losses_t, best_losses_target




# ======== EXECUTION ========
key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

temp1=time.time()
def prepare_input(key,cond_leng):
    keys = jax.random.split(key, cond_leng)
    kind_idx = jnp.arange(cond_leng)
    return batched_init_polarpoint(Nstates, Ndataq, kind_idx, keys, epsilon)
all_input = prepare_input(key,len(mus))
print("Preparation time: ", time.time()-temp1)

temp2=time.time()
def scramble_input(all_initial_states, sc_param, Ndata, cond_leng, key):
    keys = jax.random.split(key, cond_leng)
    return batched_scrambling(all_initial_states, sc_param, Ndata, keys)
all_scrambled = scramble_input(all_input,schedule("quadratic", T), Ndataq, len(mus), subkey2)
print("Scrambling time: ", time.time()-temp2)

all_losses_t, all_losses_target = [],[]
bck_distr, parameters, all_losses_t, all_losses_target, best_losses_t, best_losses_target = denoising(all_scrambled, mus, MMD, Nopt_loops, Nancilla, Ndataq, L, subkey3)
print(f"Training completed for {len(bck_distr)} conditioning types")


np.save("params.npy",np.array(parameters))
np.save("bck_distr.npy",np.array(bck_distr))
np.save("all_input.npy", np.array(all_input)) 
np.save("conditioning_params.npy", np.array(mus))
np.save("all_losses_t.npy", np.array(all_losses_t))
np.save("all_losses_target.npy", np.array(all_losses_target))
np.save("best_losses_t.npy",       np.array(best_losses_t))
np.save("best_losses_target.npy", np.array(best_losses_target))



import math
import qutip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
import numpy as np



# ────────────────────────   PLOTING TRAINING RESULTS   ────────────────────────
def plot_bloch_qutip(bck_distr, polar_kinds):
    final = np.array(bck_distr[0])   # (Npoints, Nstates, 2)
    Npoints = len(polar_kinds)
    # Colors automatically
    colors = plt.cm.hsv(np.linspace(0, 0, Npoints))
    buffers = []

    for i, (kind, color) in enumerate(zip(polar_kinds, colors)):
        states = final[i]
        # ── Bloch vectors ────────────────────────────────────────
        points = []
        for sv in states:
            ket = qutip.Qobj(sv.reshape(2, 1))
            dm  = ket * ket.dag()
            bx = float(np.real((dm * qutip.sigmax()).tr()))
            by = float(np.real((dm * qutip.sigmay()).tr()))
            bz = float(np.real((dm * qutip.sigmaz()).tr()))
            points.append([bx, by, bz])
        points = np.array(points).T
        # ── Individual Bloch sphere ──────────────────────────────
        b = qutip.Bloch()
        b.point_color  = [color]
        b.point_marker = ['o']
        b.point_size   = [8]
        b.sphere_alpha = 0.05
        b.vector_color = ['red']
        b.add_points(points, meth='s')
        b.render()
        b.fig.suptitle(f"|{kind}⟩", fontsize=13)
        buf = BytesIO()
        b.fig.savefig(
            buf,
            format='png',
            dpi=120,
            bbox_inches='tight'
        )
        buf.seek(0)
        buffers.append(buf)
        plt.close(b.fig)

    # ── Automatic grid size ─────────────────────────────────────
    ncols = math.ceil(math.sqrt(Npoints))
    nrows = math.ceil(Npoints / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5*ncols, 5*nrows)
    )
    axes = np.array(axes).reshape(-1)

    # ── Fill plots ──────────────────────────────────────────────
    for ax, buf in zip(axes, buffers):
        img = mpimg.imread(buf)
        ax.imshow(img)
        ax.axis('off')

    # ── Hide empty axes ─────────────────────────────────────────
    for ax in axes[len(buffers):]:
        ax.axis('off')
    fig.suptitle("Denoised distributions", fontsize=18)
    plt.tight_layout()
    plt.savefig(
        "bloch_denoised.png",
        dpi=150,
        bbox_inches='tight'
    )
    plt.show()

plot_bloch_qutip(bck_distr, polar_kinds)







# ────────────────────────   PLOTING LOSSES   ────────────────────────
CLR_T      = "#3973f0"   
CLR_SM = "#e74d4d"   

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

ax_t, ax_f = axes

for i, (losses_t, losses_target) in enumerate(zip(all_losses_t, all_losses_target)):
    # ── loss t→t-1 ─────────────
    data_t = np.array(losses_t, dtype=float)
    xs_t   = i * len(data_t) + np.arange(len(data_t))

    ax_t.scatter(xs_t, data_t, color=CLR_T, alpha=0.1, s=0.4, linewidths=0)
    w  = max(1, len(data_t) // 20)
    sm = np.convolve(data_t, np.ones(w) / w, mode="valid")
    ax_t.plot(i * len(data_t) + np.arange(len(sm)), sm, color=CLR_SM, linewidth=1.2)

    # ── loss t→0 ──────────────
    if losses_target:
        vals_f  = np.array(losses_target, dtype=float)
        steps_f = np.arange(len(vals_f)) * 50 + i * len(data_t)  # reconstruye el eje x

        ax_f.scatter(steps_f, vals_f, color=CLR_T, alpha=0.6, s=8, linewidths=0, zorder=3)

        if len(vals_f) >= 3:
            w_f  = max(1, len(vals_f) // 5)
            sm_f = np.convolve(vals_f, np.ones(w_f) / w_f, mode="valid")
            xs_sm_f = steps_f[:len(sm_f)] + w_f // 2 * 50
            ax_f.plot(xs_sm_f, sm_f, color=CLR_SM, linewidth=1.2, zorder=4)
# Separadores entre denoising steps
T = len(all_losses_t)
Nopt = len(all_losses_t[0]) if all_losses_t else 1

for ax in axes:
    for k in range(1, T):
        ax.axvline(k * Nopt, color="#bbbbbb", linewidth=0.6, linestyle=":")
    ax.set_yscale("log")
    ax.set_xlabel("Optimization step", fontsize=11)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
    # Etiquetas de denoising step en el eje x
    ax.set_xticks([k * Nopt + Nopt // 2 for k in range(T)])
    ax.set_xticklabels([f"$t_{{{T-k}}}$" for k in range(T)], fontsize=7)

ax_t.set_ylabel(r"$\mathcal{L}(t \to t-1)$", fontsize=11)
ax_f.set_ylabel(r"$\mathcal{L}(t \to 0)$",   fontsize=11)
ax_t.set_title("t vs t-1 loss",  fontsize=11)
ax_f.set_title("t vs target loss",    fontsize=11)

plt.tight_layout()
plt.savefig("losses.png", dpi=200, bbox_inches="tight")
print("Losses plot saved in losses.png")






# ────────────────────────   PLOTING FORWARD/BACKWARD PROCESS STEPS   ────────────────────────
def plot_bloch_grid(steps, color, title, label, filename, ncols=7):
    """
    Renders a grid of Bloch spheres, 3 rows × ncols columns.
    steps: list of (Nstates, 2) arrays, length T+1
    """
    n      = len(steps)
    nrows  = -(-n // ncols)   # ceiling division → 3 rows for 21 panels with ncols=7

    # ── Render each sphere into a buffer ─────────────────────────
    def render_sphere(states, t):
        points = np.array([bloch_vector(sv) for sv in states]).T  # (3, Nstates)
        b = qutip.Bloch()
        b.point_color  = [color]
        b.point_marker = ['o']
        b.point_size   = [8]
        b.sphere_alpha = 0.05
        b.frame_alpha  = 0.1
        b.add_points(points, meth='s')
        b.render()
        b.fig.suptitle(f"t={t}", fontsize=7, y=0.98)
        buf = BytesIO()
        b.fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        plt.close(b.fig)
        return buf

    bufs = [render_sphere(steps[t], t) for t in range(n)]

    # ── Combine into nrows × ncols grid ──────────────────────────
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 1.8, nrows * 2.2))
    axes = axes.flatten()

    for ax, buf in zip(axes, bufs):
        ax.imshow(mpimg.imread(buf))
        ax.axis('off')

    # Hide unused panels if n < nrows*ncols
    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle(f"{title}  |  polar point |{label}⟩",
                 fontsize=13, y=1.01, color=color, fontweight='bold')
    plt.subplots_adjust(wspace=0.02, hspace=0.05)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved → {filename}")


def bloch_vector(sv):
    ket = qutip.Qobj(sv.reshape(2, 1))
    dm  = ket * ket.dag()
    return [float(np.real((dm * op).tr()))
            for op in [qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]]


def plot_evolution(all_scrambled, bck_distr, dist_idx=0, ncols=7):
    """
    Two separate figures: one for scrambling, one for denoising.
    Each with 3 rows × ncols columns for T=20 (21 panels).

    all_scrambled: (6, T+1, Nstates, 2)
    bck_distr:     list of (6, Nstates, 2), length T+1, already reversed
    """
    label = polar_kinds[dist_idx]
    T1    = all_scrambled.shape[1]   # T+1

    scr_steps = [np.array(all_scrambled[dist_idx, t]) for t in range(T1)]
    den_steps = [np.array(bck_distr[t][dist_idx])     for t in range(T1)]

    plot_bloch_grid(
        steps    = scr_steps,
        color    = 'royalblue',
        title    = "Forward scrambling",
        label    = label,
        filename = f"scrambling_{label}.png",
        ncols    = ncols
    )

    plot_bloch_grid(
        steps    = den_steps,
        color    = 'tomato',
        title    = "Backward denoising",
        label    = label,
        filename = f"denoising_{label}.png",
        ncols    = ncols
    )


from io import BytesIO
import matplotlib.image as mpimg
plot_evolution(all_scrambled, bck_distr, dist_idx=1)  
