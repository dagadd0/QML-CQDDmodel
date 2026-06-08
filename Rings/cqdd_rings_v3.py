# Core libraries
import tensorcircuit as tc
import jax
import jax.numpy as jnp
import optax
import numpy as np
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import functools
import time

tc.set_dtype("complex64")
tc.set_backend("jax")

Nstates = N_points = Npoints= 500    # Number of elements in the distribution
Nqubits=4                 # Number of qubits of each circuit element (including ancilla)
Nancilla=3                  # Number of ancillas
Ndataq = 1                  # Number of data Qubits
dim=2**Nqubits              # Dimension of the state of each circuit
T = 20                      # Steps of Scrambling
L= 25                       # Layers of at each step of denoising
key = jax.random.PRNGKey(0) # Key for deterministic randomness given by (seed)
epsilon = 0.04              # Epsilon for the polar points distribution, determines amount of noise
Nopt_loops = 15000           # Number of parameter optimization loops
opt = optax.adam(learning_rate=0.01)


#mus = jnp.array([[0,0], [jnp.pi/2,0]])
#mus = jnp.array([[0,0], [3*jnp.pi/8,0], [3*jnp.pi/4,0], [9*jnp.pi/8,0], [12*jnp.pi/8,0], [15*jnp.pi/8,0]])
mus = jnp.array([[0,0], [jnp.pi/4,0], [2*jnp.pi/4,0], [3*jnp.pi/4,0], [4*jnp.pi/4,0], [5*jnp.pi/4,0], [6*jnp.pi/4,0], [7*jnp.pi/4,0], [8*jnp.pi/4,0]])
#mus = jnp.array([[0,0], [2*jnp.pi/4,0], [4*jnp.pi/4,0], [6*jnp.pi/4,0], [8*jnp.pi/4,0]])
"""mus = jnp.array([
    [0*jnp.pi/4, 0*jnp.pi/4],
    [0*jnp.pi/4, 2*jnp.pi/4],
    [0*jnp.pi/4, 4*jnp.pi/4],
    [0*jnp.pi/4, 6*jnp.pi/4],

    [2*jnp.pi/4, 0*jnp.pi/4],
    [2*jnp.pi/4, 2*jnp.pi/4],
    [2*jnp.pi/4, 4*jnp.pi/4],
    [2*jnp.pi/4, 6*jnp.pi/4],

    [4*jnp.pi/4, 0*jnp.pi/4],
    [4*jnp.pi/4, 2*jnp.pi/4],
    [4*jnp.pi/4, 4*jnp.pi/4],
    [4*jnp.pi/4, 6*jnp.pi/4],

    [6*jnp.pi/4, 0*jnp.pi/4],
    [6*jnp.pi/4, 2*jnp.pi/4],
    [6*jnp.pi/4, 4*jnp.pi/4],
    [6*jnp.pi/4, 6*jnp.pi/4],

    [1*jnp.pi/4, 1*jnp.pi/4],
    [3*jnp.pi/4, 3*jnp.pi/4],
    [5*jnp.pi/4, 5*jnp.pi/4],
    [7*jnp.pi/4, 7*jnp.pi/4],
])
"""
mus = jnp.array([
    [0.1 * jnp.pi, 0],   # cerca polo norte  (era z =  0.95)
    [0.3 * jnp.pi, 0],   # hemisferio norte  (era z =  0.5)
    [0.5  * jnp.pi, 0],   # ecuador           (era z =  0)
    [0.7 * jnp.pi, 0],   # hemisferio sur    (era z = -0.5)
    [0.9 * jnp.pi, 0],   # cerca polo sur    (era z = -0.95)
])

# --------   PREPARATION OF INITIAL/TARGET STATES DISTRIBUTION  --------

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
    mu1, mu2 = mu
    phi = jax.random.uniform(
        key,
        shape=(Nstates,),
        minval=0,
        maxval=2*jnp.pi
    )
    states = tilted_ring(phi, mu1, mu2)
    return states
batched_init_rings= jax.vmap(init_ring, in_axes=(None,0,0))


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

def init_ring_theta(Nstates, mu, key):
    mu1, mu2 = mu
    phi = jax.random.uniform(
        key,
        shape=(Nstates,),
        minval=0,
        maxval=2 * jnp.pi
    )
    return tilted_ring_theta(phi, mu1, mu2)

batched_init_rings_z = jax.vmap(init_ring_theta, in_axes=(None, 0, 0))


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
    s_in, s_fin = 0.005, 5
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

@functools.partial(jax.jit, static_argnames=())
def MMD_von_mises(X, Y, kappa=5.0):
    """
    MMD con kernel de Von Mises: k(ψ,φ) = exp(κ·|<ψ|φ>|²)
    
    Args:
        X: (N, 2) estados cuánticos generados
        Y: (M, 2) estados cuánticos objetivo
        kappa: parámetro de concentración del kernel
    Returns:
        escalar ≥ 0
    """
    N, M = X.shape[0], Y.shape[0]

    fXY = jnp.abs(X @ Y.conj().T) ** 2 
    fXX = jnp.abs(X @ X.conj().T) ** 2  
    fYY = jnp.abs(Y @ Y.conj().T) ** 2 

    # Kernel de Von Mises
    kXY = jnp.exp(kappa * fXY)
    kXX = jnp.exp(kappa * fXX)
    kYY = jnp.exp(kappa * fYY)


    mXY = jnp.mean(kXY)
    mXX = jnp.sum(kXX) / (N * N + 1e-10)
    mYY = jnp.sum(kYY) / (M * M + 1e-10)

    return mXX + mYY - 2.0 * mXY


@functools.partial(jax.jit, static_argnames=())
def MMD_multiscale(X, Y, kappas=(1, 10.0, 30.0)):
    """
    MMD con kernel multi-escala: suma de Von Mises con varias concentraciones.
    
    Captura simultáneamente estructura gruesa (κ pequeño) y fina (κ grande).
    Más robusto que un solo κ fijo.
    
    Args:
        X, Y: (N, 2) estados cuánticos
        kappas: tupla de concentraciones (static para JIT)
    Returns:
        escalar ≥ 0
    """
    return sum(MMD_von_mises(X, Y, k) for k in kappas)



@functools.partial(jax.jit, static_argnames=())
def coulomb_distance(X, Y, delta=0.01):
    """
    Distancia de Coulomb entre distribuciones de estados cuánticos.
    k(ψ,φ) = 1/(1 - |<ψ|φ>|² + δ)
    δ regulariza la singularidad cuando ψ=φ.
    δ ∈ [0.01, 0.1] recomendado.
    Particularmente buena para anillos porque penaliza fuertemente
    estados que deberían estar cerca pero están lejos.
    """
    N, M = X.shape[0], Y.shape[0]

    fXY = jnp.abs(X @ Y.conj().T) ** 2
    fXX = jnp.abs(X @ X.conj().T) ** 2
    fYY = jnp.abs(Y @ Y.conj().T) ** 2

    kXY =  1.0 / (1.0 - fXY + delta)
    kXX =  1.0 / (1.0 - fXX + delta)
    kYY =  1.0 / (1.0 - fYY + delta)

    mXY =  jnp.mean(kXY)
    mXX =  jnp.sum(kXX) / (N * (N ) + 1e-10)
    mYY =  jnp.sum(kYY ) / (M * (M) + 1e-10)

    return + mXX + mYY - 2.0 * mXY



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
current_withancilla = jax.vmap(
        lambda states: batched_addancilla(states, Nancilla)
    )


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
    key, subkey1, subkey2 = jax.random.split(key,3)
    c = tc.Circuit(Nqubits, inputs=current_state)
    for i in range(Nancilla):
        c.ry(Ndataq+i, theta = (mu[0]))
        c.rz(Ndataq+i, theta = (mu[1]))
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
    keys = jax.random.split(key,Nstates)
    transformed_states = batched_transform(theta, mu, current, Nancilla, Ndataq, Nqubits, keys) # Returns Shape: [Npoints, 2**Nqubits]
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

def _eval_loss_target(params, mus, current_states, final_states, kind, Nancilla, Ndataq, Nqubits, key):
    keys = jax.random.split(key, len(mus))
    return jnp.mean(jax.vmap(
        lambda mu, current, target, k: loss(params, mu, current, target, kind, Nancilla, Ndataq, Nqubits, k),
        in_axes=(0, 0, 0, 0)
    )(mus, current_states, final_states, keys))
eval_loss_target = jax.jit(_eval_loss_target, static_argnums=(4, 5, 6, 7))

def denoising_step(params, mus, opt_state, current_states, trg_states, final_states,
                   kind, opt_loops, Nancilla, Ndataq, Nqubits, key):
    losses_t, losses_target = [],[]
    best_loss   = 0
    best_params = params

    current_states_withancilla = current_withancilla(current_states)
    def scan_body(carry, _):
        params, opt_state, key, best_params, best_loss = carry
        key, subkey, eval_key = jax.random.split(key,3)
        new_params, new_opt_state, mean_loss = train_step(
            params, mus, opt_state, current_states_withancilla, trg_states,
            kind, Nancilla, Ndataq, Nqubits, subkey)
        improved = mean_loss < best_loss
        best_params = jax.lax.cond(improved, lambda: new_params, lambda: best_params)
        best_loss   = jnp.where(improved, mean_loss, best_loss)

        eval_keys = jax.random.split(eval_key, len(mus))
        loss_vs_target = [0]
        """jnp.mean(jax.vmap(
            lambda mu, current, target, k: loss(
                new_params, mu, current, target,
                kind, Nancilla, Ndataq, Nqubits, k),
            in_axes=(0, 0, 0, 0)
        )(mus, current_states_withancilla, final_states, eval_keys))"""

        return (new_params, new_opt_state, key, best_params, best_loss), (mean_loss, loss_vs_target)

    init = (params, opt_state, key, params, jnp.array(jnp.inf))
    (final_params, opt_state, key, best_params, best_loss), (losses_t, losses_target) = jax.lax.scan(
        scan_body, init, None, length=opt_loops)

    # Final transform: vmap over all the cases (mu, states, keys)
    final_keys = jax.random.split(key, len(mus) * Nstates).reshape(mus.shape[0], -1, 2)
    batched_final_transform = jax.vmap(
        lambda mu, states, keys: batched_transform(
            best_params, mu, states,
            Nancilla, Ndataq, Nqubits, keys), in_axes=(0, 0, 0))
    transformed_states = batched_final_transform(mus, current_states_withancilla, final_keys) # shape: (len(mus), Nstates, 2**Nqubits)
    return transformed_states, best_params, opt_state, losses_t, losses_target, best_loss
denoising_step = jax.jit(denoising_step, static_argnums=(6, 7, 8, 9, 10))

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
    all_losses_t, all_losses_target, best_loss_per_step= [],[], []
    bck_distr = [all_target_distr[:,T]]
    final_states = all_target_distr[:,0]

    for i in range(T):
        key, subkey = jax.random.split(key)
        params    = jax.random.normal(subkey, (L, 2*Nqubits))   # single shared params
        opt_state = opt.init(params)

        key, subkey = jax.random.split(key,2)
        trg_states = all_target_distr[:,T-i-1]    # Shape: [6, Npoints, 2**Ndataq]
        current_states = bck_distr[i]       # Shape: [6, Npoints, 2**Ndataq]
        
        temp3=time.time()
        next_states, params, opt_state, losses_t, losses_target, best_loss = denoising_step(
                    params, mus, opt_state,
                    current_states, trg_states, final_states,
                    dist_kind, opt_loops, 
                    Nancilla, Ndataq, Nqubits, subkey
                )
        all_losses_t.append(losses_t)
        all_losses_target.append(losses_target)
        best_loss_per_step.append(float(best_loss))
        parameters.append(params)
        bck_distr.append(next_states)
        print(f"Denoising step {i+1}, time:  {time.time()-temp3:.4f}")
    bck_distr = bck_distr[::-1]
    return bck_distr, parameters, all_losses_t, all_losses_target, best_loss_per_step



# ======== EXECUTION ========
key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

temp1=time.time()
def prepare_input(key,cond_leng):
    keys = jax.random.split(key, cond_leng)
    return batched_init_rings_z(Nstates, mus, keys)
all_input = prepare_input(key,len(mus))
print(len(all_input))
print("Preparation time: ", time.time()-temp1)

temp2=time.time()
def scramble_input(all_initial_states, sc_param, Ndata, cond_leng, key):
    keys = jax.random.split(key, cond_leng)
    return batched_scrambling(all_initial_states, sc_param, Ndata, keys)
all_scrambled = scramble_input(all_input,schedule("quadratic", T), Ndataq, len(mus), subkey2)
print(len(all_scrambled))
print("Scrambling time: ", time.time()-temp2)
all_losses_t, all_losses_target = [],[]
bck_distr, parameters, all_losses_t, all_losses_target, best_loss_per_step = denoising(all_scrambled, mus, coulomb_distance, Nopt_loops, Nancilla, Ndataq, L, subkey3)
print(f"Training completed for {len(bck_distr)} conditioning types")


np.save("params.npy",np.array(parameters))
np.save("bck_distr.npy",np.array(bck_distr))
np.save("losses_t.npy",np.array(all_losses_t))
np.save("losses_target.npy", np.array(all_losses_target)) 
np.save("best_losses_t.npy", np.array(best_loss_per_step)) 
np.save("all_input.npy",         np.array(all_input))          # <-- nuevo
np.save("conditioning_params.npy", np.array(mus))      



import math
import qutip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
import numpy as np



# ────────────────────────   PLOTING TRAINING RESULTS   ────────────────────────
def plot_bloch_qutip(bck_distr, mus):
    final = np.array(bck_distr[0])   # (Npoints, Nstates, 2)
    Npoints = len(mus)
    # Colors automatically
    colors = plt.cm.hsv(np.linspace(0, 0, Npoints))
    buffers = []

    for i, (kind, color) in enumerate(zip(mus, colors)):
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

plot_bloch_qutip(bck_distr, mus)



# ────────────────────────   PLOTING LOSSES   ────────────────────────

import numpy as np
import matplotlib.pyplot as plt

CLR_T  = "#3973f0"
CLR_SM = "#e74d4d"

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
ax_t, ax_f = axes


# ─────────────────────────────────────────────
# Dibujar curvas
# ─────────────────────────────────────────────

for i, (losses_t, losses_target) in enumerate(
    zip(all_losses_t, all_losses_target)
):

    # ── loss t→t−1 ────────────────────────────

    data_t = np.array(losses_t, dtype=float)

    # evitar problemas con log
    data_t = np.maximum(data_t, 1e-20)

    xs_t = i * len(data_t) + np.arange(len(data_t))

    ax_t.scatter(
        xs_t,
        data_t,
        color=CLR_T,
        alpha=0.1,
        s=0.4,
        linewidths=0
    )

    # suavizado en espacio log
    w = max(1, len(data_t)//20)

    log_data = np.log10(data_t)

    sm = np.convolve(
        log_data,
        np.ones(w)/w,
        mode="valid"
    )

    sm = 10**sm

    ax_t.plot(
        i*len(data_t) + np.arange(len(sm)),
        sm,
        color=CLR_SM,
        linewidth=1.2
    )


    # ── loss t→0 ─────────────────────────────

    if len(losses_target) > 0:

        vals_f = np.array(
            losses_target,
            dtype=float
        )

        vals_f = np.maximum(vals_f, 1e-20)

        steps_f = (
            np.arange(len(vals_f))*50
            + i*len(data_t)
        )

        ax_f.scatter(
            steps_f,
            vals_f,
            color=CLR_T,
            alpha=0.6,
            s=8,
            linewidths=0,
            zorder=3
        )

        if len(vals_f) >= 3:

            w_f = max(
                1,
                len(vals_f)//5
            )

            log_vals = np.log10(vals_f)

            sm_f = np.convolve(
                log_vals,
                np.ones(w_f)/w_f,
                mode="valid"
            )

            sm_f = 10**sm_f

            xs_sm_f = (
                steps_f[:len(sm_f)]
                + w_f//2*50
            )

            ax_f.plot(
                xs_sm_f,
                sm_f,
                color=CLR_SM,
                linewidth=1.2,
                zorder=4
            )


# ─────────────────────────────────────────────
# Escalas automáticas independientes
# ─────────────────────────────────────────────

vals_t = np.concatenate([
    np.array(x, dtype=float)
    for x in all_losses_t
])

vals_t = vals_t[
    np.isfinite(vals_t)
    & (vals_t > 0)
]

"""vals_f = np.concatenate([
    np.array(x, dtype=float)
    for x in all_losses_target
    if len(x) > 0
])"""
"""
vals_f = vals_f[
    np.isfinite(vals_f)
    & (vals_f > 0)
]"""


ax_t.set_yscale("log")
ax_f.set_yscale("log")

ax_t.set_ylim(
    np.min(vals_t)*0.8,
    np.max(vals_t)*1.2
)

"""ax_f.set_ylim(
    np.min(vals_f)*0.8,
    np.max(vals_f)*1.2
)"""


# ─────────────────────────────────────────────
# Formato
# ─────────────────────────────────────────────

T = len(all_losses_t)
Nopt = len(all_losses_t[0]) if T > 0 else 1

for ax in axes:

    for k in range(1, T):
        ax.axvline(
            k*Nopt,
            color="#bbbbbb",
            linewidth=0.6,
            linestyle=":"
        )

    ax.set_xlabel(
        "Optimization step",
        fontsize=11
    )

    ax.grid(
        True,
        which="both",
        linewidth=0.4,
        alpha=0.5
    )

    ax.set_xticks([
        k*Nopt + Nopt//2
        for k in range(T)
    ])

    ax.set_xticklabels(
        [f"$t_{{{T-k}}}$" for k in range(T)],
        fontsize=7
    )


ax_t.set_ylabel(
    r"$\mathcal{L}(t \to t-1)$",
    fontsize=11
)

ax_f.set_ylabel(
    r"$\mathcal{L}(t \to 0)$",
    fontsize=11
)

ax_t.set_title(
    "t vs t-1 loss",
    fontsize=11
)

ax_f.set_title(
    "t vs target loss",
    fontsize=11
)


plt.tight_layout(
    pad=1.5,
    rect=[0,0.05,1,1]
)

plt.savefig(
    "losses.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

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
    label = mus[dist_idx]
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
