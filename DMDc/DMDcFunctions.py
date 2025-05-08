# File: DMDcFunctions.py
import os
import h5py
import math
import numpy as np
from pydmd import DMDc
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Directory for saving and loading intermediate files
script_dir = os.path.dirname(os.path.realpath(__file__))
reduced_dir  = os.path.join(script_dir, 'ReducedModels')
os.makedirs(reduced_dir, exist_ok=True)

#
# These three functions (respectively):
# 1.load_h5_data(): load and save as numpy files the simulation's span averaged velocity fields, actuation, and dt
# 2.perform_dmdc(): Perform DMDc via the DMDc software from the Brunton lab 
# 3.create_dmdc_model(): Create the DMDc model from which the A and B matrices are saved
#

# Load HDF5 data, flatten U snapshots, and subsample
def load_h5_data(span_avg_file, traj_file, SAint, velocity_components=[1,2,3]):
    # Read span-averaged snapshots
    with h5py.File(span_avg_file, 'r') as f:
        data = f['span_average'][:]            # shape (m, comps, nx, ny)
    # Read control inputs and time steps
    with h5py.File(traj_file, 'r') as f:
        u_full = f['jet_amplitude'][:]        # shape (m,)
        dt_full = f['dt'][:]                  # shape (m,)

    # Save initial (U,V,W) snapshot for later reconstruction
    snapshots0 = data[0, velocity_components, :, :]  # shape = (3, nx, ny)
    np.save(os.path.join(script_dir, 'init_orig_state.npy'), snapshots0)

    m, _, nx, ny = data.shape
    # Stack U,V,W then flatten each snapshot to a (3*nx*ny)-vector
    snapshots = data[:, velocity_components, :, :]     # (m, 3, nx, ny)
    snapshots_flat = snapshots.reshape(m, -1).T        # (3*nx*ny, m)

    # Subsample both time and control
    u_sub = u_full[::SAint]                      # (m_sub,)
    dt_sub = dt_full[::SAint]                    # (m_sub,)

    # Align for discrete DMDc: drop last control and keep corresponding snapshots
    u = u_sub[:-1]                                   # (m_sub-1,)
    snapshots_flat = snapshots_flat[:, :u.shape[0]+1]# (n_states, m_sub)

    # Reshape control to (1, m_sub-1)
    u = u[np.newaxis, :]

    # Save subsampled control and dt
    np.save(os.path.join(script_dir, 'U.npy'), u)
    # We save dt as a scalar (assume constant step)
    np.save(os.path.join(script_dir, 'dt.npy'), dt_sub)

    return snapshots_flat, u

# Compute DMDc A and B matrices
def perform_dmdc(snapshots, u, svd_rank=0):
    dmdc = DMDc(svd_rank=svd_rank)
    dmdc.fit(snapshots, u)
    A_reduced = dmdc.operator.as_numpy_array
    DMDcBasis = dmdc.basis           # shape (n_features, r)
    B_full = dmdc.B           # shape (n_features, m)
    B_reduced = DMDcBasis.T @ B_full # shape (r, m)
    init_full = np.load('init_orig_state.npy')
    init_snap_full_flat = init_full.ravel()
    init_snap_reduced = DMDcBasis.T @ init_snap_full_flat
    print("Init_snap_reduced:", init_snap_reduced.shape)
    print("A_reduced:", A_reduced.shape)
    print("B_reduced:", B_reduced.shape)
    return A_reduced, B_reduced, init_snap_reduced, DMDcBasis

# Create and save DMDc model
def create_dmdc_model(span_avg_file, traj_file, SA_int, velocity_components=[1,2,3]):
    snapshots, u = load_h5_data(span_avg_file, traj_file, SA_int, velocity_components)
    print(f"Snapshots matrix shape: {snapshots.shape}")
    print(f"Control matrix shape: {u.shape}")
    A_red, B_red, init_snap_red, DMDcBasis = perform_dmdc(snapshots, u)
    np.save(os.path.join(script_dir, 'Init_snap_red.npy'), init_snap_red)
    np.save(os.path.join(script_dir, 'A_red_matrix.npy'), A_red)
    np.save(os.path.join(script_dir, 'B_red_matrix.npy'), B_red)
    np.save(os.path.join(script_dir, 'DMDcBasis.npy'), DMDcBasis)
    print(f"Saved A_red_matrix.npy ({A_red.shape}), B_red_matrix.npy ({B_red.shape}), Init_snap_red.npy ({init_snap_red.shape}), and DMDcBasis.npy ({DMDcBasis.shape})")
    return DMDcBasis


# A quick and dirty method to produce the truncated ROM Models
# 
# 2.perform_dmdc(): Perform DMDc via the DMDc software from the Brunton lab 
# 3.create_dmdc_model(): Create the DMDc model from which the A and B matrices are saved
#
# Compute DMDc A and B matrices
def perform_truncated_dmdc(snapshots, u, svd_rank=0):
    dmdc = DMDc(svd_rank=svd_rank)
    dmdc.fit(snapshots, u)
    A_reduced = dmdc.operator.as_numpy_array
    DMDcBasis = dmdc.basis           # shape (n_features, r)
    B_full = dmdc.B           # shape (n_features, m)
    B_reduced = DMDcBasis.T @ B_full # shape (r, m)
    init_full = np.load(os.path.join(script_dir, 'init_orig_state.npy'))
    init_snap_full_flat = init_full.ravel()
    init_snap_reduced = DMDcBasis.T @ init_snap_full_flat

    return A_reduced, B_reduced, init_snap_reduced, DMDcBasis

# Create and save DMDc model
def create_truncated_dmdc_model(span_avg_file, traj_file, SA_int, velocity_components=[1,2,3]):
    snapshots, u = load_h5_data(span_avg_file, traj_file, SA_int, velocity_components)
    # 2) compute SVD once to get singular values
    U, S, Vh = np.linalg.svd(snapshots, full_matrices=False)
    energy   = np.cumsum(S**2) / np.sum(S**2)

    # 3) for each desired energy level, find r and build/save ROM
    thresholds = [0.99, 0.96, 0.93, 0.90]
    for pct in thresholds:
        # find smallest r so that energy[:r] ≥ pct
        r = np.searchsorted(energy, pct) + 1
        label = f"{int(pct*100)}pct"

        # build ROM
        A_r, B_r, init_r, Phi_r = perform_truncated_dmdc(snapshots, u, svd_rank=r)

        # save with unique names in ReducedModels/
        np.save(os.path.join(reduced_dir, f'Init_snap_red_{label}.npy'),   init_r)
        np.save(os.path.join(reduced_dir, f'A_red_matrix_{label}.npy'),    A_r)
        np.save(os.path.join(reduced_dir, f'B_red_matrix_{label}.npy'),    B_r)
        np.save(os.path.join(reduced_dir, f'DMDcBasis_{label}.npy'),      Phi_r)

        print(f"[{label}]  r={r:3d} modes → "
              f"Init_snap_red_{label}.npy, A_red_matrix_{label}.npy, "
              f"B_red_matrix_{label}.npy, DMDcBasis_{label}.npy")

    # optional: return a dict of bases if you need them afterwards
    return


#
# 1. dmdc_sim(): This function simulates the ROM provided by DMDc and saves the output as an h5 file structured similarly to the original "span_averages.h5" file 
#

# Reconstruct U snapshots and write to HDF5 in original format
def dmdc_sim(A, B, U, dt, x0, Ur, output_h5='dmdc_span_averages.h5'):
    # x0: initial U snapshot (2D array of shape nx×ny)
    x0_elements = x0.shape
    print(f'x0_elements: {x0_elements}')


    # Discrete simulation: x_{k+1} = A x_k + B u_k
    n_states = A.shape[0]
    print(f'n_states: {n_states}')
    n_steps = U.shape[1]
    print(f'n_steps: {n_steps}')
    # Flatten initial state
    X = np.zeros((n_states, n_steps+1))
    X[:, 0] = x0
    for k in range(n_steps):
        X[:, k+1] = A.dot(X[:, k]) + B.dot(U[:, k])

    # Reshape to (time, 1, nx=600, ny=208)
    x_full_flat = Ur.dot(X)

    # Load original grid dims (may be U only or [U,V,W])
    init_orig_state = np.load(os.path.join(script_dir, 'init_orig_state.npy'))
    if init_orig_state.ndim == 2:
        # legacy U-only case
        n_comp = 1
        nx, ny = init_orig_state.shape
    else:
        # U,V,W stacked
        n_comp, nx, ny = init_orig_state.shape

    # Reconstruct 4D array in one step using C-order inversion of flattening
    T = n_steps + 1
    flat_TN = x_full_flat.T               # shape: (T, n_states = n_comp*nx*ny)
    data = flat_TN.reshape((T, n_comp, nx, ny), order='C')

    # Write to HDF5
    out_path = os.path.join(script_dir, output_h5)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('span_average', data=data)
        f.create_dataset('dt', data=dt)
    print(f"Wrote reconstructed snapshots to {out_path}, shape={data.shape}")
    return out_path


#
# 1. Model_Reduction_Error(): This returns the root squared error between the ROM and FOM fields
#
def Model_Red_Error(span_avg_file):
    # Read span-averaged snapshots
    with h5py.File(span_avg_file, 'r') as f:
        FOMdata = f['span_average'][:, 1:4, :, :]            # shape (m, comps, nx, ny)
    with h5py.File(f"{script_dir}/dmdc_span_averages.h5", 'r') as f:
        ROMdata = f['span_average'][:]            # shape (m, comps, nx, ny)
    RSEdata = np.abs(FOMdata-ROMdata)/np.abs(FOMdata)
    np.save(os.path.join(script_dir, 'SquaredError.npy'), RSEdata)
    return RSEdata

#
# 1. animate_error(): This returns an animation of RSE between ROM and FOM fields, calculated above
#
def animate_error(err, comp=0, interval=100, cmap='viridis'):

    nt, ncomp, nx, ny = err.shape
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(err[0,comp].T, origin='lower', cmap=cmap,
                   vmin=0, vmax=np.nanmax(err))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Relative error')
    title = ax.set_title(f"t = 0 (frame 0)")

    def update(i):
        im.set_data(err[i,comp])
        title.set_text(f"t = {i} (frame {i})")
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=nt,
                                  interval=interval, blit=True)
    return ani


#
# 1. Q_creation(): This creates Q, used in the optimal control cost function, by taking in a weighting matrix and applying the DMDc basis 
#

def Q_creation(alpha):
    FOM_field = np.load(os.path.join(script_dir, 'init_orig_state.npy'))
    print(FOM_field[0,0,:])
    _, Nx, Ny = np.shape(FOM_field)
    wall_frac = 2/3
    Ny_wr = int(round(wall_frac * Ny))
    j = np.arange(Ny_wr)
    nabla = j / (Ny_wr - 1)

    if abs(alpha) < 1e-12:
        # linear limit
        phi = 1.0 - nabla
    else:
        # sinh‑stretch
        phi = np.sinh(alpha * (1 - nabla)) / np.sinh(alpha)

    # pad with zeros to reach length Ny
    phi = np.pad(phi, (0, Ny - Ny_wr), 'constant', constant_values=0)

    # create matrix with each column equal to phi (shape: Ny x Nx)
    Q_full = np.tile(phi[:, None], (1, Nx))

    Q_xy = Q_full.T
    Q_stack = np.stack([Q_xy, Q_xy, Q_xy], axis=0)    # shape = (3, Nx, Ny)
    Q_vec = Q_stack.reshape(-1, order='C')

    print(f'The first five near-wall values of Q vec are {phi[:5]}')
    print(f'The last five boundary layer values of Q vec are {phi[-5:]}')
    print(f'Q matrix shape: {Q_full.shape}')

    # project onto DMDc basis if provided
    DMDcBasis = np.load(os.path.join(script_dir, 'DMDcBasis.npy'))
    # DMDcBasis: (n_features, r) basis matrix from DMDc
    # Q_reduced = DMDcBasis.T @ Q_vec   # (r, Nx)
    Phi_weighted = DMDcBasis * Q_vec[:, None]     # (N, r)
    Q_reduced = DMDcBasis.T @ Phi_weighted        # (r, r)
    print(f'Q_reduced shape: {Q_reduced.shape}')
    np.save(os.path.join(script_dir, 'Q.npy'), Q_reduced)
    R = np.array([[1]])
    print(f'R shape: {R.shape}')
    np.save(os.path.join(script_dir, 'R.npy'), R)
    return Q_reduced


#
# 1. K_calc(): This returns P and K, the later of which is the controller gain 
#

def K_calc(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    print(f'P has been calculated: {P.shape}')
    K = np.linalg.solve(R, B.T @ P)
    print(f'K has been calculated. Shape {K.shape}')
    print(f'u(t) = -Kx(t)')
    np.save(os.path.join(script_dir, 'P.npy'), P)
    np.save(os.path.join(script_dir, 'K.npy'), K)
    return P, K