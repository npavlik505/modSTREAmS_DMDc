# Functions To determine accuracy of ROM

# File: DMDcFunctions.py
import os
import h5py
import math
import numpy as np
from pydmd import DMDc
from scipy.linalg import solve_discrete_are
from scipy.linalg import solve_continuous_lyapunov
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from control import ss

# Directory for saving and loading intermediate files
script_dir = os.path.dirname(os.path.realpath(__file__))

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

# Compute reduced initial state
def init_snapshot_red(init_full):
    DMDcBasis = np.load('DMDcBasis.npy')
    init_full = np.load('init_orig_state.npy')
    init_snap_full_flat = init_full.ravel()
    init_snap_red = DMDcBasis.T @ init_snap_full_flat
    print("Init_snap_reduced:", init_snap_red.shape)
    np.save(os.path.join(script_dir, 'Init_snap_red.npy'), init_snap_red)
    return init_snap_red

#
# 1. dmdc_sim(): This function simulates the ROM provided by DMDc and saves the output as an h5 file structured similarly to the original "span_averages.h5" file 
#

# Reconstruct U snapshots and write to HDF5 in original format
def dmdc_sim(A, B, U, dt, x0, DMDcBasis, output_h5='dmdc_span_averages.h5'):
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
    x_full_flat = DMDcBasis.dot(X)

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


def Erms(snapshots_flat_FOM, snapshots_flat_ROM):
    """
    Compute the (absolute) root‐mean‐square error between ROM and FOM snapshots:
      E_rms = sqrt( mean( (Y_rom − Y_fom)^2 ) )
    """
    diff = snapshots_flat_ROM - snapshots_flat_FOM
    mse  = np.mean(diff**2)
    return np.sqrt(mse)

def NRMSE(snapshots_flat_FOM, snapshots_flat_ROM):
    """
    Compute the normalized RMS error:
      NRMSE = E_rms / RMS_fom,
    where RMS_fom = sqrt( mean( Y_fom^2 ) ).
    """
    E_rms = Erms(snapshots_flat_FOM, snapshots_flat_ROM)
    denom = np.sqrt(np.mean(snapshots_flat_FOM**2))
    if denom == 0:
        return np.nan
    return E_rms / denom

def Efrobenius(snapshots_flat_FOM, snapshots_flat_ROM):
    """
    Compute the Frobenius‐norm error ratio:
      E_frob = ||Y_rom − Y_fom||_F / ||Y_fom||_F.
    """
    num = np.linalg.norm(snapshots_flat_ROM - snapshots_flat_FOM, ord='fro')
    den = np.linalg.norm(snapshots_flat_FOM,             ord='fro')
    if den == 0:
        return np.nan
    return num / den

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
# H2 and Hinf Calculations
# The first function loads the ROMs from the ReducedModels folder, so make sure it is included in the CWD
# The second and third function uses Ch.4 methods to compute the H2 and Hinf norms
# The fourth function simply configures the first three to produce the H2 and Hinf values for each ROM 
#
rom_folder   = os.path.join(os.path.dirname(__file__), 'ReducedModels')
energy_labels = ['99pct', '96pct', '93pct', '90pct']

def load_rom_ss(rom_dir, label):
    """
    Load A_r, B_r and build a StateSpace with C=I, D=0.
    """
    A = np.load(os.path.join(rom_dir, f'A_red_matrix_{label}.npy'))
    B = np.load(os.path.join(rom_dir, f'B_red_matrix_{label}.npy'))
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    C = np.eye(n_states)
    D = np.zeros((n_states, n_inputs))
    return ss(A, B, C, D)

def h2_norm_via_lyap(sys):
    """
    Compute H2 norm by solving A X + X A^T + B B^T = 0,
    then ||G||_H2 = sqrt(trace(C X C^T + D D^T)).
    """
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    # Solve A X + X A^T + B B^T = 0  =>  solve_continuous_lyapunov(A, -B B^T)
    X = solve_continuous_lyapunov(A, -B @ B.T)
    # H2 norm
    return np.sqrt(np.trace(C @ X @ C.T + D @ D.T))


def hinf_norm_approx(sys, wmin=1e-3, wmax=1e3, npts=500):
    """
    Approximate H∞ norm by frequency sweep:
      ‖G‖∞ ≈ max_{ω∈logspace[wmin,wmax]} σ_max(G(jω))
    Returns (approx_norm, ω_at_max).
    """
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    freqs = np.logspace(np.log10(wmin), np.log10(wmax), npts)
    max_gain = 0.0
    w_at_max  = freqs[0]
    I = np.eye(A.shape[0])
    
    for w in freqs:
        G = C @ np.linalg.inv(1j*w * I - A) @ B + D
        sigma = np.linalg.svd(G, compute_uv=False)[0]
        if sigma > max_gain:
            max_gain = sigma
            w_at_max  = w
    
    return max_gain, w_at_max

def H2_Hinf_Analysis():
    rom_folder   = os.path.join(os.path.dirname(__file__), 'ReducedModels')
    energy_labels = ['99pct', '96pct', '93pct', '90pct']
    for label in energy_labels:
        sysr = load_rom_ss(rom_folder, label)
        
        h2_val = h2_norm_via_lyap(sysr)
        hinf_val, w_peak = hinf_norm_approx(sysr, wmin=1e-2, wmax=1e2, npts=800)
        
        print(f"[{label}]  ‖G‖₂ ≈ {h2_val:.3e}  ;  "
              f"‖G‖∞ ≈ {hinf_val:.3e}  @ ω={w_peak:.2f} rad/s")