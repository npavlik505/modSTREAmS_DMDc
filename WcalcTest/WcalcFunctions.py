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
import matplotlib.pyplot as plt
from scipy.signal import welch, csd, TransferFunction

# Directory for saving and loading intermediate files
script_dir = os.path.dirname(os.path.realpath(__file__))

#
# 1.load_h5_data(): load and save as numpy files the simulation's span averaged velocity fields, actuation, and dt
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

# def TFfromTimeSeries(snapshots_flat, U, dt, number_of_poles):
#     """
#     Estimate frequency response (Transfer Function) from time series data.
    
#     Parameters:
#         snapshots_flat: ndarray
#             Flattened difference between FOM and ROM snapshots [states x samples].
#         U: ndarray
#             Actuation input vector [1 x samples].
#         dt: float
#             Time step between samples.
#         number_of_poles: int (unused, maintained for compatibility)
#             Number of poles desired if using rational fitting (ignored here).

#     Returns:
#         G_diff_freq_response: ndarray (complex)
#             Frequency response of the difference G_diff at frequencies f.
#         f: ndarray
#             Frequencies (in Hz) at which the frequency response is computed.
#     """

#     # Ensure dt is scalar float
#     dt = float(np.atleast_1d(dt)[0])
#     fs = 1.0 / dt  # Sampling frequency

#     # Flatten input
#     u = np.ravel(U)
#     Y_diff = snapshots_flat  # shape: (n_states, n_samples)

#     # Calculate frequency response for each state separately, then average magnitude-wise
#     n_states, n_samples = Y_diff.shape
#     nperseg = min(1024, n_samples)

#     # Compute input power spectral density once (common for all states)
#     f, Puu = welch(u, fs=fs, nperseg=nperseg)

#     # Initialize array to hold individual state frequency responses
#     G_states = np.zeros((n_states, len(f)), dtype=complex)

#     for i in range(n_states):
#         _, Pyu = csd(Y_diff[i, :], u, fs=fs, nperseg=nperseg)
#         # Compute frequency response for each state
#         G_states[i, :] = Pyu / Puu

#     # Average the frequency response over all states (robust to noise)
#     G_diff_freq_response = np.mean(G_states, axis=0)

#     return G_diff_freq_response, f

def TFfromTimeSeries(snapshots_flat, U, dt, number_of_poles):
    """
    Estimate the frequency response G_diff(f) = P_yu(f) / P_uu(f)
    using Welch (PSD) and CSD, with consistent segment lengths.

    Returns
    -------
    G_diff : ndarray, complex
        Averaged frequency response across all states.
    f      : ndarray, float
        Frequencies in Hz where G_diff was evaluated.
    """

    # ensure dt is scalar
    dt = float(np.atleast_1d(dt)[0])
    fs = 1.0 / dt

    # flatten input and output/error signals
    u = np.ravel(U)                  # shape: (N,)
    Y_diff = snapshots_flat          # shape: (n_states, N+1)

    # truncate Y_diff to match u’s length
    N = len(u)
    Y_diff = Y_diff[:, :N]           # now both are length N

    # choose segment length ≤ both signals
    nperseg = min(1024, N)
    # explicitly set nfft = nperseg for both
    nfft = nperseg

    # 1) PSD of input
    f, Puu = welch(u, fs=fs,
                   nperseg=nperseg,
                   nfft=nfft)

    # 2) CSD for each state → freq response
    n_states = Y_diff.shape[0]
    G_states = np.zeros((n_states, len(f)), dtype=complex)
    for i in range(n_states):
        _, Pyu = csd(Y_diff[i, :], u,
                     fs=fs,
                     nperseg=nperseg,
                     nfft=nfft)
        G_states[i, :] = Pyu / Puu

    # 3) average across states
    G_diff = np.mean(G_states, axis=0)

    return G_diff, f



def WcalculationVec(G_DIFF: TransferFunction,
                    freqs: np.ndarray,     # in Hz
                    omega_b: float,
                    epsilon: float):
    """
    Build W(jw) = ( (1/Ms) * s + ω_b ) / ( s + ω_b*ε ) 
    on the same freq grid used by TFfromTimeSeries.

    Returns
      w   : 1D array of rad/s
      W   : complex 1D array W(jw)
      (optionally) We_tf: the TransferFunction object
    """
    # 1) Turn Hz->rad/s
    w = 2 * np.pi * freqs

    # # 2) Evaluate both transfer functions
    # _, H_DIFF = G_DIFF.freqresp(w)

    # For SISO SciPy TF you often get H.shape == (n_freq,) already.

    # 3) Pointwise error magnitude
    mag    = np.abs(G_DIFF)
    Ms     = 1.1 * np.max(mag)    # 10% safety margin

    # 4) Build s = jω and evaluate W(jω)
    s = 1j * w
    W = ((1/Ms)*s + omega_b) / (s + omega_b*epsilon)

    # 5) (Optional) also package it as a TransferFunction
    We_tf = TransferFunction([1/Ms, omega_b],
                             [1,     omega_b*epsilon])

    return w, W, We_tf

def ErrorFreqPlot(G_DIFF, w):
    # w is your rad/s vector
    mag = np.abs(G_DIFF)

    print("Max discrepancy magnitude (Hinf norm):", mag.max())

    # G_DIFF = G_FOM - G_ROM
    # max_Gdiff_mag = np.max(np.abs(G_DIFF))
    # print("Max discrepancy magnitude (Hinf norm):", max_Gdiff_mag)
    # print("Verify close match with Hinf norm values from DMDcAnalysis")

    plt.figure()
    plt.semilogx(w, 20 * np.log10(mag), label='ROM-FOM absolute error') # np.log10(mag / np.abs(H_FOM)), label='ROM-FOM relative error')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.legend()
    plt.show()

def WeightFreqPlot(f, W_vec):
    plt.semilogx(f, 20*np.log10(np.abs(W_vec)), label='Weight |W(jω)|')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Magnitude [dB]')
    plt.legend()
    plt.grid(True)
    plt.show()

def SmallGainPracticalCheck(G_DIFF, W_vec, w):
    """
    Perform the practical small‐gain check:
      1. Compute the FOM–ROM error Δ(jω) = G_FOM(jω) – G_ROM(jω)
      2. Plot 20·log10|Δ|, 20·log10|W|, and their sum 20·log10|Δ·W|
      3. Draw a 0 dB line and report any frequency where sum ≥ 0 dB

    Parameters
    ----------
    G_FOM : control.TransferFunction or similar
        The full-order model.
    G_ROM : control.TransferFunction or similar
        The reduced-order model.
    W_vec : array_like
        Samples of W(jω) at the frequencies in `w`.
    w : array_like
        Frequency vector (rad/s).

    Returns
    -------
    total_db : ndarray
        The pointwise sum in dB: 20·log10|Δ(jω)| + 20·log10|W(jω)|.
    """
    # # Compute frequency responses
    # _, H_DIFF = G_DIFF.freqresp(w)
    # H_diff = H_DIFF.flatten()

    # Convert to dB
    error_db  = 20 * np.log10(np.abs(G_DIFF))
    weight_db = 20 * np.log10(np.abs(W_vec).flatten())
    total_db  = error_db + weight_db

    # Plot
    plt.figure()
    plt.semilogx(w, error_db,  label='Error 20·log₁₀|Δ(jω)|', linewidth=1)
    plt.semilogx(w, weight_db, label='Weight 20·log₁₀|W(jω)|', linewidth=1)
    plt.semilogx(w, total_db,  label='Sum 20·log₁₀|Δ·W|', linewidth=2)
    plt.axhline(0, color='k', linestyle='--', label='0 dB line')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    plt.show()

    # Check small‐gain condition
    if np.all(total_db < 0):
        print("✅ Small‐gain condition satisfied: |Δ·W| < 1 over all ω.")
    else:
        viol_idx = np.where(total_db >= 0)[0]
        first_viol = w[viol_idx[0]]
        print(f"❌ Small‐gain violated at {len(viol_idx)} frequencies; "
              f"first violation at ω = {first_viol:.2f} rad/s.")

    return total_db
