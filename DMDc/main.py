# main analysis script
import numpy as np
import DMDcFunctions

# # File: main.py
if __name__ == '__main__':
    # Define paths to the FOM span_averages.H5 file and the corresponding actuation values, as well as the interval between writing the span-averaged data
    span_avg = '/home/nate/Desktop/DMDc/output/distribute_save/span_averages.h5'
    traj     = '/home/nate/Desktop/DMDc/output/distribute_save/trajectories.h5'
    SA_int = 5 # Note: Inverval can be found under "config" section of justfile

    # Build ROM (including U,V,W velocity components) and get grid dims and deltat
    # here comps = [1,2,3] correspond to U, V, W in span_average
    comps = [1, 2, 3]
    DMDcBasis = DMDcFunctions.create_dmdc_model(span_avg, traj, SA_int, comps)

    # Load saved model and inputs
    A  = np.load('A_red_matrix.npy')
    B  = np.load('B_red_matrix.npy')
    U  = np.load('U.npy')
    dt_vals = np.load('dt.npy')
    print(f'dt_vals shape: {np.shape(dt_vals)}')
    dt = dt_vals[0]
    x0 = np.load('Init_snap_red.npy')

    # Write DMDc ROM output to HDF5
    DMDcFunctions.dmdc_sim(A, B, U, dt, x0, DMDcBasis)

    # Calculate the W (performance weighting) and R (actuation weighting) matrices
    phi  = DMDcFunctions.Q_creation(alpha=0.0)   # Alpha determines path from 1 to 0. Alpha = 0 is linear, higher alpha values indicate shorter,...
                                                # steeper transition near boundary layer.

    # Load Q and R for to finally calculate K, for the controller U(t)
    Q  = np.load('Q.npy')
    R  = np.load('R.npy')

    # Calculate K
    DMDcFunctions.K_calc(A, B, Q, R)

    DMDcFunctions.create_truncated_dmdc_model(span_avg_file = span_avg, traj_file = traj, SA_int=SA_int)

    # # Calculate the square root of the the error between the ROM and FOM squared
    # error = DMDcFunctions.Model_Red_Error(span_avg) 
    # # Animate the error across time, with comp = [0, 1, 2] corresponding to velocity component [U, V, W] and save as MP4 using ffmpeg
    # comp = 0
    # err_ani = DMDcFunctions.animate_error(error, comp, interval=50)
    # err_ani.save(f'error_{comp}_velocity.mp4', fps=20, dpi=150)

