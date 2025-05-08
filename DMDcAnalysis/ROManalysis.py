# main analysis script
import numpy as np
import AnalysisFunctions
import os

# # File: main.py
if __name__ == '__main__':
    # Define paths to the FOM span_averages.H5 file and the corresponding actuation values, as well as the interval between writing the span-averaged data
    span_avg = '/home/nate/Desktop/DMDc/output/distribute_save/span_averages.h5'
    traj     = '/home/nate/Desktop/DMDc/output/distribute_save/trajectories.h5'
    SA_int = 5 # Note: Inverval can be found under "config" section of justfile
    compsFOM = [1, 2, 3] # comps = [1,2,3] correspond to U, V, W in span_average
    compsROM = [0, 1, 2] # comps = [0,1,2] correspond to U, V, W in DMDc_span_average
    compError = [0] # comps = [0,1,2] correspond to U, V, W in the 2D RSE
    
    # Calculate the flattened snapshots of the FOM for Frobenius Analysis  
    snapshots_flat_FOM, U = AnalysisFunctions.load_h5_data(span_avg_file = span_avg, traj_file = traj, SAint = SA_int, velocity_components=compsFOM)

    # Load saved ROM components and input (collected from )
    A  = np.load('A_red_matrix.npy')
    B  = np.load('B_red_matrix.npy')
    DMDcBasis = np.load('DMDcBasis.npy')
    dt_vals = np.load('dt.npy')
    print(f'dt_vals shape: {np.shape(dt_vals)}')
    dt = dt_vals[0]
    x0_full = np.load('init_orig_state.npy')

    # Calculate and then load the reduced state initial snapshot
    x0 = AnalysisFunctions.init_snapshot_red(init_full=x0_full)

    # Write DMDc ROM output to HDF5
    AnalysisFunctions.dmdc_sim(A, B, U, dt, x0, DMDcBasis)

    # Calculate the flattened snapshots of the ROM for Frobenius Analysis  
    snapshots_flat_ROM, U = AnalysisFunctions.load_h5_data(span_avg_file = 'dmdc_span_averages.h5', traj_file = traj, SAint = SA_int, velocity_components=compsROM)

    Erms = AnalysisFunctions.Erms(snapshots_flat_FOM = snapshots_flat_FOM , snapshots_flat_ROM = snapshots_flat_ROM)
    print(f'Erms: {Erms}')
    NRMSE = AnalysisFunctions.NRMSE(snapshots_flat_FOM = snapshots_flat_FOM , snapshots_flat_ROM = snapshots_flat_ROM)
    print(f'NRMSE: {NRMSE}')
    Efrob = AnalysisFunctions.Efrobenius(snapshots_flat_FOM = snapshots_flat_FOM , snapshots_flat_ROM = snapshots_flat_ROM)
    print(f'Efrobenius: {Efrob}')


    # # Calculate the square root of the the error between the ROM and FOM squared
    # error = AnalysisFunctions.Model_Red_Error(span_avg) 
    # # Animate the error across time, with comp = [0, 1, 2] corresponding to velocity component [U, V, W] and save as MP4 using ffmpeg
    # comp = 0
    # err_ani = AnalysisFunctions.animate_error(error, comp, interval=50)
    # err_ani.save(f'error_{comp}_velocity.mp4', fps=20, dpi=150)


    
    # Produce the H2 and Hinf values for each ROM 
    AnalysisFunctions.H2_Hinf_Analysis()