# main script for calculating weighting function W using WcalcFunctions.py
import os
import numpy as np
import WcalcFunctions
import matplotlib.pyplot as plt



if __name__ == '__main__':
    ROM_reduction_level = '93' # Choose as string 99, 96, 93, 90 (represents energy after mode reduction). 93 chosen here because it has minimum H2 and equal Hinf norm compared to the rest

    SA_int = 5 # Note: Inverval can be found under "config" section of justfile
    traj = 'DMDcAnalysisFiles/trajectories.h5'
    # Load saved ROM components and input (collected from )
    A_ROM = np.load(f'ReducedModels/A_red_matrix_{ROM_reduction_level}pct.npy')
    B_ROM = np.load(f'ReducedModels/B_red_matrix_{ROM_reduction_level}pct.npy')
    DMDcBasis = np.load(f'ReducedModels/DMDcBasis_{ROM_reduction_level}pct.npy')
    dt = np.load('DMDcAnalysisFiles/dt.npy')
    x0_red = np.load(f'ReducedModels/Init_snap_red_{ROM_reduction_level}pct.npy')
    compsFOM = [1, 2, 3] # comps = [1,2,3] correspond to U, V, W in span_average
    compsROM = [0, 1, 2] # comps = [0,1,2] correspond to U, V, W in DMDc_span_average

    # Load and flatten the FOM and ROM snapshots and appropriately downsample actuation values
    snapshots_flat_FOM, U = WcalcFunctions.load_h5_data(span_avg_file = 'DMDcAnalysisFiles/span_averages.h5', traj_file = traj, SAint = SA_int, velocity_components=compsFOM) # FOM

    print(snapshots_flat_FOM.shape)
    print(snapshots_flat_FOM[0:100])


    # Write DMDc ROM output to HDF5
    WcalcFunctions.dmdc_sim(A_ROM, B_ROM, U, dt, x0_red, DMDcBasis)

    # Calculate the flattened snapshots of the ROM for Frobenius Analysis  
    snapshots_flat_ROM, U = WcalcFunctions.load_h5_data(span_avg_file = 'dmdc_span_averages.h5', traj_file = traj, SAint = SA_int, velocity_components=compsROM) #ROM

    #Debug
    print(snapshots_flat_ROM.shape)
    print(snapshots_flat_ROM[0:100])

    # Good up to here, error likely resides in TFfromTimeSeries function

    snapshots_flat_diff = snapshots_flat_FOM - snapshots_flat_ROM

    # Calculate the transfer function for both FOM and ROM
    G_diff_response, f = WcalcFunctions.TFfromTimeSeries(snapshots_flat=snapshots_flat_diff, U=U, dt=dt, number_of_poles=4)

    # Weight matrix calculation from book
    WcalcFunctions.ErrorFreqPlot(G_diff_response, 2*np.pi*f)

    freq1, W_vec1, W_tf1 = WcalcFunctions.WcalculationVec1(G_diff_response, f, omega_b = 40.0, epsilon = 0.1)
    print(f'freq (omega): {freq1}')
    print(f'W_vec (Weights in vector form): {W_vec1}')
    print(f'W_freq (Weights in TF form): {W_tf1}')

    WcalcFunctions.WeightFreqPlot(2*np.pi*f, W_vec1)

    total_db1 = WcalcFunctions.SmallGainPracticalCheck(G_diff_response, W_vec1, w = 2*np.pi*f)

    # Weight matrix adjustment 1
    freq2, W_vec2, W_tf2 = WcalcFunctions.WcalculationVec2(G_diff_response, f, omega_b = 40.0, omega_h = 200.0, epsilon = 0.1)
    # freq2, W_vec2, W_tf2 = WcalcFunctions.WcalculationVec2(G_diff_response, f, omega_h = 10.0, epsilon = 1.0, slope_db_per_dec=-50)
    print(f'freq (omega): {freq2}')
    print(f'W_vec (Weights in vector form): {W_vec2}')
    print(f'W_freq (Weights in TF form): {W_tf2}')

    WcalcFunctions.WeightFreqPlot(2*np.pi*f, W_vec2)

    total_db2 = WcalcFunctions.SmallGainPracticalCheck(G_diff_response, W_vec2, w = 2*np.pi*f)

    # Weight matrix adjustment 2
    freq3, W_vec3, W_tf3 = WcalcFunctions.WcalculationVec3(G_diff_response, f, omega_b = 40.0, epsilon = 0.1)
    print(f'freq (omega): {freq3}')
    print(f'W_vec (Weights in vector form): {W_vec3}')
    print(f'W_freq (Weights in TF form): {W_tf3}')

    WcalcFunctions.WeightFreqPlot(2*np.pi*f, W_vec3)

    total_db3 = WcalcFunctions.SmallGainPracticalCheck(G_diff_response, W_vec3, w = 2*np.pi*f)
