from data.load_data import load_cache_behavior_neuropixel, get_trial_df
import numpy as np
from numpy import linalg as LA
import my_functions_shared_variability as mf

#loading cache and selecting the session
cache = load_cache_behavior_neuropixel()

number_of_sessions = 10 #NUMBER OF DIFFERENT SESSIONS 

sessions_to_iterate = cache.get_ecephys_session_table().index.values[:number_of_sessions]

#Brain region and quality matrix
#my_area = 'VISp'
#my_area = 'VISl'

#my_areas = ['VISl', 'VISal', 'VISam', 'VISpm']
my_areas = ['CA1', 'CA3','DG']

amplitude_cutoff_maximum = 0.1
presence_ratio_minimum = 0.9
isi_violations_maximum = 0.5

num_images = 8
num_pairs = int(num_images*(num_images-1)/2)

all_angles_signal_noise_first_ACTIVE = np.zeros((number_of_sessions,num_pairs*num_images)) #angles between signal axes and noise correlations axes
all_angles_signal_noise_first_PASSIVE = np.zeros((number_of_sessions,num_pairs*num_images))
all_angles_signal_noise_second_ACTIVE = np.zeros((number_of_sessions,num_pairs*num_images))
all_angles_signal_noise_second_PASSIVE = np.zeros((number_of_sessions,num_pairs*num_images))

angles_signal_axes_ACT_PASS = np.zeros((number_of_sessions, num_pairs)) #angles between signal axes across conditions
angles_first_eigen_NC_ACT_PASS = np.zeros((number_of_sessions, num_images)) #angles between first eigenvectors of the noise correlations across conditions
angles_second_eigen_NC_ACT_PASS = np.zeros((number_of_sessions, num_images)) #same as before, but with the second eigenvectors

#loop over sessions
for l in range(0,number_of_sessions):
    
    print(l)
    
    session_id = sessions_to_iterate[l]
    session = cache.get_ecephys_session(session_id)
    trial_df = get_trial_df(session)
    
    
    ###
    #select go-hit trials or go-miss trials
    trials = session.trials
    trial_df['stim_id'] = trial_df.index.values

    #merge with trials features
    m_df = trial_df.merge(right=trials, on='trials_id')
    m_df.index = m_df.stim_id
    m_df = m_df.sort_index()

    #build the vector to get the type of trial per each trial

    idx_hit_mode = np.zeros(trial_df.shape[0])
    idx_miss_mode = np.zeros(trial_df.shape[0])

    for i in range(0,trial_df.shape[0]):
        idx_hit_mode[i] = m_df.iloc[i].hit
        idx_miss_mode[i] = m_df.iloc[i].miss
    ###
    
    
    #VISp_units = mf.get_spike_data(cache, session, my_area, amplitude_cutoff_maximum, presence_ratio_minimum, isi_violations_maximum)
    
    VISp_units = mf.get_spike_data_multiple_areas(cache, session, my_areas, amplitude_cutoff_maximum, presence_ratio_minimum, isi_violations_maximum)


    #get firing rates ( and their normalized versions, where each firing rate is normalized with respect to its maximum)
    
    firing_rates, norm_firing_rates, idx_image_int, idx_active, idx_image_order = mf.get_norm_firing_rates_per_stim_ID(VISp_units, session, trial_df)
    
    #select only the same stimuli with their presentation order

    #ACTIVE

    mean_firing_rates_per_image_int_ACTIVE, mean_firing_rates_per_image_int_and_order_ACTIVE = mf.grouping_stimuli(VISp_units, firing_rates, trial_df, idx_active, idx_image_int, idx_image_order, True, True, False, idx_hit_mode, idx_miss_mode)

    #PASSIVE

    mean_firing_rates_per_image_int_PASSIVE, mean_firing_rates_per_image_int_and_order_PASSIVE = mf.grouping_stimuli(VISp_units, firing_rates, trial_df, idx_active, idx_image_int, idx_image_order, False, True, False, idx_hit_mode, idx_miss_mode)
    

    #build the signal axes

    num_units = len(VISp_units.index.values)

    signal_axes_ACTIVE = np.zeros((num_units, num_pairs))
    signal_axes_PASSIVE = np.zeros((num_units, num_pairs))

    idx_pair = 0
    for i in range(0,num_images):
        for j in range(i+1,num_images):

            signal_axes_ACTIVE[:,idx_pair] = mean_firing_rates_per_image_int_ACTIVE[:,i] - mean_firing_rates_per_image_int_ACTIVE[:,j]
            signal_axes_PASSIVE[:,idx_pair] = mean_firing_rates_per_image_int_PASSIVE[:,i] - mean_firing_rates_per_image_int_PASSIVE[:,j]

            idx_pair = idx_pair +1
            
    
    #extract eigenvectors from noise correlations

    #first, compute NOISE correlations

    #ACTIVE - hit trials only

    noise_correlations_per_image_ACTIVE, noise_correlations_with_image_order_ACTIVE, mean_noise_correlations_ACTIVE, mean_noise_correlations_per_image_order_ACTIVE = mf.get_noise_correlations(trial_df, VISp_units, firing_rates, idx_image_int, idx_active, idx_image_order, True, True, False, idx_hit_mode, idx_miss_mode)

    #PASSIVE - hit trials only

    noise_correlations_per_image_PASSIVE, noise_correlations_with_image_order_PASSIVE, mean_noise_correlations_PASSIVE, mean_noise_correlations_per_image_order_PASSIVE = mf.get_noise_correlations(trial_df, VISp_units, firing_rates, idx_image_int, idx_active, idx_image_order, False, True, False, idx_hit_mode, idx_miss_mode)

    #Eigenvectors per each image

    eigenvectors_matrices_NC_A = np.zeros((num_units,num_units,num_images))
    eigenvectors_matrices_NC_P = np.zeros((num_units,num_units,num_images))

    eigenvalues_matrix_NC_A = np.zeros((num_units,num_images))
    eigenvalues_matrix_NC_P = np.zeros((num_units,num_images))

    for i in range(0,num_images):
        
        #active
        eigenvalues_matrix_NC_A_tmp, eigenvectors_matrices_NC_A_tmp = LA.eig(np.nan_to_num(noise_correlations_per_image_ACTIVE[:,:,i], nan = 0))
        
        eigenvalues_matrix_NC_A[:,i] = eigenvalues_matrix_NC_A_tmp
        eigenvectors_matrices_NC_A[:,:,i] = eigenvectors_matrices_NC_A_tmp
    
        #passive
        eigenvalues_matrix_NC_P_tmp, eigenvectors_matrices_NC_P_tmp = LA.eig(np.nan_to_num(noise_correlations_per_image_PASSIVE[:,:,i], nan = 0))
        
        eigenvalues_matrix_NC_P[:,i] = eigenvalues_matrix_NC_P_tmp
        eigenvectors_matrices_NC_P[:,:,i] = eigenvectors_matrices_NC_P_tmp
        
    
    #angles between signal axes and noise correlations (1st vector)

    idx_angle = 0
    for i in range(0,num_pairs):
        for j in range(0,num_images):
            all_angles_signal_noise_first_ACTIVE[l,idx_angle] = mf.angle_between_vectors(signal_axes_ACTIVE[:,i], eigenvectors_matrices_NC_A[:,0,j])
            all_angles_signal_noise_first_PASSIVE[l,idx_angle] = mf.angle_between_vectors(signal_axes_PASSIVE[:,i], eigenvectors_matrices_NC_P[:,0,j])

            idx_angle = idx_angle + 1
    
    #same, but with the second eigenvectors   
    
    idx_angle = 0
    for i in range(0,num_pairs):
        for j in range(0,num_images):
            all_angles_signal_noise_second_ACTIVE[l,idx_angle] = mf.angle_between_vectors(signal_axes_ACTIVE[:,i], eigenvectors_matrices_NC_A[:,1,j])
            all_angles_signal_noise_second_PASSIVE[l,idx_angle] = mf.angle_between_vectors(signal_axes_PASSIVE[:,i], eigenvectors_matrices_NC_P[:,1,j])

            idx_angle = idx_angle + 1
    
    
    #angles between signal axes in active and passive case
    
    angles_signal_axes_ACT_PASS[l,:] = mf.compute_angles_between_matrices_row_vectors(signal_axes_ACTIVE.T, signal_axes_PASSIVE.T)
    
    #angles between FIRST eigenvectors in active and passive case
    
    angles_first_eigen_NC_ACT_PASS[l,:] = mf.compute_angles_between_matrices_row_vectors(eigenvectors_matrices_NC_A[:,0,:].T, eigenvectors_matrices_NC_P[:,0,:].T)

    #angles between SECOND eigenvectors in active and passive case
    
    angles_second_eigen_NC_ACT_PASS[l,:] = mf.compute_angles_between_matrices_row_vectors(eigenvectors_matrices_NC_A[:,1,:].T, eigenvectors_matrices_NC_P[:,1,:].T)


np.savetxt("angles_between_signal_and_FIRST_noise_ACTIVE_HIPP.txt",all_angles_signal_noise_first_ACTIVE, delimiter=' ', comments='')
np.savetxt("angles_between_signal_and_SECOND_noise_ACTIVE_HIPP.txt",all_angles_signal_noise_second_ACTIVE, delimiter=' ', comments='')

np.savetxt("angles_between_signal_and_FIRST_noise_PASSIVE_HIPP.txt",all_angles_signal_noise_first_PASSIVE, delimiter=' ', comments='')
np.savetxt("angles_between_signal_and_SECOND_noise_PASSIVE_HIPP.txt",all_angles_signal_noise_second_PASSIVE, delimiter=' ', comments='')

np.savetxt("angles_between_signal_axes_HIPP.txt",angles_signal_axes_ACT_PASS, delimiter=' ', comments='')
np.savetxt("angles_between_FIRST_noise_axes_HIPP.txt",angles_first_eigen_NC_ACT_PASS, delimiter=' ', comments='')
np.savetxt("angles_between_SECOND_noise_axes_HIPP.txt",angles_second_eigen_NC_ACT_PASS, delimiter=' ', comments='')




    