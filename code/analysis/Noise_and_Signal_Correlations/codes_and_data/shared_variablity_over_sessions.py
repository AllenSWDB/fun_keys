from data.load_data import load_cache_behavior_neuropixel, get_trial_df
import numpy as np
import matplotlib.pyplot as plt # used for plotting data
from numpy import linalg as LA
from numpy.linalg import matrix_rank
from scipy.stats import pearsonr
import my_functions_shared_variability as mf


#loading cache and selecting the session
cache = load_cache_behavior_neuropixel()

number_of_sessions = 60
sessions_to_iterate = cache.get_ecephys_session_table().index.values[:number_of_sessions]

#sessions_to_iterate = [1053925378, 1081090969, 1064415305, 1108334384, 1115356973]

#Brain region and quality matrix
my_area = 'VISl'
#my_areas = ['VISl', 'VISal', 'VISam', 'VISpm']

amplitude_cutoff_maximum = 0.1
presence_ratio_minimum = 0.9
isi_violations_maximum = 0.5

mean_NC_ACTIVE = np.zeros(number_of_sessions)
mean_SC_ACTIVE = np.zeros(number_of_sessions)
mean_NC_PASSIVE = np.zeros(number_of_sessions)
mean_SC_PASSIVE= np.zeros(number_of_sessions)

is_orthogonal_matrix = np.zeros(number_of_sessions)
determinant_rot_matrix = np.zeros(number_of_sessions)

pearson_ACTIVE = np.zeros(number_of_sessions)
pearson_PASSIVE = np.zeros(number_of_sessions)
p_value_ACTIVE = np.zeros(number_of_sessions)
p_value_PASSIVE = np.zeros(number_of_sessions)

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
    
    VISp_units = mf.get_spike_data(cache, session, my_area, amplitude_cutoff_maximum, presence_ratio_minimum, isi_violations_maximum)
    
    #VISp_units = mf.get_spike_data_multiple_areas(cache, session, my_areas, amplitude_cutoff_maximum, presence_ratio_minimum, isi_violations_maximum)

    #get firing rates ( and their normalized versions, where each firing rate is normalized with respect to its maximum)
    
    firing_rates, norm_firing_rates, idx_image_int, idx_active, idx_image_order = mf.get_norm_firing_rates_per_stim_ID(VISp_units, session, trial_df)

    #select only the same stimuli with their presentation order

    #ACTIVE

    mean_firing_rates_per_image_int_ACTIVE, mean_firing_rates_per_image_int_and_order_ACTIVE = mf.grouping_stimuli(VISp_units, firing_rates, trial_df, idx_active, idx_image_int, idx_image_order, True, True, False, idx_hit_mode, idx_miss_mode)

    #PASSIVE

    mean_firing_rates_per_image_int_PASSIVE, mean_firing_rates_per_image_int_and_order_PASSIVE = mf.grouping_stimuli(VISp_units, firing_rates, trial_df, idx_active, idx_image_int, idx_image_order, False, True, False, idx_hit_mode, idx_miss_mode)
    
    #NOISE correlations

    #ACTIVE

    noise_correlations_per_image_ACTIVE, noise_correlations_with_image_order_ACTIVE, mean_noise_correlations_ACTIVE, mean_noise_correlations_per_image_order_ACTIVE = mf.get_noise_correlations(trial_df, VISp_units, firing_rates, idx_image_int, idx_active, idx_image_order, True, True, False, idx_hit_mode, idx_miss_mode)

    #PASSIVE

    noise_correlations_per_image_PASSIVE, noise_correlations_with_image_order_PASSIVE, mean_noise_correlations_PASSIVE, mean_noise_correlations_per_image_order_PASSIVE = mf.get_noise_correlations(trial_df, VISp_units, firing_rates, idx_image_int, idx_active, idx_image_order, False, True, False, idx_hit_mode, idx_miss_mode)
    
    #SIGNAL correlations
#for signal correlations we just need to run np.corrcoef(mean_norm_firing_rates_per_image_int) or np.corrcoef(mean_norm_firing_rates_per_image_int_and_order[:,:,i]) if we want to divide them by order

    #ACTIVE 

    signal_correlations_ACTIVE = np.corrcoef(mean_firing_rates_per_image_int_ACTIVE)
    
    #signal_correlations_per_image_order_ACTIVE = mf.get_signal_correlations_per_image_order(VISp_units, mean_firing_rates_per_image_int_and_order_ACTIVE)

    #PASSIVE 

    signal_correlations_PASSIVE = np.corrcoef(mean_firing_rates_per_image_int_PASSIVE)
    
    #signal_correlations_per_image_order_PASSIVE = mf.get_signal_correlations_per_image_order(VISp_units, mean_firing_rates_per_image_int_and_order_PASSIVE)

    #SIGNAL and NOISE correlations --> comparison

    ncorr_ACTIVE = np.matrix.flatten(mean_noise_correlations_ACTIVE)
    indices_ACTIVE = np.where(ncorr_ACTIVE > 0.99) #remove the ones along the diagonal
    ncorr_ACTIVE = np.delete(ncorr_ACTIVE, indices_ACTIVE)

    ncorr_PASSIVE = np.matrix.flatten(mean_noise_correlations_PASSIVE)
    indices_PASSIVE = np.where(ncorr_PASSIVE > 0.99) #remove the ones along the diagonal
    ncorr_PASSIVE = np.delete(ncorr_PASSIVE, indices_PASSIVE)

    scorr_ACTIVE = np.matrix.flatten(signal_correlations_ACTIVE)
    scorr_ACTIVE = np.delete(scorr_ACTIVE,indices_ACTIVE)

    scorr_PASSIVE = np.matrix.flatten(signal_correlations_PASSIVE)
    scorr_PASSIVE = np.delete(scorr_PASSIVE,indices_PASSIVE)
    
    #orthogonal matrix
    
    #eigenvalues_SC_A, eigenvectors_SC_A = LA.eig(np.nan_to_num(signal_correlations_ACTIVE, nan=0))
    eigenvalues_NC_A, eigenvectors_NC_A = LA.eig(np.nan_to_num(mean_noise_correlations_ACTIVE, nan = 0))

    #eigenvalues_SC_P, eigenvectors_SC_P = LA.eig(np.nan_to_num(signal_correlations_PASSIVE, nan = 0))
    eigenvalues_NC_P, eigenvectors_NC_P = LA.eig(np.nan_to_num(mean_noise_correlations_PASSIVE, nan = 0))
    
    rot_matrix_tmp = np.dot(eigenvectors_NC_A, eigenvectors_NC_P.T)
    is_orthogonal_matrix[l] = np.allclose(np.dot(rot_matrix_tmp, rot_matrix_tmp.T), np.identity(rot_matrix_tmp.shape[0]))
    
    determinant_rot_matrix[l] = LA.det(rot_matrix_tmp)
    
    #saving data
    
    mean_NC_ACTIVE[l] = np.nanmean(ncorr_ACTIVE)
    mean_SC_ACTIVE[l] = np.nanmean(scorr_ACTIVE)
    mean_NC_PASSIVE[l] = np.nanmean(ncorr_PASSIVE)
    mean_SC_PASSIVE[l] = np.nanmean(scorr_PASSIVE)

    #compute pearson corr
    pearson_ACTIVE[l] = pearsonr(ncorr_ACTIVE[np.isnan(ncorr_ACTIVE)==False],scorr_ACTIVE[np.isnan(ncorr_ACTIVE)==False])[0]
    pearson_PASSIVE[l] = pearsonr(ncorr_PASSIVE[np.isnan(ncorr_PASSIVE)==False],scorr_PASSIVE[np.isnan(ncorr_PASSIVE)==False])[0]
    
    p_value_ACTIVE[l] = pearsonr(ncorr_ACTIVE[np.isnan(ncorr_ACTIVE)==False],scorr_ACTIVE[np.isnan(ncorr_ACTIVE)==False])[1]
    p_value_PASSIVE[l] = pearsonr(ncorr_ACTIVE[np.isnan(ncorr_ACTIVE)==False],scorr_ACTIVE[np.isnan(ncorr_ACTIVE)==False])[1]
                                
    print("pearson - ACTIVE:")
    print(pearson_ACTIVE[l])
    print("pearson - PASSIVE:")
    print(pearson_PASSIVE[l])
    
    print("p values:")
    print(p_value_ACTIVE[l])
    print(p_value_PASSIVE[l])
    
data_to_be_saved = np.column_stack((mean_NC_ACTIVE, mean_SC_ACTIVE, mean_NC_PASSIVE, mean_SC_PASSIVE, is_orthogonal_matrix,  determinant_rot_matrix, pearson_ACTIVE, pearson_PASSIVE, p_value_ACTIVE, p_value_PASSIVE))


np.savetxt("signal_noise_corr_comparison_VISl_HIGH.txt", data_to_be_saved, delimiter=' ', comments='')

    