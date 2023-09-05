from data.load_data import load_cache_behavior_neuropixel, get_trial_df
import numpy as np
import matplotlib.pyplot as plt # used for plotting data
from numpy import linalg as LA
from numpy.linalg import matrix_rank
from scipy.stats import pearsonr

#FUNCTIONS 

#get the spike data
def get_spike_data(cache, session, my_area, amplitude_cutoff_maximum = 0.1, presence_ratio_minimum = 0.9, isi_violations_maximum = 0.5):

    # get all units for this session; apply quality metrics [spike sorting]
    units = session.get_units(
        amplitude_cutoff_maximum = amplitude_cutoff_maximum, 
        presence_ratio_minimum = presence_ratio_minimum,
        isi_violations_maximum = isi_violations_maximum
    )

    # merge to channel data to match units to brain regions
    channels = cache.get_channel_table()
    units_channels = units.merge(channels, left_on='peak_channel_id', right_index=True)

    # Filter by region
    my_units = units_channels.loc[units_channels.structure_acronym == my_area]
    
    return my_units

#extracting the firing rates
def get_norm_firing_rates_per_stim_ID(my_units, trial_df):
    
    num_selected_trials = 4 #this only works for a trial_df in which we select only the last four trials before the image changes
    
    num_trials = trial_df.shape[0]
    num_units = len(VISp_units.index.values)
    firing_rates = np.zeros((num_units,num_trials))
    idx_image_int = np.zeros(num_trials)
    idx_active = np.zeros(num_trials)
    idx_image_order = np.zeros(num_trials)
    
    for i in range(0,num_units):
        for j in range(0,num_trials):

            spike_times = session.spike_times[VISp_units.index.values[i]]
            sum_spikes_tmp = len(spike_times[(spike_times > trial_df.iloc[j].start_time) & (spike_times < trial_df.iloc[j].end_time)])
            firing_rates[i,j] = sum_spikes_tmp/(trial_df.iloc[j].end_time-trial_df.iloc[j].start_time)
    
    for j in range(0,num_trials):
        idx_image_int[j] = trial_df.iloc[j].image_int
        idx_active[j] = trial_df.iloc[j].active    
        idx_image_order[j] = j%num_selected_trials
   
    max_rates = firing_rates.max(axis=1)
    norm_firing_rates = firing_rates/max_rates[:,None] 
    
    return firing_rates, norm_firing_rates, idx_image_int, idx_active, idx_image_order

#grouping together same stimuli
def grouping_stimuli(my_units, norm_firing_rates, trial_df, idx_active, idx_image_int, idx_image_order, active_mode):
    
    images = np.unique(trial_df.image_int.values)
    num_images = len(images)
    num_trials = trial_df.shape[0]
    num_units = len(my_units.index.values)  
    max_order = 4 #this works only for the dataframe in which we select only the last four presentations before the stimulus changes
    
    mean_norm_firing_rates_per_image_int = np.zeros((num_units,num_images))
    
    for i in range(0,num_units):
        for j in range(0,num_images):        
            mean_norm_firing_rates_per_image_int[i,j] = np.mean(norm_firing_rates[i,(idx_image_int == j) & (idx_active == active_mode)])
       
    
    mean_norm_firing_rates_per_image_int_and_order = np.zeros((num_units,num_images,max_order))
    
    for i in range(0,num_units):
        for j in range(0,num_images):   
            for k in range(0,max_order):
                mean_norm_firing_rates_per_image_int_and_order[i,j,k] = np.mean(norm_firing_rates[i,(idx_image_int == j) & (idx_active == active_mode) & (idx_image_order == k)])
       
    
    return mean_norm_firing_rates_per_image_int, mean_norm_firing_rates_per_image_int_and_order

#get NOISE correlations
#for signal correlations we just need to run np.corrcoef(mean_norm_firing_rates_per_image_int) or np.corrcoef(mean_norm_firing_rates_per_image_int_and_order[:,:,i]) if we want to divide them by order
#for total correlations instead, we just need to compute np.corrcoef(norm_firing_rates) (Not total sure wheter it makes completely sense, given that we're chunking time)

def get_noise_correlations(trial_df, my_units, norm_firing_rates, idx_image_int, idx_active, idx_image_order, active_mode):
    
    images = np.unique(trial_df.image_int.values)
    num_images = len(images)
    num_trials = trial_df.shape[0]
    num_units = len(my_units.index.values)  
    max_order = 4 #this works only for the dataframe in which we select only the last four presentations before the stimulus changes
    
    noise_correlations_per_image = np.zeros((num_units,num_units,num_images)) #initialize our array of correlation matrices for each stim
    
    for i in range(0,num_images):
            noise_correlations_per_image[:,:,i] = np.corrcoef(norm_firing_rates[:,(idx_image_int == i) & (idx_active == active_mode)])

    mean_noise_correlations = np.nanmean(noise_correlations_per_image,axis=2)
    
    noise_correlations_with_image_order = np.zeros((num_units,num_units,num_images, max_order)) #initialize our array of correlation matrices for each stim

    for i in range(0,num_images):
        for j in range(0,max_order):
            noise_correlations_with_image_order[:,:,i,j] = np.corrcoef(norm_firing_rates[:,(idx_image_int == i) & (idx_active == active_mode) & (idx_image_order == j)])
            
    mean_noise_correlations_per_image_order = np.nanmean(noise_correlations_with_image_order,axis=2)
    
    return noise_correlations_per_image, noise_correlations_with_image_order, mean_noise_correlations, mean_noise_correlations_per_image_order

def get_signal_correlations_per_image_order(my_units, mean_norm_firing_rates_per_image_int_and_order):
        
    num_units = len(my_units.index.values)  
    max_order = 4 #this works only for the dataframe in which we select only the last four presentations before the stimulus changes
    signal_corr_per_image_order = np.zeros((num_units, num_units, max_order))
        
    for i in range(0, max_order):
        signal_corr_per_image_order[:,:,i] = np.corrcoef(mean_norm_firing_rates_per_image_int_and_order[:,:,i])
            
    return signal_corr_per_image_order
   
    
#loading cache and selecting the session
cache = load_cache_behavior_neuropixel()
sessions_to_iterate = cache.get_ecephys_session_table().index.values[:10]

mean_NC_ACTIVE = np.zeros(len(session_to_iterate))
mean_SC_ACTIVE = np.zeros(len(session_to_iterate))
mean_NC_PASSIVE = np.zeros(len(session_to_iterate))
mean_SC_PASSIVE= np.zeros(len(session_to_iterate))

pearson_ACTIVE = np.zeros(len(session_to_iterate))
pearson_PASSIVE = np.zeros(len(session_to_iterate))
p_value_ACTIVE = np.zeros(len(session_to_iterate))
p_value_PASSIVE = np.zeros(len(session_to_iterate))


for l in range(0,len(session_to_iterate)):
    
    session_id = session_to_iterate[l]
    session = cache.get_ecephys_session(session_id)
    trial_df = get_trial_df(session)
    
    #get the units
    my_area = 'VISp'
    amplitude_cutoff_maximum = 0.1
    presence_ratio_minimum = 0.9
    isi_violations_maximum = 0.5
    VISp_units = get_spike_data(cache, session, my_area, amplitude_cutoff_maximum, presence_ratio_minimum, isi_violations_maximum)

    #get normalized firing rates (each firing rate is normalized, with respect to its maximum)
    firing_rates, norm_firing_rates, idx_image_int, idx_active, idx_image_order = get_norm_firing_rates_per_stim_ID(VISp_units, trial_df)

    #select only the same stimuli with their presentation order

    #ACTIVE

    mean_norm_firing_rates_per_image_int_ACTIVE, mean_norm_firing_rates_per_image_int_and_order_ACTIVE = grouping_stimuli(VISp_units, norm_firing_rates, trial_df, idx_active, idx_image_int, idx_image_order, True)

    #PASSIVE

    mean_norm_firing_rates_per_image_int_PASSIVE, mean_norm_firing_rates_per_image_int_and_order_PASSIVE = grouping_stimuli(VISp_units, norm_firing_rates, trial_df, idx_active, idx_image_int, idx_image_order, False)
    
    #NOISE correlations

    #ACTIVE

    noise_correlations_per_image_ACTIVE, noise_correlations_with_image_order_ACTIVE, mean_noise_correlations_ACTIVE, mean_noise_correlations_per_image_order_ACTIVE = get_noise_correlations(trial_df, VISp_units, norm_firing_rates, idx_image_int, idx_active, idx_image_order, True)

    #PASSIVE

    noise_correlations_per_image_PASSIVE, noise_correlations_with_image_order_PASSIVE, mean_noise_correlations_PASSIVE, mean_noise_correlations_per_image_order_PASSIVE = get_noise_correlations(trial_df, VISp_units, norm_firing_rates, idx_image_int, idx_active, idx_image_order, False)
    
    #SIGNAL correlations
#for signal correlations we just need to run np.corrcoef(mean_norm_firing_rates_per_image_int) or np.corrcoef(mean_norm_firing_rates_per_image_int_and_order[:,:,i]) if we want to divide them by order

    #ACTIVE 

    signal_correlations_ACTIVE = np.corrcoef(mean_norm_firing_rates_per_image_int_ACTIVE)
    signal_correlations_per_image_order_ACTIVE = get_signal_correlations_per_image_order(VISp_units, mean_norm_firing_rates_per_image_int_and_order_ACTIVE)

    #PASSIVE 

    signal_correlations_PASSIVE = np.corrcoef(mean_norm_firing_rates_per_image_int_PASSIVE)
    signal_correlations_per_image_order_PASSIVE = get_signal_correlations_per_image_order(VISp_units, mean_norm_firing_rates_per_image_int_and_order_PASSIVE)

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
    
    mean_NC_ACTIVE[l] = np.nanmean(ncorr_ACTIVE)
    mean_SC_ACTIVE[l] = np.nanmean(scorr_ACTIVE)
    mean_NC_PASSIVE[l] = np.nanmean(ncorr_PASSIVE)
    mean_SC_PASSIVE[l] = np.nanmean(scorr_PASSIVE)

    print(mean_NC_ACTIVE[l])
    print(mean_NC_PASSIVE[l])
    print(mean_SC_ACTIVE[l])
    print(mean_SC_PASSIVE[l])
    
    #compute pearson corr
    pearson_ACTIVE[l] = pearsonr(ncorr_ACTIVE[np.isnan(ncorr_ACTIVE)==False],scorr_ACTIVE[np.isnan(ncorr_ACTIVE)==False])[0]
    pearson_PASSIVE[l] = pearsonr(ncorr_PASSIVE[np.isnan(ncorr_PASSIVE)==False],scorr_PASSIVE[np.isnan(ncorr_PASSIVE)==False])[0]
    
    p_value_ACTIVE[l] = pearsonr(ncorr_ACTIVE[np.isnan(ncorr_ACTIVE)==False],scorr_ACTIVE[np.isnan(ncorr_ACTIVE)==False])[1]
    p_value_PASSIVE[l] = pearsonr(ncorr_ACTIVE[np.isnan(ncorr_ACTIVE)==False],scorr_ACTIVE[np.isnan(ncorr_ACTIVE)==False])[1]

    print(pearson_ACTIVE[l])
    print(pearson_PASSIVE[l])
    
    print(p_value_ACTIVE[l])
    print(p_value_PASSIVE[l])
    

   

