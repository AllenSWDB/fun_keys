import numpy as np
from numpy import linalg as LA

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
def get_norm_firing_rates_per_stim_ID(my_units, session, trial_df):
    
    num_selected_trials = 4 #this only works for a trial_df in which we select only the last four trials before the image changes
    
    num_trials = trial_df.shape[0]
    num_units = len(my_units.index.values)
    firing_rates = np.zeros((num_units,num_trials))
    idx_image_int = np.zeros(num_trials)
    idx_active = np.zeros(num_trials)
    idx_image_order = np.zeros(num_trials)
    
    for i in range(0,num_units):
        for j in range(0,num_trials):

            spike_times = session.spike_times[my_units.index.values[i]]
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
def grouping_stimuli(my_units, norm_firing_rates, trial_df, idx_active, idx_image_int, idx_image_order, active_mode, hit_mode, miss_mode, idx_hit_mode, idx_miss_mode):
    
    images = np.unique(trial_df.image_int.values)
    num_images = len(images)
    num_trials = trial_df.shape[0]
    num_units = len(my_units.index.values)  
    max_order = 4 #this works only for the dataframe in which we select only the last four presentations before the stimulus changes
    
    mean_norm_firing_rates_per_image_int = np.zeros((num_units,num_images))
    
    for i in range(0,num_units):
        for j in range(0,num_images):        
            mean_norm_firing_rates_per_image_int[i,j] = np.mean(norm_firing_rates[i,(idx_image_int == j) & (idx_active == active_mode) & (idx_hit_mode == hit_mode) & (idx_miss_mode == miss_mode)])
       
    
    mean_norm_firing_rates_per_image_int_and_order = np.zeros((num_units,num_images,max_order))
    
    for i in range(0,num_units):
        for j in range(0,num_images):   
            for k in range(0,max_order):
                mean_norm_firing_rates_per_image_int_and_order[i,j,k] = np.mean(norm_firing_rates[i,(idx_image_int == j) & (idx_active == active_mode) & (idx_image_order == k) & (idx_hit_mode == hit_mode) & (idx_miss_mode == miss_mode)])
       
    
    return mean_norm_firing_rates_per_image_int, mean_norm_firing_rates_per_image_int_and_order

#get NOISE correlations
#for signal correlations we just need to run np.corrcoef(mean_norm_firing_rates_per_image_int) or np.corrcoef(mean_norm_firing_rates_per_image_int_and_order[:,:,i]) if we want to divide them by order
#for total correlations instead, we just need to compute np.corrcoef(norm_firing_rates) (Not total sure wheter it makes completely sense, given that we're chunking time)

def get_noise_correlations(trial_df, my_units, norm_firing_rates, idx_image_int, idx_active, idx_image_order, active_mode, hit_mode, miss_mode, idx_hit_mode, idx_miss_mode):
    
    images = np.unique(trial_df.image_int.values)
    num_images = len(images)
    num_trials = trial_df.shape[0]
    num_units = len(my_units.index.values)  
    max_order = 4 #this works only for the dataframe in which we select only the last four presentations before the stimulus changes
    
    noise_correlations_per_image = np.zeros((num_units,num_units,num_images)) #initialize our array of correlation matrices for each stim
    
    for i in range(0,num_images):
            noise_correlations_per_image[:,:,i] = np.corrcoef(norm_firing_rates[:,(idx_image_int == i) & (idx_active == active_mode) & (idx_hit_mode == hit_mode) & (idx_miss_mode == miss_mode)])

    mean_noise_correlations = np.nanmean(noise_correlations_per_image,axis=2)
    
    noise_correlations_with_image_order = np.zeros((num_units,num_units,num_images, max_order)) #initialize our array of correlation matrices for each stim

    for i in range(0,num_images):
        for j in range(0,max_order):
            noise_correlations_with_image_order[:,:,i,j] = np.corrcoef(norm_firing_rates[:,(idx_image_int == i) & (idx_active == active_mode) & (idx_image_order == j) & (idx_hit_mode == hit_mode) & (idx_miss_mode == miss_mode)])
            
    mean_noise_correlations_per_image_order = np.nanmean(noise_correlations_with_image_order,axis=2)
    
    return noise_correlations_per_image, noise_correlations_with_image_order, mean_noise_correlations, mean_noise_correlations_per_image_order

def get_signal_correlations_per_image_order(my_units, mean_norm_firing_rates_per_image_int_and_order):
        
    num_units = len(my_units.index.values)  
    max_order = 4 #this works only for the dataframe in which we select only the last four presentations before the stimulus changes
    signal_corr_per_image_order = np.zeros((num_units, num_units, max_order))
        
    for i in range(0, max_order):
        signal_corr_per_image_order[:,:,i] = np.corrcoef(mean_norm_firing_rates_per_image_int_and_order[:,:,i])
            
    return signal_corr_per_image_order

def angle_between_vectors(vector1, vector2):
    # Compute the dot product between the two vectors
    dot_product = np.dot(vector1, vector2)

    # Compute the magnitudes (norms) of the vectors
    norm1 = LA.norm(vector1)
    norm2 = LA.norm(vector2)

    # Calculate the cosine of the angle between the vectors
    cosine_theta = dot_product / (norm1 * norm2)

    # Use arccosine to get the angle in radians
    angle_rad = np.arccos(np.clip(cosine_theta, -1.0, 1.0))

    # Convert the angle from radians to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def compute_angles_between_matrices_row_vectors(matrix1, matrix2):
    """
    Compute the angles between corresponding vectors in two matrices.

    Parameters:
    matrix1 (numpy.ndarray): The first matrix where each row is a vector.
    matrix2 (numpy.ndarray): The second matrix where each row is a vector.
    
    Returns:
    numpy.ndarray: An array of angles (in degrees) between corresponding vectors.
    """
    # Normalize the vectors in both matrices (optional but recommended)
    matrix1_normalized = matrix1 / np.linalg.norm(matrix1, axis=1)[:, np.newaxis]
    matrix2_normalized = matrix2 / np.linalg.norm(matrix2, axis=1)[:, np.newaxis]

    # Compute the dot product between corresponding vectors
    dot_products = np.sum(matrix1_normalized * matrix2_normalized, axis=1)

    # Calculate the angles using the arccosine function
    angles_rad = np.arccos(np.clip(dot_products, -1.0, 1.0))
    
    # Convert angles to degrees
    angles_deg = np.degrees(angles_rad)

    return angles_deg
 
def calculate_angles_between_column_vectors(matrix):
    """
    Calculate the angles between all possible pairs of vectors in a matrix.

    Parameters:
    matrix (numpy.ndarray): The input matrix where each column is a vector.

    Returns:
    numpy.ndarray: A symmetric matrix where each element (i, j) contains the angle (in degrees)
                  between vector i and vector j.
    """
    # Normalize the vectors (optional, but recommended for angle calculations)
    matrix_normalized = matrix / np.linalg.norm(matrix, axis=0)

    # Get the number of vectors (columns)
    num_vectors = matrix_normalized.shape[1]

    # Initialize an array to store the angles
    angles = np.zeros((num_vectors, num_vectors))

    # Compute the angles between all possible pairs of vectors
    for i in range(num_vectors):
        for j in range(i, num_vectors):
            if i == j:
                angles[i, j] = 0.0  # Angle between a vector and itself is 0
            else:
                # Calculate the dot product between vectors i and j
                dot_product = np.dot(matrix_normalized[:, i], matrix_normalized[:, j])

                # Calculate the angle using the arccosine function
                angle_rad = np.arccos(dot_product)

                # Convert the angle to degrees
                angle_deg = np.degrees(angle_rad)

                # Store the angle in the array (symmetric matrix)
                angles[i, j] = angle_deg
                angles[j, i] = angle_deg  # Angle matrix is symmetric

    return angles

