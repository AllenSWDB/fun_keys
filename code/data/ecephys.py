""" Functions to work with behavior neuropixel data
"""
import pandas as pd
from tqdm import tqdm
import xarray as xr
import numpy as np

def get_unit_counts_per_session(cache,
                         select_order = ['VISp', 'VISal', 'VISam', 'VISrl',
                                         'VISl', 'LP', 'LGd', 'LGv', 'CA1',
                                         'DG', 'VISl']
                               ):
    """ Returns a pandas dataframe with simultaneously recorded units
        in the specified areas
    """
    # get all recorded units
    sessions = cache.get_ecephys_session_table()
    all_units = cache.get_unit_table()
    
    # count simultaneously recorded neurons in each session
    session_ids = sessions.index.values
    count_series = list()

    for i, ses_id in enumerate(session_ids):
        ses_units = all_units[ all_units['ecephys_session_id'] == ses_id ]
        neuron_counts = ses_units.groupby('structure_acronym').structure_acronym.count()
        neuron_counts.name = ses_id
        count_series.append( neuron_counts.to_frame().T )

    sim_units = pd.concat(count_series)
    
    ordered_brain_areas = sim_units[select_order]
    
    return ordered_brain_areas


def simplify_names( name ):
    """ Group brain areas into a simpler classification
    
    Usage example:
    units['easy_name'] = units.structure_acronym.apply( simplify_names )
    
    """
    if name in ['VISp']:
        return '1__V1'
    if name in ['VISal', 'VISam', 'VISl', 'VISpm', 'VISrl']:
        return '2__V2'
    if name in ['CA1', 'CA3', 'DG']:
        return '4__HP'
    if name in ['LP']:
        return '3__TH_2'
    if name in ['LGd', 'LGv']:
        return '0__TH_1'
    
    return '5__other'


def add_cortical_layer_to_units( units ):
    """ This function requires the vbn_supplemental_tables to be mapped
    to the data folder of the CodeOcean capsule!
    
    Usage example:
    units = session.get_units(
        amplitude_cutoff_maximum = 0.1, 
        presence_ratio_minimum = 0.9,
        isi_violations_maximum = 0.5
        )
    units = add_cortical_layer_to_units( units )
    """
    
    supp_data_file = '/data/vbn_supplemental_tables/supplemental_tables/master_unit_table.csv'
    if not os.path.exists(supp_data_file):
        raise Exception('Check if vbn_supplemental_tables is mapped to the capsule!')
    
    df_supp = pd.read_csv(supp_data_file)
    df_part = df_supp[['unit_id','cortical_layer']]
    
    # add the layer cortical layer information to the table
    units = units.merge(right=df_part, left_on='id', right_on='unit_id')
    return units


def get_spike_matrix_10ms(units, session):
    """ Returns spike matrix (nr_units, nr_timesteps) as xarray for whole recording
    
    units: unit table with unit_ids
    session: session table
    """
    
    max_index = 10000000
    nr_neurons = len(units)
    unit_ids = units.index.values

    # spikes sampled at 100Hz to save memory
    spike_train = np.zeros( (nr_neurons, int(max_index/10) ) )
    spike_time = np.arange( spike_train.shape[1] ) / 100

    spikes_all = session.spike_times

    for n, unit_id in tqdm(enumerate(unit_ids)):
        # spikes at 1kHz to have only one spike per bin
        spikes_1kHz = np.zeros( max_index )
        spikes = spikes_all[unit_id]
        spike_index = (spikes * 1000).astype(int)
        spikes_1kHz[spike_index] = 1

        # add spikes up in 10ms window
        spike_train[n,:] = np.add.reduceat(spikes_1kHz, np.arange(0,max_index,10))

    data_xr = xr.DataArray(spike_train, dims=("unit_id", "time"), coords={"unit_id": unit_ids, "time": spike_time})
    return data_xr
    
    
    
def easy_spike_matrix_and_unit_table(cache, session, with_layer=False):
    """ Wrapper function to get spike_matrix (10ms sampling) and simple unit table
    
    Usage:
    from data.load_data import *
    from data import ecephys
    cache = load_cache_behavior_neuropixel()
    session_id = 1139846596
    session = cache.get_ecephys_session(session_id)
    data_xr, units = ecephys.easy_spike_matrix_and_unit_table(cache, session)
    
    """
    
    # load units
    units = session.get_units(
        amplitude_cutoff_maximum = 0.1, 
        presence_ratio_minimum = 0.9,
        isi_violations_maximum = 0.5
    )

    # merge to channel data to match units to brain regions
    channels = cache.get_channel_table()
    units = units.merge(channels, left_on='peak_channel_id', right_index=True)
    
    # add easy name
    units['easy_name'] = units.structure_acronym.apply( simplify_names )
    
    # get xarray with spike matrix
    data_xr = get_spike_matrix_10ms(units, session)
    
    if with_layer == False:
        
        units_reduced = units[['structure_acronym', 'easy_name', 'firing_rate', 'peak_channel_id',
                       'probe_vertical_position', 'anterior_posterior_ccf_coordinate',
                       'dorsal_ventral_ccf_coordinate', 'left_right_ccf_coordinate', ]]
        return data_xr, units_reduced
    
    # otherwise add layer information
    units = add_cortical_layer_to_units( units )
    
    units_reduced = units[['structure_acronym', 'easy_name', 'cortical_layer',
                           'firing_rate', 'peak_channel_id',
               'probe_vertical_position', 'anterior_posterior_ccf_coordinate',
               'dorsal_ventral_ccf_coordinate', 'left_right_ccf_coordinate', ]]
    return data_xr, units_reduced
    