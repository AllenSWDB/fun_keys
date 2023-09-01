""" Functions to work with behavior neuropixel data
"""
import pandas as pd
from tqdm import tqdm
import xarray as xr
import numpy as np

import os

def get_unit_counts_per_session(cache,
                         select_order = ['VISp', 'VISal', 'VISam', 'VISrl',
                                         'VISl', 'VISpm', 'LP', 'LGd', 'LGv', 'CA1',
                                         'DG', ]
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
    if name in ['LGd', 'LGv']:       return 'LGN'
    if name in ['VISp']:             return 'VISp'
    if name in ['VISl', 'VISal']:    return 'VIS_lat'
    if name in ['VISam', 'VISpm']:   return 'VIS_med'
    if name in ['VISrl']:            return 'PTLp'
    if name in ['LP']:               return 'LP'
    if name in ['CA1', 'CA3', 'DG']: return 'HP'
    return 'other'

def simplify_names_to_int( name ):
    """ Assign custom integer of brain areas into a simpler classification
    
    Usage example:
    units['area_int'] = units.structure_acronym.apply( simplify_names_to_int )
    
    """
    if name in ['LGd', 'LGv']:       return 0
    if name in ['VISp']:             return 1
    if name in ['VISl', 'VISal']:    return 2
    if name in ['VISam', 'VISpm']:   return 3
    if name in ['VISrl']:            return 4
    if name in ['LP']:               return 5
    if name in ['CA1', 'CA3', 'DG']: return 6
    return 7

def easy_layer(layer):
    """ Simplify layer assignment """
    if layer in ['2/3', '4']:
        return 'sup'
    if layer in ['5', '6a', '6b']:
        return 'deep'
    else:
        return layer # nan
    
    
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
    units.index = units['unit_id']
    
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

    data_xr = xr.DataArray(spike_train.astype(np.int16),
                           dims=("unit_id", "time"),
                           coords={"unit_id": unit_ids, "time": spike_time})
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
    units['area_int'] = units.structure_acronym.apply(  simplify_names_to_int )
    
    # get xarray with spike matrix
    data_xr = get_spike_matrix_10ms(units, session)
    
    units['FS']=False
    units['FS'][units['waveform_duration']<.4]=True
    
    if with_layer == False:
        
        units_reduced = units[['structure_acronym', 'easy_name', 'area_int', 
                               'firing_rate', 'peak_channel_id',
                               'probe_vertical_position',
                               'anterior_posterior_ccf_coordinate',
                               'dorsal_ventral_ccf_coordinate',
                               'left_right_ccf_coordinate','FS' ]]
        return data_xr, units_reduced
    
    # otherwise add layer information
    units = add_cortical_layer_to_units( units )
    units['easy_layer'] = units.cortical_layer.apply( easy_layer )
    
    units_reduced = units[['structure_acronym', 'easy_name', 'area_int', 
                           'easy_layer', 'cortical_layer',
                           'firing_rate', 'peak_channel_id',
                           'probe_vertical_position',
                           'anterior_posterior_ccf_coordinate',
                           'dorsal_ventral_ccf_coordinate',
                           'left_right_ccf_coordinate','FS',]]
    return data_xr, units_reduced
    
    
    
    
def get_stim_xarray(trial_df, data_xr, image_int, active, start_dt=0, end_dt=0.5):
    """ Return slices from the large data_xr at specific conditions and image numbers
    
    Returns xarray (nr_stims, nr_units, time) for given condition
    
    Usage:
    from data import load_data, ecephys
    trial_df = load_data.get_trial_df(session)
    data_xr, units = ecephys.easy_spike_matrix_and_unit_table(cache, session)
    image_int = 3    # can be between 0 and 7
    active = True    # can be True or False
    start_dt = -0.1  # will be added to 0, negative values for stim before
    end_dt = 0.5     # end time (in seconds)
    
    stim_xr = get_stim_xarray(trial_df, data_xr, image_int, active, start_dt, end_dt)
    """
    
    sel_trial = trial_df[ (trial_df.active==active) & (trial_df.image_int==image_int) ]
    sel_data = list()
    sel_stim_id = list()

    for ind, start_t in sel_trial.start_time.items():
        sel_stim_id.append(ind)
        startInd = np.searchsorted(data_xr.time, start_t+start_dt)
        endInd = np.searchsorted(data_xr.time, start_t+end_dt)  
        sel_data.append( data_xr[:,startInd:endInd] )
        
    stim_arr = np.stack( [ar.data for ar in sel_data] )
    time_part = sel_data[0].time.data - sel_data[0].time.data[0] + start_dt

    stim_xr = xr.DataArray(stim_arr.astype(np.int16),
                           dims=("stim_id", "unit_id", "time"),
                           coords={"stim_id": sel_stim_id, 
                                   "unit_id": data_xr.unit_id, "time": time_part}
                          )
    
    # add meta-information
    stim_xr.attrs['active'] = active
    stim_xr.attrs['image_int'] = image_int
    
    period = 'Active' if active else 'Passive'
    stim_xr.name = period + ', Image {}'.format(image_int)
    
    return stim_xr


def get_all_stim_xarrays(trial_df, data_xr, start_dt=0, end_dt=0.5):
    """ Returns a list with stim_xr for active/passive and all 8 images
    
    Element 0 in list: active, img 0
    Element 1 in list: active, img 1
    Element 8 in list: passive, img 0
    
    from data import load_data, ecephys
    trial_df = load_data.get_trial_df(session)
    data_xr, units = ecephys.easy_spike_matrix_and_unit_table(cache, session)
    start_dt = 0     # will be added to 0, negative values for times before stim onset
    end_dt = 0.5     # end time (in seconds)
    
    all_stim_xrs = get_all_stim_xarrays(trial_df, data_xr, start_dt, end_dt)
    """
    
    all_stim_xrs = list()
    for active in [True, False]:
        for image_int in range(8):
            stim_xr = get_stim_xarray(trial_df, data_xr, image_int,
                                              active, start_dt=start_dt, end_dt=end_dt)
            all_stim_xrs.append( stim_xr )
        
    return all_stim_xrs


def get_presentation_xarray(presentation_table, data_xr, start_dt=0, end_dt=0.5,
                            only_active_passive=True):
    """ Return one data_xr with all presentations (all image presentations)
    
    Usage:
    from data import load_data, ecephys
    presentation_table = session.stimulus_presentations
    data_xr, units = ecephys.easy_spike_matrix_and_unit_table(cache, session)
    start_dt = 0      # will be added to 0, negative values for stim before
    end_dt = 0.75     # end time (in seconds)
    
    pres_xr = get_presentation_xarray(presentation_table, data_xr, start_dt, end_dt)
    """
    
    if only_active_passive:
        pt = presentation_table
        presentation_table = pt.loc[(pt.stimulus_block==0)|(pt.stimulus_block==5)]
        
    sel_trial = presentation_table
    sel_data = list()
    sel_stim_id = list()

    for ind, start_t in sel_trial.start_time.items():
        sel_stim_id.append(ind)
        startInd = np.searchsorted(data_xr.time, start_t+start_dt)
        endInd = np.searchsorted(data_xr.time, start_t+end_dt)  
        sel_data.append( data_xr[:,startInd:endInd] )
        
    stim_arr = np.stack( [ar.data for ar in sel_data] )
    time_part = sel_data[0].time.data - sel_data[0].time.data[0] + start_dt

    pres_xr = xr.DataArray(stim_arr.astype(np.int16), dims=("stim_id", "unit_id", "time"),
                           coords={"stim_id": sel_stim_id, 
                                   "unit_id": data_xr.unit_id, "time": time_part}
                          )
    
    # add meta-information
    pres_xr.name = 'All stim presentations'
    
    return pres_xr


def find_FS_units(unit_table):
    unit_table['FS']=False
    unit_table['FS'][unit_table['waveform_duration']<.4]=True
    
    return unit_table