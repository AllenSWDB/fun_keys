""" Functions to work with behavior neuropixel data
"""
import pandas as pd

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
