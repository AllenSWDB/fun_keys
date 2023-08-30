""" Helper functions to load data

FunKeys """


'''
How to use:
from data.load_data import *
cache = load_cache_behavior_neuropixel()
session_id = 1139846596
session = cache.get_ecephys_session(session_id)
trial_df = get_trial_df(session)
'''


import platform
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache
import numpy as np
import pandas as pd
import scipy as sp


def load_cache_behavior_neuropixel():
    """ Load cache object for visual behavior neuropixel dataset
    
    Returns a cache object
    """
    
    platstring = platform.platform()

    if 'Darwin' in platstring:
        # macOS 
        data_root = "/Volumes/Brain2023/"
    elif 'Windows'  in platstring:
        # Windows (replace with the drive letter of USB drive)
        data_root = "E:/"
    elif ('amzn' in platstring):
        # then on CodeOcean
        data_root = "/data/"
    else:
        # then your own linux platform
        # EDIT location where you mounted hard drive
        data_root = "/media/$USERNAME/Brain2023/"
        
    cache_dir  = data_root
    cache = VisualBehaviorNeuropixelsProjectCache.from_local_cache(cache_dir=cache_dir, use_static_cache=True)

    return cache


def get_trial_df(session):
    '''
    Function to get trials to analyze (4 before change, no omissions) for
        given session.
    
    Takes session as input
    
    Returns dataframe with all instances of 4 stims before change for active
        and passive context. Includes active (bool), trials_id (one int per set
        of 4 stims), start_time, end_time, image_int (image code), image_name
        (actual image identifier)
    '''
    
    # get all stim presentations and find change indices
    stim_df = session.stimulus_presentations
    pres_df = stim_df.loc[(stim_df.stimulus_block==0)|(stim_df.stimulus_block==5)]
    change_inds = pres_df.loc[pres_df.is_change==True].index

    # get int to represent each image (0-7 for images, -10 for omission)
    images = sorted( pres_df.image_name.unique() )
    image_to_int = dict()
    for i, image in enumerate(images):
        if image=='omitted':
            image_to_int[image] = -10
        else:
            image_to_int[image] = i
    pres_df['image_int'] = pres_df.image_name.apply( lambda img: image_to_int[img] )

    # get indices of last 4 images before change
    trials = np.zeros((len(change_inds), 4))
    for trial,ind in enumerate(change_inds):
        trials[trial,:] = np.arange(ind-4,ind)

    # remove cases where 1 of the 4 stims was removed
    trials_to_keep = []
    for trial in trials:
        if pres_df.loc[trial].omitted.sum() == 0: #if no stims omitted
            trials_to_keep.append(trial)
    # trial_inds_arr = np.vstack(trials_to_keep)
    trial_inds_vec = np.concatenate(trials_to_keep)

    # filter to get useful columns + trials of interest
    df = pres_df.loc[trial_inds_vec].filter([
        'active', 
        'trials_id',
        'start_time',
        'end_time',
        'image_int',
        'image_name'
    ])

    # df.loc[df.active==True] to get active session trials
    # df.loc[df.active==False] to get passive session trials

    return df



### WIP
def make_behavior_table(session, df):
    '''
    Input session and stim/pres df with selected trials to analyze (4 before change)
    '''
    
    # Get timestamps corresponding to go trials
    trial_start = df.start_time
    trial_stop = df.end_time

    # Get running speed and corresponding timestamps
    running_time = session.running_speed.timestamps
    running_speed = session.running_speed.speed
    mean_speed = [np.nanmean(running_speed[np.logical_and(s1 <= running_time, running_time <= s2)]) for s1, s2 in zip(trial_start, trial_stop)]

    # Get pupil size and corresponding timestamps
    pupil_time = session.eye_tracking.timestamps
    pupil_area = session.eye_tracking.pupil_area
    mean_pupil_area = [np.nanmean(pupil_area[np.logical_and(s1 <= pupil_time, pupil_time <= s2)]) for s1, s2 in zip(trial_start, trial_stop)]
    # impute missing values
    inds = np.where(np.isnan(mean_pupil_area))[0]
    for i in inds:
        mean_pupil_area[i] = np.nanmean(mean_pupil_area[i-1:i+1])

    # Get lick counts
    # lick_count = session.trials.apply(lambda row : len(row['lick_times']), axis = 1)

    # Calculate hit rate
    # hit_rate = session.trials.hit.rolling(10).mean().values
    # hit_rate[:9] = 0 #otherwise these will be nans

    # Construct a dataframe
    behavior_data = pd.DataFrame({
                'Mean speed': mean_speed, 
                'Mean pupil area': mean_pupil_area})
    
    return behavior_data


def make_behavior_table_active(session,trial_df):
    '''
    Input session and stim/pres df with selected trials to analyze.
    This finds the matching trial IDs in the trial metadata
    and extracts behavioral variables per trial - NOT per stimulus presentation.
    '''
    #get trial IDs from the trial_df (i.e. the df based on stim presentations)
    trialsWeWant=np.unique(trial_df.trials_id.values).tolist()
    
    # find the trials in the trial metadata that correspond to those in the stime-defined trial df
    trial_metadata=session.trials
    trial_metadata=trial_metadata.loc[trialsWeWant]
    trial_metadata=trial_metadata[trial_metadata.go]
    
    # Get timestamps corresponding to go trials
    trial_start = trial_metadata.start_time
    trial_stop = trial_metadata.stop_time

    # Get running speed and corresponding timestamps
    running_time = session.running_speed.timestamps
    running_speed = session.running_speed.speed
    mean_speed = [np.nanmean(running_speed[np.logical_and(s1 <= running_time, running_time <= s2)]) for s1, s2 in zip(trial_start, trial_stop)]

    # Get pupil size and corresponding timestamps
    pupil_time = session.eye_tracking.timestamps
    pupil_area = session.eye_tracking.pupil_area
    mean_pupil_area = [np.nanmean(pupil_area[np.logical_and(s1 <= pupil_time, pupil_time <= s2)]) for s1, s2 in zip(trial_start, trial_stop)]
    # impute missing values
    inds = np.where(np.isnan(mean_pupil_area))[0]
    for i in inds:
        mean_pupil_area[i] = np.nanmean(mean_pupil_area[i-1:i+1])
    Z_mean_pupil = sp.stats.zscore(mean_pupil_area)
    
    #get lick count
    lick_count = trial_metadata.apply(lambda row : len(row['lick_times']), axis = 1)
    
    #get rolling average hit rate. Right now window is set to 5 trials.
    hit_rate = trial_metadata.hit.rolling(5).mean() #we are choosing to do a rolling avrg over 5 trials
    
    #get true/false hit for each trial as a number
    hit_bool = trial_metadata['hit'].astype(int)

    # Construct a dataframe
    behavior_data = pd.DataFrame({
                'Mean speed': mean_speed, 
                'Mean pupil area': mean_pupil_area,
                'Zscored mean pupil area': Z_mean_pupil,
                'Lick count': lick_count,
                'Rolling mean hit rate': hit_rate,
                'Hit/miss this trial' : hit_bool,
    })
    
    return behavior_data