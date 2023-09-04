import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def get_spikes(cache, session, trial_df, region='VISp', post_stim_dur=0):
    
    spike_times = session.spike_times  # get spike times dict for all units recorded

    # get unit metadata for this session + apply quality metrics
    # PARAMS
    all_units = session.get_units(
        amplitude_cutoff_maximum = 0.1, 
        presence_ratio_minimum = 0.9,
        isi_violations_maximum = 0.5)

    # merge to channel data to match units to brain regions
    channels = cache.get_channel_table()
    unit_channels = all_units.merge(channels, left_on='peak_channel_id', right_index=True)

    # to filter by region
    if region is not None:
        units_df = unit_channels.loc[unit_channels.structure_acronym==region] #PARAMS
    else:
        units_df = unit_channels

    # get spike counts within each stim pres window
    spike_mat = np.zeros([len(trial_df), len(units_df)])

    for i,unit in enumerate(units_df.index): #for each neuron 1:N...
        spikes = spike_times[unit]
        counts = [] #initialize column vector

        for start,end in zip(trial_df.start_time, trial_df.end_time): #for each stim presentation...
            startInd = np.searchsorted(spikes, start)
            endInd = np.searchsorted(spikes, end+post_stim_dur)
            rel_spike_times = spikes[startInd:endInd]-start #relative spike times in this window
            count = len(rel_spike_times)
            counts.append(count) #append spike counts for this stim pres

        spike_mat[:,i] = counts #add column vector of FRs for this neuron to spike matrix

    counts_df = pd.DataFrame(
        data = spike_mat,
        index = trial_df.index,
        columns = units_df.index
    )
    counts_df['abs_trial_id'] = trial_df.abs_trial_id
    
    return counts_df



def get_dm(counts_df, state_stims, agg_over_tr=True):
    ''' get max-normalized design matrix
    '''
    if agg_over_tr:
        FRs = counts_df.loc[state_stims].groupby('abs_trial_id').agg('mean')
        FRs_normed = FRs / FRs.max()
    else:
        FRs_normed = counts_df.loc[state_stims] / counts_df.loc[state_stims].max()
    
    return FRs_normed



def do_pca(dm):
    # input should be features X samples
    cov_mat = np.cov(dm)
    evals, evecs = np.linalg.eig(cov_mat)

    return evals, evecs



def participation_ratio(evals):
    return (np.sum(evals)**2) / np.sum(evals**2)



def get_pca_dict(dm, inds, trial_df):
    
    evals, evecs = do_pca(dm.T)
    pr = participation_ratio(evals)

    # mean-center the data + project into PC space
    cent = dm - dm.mean()
    project = np.dot(np.transpose(evecs[:,0:3]), cent.T)

    # stim analysis
    int_labels = trial_df.loc[inds].groupby(['abs_trial_id', 'image_int']).agg('max').index.get_level_values(1).to_numpy()
    img_labels = trial_df.loc[inds].groupby(['abs_trial_id', 'image_name']).agg('max').index.get_level_values(1).to_numpy()
    ss_stim = silhouette_score(X=dm,labels=img_labels)

    # time analysis
    trial_labels = trial_df.loc[inds].groupby('abs_trial_id').agg('max').index.to_numpy()
    trial_splits = np.array_split(trial_labels, 3) # split time into 1st, 2nd, 3rd chunks for classification
    splits_labels = []
    for split in range(3):
        splits_labels += ([str(split)] * len(trial_splits[split]))
    ss_time = silhouette_score(X=dm,labels=splits_labels)
    
    pca_dict = {
            'evals' : evals,
            'evecs' : evecs,
            'PR' : pr,
            'SS stim' : ss_stim,
            'SS time' : ss_time,
            'project' : project,
            'stim labels' : int_labels,
            'time labels' : trial_labels}
    
    return pca_dict



def create_session_dict(trial_df, counts_df):
    session_dict = {}
    
    # full PCA
    f_inds = counts_df.index
    f_dm = get_dm(counts_df, f_inds)
    f_pca_dict = get_pca_dict(f_dm, f_inds, trial_df)
    session_dict['full'] = f_pca_dict
    
    state_labels = trial_df.loc[f_inds].groupby(['abs_trial_id', 'image_int']).agg('max').state.to_numpy()
    ss_state = silhouette_score(X=f_dm,labels=state_labels)
    session_dict['full']['state labels'] = [int(s) for s in state_labels]
    session_dict['full']['ss state'] = ss_state
    
    # state PCAs
    states = trial_df.state.unique()
    for state in states:
        state_inds = trial_df.loc[trial_df.state==state].index
        state_dm = get_dm(counts_df, state_inds)
        state_pca_dict = get_pca_dict(state_dm, state_inds, trial_df)
        session_dict[state] = state_pca_dict
    
    session_dict['trial df'] = trial_df
    
    return session_dict



def plot_state_pcas(session_dict):

    states = session_dict['trial df'].state.unique()
    
    f1,ax1 = plt.subplots(1,len(states), figsize=(6,3.2))
    f2,ax2 = plt.subplots(1,len(states), figsize=(6,3))

    for state in states:

        s = session_dict[state]

        ax1[int(state)].scatter(s['project'][0], s['project'][1], c=s['stim labels'])
        ax1[int(state)].set(xlabel='PC1',ylabel='PC2',title='state '+state)

        ax2[int(state)].scatter(s['project'][0], s['project'][1], c=s['time labels'])
        ax2[int(state)].set(xlabel='PC1',ylabel='PC2',title='state '+state)

    f1.suptitle('colored by stimulus')
    f1.tight_layout()
    f2.suptitle('colored by time')
    f2.tight_layout()



def plot_full_pca(session_dict):

    s = session_dict['full']
    f,ax = plt.subplots(1,3, figsize=(8,3.2))

    ax[0].scatter(s['project'][0], s['project'][1], c=s['stim labels'])
    ax[0].set(xlabel='PC1',ylabel='PC2',title='colored by stim')

    ax[1].scatter(s['project'][0], s['project'][1], c=s['time labels'])
    ax[1].set(xlabel='PC1',ylabel='PC2',title='colored by trial')

    ax[2].scatter(s['project'][0], s['project'][1], c=s['state labels'])
    ax[2].set(xlabel='PC1',ylabel='PC2',title='colored by state')

    f.tight_layout()