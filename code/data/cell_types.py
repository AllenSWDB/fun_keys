""" Functions to identify and label cell types in behavior neuropixel data
    Only works for certain sessions
"""
import pandas as pd
from tqdm import tqdm
import xarray as xr
import numpy as np
import scipy as sp
import statsmodels.stats.multitest as smStats
import os
import matplotlib.pyplot as plt



def find_FS_units(unit_table):
    unit_table['FS']=False
    unit_table['FS'][unit_table['waveform_duration']<.4]=True
    
    return unit_table

def makePSTH(spikes, startTimes, windowDur, binSize=0.001):
    '''
    Convenience function to compute a peri-stimulus-time histogram
    (see section 7.2.2 here: https://neuronaldynamics.epfl.ch/online/Ch7.S2.html)
    INPUTS:
        spikes: spike times in seconds for one unit
        startTimes: trial start times in seconds; the first spike count 
            bin will be aligned to these times
        windowDur: trial duration in seconds
        binSize: size of spike count bins in seconds
    OUTPUTS:
        Tuple of (PSTH, bins), where:
            PSTH gives the trial-averaged spike rate for 
                each time bin aligned to the start times;
            bins are the bin edges as defined by numpy histogram
    '''
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros(bins.size-1)
    for start in startTimes:
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]
    
    counts = counts/len(startTimes)
    return counts/binSize, bins[:-1]


def make_neuron_time_trials_array(units, spike_times, stim_table, 
                                   time_before, trial_duration,
                                   bin_size=0.001):
    '''
    Function to make a 3D array with dimensions [neurons, time bins, trials] to store
    the spike counts for stimulus presentation trials. 
    INPUTS:
        units: dataframe with unit info (same form as session.units table)
        stim_table: dataframe whose indices are trial ids and containing a
            'start_time' column indicating when each trial began
        time_before: seconds to take before each start_time in the stim_table
        trial_duration: total time in seconds to take for each trial
        bin_size: bin_size in seconds used to bin spike counts 
    OUTPUTS:
        unit_array: 3D array storing spike counts. The value in [i,j,k] 
            is the spike count for neuron i at time bin j in the kth trial.
        time_vector: vector storing the trial timestamps for the time bins
    '''
    # Get dimensions of output array
    neuron_number = len(units)
    trial_number = len(stim_table)
    num_time_bins = int(trial_duration/bin_size)
    
    # Initialize array
    unit_array = np.zeros((neuron_number, num_time_bins, trial_number))
    
    # Loop through units and trials and store spike counts for every time bin
    for u_counter, (iu, unit) in enumerate(units.iterrows()):
        
        # grab spike times for this unit
        unit_spike_times = spike_times[iu]
        
        # now loop through trials and make a PSTH for this unit for every trial
        for t_counter, (it, trial) in enumerate(stim_table.iterrows()):
            trial_start = trial.start_time - time_before
            unit_array[u_counter, :, t_counter] = makePSTH(unit_spike_times, 
                                                            [trial_start], 
                                                            trial_duration, 
                                                            binSize=bin_size)[0]
    
    # Make the time vector that will label the time axis
    time_vector = np.arange(num_time_bins)*bin_size - time_before
    
    return unit_array, time_vector

def find_SST_cells(session_id,session):
    'assumes that the cache is already loaded for this sessions'
    if session_id == 1108334384 or session_id== 1115356973:
        print('SST session - will find SST cells')
        
        units =     units = session.get_units(amplitude_cutoff_maximum = 0.1, 
                                              presence_ratio_minimum = 0.9,
                                              isi_violations_maximum = 0.5)
        channels = session.get_channels()
        units = units.merge(channels, left_on='peak_channel_id', right_index=True)
        print('Got units table and merged with channels table')
        
        spike_times = session.spike_times
        spike_times
        opto_table = session.optotagging_table
        print('Got opto table')
        #Make 3D array
        time_before_laser = 0.5
        trial_duration = 1.5
        bin_size = 0.001
        time_array = np.arange(-time_before_laser,trial_duration-time_before_laser,bin_size)

        #opto_array has spike counts for each opto trial
        opto_array,time = make_neuron_time_trials_array(units, spike_times,opto_table, 
                                                        time_before_laser, trial_duration, bin_size)

        #Grab the short pulse + high power trials
        duration = opto_table.duration.min() #choose the short duration
        level = opto_table.level.max() #choose the high power

        #Find the indicies of trials with this duration and level
        sel_trials=((opto_table['duration']==duration)&(opto_table['level']==level)).values

        #Average over these selected trials from the opto array
        mean_opto_responses=np.nanmean(opto_array[:,:,sel_trials],2)

        optoRespByTrial = opto_array[:,:,sel_trials]
        optoRespByTrial.shape #nUnit x nTimepoints x nSelectedTrials

        # slice our data array to take the baseline period (before the laser turns on)
        baseline_time_idx = (time_array>=-0.010)&(time_array<-0.002)

        # then average over this time window in the mean_opto_responses to get the baseline rate for each unit
        baseline_rate = np.mean(optoRespByTrial[:,baseline_time_idx,:],1)
        #nTrials nUnits
        baseline_rate=baseline_rate.T

        # do the same for the period when the laser was on to get the evoked rate for each unit, 
        #between 1ms and 9ms after the onset of the laser
        evoked_rate_idx = (time_array>=0.001)&(time_array<0.009)
        evoked_rate = np.mean(optoRespByTrial[:,evoked_rate_idx,:],1)
        evoked_rate=evoked_rate.T

        # do the wilcoxen test for each unit
        optoT=[]
        optoP=[]
        print('running wilcoxen test')
        # I now have the a nUnits array of baseline rates and an nUnits array of evoked rates for the selected stim paramters
        for iCell in range(evoked_rate.shape[1]):
            thisBsln = baseline_rate[:,iCell]
            thisEvoked = evoked_rate[:,iCell]   
            try:
                this_optoT,this_optoP = sp.stats.wilcoxon(thisBsln,thisEvoked,alternative='less',zero_method='pratt')
                optoT.append(this_optoT)
                optoP.append(this_optoP)
            except(ValueError):
                optoT.append(0)
                optoP.append(1)
        #correct for repeat testing
        optoH,optoP_corrected,x1,x2=smStats.multipletests(optoP, alpha=0.05, method='hs')
        SST_opto_responses = mean_opto_responses[optoH==True]
        units['SST']=optoH

        return units

    else:
        print('Not applicable to this session')