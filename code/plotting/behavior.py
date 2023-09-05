""" Plotting functions of behavior

Partly copied from: https://allenswdb.github.io/visual-behavior/VB-BehaviorSessionData.html
"""
import numpy as np

def plot_running(ax, behavior_session, initial_time, final_time):
    '''
    a simple function to plot running speed between two specified times on a specified axis
    inputs:
        ax: axis on which to plot
        intial_time: initial time to plot from
        final_time: final time to plot to
    '''
    running_sample = behavior_session.running_speed.copy()
    running_sample = running_sample[(running_sample.timestamps >= initial_time) & 
                                    (running_sample.timestamps <= final_time)] 
    ax.plot(running_sample['timestamps'],
            running_sample['speed'])

def plot_licks(ax, behavior_session, initial_time, final_time):
    '''
    a simple function to plot licks as dots between two specified times on a specified axis
    inputs:
        ax: axis on which to plot
        intial_time: initial time to plot from
        final_time: final time to plot to
    '''
    licking_sample = behavior_session.licks.copy()
    licking_sample = licking_sample[(licking_sample.timestamps >= initial_time) & 
                                    (licking_sample.timestamps <= final_time)]     
    ax.plot(licking_sample['timestamps'], np.zeros_like(licking_sample['timestamps']),
            marker = 'o', color = 'black', linestyle = 'none')
    
def plot_rewards(ax, behavior_session, initial_time, final_time):
    '''
    a simple function to plot rewards between two specified times as blue diamonds on a specified axis
    inputs:
        ax: axis on which to plot
        intial_time: initial time to plot from
        final_time: final time to plot to
    '''
    rewards_sample = behavior_session.rewards.copy()
    rewards_sample = rewards_sample[(rewards_sample.timestamps >= initial_time) & 
                                    (rewards_sample.timestamps <= final_time)]      
    ax.plot(rewards_sample['timestamps'], np.zeros_like(rewards_sample['timestamps']),
            marker = 'd', color = 'red', linestyle = 'none', markersize = 12, alpha = 0.9)
    
def plot_stimuli(ax, behavior_session, initial_time, final_time):
    '''
    a simple function to plot stimuli as colored vertical spans on a s
    inputs:
        ax: axis on which to plot
        intial_time: initial time to plot from
        final_time: final time to plot to
    '''
    stimulus_presentations_sample = behavior_session.stimulus_presentations.copy()
    stimulus_presentations_sample = stimulus_presentations_sample[(stimulus_presentations_sample.end_time >= initial_time) & 
                                    (stimulus_presentations_sample.start_time <= final_time)] 
    
    clr = {'omitted':'k', 'im104_r':'C0', 'im114_r':'C1', 'im111_r':'C2', 'im024_r':'C3', 'im034_r':'C4',
       'im087_r':'C5', 'im005_r':'C6', 'im083_r':'C7', np.nan:'k'}
    for idx, stimulus in stimulus_presentations_sample.iterrows():
        ax.axvspan(stimulus['start_time'], stimulus['end_time'], color=clr[stimulus.image_name], alpha=0.4)
        
        
def plot_behavior_all(ax, behavior_session, initial_time, final_time):
    """ """
    plot_running(ax, behavior_session, initial_time, final_time)
    plot_licks(ax, behavior_session, initial_time, final_time)
    plot_rewards(ax, behavior_session, initial_time, final_time)
    plot_stimuli(ax, behavior_session, initial_time, final_time)