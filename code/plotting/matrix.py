""" Plotting of matrices """

import matplotlib.pyplot as plt
import numpy as np


def custom_area_cmap():
    """ """
    cmap = plt.cm.tab10  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(7)]

    # create the new map
    import matplotlib as mpl
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'tab7', cmaplist, 7)
    return cmap


def plot_cor_with_areas(cc_mat, data_xr, units, scale=0.3):
    """ """
    cmap = custom_area_cmap()
    
    f, (ax0, ax1) = plt.subplots(2, 2,
             gridspec_kw={'width_ratios': [10, 1],
                          'height_ratios': [1, 10]},
             figsize=(12,12))  

    area_int = units.loc[ data_xr.unit_id ].area_int
    ax0[0].imshow( area_int.values[None,:], aspect='auto', cmap=cmap)
    ax1[1].imshow( area_int.values[:,None], aspect='auto', cmap=cmap)

    ax1[0].imshow(cc_mat, vmin=-scale, vmax=scale,
                  cmap='seismic', aspect='auto')
    ax1[0].set_xlabel('Neurons')
    ax1[0].set_ylabel('Neurons')
    
    ax0[1].axis('off')
    ax0[0].axis('off')
    ax1[1].axis('off')
    
    
    
def plot_cor_with_clusters(cc_mat, clusters, scale=0.3):
    """ """
    cmap = 'gist_rainbow'
    
    f, (ax0, ax1) = plt.subplots(2, 2,
             gridspec_kw={'width_ratios': [10, 1],
                          'height_ratios': [1, 10]},
             figsize=(12,12))  

    
    ax0[0].imshow( clusters[None,:], aspect='auto', cmap=cmap)
    ax1[1].imshow( clusters[:,None], aspect='auto', cmap=cmap)

    ax1[0].imshow(cc_mat, vmin=-scale, vmax=scale,
                  cmap='seismic', aspect='auto')

    ax0[1].axis('off')
    ax0[0].axis('off')
    ax1[1].axis('off')
    

def plot_two_mat_with_areas(mat_active, mat_passive, scale=0.1,
                           title_1='Active', title_2='Passive'):
    """ """
    
    plt.figure(figsize=(8,5))
    plt.subplot(1,2,1)
    plt.imshow(mat_active, vmin=0, vmax=scale, cmap='viridis')
    plt.gca().grid(False)
    
    area_order = ['LGN', 'VISp', 'VIS_lat', 'VIS_med', 'PTLp', 'LP', 'HP']
    _ = plt.xticks( np.arange(7), area_order, rotation=45)
    _ = plt.yticks( np.arange(7), area_order)
    plt.title(title_1)

    plt.subplot(1,2,2)
    plt.imshow(mat_passive, vmin=0, vmax=scale, cmap='viridis')
    plt.title(title_2)
    plt.gca().grid(False)
    # plt.colorbar()

    _ = plt.xticks( np.arange(7), area_order, rotation=45)
    _ = plt.yticks( np.arange(7), ['']*7)    