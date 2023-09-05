""" Helper functions to group / cluster units """

import scipy.cluster.hierarchy as sch
import numpy as np

def cluster_by_area( input_xr, units, sort_by_area=True):
    
    sorted_ids = list()
    d = input_xr
    
    # exclude neurons without any response
    d = d[ np.std(d, axis=1) > 0 ]

    # remove entries with only zeros
    area_int = units.loc[ d.unit_id ].area_int

    if sort_by_area:
        for area_i in range(7):
            unit_ids = d.unit_id[ area_int.values == area_i ]
            part = d.loc[ unit_ids ]

            pairwise_distances = sch.distance.pdist(part, metric='correlation')
            linkage = sch.linkage(pairwise_distances, method='ward')
            idx_to_cluster_array = sch.fcluster(linkage, t=0, criterion='distance')
            idx = np.argsort(idx_to_cluster_array)
            sorted_ids.extend( unit_ids[idx].values )
    else:
        unit_ids = d.unit_id      # all ids, just for consistency with above
        part = d.loc[ unit_ids ]

        pairwise_distances = sch.distance.pdist(part, metric='correlation')
        linkage = sch.linkage(pairwise_distances, method='ward')
        idx_to_cluster_array = sch.fcluster(linkage, t=0, criterion='distance')
        idx = np.argsort(idx_to_cluster_array)
        sorted_ids.extend( unit_ids[idx].values )
    
    return sorted_ids


    
def calc_mean_per_area(cc_mat, area_int, nr_areas=7,
                      absolute=False):
    """ """
    if absolute:
        cc_mat = np.abs( cc_mat )
        
    cor_by_area = np.zeros((nr_areas,nr_areas))
    
    for i in range(nr_areas):
        for j in range(nr_areas):
            sel = cc_mat[area_int==i,:][:,area_int==j]
            if i == j:
                # remove diagonal
                sel = sel[ ~np.eye(sel.shape[0],dtype=bool) ]

            cor_by_area[i,j] = np.nanmean( sel )

    return cor_by_area
