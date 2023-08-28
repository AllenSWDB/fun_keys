""" Helper functions to load data

FunKeys """

import platform
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache


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
