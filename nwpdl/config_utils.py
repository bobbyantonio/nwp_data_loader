import os
import sys
import copy
import numpy as np
import types
import math
from pathlib import Path

HOME = Path(__file__).parents[1]
CONFIG_FOLDER = HOME / 'config'

from .utils import load_yaml_file

def read_config(config_filename: str=None, config_folder: str=CONFIG_FOLDER) -> dict:

    config_path = os.path.join(config_folder, config_filename)
    try:
        config_dict = load_yaml_file(config_path)
    except FileNotFoundError as e:
        print(e)
        print(f"You must set {config_filename} in the config folder.")
        sys.exit(1)       
    
    return config_dict


def read_data_config(config_filename: str='data_config.yaml', config_folder: str=CONFIG_FOLDER,
                     data_config_dict: dict=None) -> dict:
    if data_config_dict is None:
        
        data_config_dict = read_config(config_filename=config_filename, config_folder=config_folder)
    
    data_config_ns = types.SimpleNamespace(**data_config_dict)
    data_config_ns.load_constants = data_config_dict.get('load_constants', True)
    data_config_ns.input_channels = len(data_config_ns.input_fields)
    data_config_ns.class_bin_boundaries = data_config_dict.get('class_bin_boundaries')
    
    if data_config_ns.class_bin_boundaries is not None:
        if 0 not in data_config_ns.class_bin_boundaries:
            data_config_ns.class_bin_boundaries = [0] + data_config_ns.class_bin_boundaries
            
        if len(data_config_ns.class_bin_boundaries) != data_config_ns.num_classes:
            raise ValueError('Class bin boundary lenght not consistent with number of classes')
    
    # For backwards compatability
    if isinstance(data_config_ns.constant_fields, int):
        data_config_ns.constant_fields = ['orography', 'lsm']
    
    data_config_ns.normalise_inputs = data_config_dict.get('normalise_inputs', False)
    
    if data_config_dict.get('normalise_outputs') is False:
            data_config_ns.output_normalisation = None
    else:
        data_config_ns.output_normalisation = data_config_dict.get('output_normalisation', "log")
        
    # Infer input image dimensions if not given
    latitude_range, longitude_range = get_lat_lon_range_from_config(data_config=data_config_ns)
    data_config_ns.input_image_height = len(latitude_range)
    data_config_ns.input_image_width = len(longitude_range)

    return data_config_ns

def get_data_paths(config_folder: str=CONFIG_FOLDER, data_config: types.SimpleNamespace=None):
    
    if data_config is None:
        data_config = read_data_config(config_folder=config_folder)
        
    if isinstance(data_config, dict):
        data_config = types.SimpleNamespace(**copy.deepcopy(data_config))
        
    data_paths = data_config.paths[data_config.data_paths]
    return data_paths

        
def get_lat_lon_range_from_config(data_config=None):
    
    if data_config is None:
        data_config = read_data_config()
    
    min_latitude = data_config.min_latitude
    max_latitude = data_config.max_latitude
    latitude_step_size = data_config.latitude_step_size
    min_longitude = data_config.min_longitude
    max_longitude = data_config.max_longitude
    longitude_step_size = data_config.longitude_step_size
    
    latitude_range=np.arange(min_latitude, max_latitude, latitude_step_size)
    longitude_range=np.arange(min_longitude, max_longitude, longitude_step_size)
    
    return latitude_range, longitude_range
