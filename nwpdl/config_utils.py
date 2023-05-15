import os
import sys
import numpy as np
import types
import math
from pathlib import Path

HOME = Path(__file__).parents[1]
CONFIG_FOLDER = HOME / 'config'

from .utils import load_yaml_file

def read_config(config_filename=None, config_folder=CONFIG_FOLDER):
    
    if config_filename is None:
        config_filename = 'local_config.yaml'
        
    config_path = os.path.join(config_folder, config_filename)
    try:
        localconfig = load_yaml_file(config_path)
    except FileNotFoundError as e:
        print(e)
        print(f"You must set {config_filename} in the main folder. Copy local_config-example.yaml and adjust appropriately.")
        sys.exit(1)       
    
    return localconfig


def get_data_paths(config_folder=CONFIG_FOLDER):
    data_config_paths = os.path.join(config_folder, 'data_paths.yaml')

    try:
        all_data_paths = load_yaml_file(data_config_paths)
    except FileNotFoundError as e:
        print(e)
        print("data_paths.yaml not found. Should exist in main folder")
        sys.exit(1)

    lc = read_config()['LOCAL']
    data_paths = all_data_paths[lc['data_paths']]
    return data_paths
