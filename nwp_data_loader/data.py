""" Functions for handling data loading and saving. """
from genericpath import isfile
import os
import re
import pickle
import logging
import netCDF4
from calendar import monthrange
from tqdm import tqdm
from glob import glob
from typing import Iterable

from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from argparse import ArgumentParser

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

from dsrnngan.utils import read_config

DATA_PATHS = read_config.get_data_paths()

# Raw paths for data that has not been regridded or cut down to correct lat/lon range

IMERG_PATH = DATA_PATHS["GENERAL"].get("IMERG", '')
NIMROD_PATH = DATA_PATHS["GENERAL"].get("NIMROD", '')
ERA5_PATH = DATA_PATHS["GENERAL"].get("ERA5", '')
IFS_PATH = DATA_PATHS["GENERAL"].get("IFS", '')
OROGRAPHY_PATH = DATA_PATHS["GENERAL"].get("OROGRAPHY")
LSM_PATH = DATA_PATHS["GENERAL"].get("LSM")
CONSTANTS_PATH = DATA_PATHS["GENERAL"].get("CONSTANTS")

FIELD_TO_HEADER_LOOKUP_IFS = {'tp': 'sfc',
                              'cp': 'sfc',
                              'sp': 'sfc',
                              'tisr': 'sfc',
                              'cape': 'sfc',
                              'tclw': 'sfc',
                              'tcwv': 'sfc',
                              'u700': 'winds',
                              'v700': 'winds',
                              #   'cdir': 'missing',
                              #   'tcrw': 'missing'
                              }

# 'cin', Left out for the moment as contains a lot of nulls
all_ifs_fields = ['2t', 'cape',  'cp', 'r200', 'r700', 'r950', 
                  'sp', 't200', 't700', 'tclw', 'tcwv', 'tisr', 'tp', 
                  'u200', 'u700', 'v200', 'v700', 'w200', 'w500', 'w700']

all_fcst_hours = np.array(range(24))

# TODO: change this to behave like the IFS data load (to allow other vals of v, u etc)
# TODO: move to a config
VAR_LOOKUP_ERA5 = {'tp': {'folder': 'total_precipitation', 'suffix': 'day',
                               'negative_vals': False},
                        'q': {'folder': 'shum', 'subfolder': 'shum700', 'suffix': 'day_1deg',
                              'normalisation': 'standardise', 'negative_vals': False},
                        't': {'folder': 'ta', 'subfolder': 'tas700', 'suffix': 'day_1deg',
                              'normalisation': 'minmax', 'negative_vals': False},
                        'u': {'folder': 'u', 'subfolder': 'u700', 'suffix': 'day_1deg',
                              'normalisation': 'minmax'},
                        'v': {'folder': 'v', 'subfolder': 'v700', 'suffix': 'day_1deg',
                              'normalisation': 'minmax'}}

IFS_NORMALISATION_STRATEGY = {'tp': {'negative_vals': False, 'normalisation': 'log'}, 
                              'cp': {'negative_vals': False, 'normalisation': 'log'},
                  'pr': {'negative_vals': False, 'normalisation': 'log'}, 
                  'prl': {'negative_vals': False, 'normalisation': 'log'},
                  'prc': {'negative_vals': False, 'normalisation': 'log'},
                  'sp': {'normalisation': 'minmax'},
                  'u': {'normalisation': 'max'},
                  'v': {'normalisation': 'max'},
                  'w': {'normalisation': 'max'},
                  'r': {'normalisation': 'max'}, 
                  '2t': {'normalisation': 'minmax', 'negative_vals': False}, 
                  'cape': {'normalisation': 'log'}, 
                  'cin': {'normalisation': 'max'}, 
                  't': {'normalisation': 'minmax', 'negative_vals': False},
                  'tclw': {'normalisation': 'log'}, 
                  'tcwv': {'normalisation': 'max'}, 
                  'tisr': {'normalisation': 'max'}}

VAR_LOOKUP_IFS = {field: IFS_NORMALISATION_STRATEGY[re.sub(r'([0-9]*[a-z]+)[0-9]*', r'\1', field)] 
                  for field in all_ifs_fields}

all_era5_fields = list(VAR_LOOKUP_ERA5.keys())

input_field_lookup = {'ifs': all_ifs_fields, 'era5': all_era5_fields}

config = read_config.read_config()

NORMALISATION_YEAR = config['TRAIN']['normalisation_year']

min_latitude = config['DATA']['min_latitude']
max_latitude = config['DATA']['max_latitude']
latitude_step_size = config['DATA']['latitude_step_size']
min_longitude = config['DATA']['min_longitude']
max_longitude = config['DATA']['max_longitude']
longitude_step_size = config['DATA']['longitude_step_size']
DEFAULT_LATITUDE_RANGE=np.arange(min_latitude, max_latitude + latitude_step_size, latitude_step_size)
DEFAULT_LONGITUDE_RANGE=np.arange(min_longitude, max_longitude + longitude_step_size, longitude_step_size)

char_integer_re = re.compile(r'[a-zA-Z]*([0-9]+)')

def denormalise(x: np.float64):
    """
    Undo log normalisation

    Args:
        x (np.float64): log normalised float

    Returns:
        np.float64: denormalised float
    """
    return 10 ** x - 1


def log_plus_1(data_array: xr.DataArray):
    """
    Transform data according to x -> log10(1 + x)

    Args:
        data_array (xr.DataArray): Data array to transform

    Returns:
        xr.DataArray: Transformed data array
    """
    return np.log10(1 + data_array)

def infer_lat_lon_names(ds: xr.Dataset):
    """
    Infer names of latitude / longitude coordinates from the dataset

    Args:
        ds (xr.Dataset): dataset (containing one latitude coordinate 
        and one longitude coordinate)

    Returns:
        tuple: (lat name, lon name)
    """
    
    coord_names = list(ds.coords)
    lat_var_name = [item for item in coord_names if item.startswith('lat')]
    lon_var_name = [item for item in coord_names if item.startswith('lon')]
    
    assert (len(lat_var_name) == 1) and (len(lon_var_name) == 1), IndexError('Cannot infer latitude and longitude names from this dataset')

    return lat_var_name[0], lon_var_name[0]

def order_coordinates(ds: xr.Dataset):
    """
    Order coordinates of dataset -> (time, lat, lon)

    Args:
        ds (xr.Dataset): Dataset to order coordinates of

    Returns:
        xr.Dataset: Ordered dataset
    """
    
    lat_var_name, lon_var_name = infer_lat_lon_names(ds)
    
    if 'time' in list(ds.dims):
        return ds.transpose('time', lat_var_name, lon_var_name)
    else:
        return ds.transpose(lat_var_name, lon_var_name)

def make_dataset_consistent(ds: xr.Dataset):
    """
    Ensure longitude and latitude are ordered in ascending order

    Args:
        ds (xr.Dataset): dataset

    Returns:
        xr.Dataset: Dataset with data reordered
    """
    
    latitude_var, longitude_var = infer_lat_lon_names(ds)
    ds = ds.sortby(latitude_var, ascending=True)
    ds = ds.sortby(longitude_var, ascending=True)
    
    return ds

def get_obs_dates(start_date: datetime,
                  end_date: datetime,  hour: int, obs_data_source: str, 
                  data_paths=DATA_PATHS,
                  ):
    """
    Get dates for which there is observational data available
    Args:
        start_date (datetime): Start date (hourly level)
        end_date (datetime): End date (hourly level)
        obs_data_source (str): Name of data source
        data_paths (dict, optional): Dict containing data paths. Defaults to DATA_PATHS.

    Returns:
        list: list of dates
    """
    date_range = pd.date_range(start=start_date, end=end_date)
    date_range = [item.date() for item in date_range]
    
    obs_dates = set([item for item in date_range if file_exists(data_source=obs_data_source, year=item.year,
                                                    month=item.month, day=item.day,
                                                    data_paths=data_paths, hour=hour)])
    
    return sorted(obs_dates)

def get_dates(years, obs_data_source: str, 
              fcst_data_source: str,
              data_paths=DATA_PATHS):
    """
    Get dates for which there is observational and forecast data

    Args:
        years (int or list): list of years, or single integer year
        obs_data_source (str): name of observational data source
        fcst_data_source (str): name of forecast data source
        data_paths (dict, optional): Dict containing paths to data. Defaults to DATA_PATHS.

    Returns:
        list: list of valid date strings
    """
    if isinstance(years, int):
        years = [years]

    years = sorted(years)
    
    date_range = pd.date_range(start=date(years[0], 1, 1), end=date(years[-1], 12, 31))
    date_range = [item.date() for item in date_range]
    
    obs_dates = set([item for item in date_range if file_exists(data_source=obs_data_source, year=item.year,
                                                    month=item.month, day=item.day,
                                                    data_paths=data_paths)])
    fcst_dates = set([item for item in date_range if file_exists(data_source=fcst_data_source, year=item.year,
                                                    month=item.month, day=item.day,
                                                    data_paths=data_paths)])
    dates = sorted(obs_dates.intersection(fcst_dates))
        
    return [item.strftime('%Y%m%d') for item in dates]


def file_exists(data_source: str, year: int,
                month: int, day:int, hour='random',
                data_paths=DATA_PATHS):
    """
    Check if file exists

    Args:
        data_source (str): Name of data source
        year (int): year
        month (int): month
        day (int): day
        data_paths (dict, optional): dict of data paths. Defaults to DATA_PATHS.

    Returns:
        bool: True if file exists
    """
    
    if hour == 'random':
        hour = 0
        
    data_path = data_paths["GENERAL"].get(data_source.upper()) 
    
    if not data_path:
        raise ValueError(f'No path specified for {data_source} in data_paths')

    if data_source == 'nimrod':
        glob_str = os.path.join(data_path, f"{year}/*.nc")
        if len(glob(glob_str)) > 0:
            return True
        
    elif data_source == 'imerg':
        for file_type in ['.HDF5', '.nc']:
            fps = get_imerg_filepaths(year, month, day, hour, file_ending=file_type, imerg_data_dir=data_path)
            if os.path.isfile(fps[0]):
                return True

    elif data_source == 'ifs':
        fp = get_ifs_filepath('tp', loaddate=datetime(year, month, day), 
                                  loadtime='12', fcst_dir=data_path)
        if os.path.isfile(fp):
            return True

    elif data_source == 'era5':
        # These are just meaningless dates to get the filepath
        era5_fp = get_era5_path('tp', year=year, month=month, era_data_dir=data_path)
        glob_str = era5_fp
        if len(glob(glob_str)) > 0:
            return True
    else:
        raise ValueError(f'Unrecognised data source: {data_source}')
    
    return False
                
def filter_by_lat_lon(ds: xr.Dataset, 
                      lon_range: list, 
                      lat_range: list, 
                      lon_var_name:str='lon', 
                      lat_var_name:str ='lat'):
    """
    Filter dataset by latitude / longitude range

    Args:
        ds (xr.Dataset): Dataset to filter
        lon_range (list): min and max longitude values
        lat_range (list): min and max latitude values
        lon_var_name (str, optional): Name of longitude variable in dataset. Defaults to 'lon'.
        lat_var_name (str, optional): Name of latitude variable in dataset. Defaults to 'lat'.

    Returns:
        xr.Dataset: Filtered dataset
    """
    lat_var_name, lon_var_name = infer_lat_lon_names(ds)
    
    all_lat_vals = [val for val in ds[lat_var_name].values if min(lat_range) <= val <= max(lat_range)]
    all_lon_vals = [val for val in ds[lon_var_name].values if min(lon_range) <= val <= max(lon_range)]
    
    ds = ds.sel({lat_var_name: all_lat_vals})
    ds = ds.sel({lon_var_name: all_lon_vals})
   
    return ds


def interpolate_dataset_on_lat_lon(ds: xr.Dataset, 
                                   latitude_vals: list, 
                                   longitude_vals: list,
                                   interp_method:str ='bilinear'):
    """
    Interpolate dataset to new lat/lon values

    Args:
        ds (xr.Dataset): Datast to interpolate
        latitude_vals (list): list of latitude values to interpolate to
        longitude_vals (list): list of longitude values to interpolate to
        interp_method (str, optional): name of interpolation method. Defaults to 'bilinear'._

    Returns:
        xr,Dataset: interpolated dataset
    """
    
    lat_var_name, lon_var_name = infer_lat_lon_names(ds)
    
    min_lat = min(latitude_vals)
    max_lat = max(latitude_vals)

    min_lon = min(longitude_vals)
    max_lon = max(longitude_vals)
    
    existing_latitude_vals = ds[lat_var_name].values
    existing_longitude_vals = ds[lon_var_name].values
    
    if all([item in existing_latitude_vals for item in latitude_vals]) and all([item in existing_longitude_vals for item in longitude_vals]):
        ds = ds.sel({lat_var_name: latitude_vals}).sel({lon_var_name: longitude_vals})
        return ds
    
    # Filter to correct lat/lon range (faster this way)
    # Add a buffer around it for interpolation
    ds = filter_by_lat_lon(ds, [min_lon - 2, max_lon+2], [min_lat - 2, max_lat +2], 
                      lon_var_name=lon_var_name, lat_var_name=lat_var_name)

    # check enough data to interpolate
    min_actual_lat = min(ds.coords[lat_var_name].values)
    max_actual_lat = max(ds.coords[lat_var_name].values)
    min_actual_lon = min(ds.coords[lon_var_name].values)
    max_actual_lon = max(ds.coords[lon_var_name].values)

    if not ((min_actual_lat < min_lat < max_actual_lat) and (min_actual_lon < min_lon < max_actual_lon)
            and (min_actual_lat < max_lat < max_actual_lat) and (min_actual_lon < max_lon < max_actual_lon)):
        raise ValueError('Larger buffer area needed to ensure good interpolation')

    ds_out = xr.Dataset(
        {
            lat_var_name: ([lat_var_name], latitude_vals),
            lon_var_name: ([lon_var_name], longitude_vals),
        }
    )

    # Use conservative to preserve global precipitation
    regridder = xe.Regridder(ds, ds_out, interp_method)
    regridded_ds = ds.copy()
    
    # Make float vars C-contiguous (to avoid warning message and potentially improve performance)
    for var in list(regridded_ds.data_vars):
        if regridded_ds[var].values.dtype.kind == 'f':
            regridded_ds[var].values = np.ascontiguousarray(regridded_ds[var].values)
            
    regridded_ds = regridder(regridded_ds)

    return regridded_ds


def load_hdf5_file(fp: str, group_name:str ='Grid'):
    """
    Load HDF5 file

    Args:
        fp (str): file path
        group_name (str, optional): name of group to load. Defaults to 'Grid'.

    Returns:
        xr.Dataset: Dataset
    """
    ncf = netCDF4.Dataset(fp, diskless=True, persist=False)
    nch = ncf.groups.get(group_name)
    ds = xr.open_dataset(xr.backends.NetCDF4DataStore(nch))

    return ds


def preprocess(variable: str, 
               ds: xr.Dataset, 
               var_name_lookup: dict, 
               stats_dict: dict=None):
    """
    Preprocess data

    Args:
        variable (str): name of variable.
        ds (xr.Dataset): dataset containing data for named variable
        var_name_lookup (dict): dict containing normalisation strategy for each variable
        stats_dict (dict, optional): dict of values required for normalisation. Defaults to None.

    Returns:
        xr.Dataset: processed dataset
    """
    
    
    var_name = list(ds.data_vars)[0]

    normalisation_type = var_name_lookup[variable].get('normalisation')

    if normalisation_type:

        if normalisation_type == 'standardise':
            ds[var_name] = (ds[var_name] - stats_dict['mean']) / stats_dict['std']

        elif normalisation_type == 'minmax':
            ds[var_name] = (ds[var_name] - stats_dict['min']) / (stats_dict['max'] - stats_dict['min'])
        
        elif normalisation_type == 'log':
            ds[var_name] = log_plus_1(ds[var_name])
            
        elif normalisation_type == 'max':
            ds[var_name] = ds[var_name] / stats_dict['max']
        
        else:
            raise ValueError(f'Unrecognised normalisation type for variable {var_name}')

    return ds


def load_observational_data(data_source: str, *args, **kwargs):
    """
    Function to pick between various different sources of observational data

    Args:
        data_source (str): anme of data source

    Returns:
        np.ndarray: array of data
    """
    if data_source.lower() == 'imerg':
        return load_imerg(*args, **kwargs)
    else:
        raise NotImplementedError(f'Data source {data_source} not implemented yet')


def load_orography(oro_path: str=OROGRAPHY_PATH, 
                   latitude_vals: list=None, 
                   longitude_vals: list=None,
                   interpolate: bool=True):
    """
    Load orography values

    Args:
        oro_path (str, optional): path to orography data. Defaults to OROGRAPHY_PATH.
        latitude_vals (list, optional): list of latitude values to filter/interpolate to. Defaults to None.
        longitude_vals (list, optional): list of longitude values to filter/interpolate to. Defaults to None.
        interpolate (bool, optional): Whether or not to interpolate. Defaults to True.

    Returns:
        np.ndarray: orography data array
    """
    ds = xr.load_dataset(oro_path)
    
    # Note that this assumes the orography is somewhat filtered already 
    # If it is worldwide orography then normalised values will probably be too small!
    max_val = ds['h'].values.max()
       
    if latitude_vals is not None and longitude_vals is not None:
        if interpolate:
            ds = interpolate_dataset_on_lat_lon(ds, latitude_vals=latitude_vals,
                                                longitude_vals=longitude_vals,
                                                interp_method='bilinear')
        else:
            ds = filter_by_lat_lon(ds, lon_range=longitude_vals, lat_range=latitude_vals)

    ds = make_dataset_consistent(ds)

    # Normalise and clip below to remove spectral artefacts
    h_vals = ds['h'].values[0, :, :]
    h_vals[h_vals < 5] = 5.0
    h_vals = h_vals / max_val

    ds.close()

    return h_vals

def load_land_sea_mask(lsm_path=LSM_PATH, 
                       latitude_vals=None, 
                       longitude_vals=None,
                       interpolate=True):
    """
    Load land-sea mask values

    Args:
        lsm_path (str, optional): path to land-sea masj data. Defaults to LSM_PATH.
        latitude_vals (list, optional): list of latitude values to filter/interpolate to. Defaults to None.
        longitude_vals (list, optional): list of longitude values to filter/interpolate to. Defaults to None.
        interpolate (bool, optional): Whether or not to interpolate. Defaults to True.

    Returns:
        np.ndarray: land-sea mask data array
    """
    ds = xr.load_dataset(lsm_path)
    
    if latitude_vals is not None and longitude_vals is not None:
        if interpolate:
            ds = interpolate_dataset_on_lat_lon(ds, latitude_vals=latitude_vals,
                                                longitude_vals=longitude_vals,
                                                interp_method='bilinear')
        else:
            ds = filter_by_lat_lon(ds, lon_range=longitude_vals, lat_range=latitude_vals)
            
    ds = make_dataset_consistent(ds)
    
    lsm = ds['lsm'].values[0, :, :]
    
    ds.close()
    
    return lsm

def load_hires_constants(batch_size: int=1, lsm_path: str=LSM_PATH, 
                         oro_path: str=OROGRAPHY_PATH,
                         latitude_vals: list=None, 
                         longitude_vals: list=None):
    """
    Load hi resolution constants (land sea mask, orography)
    
    Note; this currently interpolates by default

    Args:
        batch_size (int, optional): Size of batch. Defaults to 1.
        lsm_path (str, optional): path to land sea mask data. Defaults to LSM_PATH.
        oro_path (str, optional): path to orography data. Defaults to OROGRAPHY_PATH.
        latitude_vals (list, optional): list of latitude values to interpolate to. Defaults to None.
        longitude_vals (list, optional): list of longitude values to interpolate to. Defaults to None.

    Returns:
        np.ndarray: array of data
    """
    # LSM 
    lsm = load_land_sea_mask(lsm_path=lsm_path, latitude_vals=latitude_vals,
                             longitude_vals=longitude_vals)
    lsm= np.expand_dims(lsm, axis=0) # Need to expand so that dims are consistent with other data

    # Orography
    z = load_orography(oro_path=oro_path, latitude_vals=latitude_vals, 
                       longitude_vals=longitude_vals)
    z = np.expand_dims(z, axis=0)
    
    return np.repeat(np.stack([z, lsm], axis=-1), batch_size, axis=0)


### These functions work with IFS / Nimrod.
# TODO: unify the functions that load data from different sources

def load_fcst_radar_batch(batch_dates: Iterable, 
                          fcst_fields: list, 
                          fcst_data_source: str, 
                          obs_data_source: str, 
                          fcst_dir: str,
                          obs_data_dir: str,
                          latitude_range: Iterable[float]=None,
                          longitude_range: Iterable[float]=None,
                          constants_dir: str=CONSTANTS_PATH,
                          constants: bool=False, hour: int=0, norm: bool=False):
    batch_x = []
    batch_y = []

    if type(hour) == str:
        if hour == 'random':
            all_fcst_hours = np.array(range(24))
            hours = all_fcst_hours[np.random.randint(24, size=[len(batch_dates)])]
        else:
            assert False, f"Not configured for {hour}"
    elif np.issubdtype(type(hour), np.integer):
        hours = len(batch_dates) * [hour]
    else:
        hours = hour

    for i, date in enumerate(batch_dates):
        h = hours[i]
        batch_x.append(load_fcst_stack(fcst_data_source, fcst_fields, date, h,
                                       latitude_vals=latitude_range, longitude_vals=longitude_range, fcst_dir=fcst_dir,
                                       norm=norm, constants_dir=constants_dir))
        
        if obs_data_source is not None:
            batch_y.append(load_observational_data(obs_data_source, date, h, log_precip=norm,
                                                latitude_vals=latitude_range, longitude_vals=longitude_range,
                                                data_dir=obs_data_dir))
    if (not constants):
        return np.array(batch_x), np.array(batch_y)
    else:
        return [np.array(batch_x), load_hires_constants(len(batch_dates))], np.array(batch_y)

def get_ifs_filepath(field: str, loaddate: datetime, 
                     loadtime: int, fcst_dir: str=IFS_PATH):
    """
    Get ifs filepath for time/data combination

    Args:
        field (str): name of field
        loaddate (datetime): load datetime
        loadtime (int): load time
        fcst_dir (str, optional): directory of forecast data. Defaults to IFS_PATH.

    Returns:
        str: filepath
    """
    filename = f"{field}_HRES_1h_EAfrica_{loaddate.strftime('%Y-%m-%d')}_{loadtime}h.nc"

    top_level_folder = re.sub(r'([0-9]*[a-z]+)[0-9]*', r'\1', field)
    subdirectory = True
    
    if top_level_folder == field:
        subdirectory = False
    
    if subdirectory:
        fp = os.path.join(fcst_dir, top_level_folder, field, filename)
    else:
        fp = os.path.join(fcst_dir, top_level_folder, filename)
        
    return fp

def get_ifs_forecast_time(year: int, 
                          month: int, 
                          day:int, hour: int):
    """
    Get the closest IFS forecast load (between 6-18h lead time) to the target
    year/month/day/hour combination

    Args:
        year (int): year
        month (int): month
        day (int): day
        hour (int): hour

    Returns:
        tuple: tuple containing (load date, load time)
    """
    
    time = datetime(year=year, month=month, day=day, hour=hour)
    
    if time.hour < 6:
        loaddate = time - timedelta(days=1)
        loadtime = '12'
    elif 6 <= time.hour < 18:
        loaddate = time
        loadtime = '00'
    elif 18 <= time.hour < 24:
        loaddate = time
        loadtime = '12'
    else:
        raise ValueError("Not acceptable time")
    
    return loaddate, loadtime
    
def load_ifs_raw(field: str, 
                 year: int, 
                 month: int,
                 day: int, hour: int, ifs_data_dir: str=IFS_PATH,
                 latitude_vals: list=None, longitude_vals: list=None, 
                 interpolate: bool=True):
    """
    Load raw IFS data (i.e without any normalisation or conversion to mm/hr)

    Args:
        field (str): name of field
        year (int): year
        month (int): month
        day (int): day  
        hour (int): hour
        ifs_data_dir (str, optional): folder with IFS data in. Defaults to IFS_PATH.
        latitude_vals (list, optional): latitude values to filter/interpolate to. Defaults to None.
        longitude_vals (list, optional): longitude values to filter/interpolate to. Defaults to None.
        interpolate (bool, optional): whether or not to interpolate. Defaults to True.
    
    Returns:
        xr.Dataset: dataset
    """
    
    if not isinstance(latitude_vals[0], np.float32):
        latitude_vals = np.array(latitude_vals).astype(np.float32)
        longitude_vals = np.array(longitude_vals).astype(np.float32)

    assert field in all_ifs_fields, ValueError(f"field must be one of {all_ifs_fields}")
    
    t = datetime(year=year, month=month, day=day, hour=hour)
    t_plus_one = datetime(year=year, month=month, day=day, hour=hour) + timedelta(hours=1)

    # Get the nearest forecast starttime
    loaddate, loadtime = get_ifs_forecast_time(year, month, day, hour)
    
    fp = get_ifs_filepath(field=field,
                          loaddate=loaddate,
                          loadtime=loadtime,
                          fcst_dir=ifs_data_dir
                          )

    ds = xr.open_dataset(fp)
    var_names = list(ds.data_vars)
    
    if np.round(ds.longitude.values.max(), 6) < np.round(max(longitude_vals), 6) or np.round(ds.longitude.values.min(), 6) > np.round(min(longitude_vals), 6):
        raise ValueError('Longitude range outside of data range')
    
    if np.round(ds.latitude.values.max(), 6) < np.round(max(latitude_vals),6) or np.round(ds.latitude.values.min(),6) > np.round(min(latitude_vals), 6):
        raise ValueError('Latitude range outside of data range')
    
    assert len(var_names) == 1, ValueError('More than one variable found; cannot automatically infer variable name')
    var_name = list(ds.data_vars)[0]
    
    # Multiplication with float32 leads to some discrepancies
    ds[var_name] = ds[var_name].astype(np.float64)
       
    # Account for cumulative fields
    if var_name in ['tp', 'cp', 'cdir', 'tisr']:
        # Output rainfall during the following hour
        ds =  ds.sel(time=t_plus_one) - ds.sel(time=t)
    else:
        ds = ds.sel(time=t)

    if latitude_vals is not None and longitude_vals is not None:
        if interpolate:
            if var_name in ['tp', 'tclw', 'cape', 'tisr', 'tcwv', 'cp']:
                interpolation_method = 'conservative'
            else:
                interpolation_method = 'bilinear'
            ds = interpolate_dataset_on_lat_lon(ds, latitude_vals=latitude_vals,
                                                longitude_vals=longitude_vals,
                                                interp_method=interpolation_method)
        else:
            ds = ds.sel(longitude=longitude_vals, method='backfill')
            ds = ds.sel(latitude=latitude_vals, method='backfill')
    
    ds = make_dataset_consistent(ds)
    ds = ds.transpose('latitude', 'longitude')
             
    return ds

def load_ifs(field: str, 
             date, hour: int, norm: bool=False, 
             fcst_dir: str=IFS_PATH, 
             var_name_lookup: dict=VAR_LOOKUP_IFS,
             latitude_vals: list=None, 
             longitude_vals: list=None, 
             constants_path: str=CONSTANTS_PATH):
    """
    Load IFS (including normalisation and conversion to mm/hr)

    Args:
        field (str): name of field
        date (str or datetime): YYYYMMDD string or datetime to forecast
        hour (int): hour to forecast
        norm (bool, optional): whether or not to normalise the data. Defaults to False.
        fcst_dir (str, optional): forecast data directory. Defaults to IFS_PATH.
        var_name_lookup (str, optional): dict with normalisation details for variables. Defaults to VAR_LOOKUP_IFS.
        latitude_vals (list, optional): latitude values to filter/interpolate to. Defaults to None.
        longitude_vals (list, optional): longitude_vals to filter/interpolate to. Defaults to None.
        constants_path (str, optional): path to constant data. Defaults to CONSTANTS_PATH.

    Returns:
        np.ndarray: data for the specified field
    """
    
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y%m%d")
        
    ds = load_ifs_raw(field, date.year, date.month, date.day, hour, ifs_data_dir=fcst_dir,
                      latitude_vals=latitude_vals, longitude_vals=longitude_vals, interpolate=True)
    
    var_name = list(ds.data_vars)[0]
    
    if not var_name_lookup[field].get('negative_vals', True):
        # Make sure no negative values
        ds[var_name] = ds[var_name].clip(min=0)
        
    if field in ['tp', 'cp', 'pr', 'prl', 'prc']:
        # precipitation is measured in metres, so multiply up
        ds[var_name] = 1000 * ds[var_name]
        
    if norm:
        stats_dict = get_ifs_stats(field, latitude_vals=latitude_vals, longitude_vals=longitude_vals,
                            use_cached=True, ifs_data_dir=fcst_dir,
                            output_dir=constants_path)
        # Normalisation here      
        ds = preprocess(field, ds, stats_dict=stats_dict, var_name_lookup=var_name_lookup)
    
    y = np.array(ds[var_name][:, :])
    
    return y


def load_fcst_stack(data_source: str, fields: list, 
                    date: str, hour: int, 
                    fcst_dir: str, 
                    constants_dir:str=CONSTANTS_PATH,
                    norm:bool=False,
                    latitude_vals:list=None, longitude_vals:list=None):
    """
    Load forecast 'stack' of all variables

    Args:
        data_source (str): source of data (e.g. ifs)
        fields (list): list of fields to load
        date (str): YYYYMMDD date string to forecast for
        hour (int): hour to forecast for
        fcst_dir (str): folder with forecast data in
        constants_dir (str, optional): folder with constants data in. Defaults to CONSTANTS_PATH.
        norm (bool, optional): whether or not to normalise the data. Defaults to False.
        latitude_vals (list, optional): list of latitude values. Defaults to None.
        longitude_vals (list, optional): list of longitude values. Defaults to None.

    Returns:
        np.ndarray: stacked array of specified variables
    """
    field_arrays = []

    if data_source == 'ifs':
        load_function = load_ifs
    elif data_source == 'era5':
        load_function = load_era5
    else:
        raise ValueError(f'Unknown data source {data_source}')

    for f in fields:
        field_arrays.append(load_function(f, date, hour, fcst_dir=fcst_dir,
                                          latitude_vals=latitude_vals, longitude_vals=longitude_vals,
                                          constants_path=constants_dir, norm=norm))
    return np.stack(field_arrays, -1)


def get_ifs_stats(field: str, latitude_vals: list, longitude_vals: list, output_dir: str=None, 
                   use_cached: bool=True, year: int=NORMALISATION_YEAR,
                   ifs_data_dir: str=IFS_PATH, hours: list=all_fcst_hours):

    min_lat = int(min(latitude_vals))
    max_lat = int(max(latitude_vals))
    min_lon = int(min(longitude_vals))
    max_lon = int(max(longitude_vals))


    # Filepath is specific to make sure we don't use the wrong normalisation stats
    fp = f'{output_dir}/IFS_norm_{field}_{year}_lat{min_lat}-{max_lat}lon{min_lon}-{max_lon}.pkl'
    var_name = None

    if use_cached and os.path.isfile(fp):
        logger.debug('Loading stats from cache')

        with open(fp, 'rb') as f:
            stats = pickle.load(f)

    else:
        print('Calculating stats')
        all_dates = list(pd.date_range(start=f'{year}-01-01', end=f'{year}-12-01', freq='D'))
        all_dates = [item.date() for item in all_dates]

        datasets = []
        
        for date in all_dates:
            year = date.year
            month = date.month
            day = date.day
            
            for hour in hours:
            
                tmp_ds = load_ifs_raw(field, year, month, day, hour, 
                                    latitude_vals=latitude_vals,
                                    longitude_vals=longitude_vals, 
                                    ifs_data_dir=ifs_data_dir,
                                    interpolate=False)
                if not var_name:
                    var_names = list(tmp_ds.data_vars)
                    assert len(var_names) == 1, ValueError('More than one variable found; cannot automatically infer variable name')
                    var_name = list(tmp_ds.data_vars)[0]

                datasets.append(tmp_ds)
        concat_ds = xr.concat(datasets, dim='time')

        stats = {'min': np.abs(concat_ds[var_name]).min().values,
                'max': np.abs(concat_ds[var_name]).max().values,
                'mean': concat_ds[var_name].mean().values,
                'std': concat_ds[var_name].std().values}

        if output_dir:
            with open(fp, 'wb') as f:
                pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)

    return stats


### Functions that work with the ERA5 data in University of Bristol


def get_era5_filepath_prefix(variable, era_data_dir=ERA5_PATH,
                             var_name_lookup=VAR_LOOKUP_ERA5):
    filepath_attrs = var_name_lookup[variable]

    if filepath_attrs.get('subfolder'):
        return os.path.join(era_data_dir, '{folder}/{subfolder}/ERA5_{subfolder}_{suffix}'.format(**filepath_attrs))
    else:
        return os.path.join(era_data_dir, '{folder}/ERA5_{folder}_{suffix}'.format(**filepath_attrs))


def get_era5_path(variable, year, month, era_data_dir=ERA5_PATH, var_name_lookup=VAR_LOOKUP_ERA5):
    suffix = get_era5_filepath_prefix(variable=variable, era_data_dir=era_data_dir, var_name_lookup=var_name_lookup)
    
    if month == '*':
        return f'{suffix}_{year}{month}.nc'
    else:
        return f'{suffix}_{year}{int(month):02d}.nc'


def load_era5_month_raw(variable, year, month, latitude_vals=None, longitude_vals=None,
                        era_data_dir=ERA5_PATH, interpolate=True):
    
    ds = xr.load_dataset(get_era5_path(variable, year=year, month=month, era_data_dir=era_data_dir))

    # Multiplication with float32 leads to some discrepancies
    ds[variable] = ds[variable].astype(np.float64)
       
    interpolation_method = 'conservative' if variable == 'tp' else 'bilinear'
    
    # only works if longitude and latitude vals specified together
    if latitude_vals is not None and longitude_vals is not None:
        if interpolate:
            ds = interpolate_dataset_on_lat_lon(ds, latitude_vals=latitude_vals,
                                                longitude_vals=longitude_vals,
                                                interp_method=interpolation_method)
        else:
            ds = filter_by_lat_lon(ds, lon_range=longitude_vals, lat_range=latitude_vals)

    # Make sure dataset is consistent with others
    ds = make_dataset_consistent(ds)
    
    if variable == 'tp':
        # Convert to mm
        ds['tp'] = ds['tp'] * 1000
    
    return ds


def load_era5_day_raw(variable: str, year: int, month: int, 
                      day: int, latitude_vals: list=None, 
                      longitude_vals: list=None,
                      era_data_dir: str=ERA5_PATH, interpolate: bool=True):

    month_ds = load_era5_month_raw(variable, year, month, latitude_vals=latitude_vals,
                                    longitude_vals=longitude_vals,
                                    era_data_dir=era_data_dir, interpolate=interpolate)

    day_ds = month_ds.sel(time=f'{year}-{int(month):02d}-{int(day):02d}')
    
    return day_ds


def get_era5_stats(variable: str, longitude_vals: list, latitude_vals: list, 
                   year: int=NORMALISATION_YEAR, output_dir: int=None,
                   era_data_dir: int=ERA5_PATH, use_cached: bool=False):
    
    min_lat = int(min(latitude_vals))
    max_lat = int(max(latitude_vals))
    min_lon = int(min(longitude_vals))
    max_lon = int(max(longitude_vals))
        
    # Filepath is spcific to make sure we don't use the wrong normalisation stats
    fp = f'{output_dir}/ERA_norm_{variable}_{year}_lat{min_lat}-{max_lat}lon{min_lon}-{max_lon}.pkl'

    if use_cached and os.path.isfile(fp):
        logger.debug('Loading stats from cache')

        with open(fp, 'rb') as f:
            stats = pickle.load(f)

    else:

        all_dates = list(pd.date_range(start=f'{year}-01-01', end=f'{year}-12-01', freq='M'))
        all_dates = [item.date() for item in all_dates]

        datasets = []

        for date in all_dates:
            year = date.year
            month = date.month
            tmp_ds = load_era5_month_raw(variable, year, month, latitude_vals=latitude_vals,
                                         longitude_vals=longitude_vals, 
                                         era_data_dir=era_data_dir,
                                         interpolate=False)

            datasets.append(tmp_ds)
        concat_ds = xr.concat(datasets, dim='time')

        stats = {'min': np.abs(concat_ds[variable]).min().values,
                'max': np.abs(concat_ds[variable]).max().values,
                'mean': concat_ds[variable].mean().values,
                'std': concat_ds[variable].std().values}

        if output_dir:
            with open(fp, 'wb') as f:
                pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)

    return stats


def load_era5(ifield, date, hour=0, log_precip=False, norm=False, fcst_dir=ERA5_PATH,
              latitude_vals=None, longitude_vals=None, var_name_lookup=VAR_LOOKUP_ERA5,
              constants_path=CONSTANTS_PATH):
    """

    Function to fetch ERA5 data, designed to match the structure of the load_fcst function, so they can be interchanged

    Args:
        latitude_vals:
        constants_path:
        ifield: field to load
        date: str, date in form YYYYMMDD
        hour: int or list, hour or hours to fetch from (Obsolete in this case as only daily data)
        log_precip: Boolean, whether to take logs of the data (Obsolete for this case as using config)
        norm: Boolean, whether to normalise (obsolete in this case as using config)
        fcst_dir:

    Returns:

    """
    dt = datetime.strptime(date, '%Y%m%d')
    ds = load_era5_day_raw(variable=ifield, year=dt.year, month=dt.month, day=dt.day,
                           latitude_vals=latitude_vals, longitude_vals=longitude_vals,
                           era_data_dir=fcst_dir)

    stats_dict = get_era5_stats(ifield, latitude_vals=latitude_vals, longitude_vals=longitude_vals,
                                use_cached=True, era_data_dir=fcst_dir,
                                output_dir=constants_path)

    if norm:
        # Normalisation here      
        ds = preprocess(ifield, ds, stats_dict=stats_dict, var_name_lookup=var_name_lookup)
    
    # Only one time index should be returned since this is daily data
    y = np.array(ds[ifield][0, :, :])
    ds.close()

    if ifield == 'tp' and log_precip == True:
        y = log_plus_1(y)

    ds.close()

    return y

def get_imerg_filepaths(year: int, month: int, day: int, 
                        hour: int, imerg_data_dir: str=IMERG_PATH, 
                        file_ending: str='.nc'):
    
    dt_start = datetime(year, month, day, hour, 0, 0) 
        
    fp1 = os.path.join(imerg_data_dir, '3B-HHR.MS.MRG.3IMERG.' + dt_start.strftime('%Y%m%d-S%H0000-E%H2959') + f'.{2*hour * 30:04d}.V06B{file_ending}')
    fp2 = os.path.join(imerg_data_dir, '3B-HHR.MS.MRG.3IMERG.' + dt_start.strftime('%Y%m%d-S%H3000-E%H5959') + f'.{(2*hour + 1) * 30:04d}.V06B{file_ending}')
    
    return [fp1, fp2]
    
def load_imerg_raw(year: int, month: int, day: int, 
                   hour:int, latitude_vals: list=None, 
                   longitude_vals: list=None,
                   imerg_data_dir: str=IMERG_PATH, file_ending: str='.nc'):

    if isinstance(latitude_vals, list):
        latitude_vals = np.array(latitude_vals)
        
    if isinstance(longitude_vals, list):
        longitude_vals = np.array(longitude_vals)
        
    if not isinstance(latitude_vals[0], np.float32):
        latitude_vals = latitude_vals.astype(np.float32)
        longitude_vals = longitude_vals.astype(np.float32)
    
    fps = get_imerg_filepaths(year, month, day, hour, imerg_data_dir=imerg_data_dir, file_ending=file_ending)
    
    datasets = []
    
    for fp in fps:
        if file_ending.lower() == '.nc':
            datasets.append(xr.load_dataset(fp))
        elif file_ending.lower() == '.hdf5':
            datasets.append(load_hdf5_file(fp, 'Grid'))
        else:
            raise IOError(f'File formats {file_ending} not supported')

    ds = xr.concat(datasets, dim='time').mean('time')
    
    if ds.lon.values.max() < max(longitude_vals) or ds.lon.values.min() > min(longitude_vals):
        raise ValueError('Longitude range outside of data range')
    
    if ds.lat.values.max() < max(latitude_vals) or ds.lat.values.min() > min(latitude_vals):
        raise ValueError('Latitude range outside of data range')

    # Note we use method nearest; the iMERG data isn't interpolated to be on
    # the same grid as the input forecast necessarily (they don't need to match exactly)
    # Add small shift otherwise it may be non-deterministic as to which side it chooses
    if longitude_vals is not None:
        ds = ds.sel(lon=longitude_vals + 1e-6, method='nearest')

    if latitude_vals is not None:
        ds = ds.sel(lat=latitude_vals + 1e-6, method='nearest')
        
    # Make sure dataset is consistent with others
    ds = make_dataset_consistent(ds)
    ds = ds.transpose('lat', 'lon', 'latv', 'lonv')

    return ds


def load_imerg(date: datetime, hour: int=18, data_dir: str=IMERG_PATH,
               latitude_vals: list=None, longitude_vals: list=None,
               log_precip: bool=False):
    """

     Function to fetch iMERG data, designed to match the structure of the load_radar function, so they can be
     interchanged

     Args:
         date: str, date in form YYYYMMDD
         hour: int or list, hour or hours to fetch from (Obsolete in this case as only daily data)
         log_precip: Boolean, whether to take logs of the data (Obsolete for this case as using config)
         radar_dir: str, directory where imerg data is stored

     Returns:

     """
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y%m%d')
        
    ds = load_imerg_raw(year=date.year, month=date.month, day=date.day, hour=hour,
                        latitude_vals=latitude_vals, longitude_vals=longitude_vals, imerg_data_dir=data_dir)

    # Take mean since data may be half hourly
    precip = ds['precipitationCal'].values
    ds.close()
    
    if log_precip:
        precip = log_plus_1(precip)

    return precip


if __name__ == '__main__':

    # Batch job for loading / saving only the relevant data

    parser = ArgumentParser(description='Subset data to desired lat/lon range')
    parser.add_argument('--fcst-data-source', choices=['ifs', 'era5'], type=str,
                        help='Source of forecast data')
    parser.add_argument('--obs-data-source', choices=['nimrod', 'imerg'], type=str,
                        help='Source of observational (ground truth) data')
    parser.add_argument('--years', default=2018, nargs='+', type=int,
                        help='Year(s) to process (space separated)')
    parser.add_argument('--months', default=list(range(1, 13)), nargs='+', type=int,
                        help='Year(s) to process (space separated)')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--min-latitude', type=int, required=True)
    parser.add_argument('--max-latitude', type=int, required=True)
    parser.add_argument('--min-longitude', type=int, required=True)
    parser.add_argument('--max-longitude', type=int, required=True)

    args = parser.parse_args()

    print(args)
    
    years = [args.years] if isinstance(args.years, int) else args.years

    latitude_vals = range(args.min_latitude -1, args.max_latitude + 1)
    longitude_vals = range(args.min_longitude -1, args.max_longitude + 1)

    if not os.path.isdir(args.output_dir):
        raise IOError('Output directory does not exist! Please create it or specify a different one')

    imerg_folders = ['IMERG', 'half_hourly', 'final']
    era5_output_dir = os.path.join(args.output_dir, 'ERA5')
    imerg_output_dir = os.path.join(args.output_dir, '/'.join(imerg_folders))

    if not os.path.isdir(era5_output_dir):
        os.mkdir(era5_output_dir)

    for _, properties in VAR_LOOKUP_ERA5.items():

        var_folder = os.path.join(era5_output_dir, properties['folder'])
        if not os.path.isdir(var_folder):
            os.mkdir(var_folder)

        subfolder = properties.get('subfolder')

        if subfolder:
            subfolder = os.path.join(var_folder, subfolder)
            if not os.path.isdir(subfolder):
                os.mkdir(subfolder)
    
    dir = args.output_dir
    for folder in imerg_folders:
        current_dir = os.path.join(dir, folder)
        if not os.path.isdir(current_dir):
            os.mkdir(current_dir)
        dir = os.path.join(dir, folder)

    all_imerg_dates = []
    for year in years:
        for month in args.months:
            for day in range(1, monthrange(year, month)[1]+1):
                for hour in range(0, 24):
                    all_imerg_dates.append((year, month, day, hour))
    
    print('starting observational data gathering')
    for date_item in tqdm(all_imerg_dates, total=len(all_imerg_dates), position=0, leave=True):

        year, month, day, hour = date_item

        fps = glob(os.path.join('/bp1/geog-tropical/data/Obs/IMERG/half_hourly/final', 
                                f'3B-HHR.MS.MRG.3IMERG.{year}{month:02d}{day:02d}-S{hour:02d}*'))
        
        if len(fps) == 0:
            print(f'No files found for {year}{month:02d}{day:02d}-S{hour:02d}')
        
        for fp in fps:
            ds =load_hdf5_file(fp)
            ds = filter_by_lat_lon(ds, lon_range=longitude_vals, 
                                   lat_range=latitude_vals)
            path_suffix = fp.split('/')[-1]
            imerg_path = os.path.join(imerg_output_dir, path_suffix).replace('.HDF5', '.nc')
            ds.to_netcdf(imerg_path)
            ds.close()

    # print("starting forcast data gathering")
    # all_era5_dates = []
    # for year in years:
    #     for month in args.months:
    #         all_era5_dates.append((year, month))
            
    # for date_item in tqdm(all_era5_dates, total=len(all_era5_dates)): 
    #     year, month = date_item              
    #     for variable in VAR_LOOKUP_ERA5:
    #         input_era5_path = get_era5_path(variable, year, month)
    #         era5_ds = xr.load_dataset(input_era5_path)
    #         era5_ds = filter_by_lat_lon(era5_ds, 
    #                                     lat_range=latitude_vals, lon_range=longitude_vals)
    #         prefix = get_era5_filepath_prefix(variable=variable, 
    #                                           era_data_dir=era5_output_dir)
    #         output_era5_path = f'{prefix}_{year}{month:02d}.nc'

    #         era5_ds.to_netcdf(output_era5_path)
    #         era5_ds.close()
