import sys, os
import unittest
import tempfile
import pickle
from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from numpy import testing

HOME = Path(__file__).parents[1]
data_folder = HOME / 'tests' / 'data'

sys.path.append(str(HOME))

from nwpdl.data import infer_lat_lon_names, load_hires_constants, load_imerg_raw, load_era5_day_raw, \
    VAR_LOOKUP_ERA5, interpolate_dataset_on_lat_lon, \
    get_era5_stats, load_fcst_stack, all_ifs_fields, load_era5, \
    load_ifs, get_imerg_filepaths, get_ifs_filepath, \
    load_fcst_radar_batch, log_plus_1, filter_by_lat_lon, load_ifs_raw, \
    VAR_LOOKUP_IFS, get_ifs_stats, file_exists, get_dates, load_land_sea_mask, load_orography


ifs_path = str(data_folder / 'IFS')
nimrod_path = str(data_folder / 'NIMROD')
constants_path = str(data_folder / 'constants')
era5_path = str(data_folder / 'ERA5')
imerg_folder = str(data_folder / 'IMERG/half_hourly/final')

def create_dummy_stats_data(year=2017):
    
    for field in all_ifs_fields:
        fp = get_ifs_filepath(field, datetime(2017, 7, 4), 12, ifs_path)
        ds = xr.load_dataset(fp)
        
        # Mock IFS stats dicts
        var_name = list(ds.data_vars)[0]
        stats = {'min': np.abs(ds[var_name]).min().values,
            'max': np.abs(ds[var_name]).max().values,
            'mean': ds[var_name].mean().values,
            'std': ds[var_name].std().values}
        
        # Saving it as 2017 since that's the defualt 
        output_fp = f'{constants_path}/IFS_norm_{field}_{year}_lat0-1lon33-34.pkl'
        with open(output_fp, 'wb') as f:
            pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)
        
        # Save for another different lat / lon range
        output_fp = f'{constants_path}/IFS_norm_{field}_{year}_lat0-0lon33-33.pkl'
        with open(output_fp, 'wb') as f:
            pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)
            
        if field == 'tp':
            # Also need to save for ERA5
            # Assuming we just need tp but if there are more ERA5 fields this needs
            # to be given more thought
            output_fp = f'{constants_path}/ERA5_norm_{field}_{year}_lat0-0lon33-33.pkl'
            with open(output_fp, 'wb') as f:
                pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)
    
class TestLoad(unittest.TestCase):
    
    def setUp(self) -> None:
        # Create dummy stats to avoid having to recalculate them
        
        self.temp_stats_dir = tempfile.TemporaryDirectory()
        self.temp_stats_dir_name = self.temp_stats_dir.name
                      
        if not os.path.isdir(os.path.join(constants_path, 'tp')):
            
            create_dummy_stats_data()
            
            for field in all_ifs_fields:
                fp = get_ifs_filepath(field, datetime(2017, 7, 4), 12, ifs_path)
                ds = xr.load_dataset(fp)
                    
                # Create mock data for calculating stats in IFS
                all_dates = list(pd.date_range(start='2017-01-01', end='2017-12-01', freq='D'))
                all_dates = [item.date() for item in all_dates]

                # create folder structure
                output_suffix = get_ifs_filepath(field, datetime(2016, 1, 1), 12, constants_path).replace(constants_path, '')
                folders = [item for item in output_suffix.split('/')[:-1] if item]

                current_folder = constants_path
                for folder in folders:
                    current_folder = os.path.join(current_folder, folder)
                    if not os.path.isdir(current_folder):
                        os.mkdir(current_folder)
                
                # save replicas of files
                for date in all_dates:
                    year = date.year
                    month = date.month
                    day = date.day
                    
                    output_path = get_ifs_filepath(field, datetime(year, month, day), 12, constants_path)

                    ds.to_netcdf(output_path)
        return super().setUp()
    
    def tearDown(self) -> None:
        self.temp_stats_dir.cleanup()
        return super().tearDown()
    
    def test_load_ifs_raw(self):
        year = 2017
        month = 7
        day = 5
        hour = 4
        
        latitude_vals = [0.05, 0.15, 0.25]
        longitude_vals = [33, 34]

        lat_coords = []
        lon_coords = []
        
        for field in all_ifs_fields:
            
            ds = load_ifs_raw(field, year, month, day, hour, ifs_data_dir=str(ifs_path),
                              latitude_vals=latitude_vals, longitude_vals=longitude_vals, 
                              interpolate=False)
            
            self.assertIsInstance(ds, xr.Dataset)
            
            data_var = list(ds.data_vars)[0]
            
            # Check only lat/lon coord, not time, and that dims are correctly ordered
            self.assertEqual(ds[data_var].values.shape, (len(latitude_vals), len(longitude_vals)))
        
            # this also checks that the longitude values are in ascending order
            testing.assert_allclose(ds.latitude.values, np.array([0.05, 0.15, 0.25]), atol=1e-7)
            testing.assert_allclose(ds.longitude.values, np.array([33.05, 34.05]), atol=1e-7)        
            
            lat_var_name, lon_var_name = infer_lat_lon_names(ds)
            
            # check that lat lon are ascending
            self.assertListEqual(list(ds[lat_var_name].values), sorted(ds[lat_var_name].values))
            self.assertListEqual(list(ds[lon_var_name].values), sorted(ds[lon_var_name].values))
            
            lat_coords.append(tuple(sorted(ds.coords[lat_var_name].values)))
            lon_coords.append(tuple(sorted(ds.coords[lon_var_name].values)))
        
        # Check lat and long coordinates are all the same
        self.assertEqual(len(set(lat_coords)), 1)
        self.assertEqual(len(set(lon_coords)), 1)

    def test_load_land_sea_mask(self):

        lsm_path = data_folder / 'constants' / 'lsm_HRES_EAfrica.nc'

        latitude_vals = np.arange(-1, 1, 0.1)
        longitude_vals = np.arange(33, 34, 0.1)
        lsm = load_land_sea_mask(lsm_path=lsm_path, latitude_vals=latitude_vals, 
                                longitude_vals=longitude_vals)
        self.assertIsInstance(lsm, np.ndarray)
        shape = lsm.shape
        
        self.assertEqual(shape[0], len(latitude_vals))
        self.assertEqual(shape[1], len(longitude_vals))
                
        # check values between 0 and 1
        self.assertLessEqual(lsm.max(), 1.0)
        self.assertGreaterEqual(lsm.min(), 0)

    def test_load_orography(self):
        
        oro_path = data_folder / 'constants' / 'h_HRES_EAfrica.nc'

        latitude_vals = np.arange(-1, 1, 0.1)
        longitude_vals = np.arange(33, 34, 0.1)
        h = load_orography(oro_path=oro_path,  latitude_vals=latitude_vals, 
                                longitude_vals=longitude_vals)
        self.assertIsInstance(h, np.ndarray)
        shape = h.shape
        
        self.assertEqual(shape[0], len(latitude_vals))
        self.assertEqual(shape[1], len(longitude_vals))
        
        # check values between 0 and 1
        self.assertLessEqual(h.max(), 1.0)
        self.assertGreaterEqual(h.min(), 0)
    
    def test_load_hires_constants(self):
        
        lsm_path = data_folder / 'constants' / 'lsm_HRES_EAfrica.nc'
        oro_path = data_folder / 'constants' / 'h_HRES_EAfrica.nc'
        
        latitude_vals = np.arange(-1, 1, 0.1)
        longitude_vals = np.arange(33, 34, 0.1)
        batch_size = 4
        
        c = load_hires_constants(oro_filename=oro_path, lsm_filename=lsm_path,
                                 batch_size=batch_size, latitude_vals=latitude_vals, 
                                 longitude_vals=longitude_vals)
        self.assertEqual(c.shape, (batch_size, len(latitude_vals), len(longitude_vals), 2))
        
        # check values between 0 and 1
        self.assertLessEqual(c.max(), 1.0)
        self.assertGreaterEqual(c.min(), 0)
    
    def test_load_era5_raw(self):

        year = 2018
        month = 12
        day = 1

        lat_coords = []
        lon_coords = []

        var_name_lookup = VAR_LOOKUP_ERA5

        for v in var_name_lookup:
    
            ds1 = load_era5_day_raw(v, year=year, month=month, day=day,
                                    latitude_vals=[0, 0.1, 0.2], longitude_vals=[33, 33.1, 33.2],
                                    era_data_dir=str(era5_path))

            self.assertIsInstance(ds1, xr.Dataset)
            
            lat_var_name, lon_var_name = infer_lat_lon_names(ds1)
            
            # check that lat lon are ascending
            self.assertListEqual(list(ds1[lat_var_name].values), sorted(ds1[lat_var_name].values))
            self.assertListEqual(list(ds1[lon_var_name].values), sorted(ds1[lon_var_name].values))
       
            lat_coords.append(tuple(sorted(ds1.coords[lat_var_name].values)))
            lon_coords.append(tuple(sorted(ds1.coords[lon_var_name].values)))

        # Check lat and long coordinates are all the same
        self.assertEqual(len(set(lat_coords)), 1)
        self.assertEqual(len(set(lon_coords)), 1)

    def test_precipitation_scaling(self):

        # Check that pre
        # year = 2018
        # month = 12
        # day = 1

        # ds_raw = load_era5_day_raw('tp', year=year, month=month, day=day, 
        #                            latitude_vals=[0, 0.1], longitude_vals=[33, 33.1],
        #                            era_data_dir=str(era5_path))
        
        # testing.assert_allclose(ds_raw['tp'].values[0], np.array([[3.36921681, 3.35620437], [3.24860298, 3.19705309]]), atol=1e-8)
        
        ## Same for IFS
        year = 2017
        month = 7
        day = 5
        hour = 18

        ds_raw = load_ifs_raw('tp', year=year, month=month, day=day, hour=hour,
                                   latitude_vals=[-1.75, -1.85], longitude_vals=[34.35, 34.45],
                                   ifs_data_dir=str(ifs_path), interpolate=False)
        
        testing.assert_allclose(ds_raw['tp'].values, np.array(np.array([[3.98702919e-04, 6.53415918e-06], [0.00081483, 0.]])), atol=1e-8)

    def test_era5_load_norm_logs(self):

        year = 2018
        month = 12
        day = 1
        lat_vals = np.arange(0, 1, 0.1)
        lon_vals = np.arange(33, 34, 0.1)

        var_name_lookup = VAR_LOOKUP_ERA5

        for v in var_name_lookup:
            if v == 'tp':
                # check log precipitation is working
                ds_tp_no_log = load_era5(v, f'{year}{month:02d}{day:02d}', hour=12, log_precip=False, norm=False,
                                         fcst_dir=str(era5_path),
                                         latitude_vals=lat_vals, longitude_vals=lon_vals, constants_path=constants_path)
                ds_tp_log = load_era5(v, f'{year}{month:02d}{day:02d}', hour=12, log_precip=True, norm=False,
                                      fcst_dir=str(era5_path),
                                      latitude_vals=lat_vals, longitude_vals=lon_vals, constants_path=constants_path)
                ds_tp_log_norm = load_era5(v, f'{year}{month:02d}{day:02d}', hour=12, log_precip=True, norm=True,
                                           fcst_dir=str(era5_path),
                                           latitude_vals=lat_vals, longitude_vals=lon_vals, constants_path=constants_path)

                testing.assert_array_equal(ds_tp_log, ds_tp_log_norm)
                testing.assert_array_equal(ds_tp_log, log_plus_1(ds_tp_no_log))

            # Try loading with normalisation
            # ds_tp_norm = load_era5(v, f'{year}{month:02d}{day:02d}', hour=12, log_precip=False, norm=True,
            #                        fcst_dir=str(era5_path),
            #                        latitude_vals=lat_vals, longitude_vals=lon_vals, constants_path=constants_path)
            # ds_tp_no_norm = load_era5(v, f'{year}{month:02d}{day:02d}', hour=12, log_precip=False, norm=False,
            #                           fcst_dir=str(era5_path),
            #                           latitude_vals=lat_vals, longitude_vals=lon_vals, constants_path=constants_path)

            # normalisation = var_name_lookup.get('normalisation')
            # stats_dict = get_era5_stats(v, output_dir=constants_path, era_data_dir=era5_path,
            #                             latitude_vals=lat_vals, longitude_vals=lon_vals, use_cached=True)

            # if normalisation == 'minmax':
            #     testing.assert_array_equal(ds_tp_norm,
            #                      (ds_tp_no_norm - stats_dict['min']) / (stats_dict['max'] - stats_dict['min']))

    def test_ifs_load_norm_logs(self):

        year = 2017
        month = 7
        day = 5
        hour=4
        
        lat_vals = np.arange(0, 1, 0.1)
        lon_vals = np.arange(33, 34, 0.1)

        var_name_lookup = VAR_LOOKUP_IFS

        for v in var_name_lookup:
            if v == 'tp':
                # check log precipitation is working
                ds_tp_no_log = load_ifs(v, f'{year}{month:02d}{day:02d}', hour=hour,  norm=False,
                                         fcst_dir=str(ifs_path),
                                         latitude_vals=lat_vals, longitude_vals=lon_vals, 
                                         constants_path=constants_path)
                ds_tp_log_norm = load_ifs(v, f'{year}{month:02d}{day:02d}', hour=hour, norm=True,
                                      fcst_dir=str(ifs_path),
                                      latitude_vals=lat_vals, longitude_vals=lon_vals, 
                                      constants_path=constants_path)

                testing.assert_array_equal(ds_tp_log_norm, log_plus_1(ds_tp_no_log))

            # Try loading with normalisation
            ds_tp_norm = load_ifs(v, f'{year}{month:02d}{day:02d}', hour=hour, norm=True,
                                   fcst_dir=str(ifs_path),
                                   latitude_vals=lat_vals, longitude_vals=lon_vals, constants_path=constants_path)
            ds_tp_no_norm = load_ifs(v, f'{year}{month:02d}{day:02d}', hour=hour, norm=False,
                                      fcst_dir=str(ifs_path),
                                      latitude_vals=lat_vals, longitude_vals=lon_vals, constants_path=constants_path)

            normalisation = var_name_lookup.get('normalisation')
            stats_dict = get_ifs_stats(v, output_dir=constants_path, ifs_data_dir=ifs_path,
                                        latitude_vals=lat_vals, longitude_vals=lon_vals)

            if normalisation == 'minmax':
                testing.assert_array_equal(ds_tp_norm,
                                 (ds_tp_no_norm - stats_dict['min']) / (stats_dict['max'] - stats_dict['min']))


    def test_filter_on_lat_lon(self):
        
        year = 2018
        month = 12
        day = 30
        hour = 18

        # Fetch raw era5 data
        raw_ds = load_era5_day_raw('tp', year=year, month=month, day=day,
                                   era_data_dir=era5_path, interpolate=False
                                   )
        
        filtered_ds = filter_by_lat_lon(raw_ds, lon_range=[33, 34], lat_range=[0, 1],
                                        lat_var_name='latitude', lon_var_name='longitude')
        
        testing.assert_array_equal(filtered_ds.latitude.values, np.arange(0, 1.25, 0.25))
        testing.assert_array_equal(filtered_ds.longitude.values, np.arange(33, 34.25, 0.25))
        
        # Check with imerg
        # Fetch raw imerg data
        latitude_vals=np.arange(-2, 2, 0.1)
        longitude_vals=np.arange(29, 31, 0.1)
        rainy_patch_ds = xr.open_dataset(os.path.join(data_folder, 'IMERG/rainy_patch.nc'))
        filtered_ds = filter_by_lat_lon(rainy_patch_ds, lon_range=longitude_vals, lat_range=latitude_vals)
        
        # This is to check that it isn't returning duplicate lat/lon values (which happens with sel and method='nearest')
        self.assertEqual(len(filtered_ds.lon.values), len(set(filtered_ds.lon.values)))
        self.assertEqual(len(filtered_ds.lat.values), len(set(filtered_ds.lat.values)))

    def test_interpolation(self):

        year = 2018
        month = 12
        day = 30
        hour = 18

        # Fetch raw imerg data
        raw_ds = load_era5_day_raw('tp', year=year, month=month, day=day,
                                   era_data_dir=era5_path, interpolate=False
                                   )
        raw_ds = raw_ds.sel(latitude=[0, 0.25, 0.5], longitude=np.arange(33, 34.5, 0.25), method='nearest')

        regridded_ds = interpolate_dataset_on_lat_lon(raw_ds, 
                                                      latitude_vals=[0.1], longitude_vals=[33.1],
                                                      interp_method='bilinear',)
        regridded_val = regridded_ds.tp.values[0][0][0]

        # Check that the linear interpolation is looking reasonable
        x1 = raw_ds.longitude.values[0]
        x2 = raw_ds.longitude.values[1]
        y1 = raw_ds.latitude.values[0]
        y2 = raw_ds.latitude.values[1]

        f11 = raw_ds.sel(longitude=x1, latitude=y1, method='nearest').tp.values[0]
        f12 = raw_ds.sel(longitude=x1, latitude=y2, method='nearest').tp.values[0]
        f21 = raw_ds.sel(longitude=x2, latitude=y1, method='nearest').tp.values[0]
        f22 = raw_ds.sel(longitude=x2, latitude=y2, method='nearest').tp.values[0]

        x = 33.1
        y = 0.1

        fxy1 = ((x2 - x) / (x2 - x1)) * f11 + ((x - x1) / (x2 - x1)) * f21
        fxy2 = ((x2 - x) / (x2 - x1)) * f12 + ((x - x1) / (x2 - x1)) * f22

        fxy = ((y2 - y) / (y2 - y1)) * fxy1 + ((y - y1) / (y2 - y1)) * fxy2

        # Not an exact match, but I'm not sure if there are other things
        # that xesmf is doing
        self.assertAlmostEqual(regridded_val, fxy, 5)

        # Check that it throws an error if the data isn't large enough to interpolate with
        with self.assertRaises(ValueError):
            interpolate_dataset_on_lat_lon(raw_ds, 
                                           latitude_vals=[-2, -1, 0], 
                                           longitude_vals=[30, 31, 32, 33])

        ##
        # Check interpolation to smaller grid with era5

    def test_imerg_fps(self):
        
        fps = get_imerg_filepaths(2019, 1, 1, 0, file_ending='.nc')
        
        self.assertEqual(fps[0].split('/')[-1], '3B-HHR.MS.MRG.3IMERG.20190101-S000000-E002959.0000.V06B.nc')
        self.assertEqual(fps[1].split('/')[-1], '3B-HHR.MS.MRG.3IMERG.20190101-S003000-E005959.0030.V06B.nc')
        
    def test_load_imerg(self):

        year = 2018
        month = 12
        day = 30
        hour = 17
        latitude_vals = [0, 0.1, 0.2]
        longitude_vals = [33, 34]

        ds = load_imerg_raw(year=year, month=month, day=day, hour=hour,
                            latitude_vals=latitude_vals,
                            longitude_vals=longitude_vals,
                            imerg_data_dir=imerg_folder)
        
        self.assertIsInstance(ds, xr.Dataset)
        
        # Check that dimensions are ordered properly
        self.assertListEqual(list(ds['precipitationCal'].dims), ['lat', 'lon'])
        self.assertEqual(ds['precipitationCal'].values.shape, (len(latitude_vals), len(longitude_vals)))
        
        # this also checks that the longitude values are in ascending order
        testing.assert_allclose(ds.lat.values, np.array([0.05, 0.15, 0.25]), atol=1e-7)
        testing.assert_allclose(ds.lon.values, np.array([33.05, 34.05]), atol=1e-7)
        
        # Check for NaNs
        self.assertFalse(np.isnan(ds['precipitationCal']).any())
        
        with self.assertRaises(ValueError):
            load_imerg_raw(year=year, month=month, day=day, hour=hour,
                            latitude_vals=latitude_vals,
                            longitude_vals=[330.05, 330.15],
                            imerg_data_dir=imerg_folder)

    # def test_era5_stats(self):

    #     var_name_lookup = VAR_LOOKUP_ERA5
    #     for v in var_name_lookup:
    #         stats = get_era5_stats(v, year=2018, era_data_dir=constants_path,
    #                                longitude_vals=[33, 34], latitude_vals=[0, 1])

    #         self.assertIsInstance(stats, dict)
    #         # Make sure negative velocities handled correctly
    #         self.assertGreater(stats['min'] + 1e-16, 0)
    #         self.assertGreater(stats['max'], 0)
    #         self.assertGreater(stats['std'], 0)
            
    # def test_ifs_stats(self):

    #     var_name_lookup = VAR_LOOKUP_IFS
    #     for v in var_name_lookup:
    #         stats = get_ifs_stats(v, year=2017,
    #                                longitude_vals=[33, 34], latitude_vals=[0, 1],
    #                                output_dir=self.temp_stats_dir.name)

    #         self.assertIsInstance(stats, dict)
    #         # Make sure negative velocities handled correctly
    #         self.assertGreater(stats['min'] + 1e-16, 0)
    #         self.assertGreater(stats['max'], 0)
    #         self.assertGreater(stats['std'], 0)

    def test_load_fcst_stack(self):
        
        longitude_vals = [33, 34]
        latitude_vals = [0, 1]

        # Check it works with IFS
        ifs_stack = load_fcst_stack('ifs', all_ifs_fields, '20170705', 4, fcst_dir=ifs_path,
                                    norm=True, latitude_vals=latitude_vals, longitude_vals=longitude_vals,
                                    constants_dir=constants_path)
        ifs_shape = ifs_stack.shape
        self.assertEqual(ifs_shape[2], len(all_ifs_fields))
        self.assertFalse(np.isnan(ifs_stack).any())

    def test_load_fcst_radar_batch(self):
        
        longitude_vals = [33, 34]
        latitude_vals = [0, 1]

        # TODO: get NIMROD data sample to enable this test to work with IFS + NIMROD
        # Check it works with IFS: 

        ifs_batch_dates = ['20170704'] 
        ifs_input_batch, imerg_batch = load_fcst_radar_batch(ifs_batch_dates, fcst_data_source='ifs', obs_data_source='imerg', fcst_fields=all_ifs_fields, 
                                          fcst_dir=ifs_path, obs_data_dir=imerg_folder, latitude_range=latitude_vals,
                                          longitude_range=longitude_vals, constants_dir=constants_path,
                                          constants=True, hour=17, norm=False)
        self.assertEqual(len(ifs_input_batch), 2)
        self.assertEqual(len(ifs_input_batch[0]), len(ifs_batch_dates))
        self.assertEqual(len(ifs_input_batch[1]), len(ifs_batch_dates))
        
        self.assertFalse(np.isnan(ifs_input_batch[0]).any())
        self.assertFalse(np.isnan(ifs_input_batch[1]).any())
        self.assertFalse(np.isnan(imerg_batch).any())

    def test_get_dates(self):
        data_paths = {'GENERAL': {
                            'IFS': str(data_folder / 'IFS'),
                            'IMERG': str(data_folder / 'IMERG/half_hourly/final'),
                            'ERA5': str(data_folder / 'ERA5'),
                            'CONSTANTS': str(data_folder / 'constants')}}
        dates = get_dates(2017, obs_data_source='imerg', fcst_data_source='ifs',
              data_paths=data_paths, hour=17)
        self.assertListEqual(['20170704'], dates)
       
    def test_file_exists(self):

        data_paths = {'GENERAL': {
                            'IFS': ifs_path,
                            'IMERG': imerg_folder,
                            'ERA5': era5_path,
                            'CONSTANTS': constants_path}}
        
        # ## ERA5
        era5_file_exists = file_exists(data_source='era5', year=2018, month=12, day=31,
                                       data_paths=data_paths)
        self.assertTrue(era5_file_exists)
        
        era5_file_exists = file_exists(data_source='era5', year=2001, month=1, day=31,
                                       data_paths=data_paths)
        self.assertFalse(era5_file_exists)
        
        ## IFS
        ifs_file_exists = file_exists(data_source='ifs', year=2017, month=7, day=4,
                                       data_paths=data_paths)
        self.assertTrue(ifs_file_exists)
        
        ifs_file_exists = file_exists(data_source='ifs', year=2017, month=1, day=31,
                                       data_paths=data_paths)
        self.assertFalse(ifs_file_exists)
        
        # ## IMERG
        imerg_file_exists = file_exists(data_source='imerg', year=2017, month=7, day=4,
                                       data_paths=data_paths, hour=17)
        self.assertTrue(imerg_file_exists)
        
        imerg_file_exists = file_exists(data_source='imerg', year=2018, month=1, day=31,
                                       data_paths=data_paths)
        self.assertFalse(imerg_file_exists)
        
if __name__ == '__main__':
    unittest.main()
