import copy
import datetime

import numpy as np
import pandas as pd
import pickle
import warnings
import tempfile
import shutil

from tint.grid_utils import get_grid_size, get_radar_info, extract_grid_data
from tint.grid_utils import parse_grid_datetime
from tint.tracks_helpers import Record, Counter
from tint.phase_correlation import get_global_shift
from tint.matching import get_pairs
from tint.objects import init_current_objects, update_current_objects
from tint.objects import get_object_prop, write_tracks
from tint.objects import post_tracks, get_system_tracks, classify_tracks
from tint.objects import get_exclusion_categories
import tint.process_ERA5 as ERA5
import tint.process_WRF as WRF
import tint.process_ACCESS as ACC
import tint.process_operational_radar as po


class Tracks(object):
    """Determine storm tracks from list of pyart grids. """

    def __init__(self, params={}, field='reflectivity'):

        # Parameter Defaults
        self.params = {
            # Margin for object matching: does not affect flow vectors.
            'SEARCH_MARGIN': 50000,  # m. CPOL cells 2500 m x 2500 m.
            # Margin around object for phase correlation.
            'FLOW_MARGIN': 40000,  # m
            # Maximum allowable global shift magnitude.
            'MAX_FLOW_MAG': 60,  # metres per second.
            # Maximum allowable disparity value.
            'MAX_DISPARITY': 999,  # float
            # Maximum magnitude of shift difference.
            'MAX_SHIFT_DISP': 60,  # metres per second.
            # Maximum magnitude of shift difference alternative.
            'MAX_SHIFT_DISP_ALT': 25,  # metres per second.
            # Gaussian smoothing parameter used in peak detection.
            'ISO_SMOOTH': 3,  # pixels
            # Altitude in m for calculating global shift.
            'GS_ALT': 2000,  # m
            # Layers to identify objects within.
            'LEVELS': np.array(
                [[500, 3500], [3500, 7500], [7500, 10000]]),  # m
            'WIND_LEVELS': None,
            # Minimum size of objects in each layer.
            'MIN_SIZE': [80, 400, 800],  # square km
            # Thresholds for object identification.
            'FIELD_THRESH': ['convective', 20, 15],  # DbZ or 'convective'.
            # Threshold to define a cell as isolated.
            'ISO_THRESH': [10, 10, 10],  # DbZ
            # Interval in the above array used for tracking.
            'TRACK_INTERVAL': 0,
            # Threshold for identifying convective "cells".
            'CELL_THRESH': 25,  # DbZ
            # Altitude to start tracking updrafts.
            'CELL_START': 3000,  # m
            # Whether to collect object based rainfall totals
            'RAIN': False,  # bool
            # Whether to save the grids associated with the accumulated totals
            'SAVE_RAIN': False,  # bool
            # Whether to include ERA5 derived fields
            'AMBIENT': None,  # None, 'WRF' or 'ERA5'
            # ERA5 base directory
            'AMBIENT_BASE_DIR': None,  # str or None
            # Time interval between radar scans
            'DT': 10,  # minutes
            # Classification thresholds
            'CLASS_THRESH': {
                'OFFSET_MAG': 10000,  # metres
                'SHEAR_MAG': 2,  # m/s
                'VEL_MAG': 5,  # m/s
                'REL_VEL_MAG': 2,  # m/s
                'ANGLE_BUFFER': 10},  # degrees
            'EXCL_THRESH': {
                'SMALL_AREA': 500,  # km^2
                'LARGE_AREA': 50000,  # km^2
                'BORD_THRESH': 0.001,  # Ratio border pixels to total pixels
                'MAJOR_AXIS_LENGTH': 100,  # km
                'AXIS_RATIO': 3,
                'DURATION': 30},  # minutes
            'INPUT_TYPE': 'GRIDS',
            'REMOTE': False,
            'AMBIENT_TIMESTEP': 1,  # hours
            'SAVE_DIR': '~/Documents',
            'REFERENCE_GRID_FORMAT': 'ODIM',
            'RESET_NEW_DAY': False,
            'REFERENCE_RADAR': 63}

        # Load user specified parameters.
        for p in params:
            if p in self.params:
                self.params[p] = params[p]
            else:
                keys = ', '.join([p for p in self.params])
                msg = '{} not a valid parameter. Choices are {}'
                msg = msg.format(p, keys)
                warnings.showwarning(msg, RuntimeWarning, 'tracks.py', 143)

        if self.params['WIND_LEVELS'] is None:
            self.params['WIND_LEVELS'] = self.params['LEVELS']

        self.field = field  # Field used for tracking.
        self.grid_size = None
        self.radar_info = None
        self.last_grid = None
        self.counter = None
        self.record = None
        self.current_objects = None
        self.tracks = pd.DataFrame()
        self.data_dic = {}
        self.ACCESS_refl = None
        self.grid_obj_day = None
        self.file_list = None

        self.reference_grid = None
        if self.params['INPUT_TYPE'] in ['ACCESS_DATETIMES', 'OPER_DATETIMES']:
            radar_num = self.params['REFERENCE_RADAR']
            if self.params['REMOTE']:
                path = '/g/data/w40/esh563/reference_grids/'
                path += 'reference_grid_{}.h5'.format(radar_num)
            else:
                path = '/home/student.unimelb.edu.au/shorte1/Documents/'
                path += 'CPOL_analysis/reference_grid_{}.h5'.format(
                    radar_num)
            self.reference_grid = ACC.get_reference_grid(
                path, params['REFERENCE_GRID_FORMAT'])
        if self.params['INPUT_TYPE'] == 'OPER_DATETIMES':
            self.tmp_dir = tempfile.mkdtemp(dir=self.params['SAVE_DIR'])

        self.__saved_record = None
        self.__saved_counter = None
        self.__saved_objects = None

    def __save(self):
        """Saves deep copies of record, counter, and current_objects. """
        self.__saved_record = copy.deepcopy(self.record)
        self.__saved_counter = copy.deepcopy(self.counter)
        self.__saved_objects = copy.deepcopy(self.current_objects)

    def __load(self):
        """Loads saved copies of record, counter, and current_objects. If new
        tracks are appended to existing tracks via the get_tracks method, the
        most recent scan prior to the addition must be overwritten to link up
        with the new scans. """
        self.record = self.__saved_record
        self.counter = self.__saved_counter
        self.current_objects = self.__saved_objects

    def save_netcdf(self, filename):
        # Convert to xarray
        ds = self.tracks.reset_index(level=['scan'], drop=True).to_xarray()
        # Bools must be converted to ints
        # Multidimensional attributes not allowed
        # Cannot have arrays as elements of dataframe,
        # e.g. mergers, parents not allowed
        # Append metadata
        return ds

    def format_next_grid(self, grids):
        if self.params['INPUT_TYPE'] == 'GRIDS':
            new_grid = next(grids)
        elif self.params['INPUT_TYPE'] == 'ACCESS_DATETIMES':
            new_datetime = next(grids)
            self.ACCESS_refl, new_grid = ACC.update_ACCESS_C(
                new_datetime, self.ACCESS_refl, self.reference_grid,
                self.params['REMOTE'])
        elif self.params['INPUT_TYPE'] == 'OPER_DATETIMES':
            new_datetime = next(grids)
            new_grid, self.file_list = po.get_grid(
                new_datetime, self.params,
                self.reference_grid, self.tmp_dir, self.file_list)
        return new_grid

    def get_next_grid(self, grid_obj2, grids, data_dic):
        """Find the next nonempty grid."""
        data = extract_grid_data(
            grid_obj2, self.field, self.grid_size, self.params)
        data_names = ['refl', 'rain', 'frames', 'cells', 'stein_class']
        for i in range(len(data_names)):
            data_dic[data_names[i] + '_new'] = data[i]
        # Skip grids that are artificially zero
        while (
                np.max(data_dic['refl']) > 30
                and np.max(data_dic['refl_new']) == 0):
            grid_obj2 = self.format_next_grid(grids)
            data = extract_grid_data(
                grid_obj2, self.field, self.grid_size, self.params)
            for i in range(len(data_names)):
                data_dic[data_names[i] + '_new'] = data[i]
            print('Skipping erroneous grid.')
        return grid_obj2, data_dic

    def get_boundary_inds(self, grid, b_path=None):
        if b_path is not None:
            with open(b_path, 'rb') as f:
                b_ind_set = pickle.load(f)
        else:
            nx = len(grid.x['data'])
            ny = len(grid.y['data'])
            s1 = [(0, i) for i in range(nx)]
            s2 = [(ny, i) for i in range(nx)]
            s3 = [(i, 0) for i in range(ny)]
            s4 = [(i, nx) for i in range(ny)]
            b_ind_set = set(s1+s2+s3+s4)
        self.params['BOUNDARY_GRID_CELLS'] = b_ind_set

    def test_new_day(self, grid_obj1, grid_obj2):
        grid_day1 = np.datetime64(str(parse_grid_datetime(grid_obj1))[:10])
        grid_day2 = np.datetime64(str(parse_grid_datetime(grid_obj2))[:10])

        new_day = grid_day1 > self.grid_obj_day
        next_day_new = grid_day2 > self.grid_obj_day

        if new_day:
            self.grid_obj_day = grid_day1
            next_day_new = False
            print('New day at {}. Resetting objects.'.format(
                grid_day1))

        return next_day_new

    def get_tracks(self, grids, b_path=None):
        """Obtains tracks given a list of pyart grid objects."""
        start_time = datetime.datetime.now()

        time_str = str(start_time)[0:-7]
        time_str = time_str.replace(" ", "_").replace(":", "_")
        time_str = time_str.replace("-", "")

        if self.record is None:
            # tracks object being initialized
            grid_obj2 = self.format_next_grid(grids)
            self.grid_size = get_grid_size(grid_obj2)
            self.radar_info = get_radar_info(grid_obj2)
            self.counter = Counter()
            self.record = Record(grid_obj2)
            self.grid_obj_day = np.datetime64(
                str(parse_grid_datetime(grid_obj2))[:10])
        else:
            # tracks object being updated
            grid_obj2 = self.last_grid
            self.grid_obj_day = np.datetime64(
                str(parse_grid_datetime(grid_obj2))[:10])
            self.tracks.drop(self.record.scan + 1)  # last scan is overwritten

        if self.current_objects is None:
            new_rain = True
        else:
            new_rain = False

        self.get_boundary_inds(grid_obj2, b_path)

        data_dic = {}

        data = extract_grid_data(
            grid_obj2, self.field, self.grid_size, self.params)
        data_names = ['refl', 'rain', 'frames', 'cells', 'stein_class']
        for i in range(len(data_names)):
            data_dic[data_names[i] + '_new'] = data[i]
        data_dic['frame_new'] = data_dic['frames_new'][
            self.params['TRACK_INTERVAL']]
        # For first grid, initialise the "current" frame to the "new" frame
        data_dic['frame'] = copy.deepcopy(data_dic['frame_new'])

        # Get Ambient
        if self.params['AMBIENT'] == 'ERA5':
            ambient_all, ambient_interp = ERA5.init_ERA5(
                grid_obj2, self.params)
            data_dic['ambient_interp'] = ambient_interp
        elif self.params['AMBIENT'] == 'WRF':
            ambient_all, ambient_interp = WRF.init_WRF(grid_obj2, self.params)
            data_dic['ambient_interp'] = ambient_interp
        elif self.params['AMBIENT'] == 'ACCESS':
            current_datetime = parse_grid_datetime(grid_obj2)
            current_datetime = np.datetime64(current_datetime)
            ambient_interp = ACC.init_ACCESS_G(
                current_datetime, self.reference_grid, self.params['REMOTE'])
            data_dic['ambient_interp'] = ambient_interp

        while grid_obj2 is not None:
            # Set current grid equal to new grid from last iteration.
            grid_obj1 = copy.deepcopy(grid_obj2)
            if not new_rain:
                # Set old old frame equal to current frame from last iteration.
                data_dic['frame_old'] = copy.deepcopy(data_dic['frame'])
            else:
                # New rain, so no relevant old frame
                data_dic['frame_old'] = None
            # Set current datasets equal to new datasets from last iteration.
            for n in data_names + ['frame']:
                data_dic[n] = data_dic[n + '_new']
            try:
                grid_obj2 = self.format_next_grid(grids)
                grid_obj2, data_dic = self.get_next_grid(
                    grid_obj2, grids, data_dic)
            except StopIteration:
                grid_obj2 = None

            if grid_obj2 is not None:

                data_dic['frame_new'] = data_dic['frames_new'][
                    self.params['TRACK_INTERVAL']]
                self.record.update_scan_and_time(grid_obj1, grid_obj2)

                # Check for gaps in record. If gap exists, tell tint to start
                # define new objects in current grid.
                # Allow a couple of missing scans
                if (
                        self.record.interval is not None
                        and self.record.interval.seconds > 1700):
                    message = 'Time discontinuity at {}.'.format(
                        self.record.time)
                    print(message)
                    new_rain = True
                    self.current_objects = None
                # current_datetime = parse_grid_datetime(grid_obj1)

                if self.params['RESET_NEW_DAY']:
                    next_day_new = self.test_new_day(grid_obj1, grid_obj2)
                    if next_day_new:
                        new_rain = True
                        print('Scan {} last before new day.'.format(
                            self.record.scan))
                        self.current_objects = None
                        continue

            else:
                # setup to write final scan
                self.__save()
                self.last_grid = grid_obj1
                self.record.update_scan_and_time(grid_obj1)
                data_dic['refl_new'] = None
                data_dic['frame_new'] = np.zeros_like(data_dic['frame'])
                data_dic['frames_new'] = np.zeros_like(data_dic['frames'])

            if np.max(data_dic['frame']) == 0:
                new_rain = True
                print('No objects found in scan {}.'.format(self.record.scan))
                self.current_objects = None
                continue

            # Update ambient winds
            if self.params['AMBIENT'] == 'ERA5':
                ambient_all, ambient_interp = ERA5.update_ERA5(
                    grid_obj1, self.params, ambient_all, ambient_interp)
                data_dic['ambient_interp'] = ambient_interp
            elif self.params['AMBIENT'] == 'WRF':
                ambient_all, ambient_interp = WRF.update_WRF(
                    grid_obj1, self.params, ambient_all, ambient_interp)
                data_dic['ambient_interp'] = ambient_interp
            elif self.params['AMBIENT'] == 'ACCESS':
                try:
                    current_datetime = parse_grid_datetime(grid_obj1)
                    current_datetime = np.datetime64(current_datetime)
                    ambient_interp = ACC.update_ACCESS_G(
                        ambient_interp, self.reference_grid,
                        current_datetime, self.params['REMOTE'])
                    data_dic['ambient_interp'] = ambient_interp
                except OSError:
                    print('Could not load ambient winds at {}'.format(
                        current_datetime))
                    print('Skipping.')
                    new_rain = True
                    self.current_objects = None
                    continue

            global_shift = get_global_shift(
                data_dic['refl'], data_dic['refl_new'], self.params)
            pairs, obj_merge_new, u_shift, v_shift = get_pairs(
                data_dic, global_shift, self.current_objects, self.record,
                self.params)

            if new_rain:
                # first nonempty scan after a period of empty scans
                self.current_objects, self.counter = init_current_objects(
                    data_dic, pairs, self.counter,
                    self.record.interval.total_seconds(), self.params)
                new_rain = False
            else:
                self.current_objects, self.counter = update_current_objects(
                    data_dic, pairs, self.current_objects, self.counter,
                    obj_merge, self.record.interval.total_seconds(),
                    self.params, grid_obj1)
            obj_merge = obj_merge_new

            obj_props = get_object_prop(
                data_dic, grid_obj1, u_shift, v_shift,
                self.field, self.record, self.params, self.current_objects)
            self.record.add_uids(self.current_objects)
            self.tracks = write_tracks(
                self.tracks, self.record, self.current_objects, obj_props)
            del global_shift, pairs, obj_props

        del grid_obj1

        if self.params['INPUT_TYPE'] == 'OPER_DATETIMES':
            shutil.rmtree(self.tmp_dir)

        self = post_tracks(self)
        self = get_system_tracks(self)
        self = classify_tracks(self)
        self = get_exclusion_categories(self)

        self.__load()
        time_elapsed = datetime.datetime.now() - start_time
        print('Time elapsed:', np.round(time_elapsed.seconds/60, 1), 'minutes')
        return
