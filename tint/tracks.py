import copy
import datetime

import numpy as np
import pandas as pd
import xarray as xr
import pickle
import warnings

from .grid_utils import get_grid_size, get_radar_info, extract_grid_data
from .helpers import Record, Counter
from .phase_correlation import get_global_shift
from .matching import get_pairs
from .objects import init_current_objects, update_current_objects
from .objects import get_object_prop, write_tracks
from .objects import post_tracks, get_system_tracks


class Tracks(object):
    """Determine storm tracks from list of pyart grids. """

    def __init__(self, params={}, field='reflectivity', b_path=None):

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
            # Gaussian smoothing parameter used in peak detection.
            'ISO_SMOOTH': 3,  # pixels
            # Altitude in m for calculating global shift.
            'GS_ALT': 1500,  # m
            # Layers to identify objects within.
            'LEVELS': np.array(
                [[3000, 3500], [3500, 7500], [7500, 10000]]),  # m
            # Minimum size of objects in each layer.
            'MIN_SIZE': [40, 400, 800],  # square km
            # Thresholds for object identification.
            'FIELD_THRESH': ['convective', 20, 15],  # DbZ or 'convective'.
            # Threshold to define a cell as isolated.
            'ISO_THRESH': [10, 10, 10],  # DbZ
            # Interval in the above array used for tracking.
            'TRACK_INTERVAL': 0,
            # Threshold for identifying "cells".
            'CELL_THRESH': 25,  # DbZ
            # Altitude to start tracking updrafts.
            'CELL_START': 3000,  # m
            # Whether to collect object based rainfall totals
            'RAIN': False,  # bool
            # Whether to save the grids associated with the accumulated totals
            'SAVE_RAIN': False}  # bool

        if b_path is not None:
            with open(b_path, 'rb') as f:
                b_ind_set = pickle.load(f)
            self.params['BOUNDARY_GRID_CELLS'] = b_ind_set
        else:
            self.params['BOUNDARY_GRID_CELLS'] = set()

        # Load user specified parameters.
        for p in params:
            if p in self.params:
                self.params[p] = params[p]
            else:
                keys = ', '.join([p for p in self.params])
                msg = '{} not a valid parameter. Choices are {}'
                msg = msg.format(p, keys)
                warnings.showwarning(msg, RuntimeWarning, 'tracks.py', 143)

        self.field = field  # Field used for tracking.
        self.grid_size = None
        self.radar_info = None
        self.last_grid = None
        self.counter = None
        self.record = None
        self.current_objects = None
        self.tracks = pd.DataFrame()
        self.data_dic = {}

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
            grid_obj2 = next(grids)
            data = extract_grid_data(
                grid_obj2, self.field, self.grid_size, self.params)
            for i in range(len(data_names)):
                data_dic[data_names[i] + '_new'] = data[i]
            print('Skipping erroneous grid.')
        return grid_obj2, data_dic

    def get_tracks(self, grids):
        """Obtains tracks given a list of pyart grid objects."""
        start_time = datetime.datetime.now()

        time_str = str(start_time)[0:-7]
        time_str = time_str.replace(" ", "_").replace(":", "_")
        time_str = time_str.replace("-", "")

        if self.record is None:
            # tracks object being initialized
            grid_obj2 = next(grids)
            self.grid_size = get_grid_size(grid_obj2)
            self.radar_info = get_radar_info(grid_obj2)
            self.counter = Counter()
            self.record = Record(grid_obj2)
        else:
            # tracks object being updated
            grid_obj2 = self.last_grid
            self.tracks.drop(self.record.scan + 1)  # last scan is overwritten

        if self.current_objects is None:
            new_rain = True
        else:
            new_rain = False

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

        while grid_obj2 is not None:
            # Set current grid equal to new grid from last iteration.
            grid_obj1 = grid_obj2
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
                grid_obj2 = next(grids)
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

        self = post_tracks(self)
        self = get_system_tracks(self)

        self.__load()
        time_elapsed = datetime.datetime.now() - start_time
        print('Time elapsed:', np.round(time_elapsed.seconds/60, 1), 'minutes')
        return
