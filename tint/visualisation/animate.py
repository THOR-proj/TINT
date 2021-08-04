import os
import shutil
import numpy as np
import copy

import tint.visualisation.figures as fig
from tint.grid_utils import parse_grid_datetime


def get_times(params, tracks, start_datetime, end_datetime):
    if params['uid_ind'] is not None:
        tmp_tracks = tracks.tracks.xs(
            (params['uid_ind'], 0), level=('uid', 'level'))
        tmp_tracks.index = tmp_tracks.index.droplevel('scan')
        date_times = tmp_tracks.index.values
        if start_datetime is not None and end_datetime is not None:
            if (
                    date_times[0] <= start_datetime
                    and date_times[-1] >= end_datetime):
                date_times = np.arange(
                    start_datetime, end_datetime, np.timedelta64(10, 'm'))
    else:
        date_times = np.arange(
            start_datetime, end_datetime, np.timedelta64(10, 'm'))
    return date_times


def check_times(grids, date_times, params):
    grid = next(grids)
    grid_time = np.datetime64(parse_grid_datetime(grid))
    if grid_time > date_times[0]:
        ind = np.argwhere(date_times == grid_time)
        if not ind:
            print('Object occurs before grids provided. Aborting')
        print('Grids start after object initialises.')
        date_times = date_times[ind:]
    else:
        counter = 0
        while grid_time < date_times[0]:
            grid = next(grids)
            grid_time = np.datetime64(parse_grid_datetime(grid))
            counter += 1
        params['winds_fn'] = params['winds_fn'][counter:]
    return grid, grid_time, date_times, params


def animate(
        tracks, grids, params, type='vertical_cross_section', fps=2,
        start_datetime=None, end_datetime=None, keep_frames=False):
    """Creates gif animation of tracked objects. """

    params = fig.check_params(params)
    styles = {
        'vertical_cross_section': fig.vertical_cross_section,
        'horizontal_cross_section': fig.horizontal_cross_section,
        'object': fig.object}
    anim_func = styles[type]

    date_times = get_times(params, tracks, start_datetime, end_datetime)

    grid, grid_time, date_times, params = check_times(
        grids, date_times, params)
    fn = type
    if params['uid_ind'] is not None:
        fn += '_{}'.format(params['uid_ind'])

    tmp_params = copy.deepcopy(params)
    tmp_params['save_dir'] += '/{}_frames'.format(fn)
    os.makedirs(tmp_params['save_dir'], exist_ok=True)

    for i in range(len(date_times)):
        print('Generating frame {}'.format(date_times[i]))
        if params['winds']:
            tmp_params['winds_fn'] = params['winds_fn'][i]
        anim_func(
            tracks, grid, tmp_params, date_time=date_times[i])
        grid = next(grids)

    make_gif_from_frames(tmp_params['save_dir'], tmp_params['save_dir'], fn, 2)


def make_gif_from_frames(frames_dir, dest_dir, basename, fps):
    print('Creating GIF - may take a few minutes.')
    os.chdir(frames_dir)
    delay = round(100 / fps)
    command = "convert -delay {} *.png -loop 0 {}.gif"
    os.system(command.format(str(delay), basename))
