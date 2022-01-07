import gc
import numpy as np
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import pyart
from pyart.core.transforms import cartesian_to_geographic
import warnings
from matplotlib import rcParams
import xarray as xr
import pandas as pd
import copy

from tint.grid_utils import get_grid_alt, parse_grid_datetime
import tint.visualisation.horizontal_helpers as hh
import tint.visualisation.vertical_helpers as vh


def init_fonts(user_params):
    # Initialise fonts
    rcParams.update({'font.family': 'serif'})
    rcParams.update({'font.serif': 'Liberation Serif'})
    rcParams.update({'mathtext.fontset': 'dejavuserif'})
    rcParams.update({'font.size': user_params['fontsize']})


def check_params(user_params):

    exclusions = [
        'small_area', 'large_area', 'intersect_border',
        'intersect_border_convective', 'duration_cond',
        'small_velocity', 'small_offset']
    params = {
        'uid_ind': None, 'cell_ind': None, 'box_rad': 0.75,
        'line_coords': False, 'center_cell': False, 'label_splits': False,
        'legend': True, 'winds': False, 'winds_fn': None,
        'colorbar_flag': True, 'direction': None, 'line_average': False,
        'crosshair': True, 'streamplot': True, 'dpi': 200, 'save_dir': None,
        'relative_winds': False, 'data_fn': None,
        'load_line_coords_winds': None, 'save_ds': False, 'alt': 3000,
        'fontsize': 20, 'leg_loc': 2, 'system_winds': ['shift'],
        'label_mergers': False, 'screen': True, 'label_type': 'velocities',
        'exclusions': exclusions, 'boundary': True}
    for p in user_params:
        if p in params:
            params[p] = user_params[p]
        else:
            keys = ', '.join([p for p in params])
            msg = '{} not a valid parameter. Choices are {}'
            msg = msg.format(p, keys)
            warnings.showwarning(msg, RuntimeWarning, 'figures.py', 26)

    return params


def get_bounding_box(tracks, grid, date_time, params):
    lon_box, lat_box, x, y = vh.get_center_coords(
        tracks, grid, params, date_time)

    box = np.array([-1 * params['box_rad'], params['box_rad']])
    box_x_bounds = (lon_box) + box
    box_y_bounds = (lat_box) + box
    return lon_box, lat_box, box_x_bounds, box_y_bounds


def add_crosshair(
        tracks, grid, date_time, params, ax, display, lon_box, lat_box):

    if params['uid_ind'] is not None:
        tmp_tracks = tracks.tracks.xs(
            (params['uid_ind'], date_time, 0), level=('uid', 'time', 'level'))
        if params['center_cell'] and (params['cell_ind'] is not None):
            cell = tmp_tracks.iloc[0]['cells'][params['cell_ind']]
            x_cell = grid.x['data'][np.array(cell)[0, 2]]
            y_cell = grid.y['data'][np.array(cell)[0, 1]]
            projparams = grid.get_projparams()
            lon_cell, lat_cell = cartesian_to_geographic(
                x_cell, y_cell, projparams)
            # Set crosshair to be at cell location
            [crosshair_lon, crosshair_lat] = [lon_cell, lat_cell]
        else:
            # Set crosshair to be at object location
            [crosshair_lon, crosshair_lat] = [lon_box, lat_box]
    else:
        # Set crosshair to be at radar position
        radar_lon = tracks.radar_info['radar_lon']
        radar_lat = tracks.radar_info['radar_lat']
        [crosshair_lon, crosshair_lat] = [radar_lon, radar_lat]
    # Plot reflectivity and crosshairs
    if params['line_coords']:
        # Plot along line cross hair
        m = np.tan(np.deg2rad(tmp_tracks['orientation'].iloc[0]))
        c = crosshair_lat - m*crosshair_lon
        crosshair_lat_0 = c
        crosshair_lat_1 = m * 360 + c
        ax.plot(
            [0, 360], [crosshair_lat_0, crosshair_lat_1], '--r', linewidth=2.0,
            zorder=2)
        # Plot across line cross hair
        m = - 1/m
        c = crosshair_lat - m*crosshair_lon
        crosshair_lat_0 = c
        crosshair_lat_1 = m * 360 + c
        ax.plot(
            [0, 360], [crosshair_lat_0, crosshair_lat_1], '--r', linewidth=2.0,
            zorder=2)
    else:
        display.plot_crosshairs(lon=crosshair_lon, lat=crosshair_lat)


def init_cross_section(
        tracks, grid, params, alt, fig, ax, date_time, figsize=(12, 12)):
    params = check_params(params)

    if alt is None:
        alt = tracks.params['LEVELS'][0, 0]

    time_ind = tracks.tracks.index.get_level_values('time')
    time_ind = np.array(time_ind).astype('datetime64[m]')
    grid_time = np.datetime64(parse_grid_datetime(grid))
    grid_time = grid_time.astype('datetime64[m]')
    if date_time is None:
        date_time = grid_time

    projection = ccrs.PlateCarree()

    # Initialise fig and ax if not passed as arguments
    if fig is None:
        init_fonts(params)
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(1, 1, 1, projection=projection)

    display = pyart.graph.GridMapDisplay(grid)
    alt_ind = get_grid_alt(tracks.grid_size, alt)

    cmap = pyart.graph.cm_colorblind.HomeyerRainbow
    [vmin, vmax] = [-8, 64]
    return [
        params, alt, fig, ax, date_time, display, alt_ind, cmap,
        vmin, vmax, projection]


def horizontal_cross_section(
        tracks, grid, params={}, alt=None, fig=None, ax=None, date_time=None):

    init_cs = init_cross_section(
        tracks, grid, params, alt, fig, ax, date_time, figsize=(8, 6))
    [
        params, alt, fig, ax, date_time, display, alt_ind, cmap,
        vmin, vmax, projection] = init_cs
    display.plot_grid(
        tracks.field, level=alt_ind, vmin=vmin, vmax=vmax, cmap=cmap,
        transform=projection, ax=ax, colorbar_label='Reflectivity [DbZ]',
        colorbar_flag=params['colorbar_flag'], zorder=1)

    ax.set_title('Altitude {} m'.format(alt))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    box = get_bounding_box(tracks, grid, date_time, params)
    if params['crosshair']:
        add_crosshair(
            tracks, grid, date_time, params, ax, display, box[0], box[1])

    tracks_times = tracks.tracks.index.get_level_values('time')
    if date_time in tracks_times:
        lgd_han = hh.add_tracked_objects(
            tracks, grid, date_time, params, ax, alt)
    else:
        lgd_han = []

    if params['winds']:
        lgd_winds = hh.add_winds(ax, tracks, grid, date_time, alt, params)
        if lgd_winds is not None:
            lgd_han.append(lgd_winds)

    if params['uid_ind'] is not None:
        ax.set_xlim(box[2][0], box[2][1])
        ax.set_ylim(box[3][0], box[3][1])

    if params['legend'] and date_time in tracks_times:
        legend = plt.legend(
            handles=lgd_han, loc='lower center', bbox_to_anchor=(0.5, -0.35),
            ncol=2, fancybox=True, shadow=True)
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor((1, 1, 1, 1))

    # Save frame and cleanup
    if params['save_dir'] is not None:
        plt.savefig(
            '{}/{}m_cross_{}.png'.format(
                params['save_dir'], params['alt'], date_time),
            bbox_inches='tight', dpi=params['dpi'], facecolor='w')
        plt.close(fig)

    return True


def vertical_cross_section(
        tracks, grid, params={}, alt=None, fig=None, ax=None, date_time=None):

    init_cs = init_cross_section(
        tracks, grid, params, alt, fig, ax, date_time, figsize=(15, 3.5))
    [
        params, alt, fig, ax, date_time, display, alt_ind, cmap,
        vmin, vmax, projection] = init_cs
    lon, lat, x, y = vh.get_center_coords(tracks, grid, params, date_time)
    x_lim = (x + np.array([-60000, 60000])) / 1000
    y_lim = (y + np.array([-60000, 60000])) / 1000
    lgd_han = []
    [dz, dy, dx] = tracks.record.grid_size

    ds = vh.format_pyart(grid)
    variables = ['reflectivity']
    if params['winds']:
        winds_ds = xr.open_dataset(params['winds_fn'])
        winds_ds = winds_ds[['U', 'V', 'W']]
        ds = xr.merge([ds, winds_ds])
        variables += ['U', 'V', 'W']

    if params['line_coords']:
        ds, tmp_tracks = vh.setup_line_coords(
            ds, tracks, params, date_time, variables)

    print('Adding reflectivity.')
    if params['direction'] == 'lat':
        display.plot_latitude_slice(
            tracks.field, lon=lon, lat=lat, colorbar_flag=True, edges=False,
            vmin=vmin, vmax=vmax, mask_outside=False, cmap=cmap, ax=ax,
            colorbar_label='Reflectivity [DbZ]')
        h_lim = x_lim
        t_string = 'Latitude Cross Section'
        x_label = 'West East Distance From Radar [km]'

    elif params['direction'] == 'lon':
        display.plot_longitude_slice(
            tracks.field, lon=lon, lat=lat, colorbar_flag=True, edges=False,
            vmin=vmin, vmax=vmax, mask_outside=False, cmap=cmap, ax=ax,
            colorbar_label='Reflectivity [DbZ]')
        h_lim = y_lim
        t_string = 'Longitude Cross Section'
        x_label = 'North South Distance From Radar [km]'

    if params['direction'] == 'parallel':
        ds_plot = ds.sel(y=y, method='nearest').squeeze()
        extent = [
            ds_plot.x[0] / 1000, ds_plot.x[-1] / 1000,
            ds_plot.z[0] / 1000, ds_plot.z[-1] / 1000]
        cs = ax.imshow(
            ds_plot.reflectivity, vmin=vmin, vmax=vmax, cmap=cmap,
            interpolation='none', origin='lower', extent=extent)
        fig.colorbar(cs, ax=ax, label='Reflectivity [DbZ]')
        h_lim = x_lim
        t_string = 'Along Line Cross Section'
        x_label = 'Along Line Distance From Radar [km]'

    elif params['direction'] == 'perpendicular':
        ds_plot, t_string = vh.setup_perpendicular_coords(
            ds, x, tmp_tracks, params, dx)
        extent = [
            ds_plot.y[0] / 1000, ds_plot.y[-1] / 1000,
            ds_plot.z[0] / 1000, ds_plot.z[-1] / 1000]
        cs = ax.imshow(
            ds_plot.reflectivity, vmin=vmin, vmax=vmax, cmap=cmap,
            interpolation='none', origin='lower', extent=extent, zorder=1)
        fig.colorbar(cs, ax=ax, label='Reflectivity [DbZ]')
        h_lim = y_lim
        x_label = 'Line Perpendicular Distance From Radar [km]'

    print('Adding stratiform offset.')
    lgd_so, so_angle = vh.add_stratiform_offset(
        ax, tracks, grid, date_time, params)
    lgd_han.append(lgd_so)

    if params['winds']:
        print('Adding winds.')
        lgd_winds, sl_angle, w_max, n_angles, n_obs, max_count = vh.add_winds(
            ds, ax, tracks, grid, date_time, params)
        if params['streamplot']:
            lgd_han.append(lgd_winds)

    if params['center_cell']:
        lgd_cell = vh.add_cell(ax, tracks, grid, date_time, params)
        lgd_han.append(lgd_cell)

    if params['legend']:
        legend = plt.legend(
            handles=lgd_han, loc='lower center', bbox_to_anchor=(0.5, -0.5),
            ncol=2, fancybox=True, shadow=True)
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor((1, 1, 1, 1))

    lim_1 = h_lim[0] - h_lim[0] % 10
    lim_2 = h_lim[1] - h_lim[1] % 10
    ax.set_xlim(lim_1, lim_2)
    ax.set_xticks(np.arange(lim_1, lim_2, 10))
    ax.set_xticklabels(np.arange(lim_1, lim_2, 10).astype(int))
    ax.set_xlabel(x_label)
    ax.set_ylim(0, 20)
    ax.set_yticks(np.arange(0, 24, 4))
    ax.set_yticklabels(
        np.round((np.arange(0, 24, 4))))
    ax.set_ylabel('Distance Above Origin [km]')
    # ax.set_title(t_string)
    ax.set_aspect(2)

    # Save frame and cleanup
    if params['save_dir'] is not None:
        plt.savefig(
            '{}/{}_cross_{}.png'.format(
                params['save_dir'], params['direction'], date_time),
            bbox_inches='tight', dpi=params['dpi'], facecolor='w')
        plt.close()

    # Save frame and cleanup
    if params['data_fn'] is not None:
        print('Saving data.')
        save_tilt_data(
            sl_angle, w_max, n_angles, n_obs, max_count, so_angle,
            date_time, params)
        if not params['load_line_coords_winds'] and params['save_ds']:
            fn = '{}/{}_{}.nc'.format(
                params['save_dir'], params['uid_ind'], date_time)
            ds.to_netcdf(fn)

    del display

    return


def save_tilt_data(
        sl_angle, w_max, n_angles, n_obs, max_count, so_angle,
        date_time, params):
    df_dic = {
        'time': [date_time], 'uid': [params['uid_ind']],
        'streamline_angle': [sl_angle], 'w_max': [w_max],
        'n_angles': [n_angles], 'n_obs': [n_obs], 'max_count': [max_count],
        'stratiform_offset_angle': [so_angle]}
    df = pd.DataFrame(df_dic)
    df.set_index(['time', 'uid'], inplace=True)
    df.sort_index(inplace=True)

    try:
        dtypes = {
            'uid': str, 'streamline_angle': float,
            'stratiform_offset_angle': float, 'w_max': float,
            'n_angles': float, 'n_obs': float, 'max_count': float}
        old_df = pd.read_csv(
            '{}/{}.csv'.format(params['save_dir'], params['data_fn']),
            dtype=dtypes, parse_dates=['time'])

        old_df.set_index(['time', 'uid'], inplace=True)
        df = old_df.append(df)
        df.sort_index(inplace=True)
    except FileNotFoundError:
        print('Creating new CSV file to store angle data.')

    fn = '{}/{}.csv'.format(params['save_dir'], params['data_fn'])
    df.reset_index(inplace=True)
    df.to_csv(fn, index=False)

    return


def two_level(tracks, grid, params, date_time=None, alt1=None, alt2=None):

    params = check_params(params)
    if alt1 is None:
        alt1 = tracks.params['GS_ALT']
    if alt2 is None:
        alt2 = tracks.params['LEVELS'][-1][0]
    grid_time = np.datetime64(parse_grid_datetime(grid))
    grid_time = grid_time.astype('datetime64[m]')
    if date_time is None:
        date_time = grid_time

    init_fonts(params)
    projection = ccrs.PlateCarree()

    print('Generating figure for {}.'.format(str(date_time)))
    if grid_time != date_time:
        msg = 'grid_time {} does not match specified date_time {}. Aborting.'
        msg = msg.format(grid_time, date_time)
        print(msg)

    # Initialise figure
    fig = plt.figure(figsize=(22, 8))
    suptitle = 'Convective and Stratiform Cloud at ' + str(grid_time)
    fig.suptitle(suptitle)

    tmp_params = copy.deepcopy(params)
    tmp_params['colorbar_flag'] = False
    # Plot frame
    ax = fig.add_subplot(1, 2, 1, projection=projection)
    horizontal_cross_section(
        tracks, grid, params=tmp_params, alt=alt1, fig=fig, ax=ax,
        date_time=date_time)

    tmp_params['colorbar_flag'] = True
    ax = fig.add_subplot(1, 2, 2, projection=projection)
    horizontal_cross_section(
        tracks, grid, params=tmp_params, alt=alt2, fig=fig, ax=ax,
        date_time=date_time)

    # Save frame and cleanup
    if params['save_dir'] is not None:
        plt.savefig(
            '{}/frame_{}.png'.format(params['save_dir'], date_time),
            bbox_inches='tight', dpi=params['dpi'], facecolor='w',
            pad_inches=0)
        plt.close()
    gc.collect()


def object(tracks, grid, params, date_time=None, alt=None):

    params = check_params(params)
    if alt is None:
        alt = tracks.params['LEVELS'][0, 0]
    grid_time = np.datetime64(parse_grid_datetime(grid))
    grid_time = grid_time.astype('datetime64[m]')
    if date_time is None:
        date_time = grid_time

    init_fonts(params)
    projection = ccrs.PlateCarree()

    print('Generating figure for {}.'.format(str(date_time)))
    if grid_time != date_time:
        msg = 'grid_time {} does not match specified date_time {}. Aborting.'
        msg = msg.format(grid_time, date_time)
        print(msg)

    # Initialise figure
    fig = plt.figure(figsize=(22, 8))
    suptitle = 'Object {}'.format(params['uid_ind'])
    if params['cell_ind']:
        suptitle += ' Cell {} '.format(params['cell_ind'])
    suptitle += 'at {}.'.format(date_time)
    fig.suptitle(suptitle, fontsize=16)

    # Plot frame
    ax = fig.add_subplot(2, 2, 1, projection=projection)
    horizontal_cross_section(
        tracks, grid, params, alt, fig, ax, date_time)

    ax = fig.add_subplot(2, 2, 3, projection=projection)
    horizontal_cross_section(
        tracks, grid, params, 9000, fig, ax, date_time)

    if params['line_coords']:
        direction_a = 'parallel'
        direction_b = 'perpendicular'
    else:
        direction_a = 'lat'
        direction_b = 'lon'

    params['direction'] = direction_a
    ax = fig.add_subplot(2, 2, 2)
    vertical_cross_section(
        tracks, grid, params, alt=None, fig=fig, ax=ax, date_time=date_time)

    params['direction'] = direction_b
    ax = fig.add_subplot(2, 2, 4)
    vertical_cross_section(
        tracks, grid, params, alt=None, fig=fig, ax=ax, date_time=date_time)

    # Save frame and cleanup
    if params['save_dir'] is not None:
        plt.savefig(
            '{}/frame_{}.png'.format(params['save_dir'], date_time),
            bbox_inches='tight', dpi=params['dpi'], facecolor='w')
        plt.show()
        plt.close()
    gc.collect()


def concat_angles(base_dir, save_path, max_uid):
    data = pd.read_csv(
        '{}/vertical_cross_section_0_frames/angles.csv'.format(base_dir))
    for i in range(1, max_uid+1):
        new_data = pd.read_csv(
            '{}/vertical_cross_section_{}_frames/angles.csv'.format(
                base_dir, i))
        data = pd.concat([data, new_data])
    data.to_csv(save_path)
    return data


def get_angle_props(angles, tracks_obj):
    angle_diff = np.abs(
        angles['streamline_angle'] - angles['stratiform_offset_angle'])
    w_max = angles['w_max'].values
    max_count = angles['max_count'].values
    n_angles = angles['n_angles'].values
    n_obs = angles['n_obs'].values
    ratio_angles = n_angles / n_obs
    ratio_bin = max_count / n_angles

    eccentricity = []
    bor_conv = []
    bor_strat = []
    area_conv = []
    area_strat = []
    u_shift = []
    v_shift = []

    for i in range(len(angles)):
        date_time = angles['time'].iloc[i]
        uid = angles['uid'].iloc[i]
        tmp_tracks_low = tracks_obj.tracks.xs(
            (date_time, str(int(uid)), 0), level=('time', 'uid', 'level'))
        tmp_tracks_high = tracks_obj.tracks.xs(
            (date_time, str(int(uid)), 2), level=('time', 'uid', 'level'))
        eccentricity.append(tmp_tracks_low['eccentricity'].iloc[0])
        u_shift.append(tmp_tracks_low['u_shift'].iloc[0])
        v_shift.append(tmp_tracks_low['v_shift'].iloc[0])
        bor_conv.append(tmp_tracks_low['touch_border'].iloc[0])
        bor_strat.append(tmp_tracks_high['touch_border'].iloc[0])
        area_conv.append(tmp_tracks_low['proj_area'].iloc[0])
        area_strat.append(tmp_tracks_high['proj_area'].iloc[0])
    [u_shift, v_shift] = [np.array(u_shift), np.array(v_shift)]
    eccentricity = np.array(eccentricity)
    [bor_conv, bor_strat] = [np.array(bor_conv), np.array(bor_strat)]
    [area_conv, area_strat] = [np.array(area_conv), np.array(area_strat)]
    speed = np.sqrt(u_shift ** 2 + v_shift ** 2)
    ratio_border_conv = bor_conv * 6.25 / area_conv
    ratio_border_strat = bor_strat * 6.25 / area_strat

    X = np.transpose(np.array([
        w_max, ratio_angles, ratio_bin, ratio_border_conv,
        ratio_border_strat, area_conv, area_strat, eccentricity,
        speed]))
    names = [
        'w_max', 'ratio_angles', 'ratio_bin', 'ratio_border_conv',
        'ratio_border_strat', 'area_conv', 'area_strat',
        'eccentricity', 'speed']

    return angle_diff, X, names


def angle_correlation(
        sl_angles, so_angles, cond, save_path, fig=None, ax=None):

    params = {'fontsize': 12}
    if fig is None:
        init_fonts(params)
        fig = plt.figure(figsize=(5, 5))
    if ax is None:
        ax = fig.add_subplot(1, 1, 1)

    ax.plot([90, 90], [0, 180], '--', color='gray')
    ax.plot([0, 180], [90, 90], '--', color='gray')
    ax.scatter(sl_angles[~cond], so_angles[~cond], marker='x', color='gray')
    ax.scatter(sl_angles[cond], so_angles[cond], marker='o', color='r')

    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 200, 20))
    ax.set_xticklabels(np.arange(0, 200, 20).astype(int))
    ax.set_xlabel('Streamline Angle [Degrees]')
    ax.set_ylim(0, 180)
    ax.set_yticks(np.arange(0, 200, 20))
    ax.set_yticklabels(np.arange(0, 200, 20).astype(int))
    ax.set_ylabel('Stratiform Offset Angle [Degrees]')
    # ax.set_title('Correlation between Stratiform Offset and Streamlines')
    ax.set_aspect(1)

    save_dir = '/home/student.unimelb.edu.au/shorte1/Documents/'
    save_dir += 'TINT_figures/correlation.png'

    plt.savefig(save_path, dpi=200, facecolor='w')
