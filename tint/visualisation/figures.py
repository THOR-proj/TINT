import gc
import numpy as np
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import pyart
from pyart.core.transforms import cartesian_to_geographic
import warnings
from matplotlib import rcParams
import xarray as xr

from tint.grid_utils import get_grid_alt, parse_grid_datetime
import tint.visualisation.horizontal_helpers as hh
import tint.visualisation.vertical_helpers as vh


def init_fonts():
    # Initialise fonts
    rcParams.update({'font.family': 'serif'})
    rcParams.update({'font.serif': 'Liberation Serif'})
    rcParams.update({'mathtext.fontset': 'dejavuserif'})
    rcParams.update({'font.size': 12})


def check_params(user_params):

    params = {
        'uid_ind': None, 'cell_ind': None, 'box_rad': 0.75,
        'line_coords': False, 'center_cell': False, 'label_splits': True,
        'legend': True, 'winds': False, 'winds_fn': None,
        'colorbar_flag': True, 'direction': None, 'line_average': False,
        'crosshair': True, 'streamplot': True, 'dpi': 200, 'save_dir': None,
        'relative_winds': False}
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
        tracks, grid, params, alt, fig, ax, date_time, figsize=(16, 16)):
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
        init_fonts()
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
        tracks, grid, params, alt, fig, ax, date_time, figsize=(10, 7))
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
    lgd_han = hh.add_tracked_objects(tracks, grid, date_time, params, ax, alt)

    if params['uid_ind'] is not None:
        ax.set_xlim(box[2][0], box[2][1])
        ax.set_ylim(box[3][0], box[3][1])

    if params['legend']:
        legend = plt.legend(handles=lgd_han, loc=2)
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor((1, 1, 1, 1))

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

    ds = vh.format_pyart(grid)
    vars = ['reflectivity']
    if params['winds']:
        winds_ds = xr.open_dataset(params['winds_fn'])
        ds = xr.merge([ds, winds_ds])
        vars += ['U', 'V', 'W']

    if params['line_coords']:
        tmp_tracks = tracks.tracks.xs(
            (date_time, params['uid_ind'], 0), level=('time', 'uid', 'level'))
        ds = vh.get_line_grid(ds, tmp_tracks['orientation'].iloc[0], vars=vars)
        if params['winds']:
            if params['relative_winds']:
                ds['U'] = ds['U'] - tmp_tracks['u_shift'].values
                ds['V'] = ds['V'] - tmp_tracks['v_shift'].values
            ds = vh.rebase_horizontal_winds(
                ds, tmp_tracks['orientation'].iloc[0])

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
        if params['line_average']:
            semi_major = tmp_tracks['semi_major'].iloc[0]
            cond = ((ds.x <= x + semi_major * 2500 / 2)
                    & (ds.x >= x - semi_major * 2500 / 2))
            ds_plot = ds.where(cond).dropna(dim='x', how='all').mean(dim='x')
            t_string = 'Line Perpendicular Mean Cross Section'
        else:
            ds_plot = ds.sel(x=x, method='nearest').squeeze()
            t_string = 'Line Perpendicular Cross Section'
        extent = [
            ds_plot.y[0] / 1000, ds_plot.y[-1] / 1000,
            ds_plot.z[0] / 1000, ds_plot.z[-1] / 1000]
        cs = ax.imshow(
            ds_plot.reflectivity, vmin=vmin, vmax=vmax, cmap=cmap,
            interpolation='none', origin='lower', extent=extent, zorder=1)
        fig.colorbar(cs, ax=ax, label='Reflectivity [DbZ]')
        h_lim = y_lim
        x_label = 'Line Perpendicular Distance From Radar [km]'

    lgd_so = vh.add_stratiform_offset(ax, tracks, grid, date_time, params)
    lgd_han.append(lgd_so)

    if params['winds']:
        lgd_winds = vh.add_winds(ds, ax, tracks, grid, date_time, params)
        if params['streamplot']:
            lgd_han.append(lgd_winds)

    if params['center_cell']:
        lgd_cell = vh.add_cell(ax, tracks, grid, date_time, params)
        lgd_han.append(lgd_cell)

    if params['legend']:
        legend = plt.legend(handles=lgd_han, loc=2)
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
    ax.set_title(t_string)
    ax.set_aspect(2)

    del display

    return


def two_level(tracks, grid, params, date_time=None, alt=None):

    params = check_params(params)
    if alt is None:
        alt = tracks.params['GS_ALT']
    grid_time = np.datetime64(parse_grid_datetime(grid))
    grid_time = grid_time.astype('datetime64[m]')
    if date_time is None:
        date_time = grid_time

    init_fonts()
    projection = ccrs.PlateCarree()

    print('Generating figure for {}.'.format(str(date_time)))
    if grid_time != date_time:
        msg = 'grid_time {} does not match specified date_time {}. Aborting.'
        msg = msg.format(grid_time, date_time)
        print(msg)

    # Initialise figure
    fig = plt.figure(figsize=(22, 8))
    suptitle = 'Convective and Stratiform Cloud at ' + str(grid_time)
    fig.suptitle(suptitle, fontsize=16)

    # Plot frame
    ax = fig.add_subplot(1, 2, 1, projection=projection)
    horizontal_cross_section(
        tracks, grid, params, alt, fig, ax, date_time)

    ax = fig.add_subplot(1, 2, 2, projection=projection)
    horizontal_cross_section(
        tracks, grid, params, 9000, fig, ax, date_time)

    # Save frame and cleanup
    if params['save_dir'] is not None:
        plt.savefig(
            '{}/frame_{}.png'.format(params['save_dir'], date_time),
            bbox_inches='tight', dpi=params['dpi'], facecolor='w')
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

    init_fonts()
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
        plt.close()
    gc.collect()
