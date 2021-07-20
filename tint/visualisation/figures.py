import gc
import numpy as np
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import copy
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
        'colorbar_flag': True, 'direction': None, 'line_average': False}
    for p in user_params:
        if p in params:
            params[p] = user_params[p]
        else:
            keys = ', '.join([p for p in params])
            msg = '{} not a valid parameter. Choices are {}'
            msg = msg.format(p, keys)
            warnings.showwarning(msg, RuntimeWarning, 'figures.py', 26)

    return params


def get_bounding_box(tracks, date_time, params):
    lon_box, lat_box, x, y = get_center_coords(tracks, params, date_time)

    box = np.array([-1 * params['box_rad'], params['box_rad']])
    box_x_bounds = (lon_box) + box
    box_y_bounds = (lat_box) + box
    return lon_box, lat_box, box_x_bounds, box_y_bounds


def add_crosshair(
        tracks, grid, date_time, params, ax, display, lon_box, lat_box):

    # Drop other objects from frame_tracks
    tmp_tracks = tracks.tracks.xs(
        (params['uid_ind'], date_time, 0), level=('uid', 'time', 'level'))

    if params['uid_ind'] is not None:
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
            [0, 360], [crosshair_lat_0, crosshair_lat_1], '--r', linewidth=2.0)
        # Plot across line cross hair
        m = - 1/m
        c = crosshair_lat - m*crosshair_lon
        crosshair_lat_0 = c
        crosshair_lat_1 = m * 360 + c
        ax.plot(
            [0, 360], [crosshair_lat_0, crosshair_lat_1], '--r', linewidth=2.0)
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


def get_center_coords(tracks, params, date_time):
    if params['uid_ind'] is not None:
        tmp_tracks = tracks.tracks.xs(
            (date_time, params['uid_ind'], 0), level=('time', 'uid', 'level'))
        lon = tmp_tracks['lon'].iloc[0]
        lat = tmp_tracks['lat'].iloc[0]
        [x, y] = [tmp_tracks['grid_x'].iloc[0], tmp_tracks['grid_y'].iloc[0]]
        if params['line_coords']:
            A = vh.get_rotation(tmp_tracks['orientation'].iloc[0])
            [x, y] = np.transpose(A).dot(np.array([x, y]))
    else:
        lon = tracks.radar_info['radar_lon']
        lat = tracks.radar_info['radar_lat']
        [x, y] = [0, 0]

    return lon, lat, x, y


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
        colorbar_flag=params['colorbar_flag'])
    # # Set labels
    ax.set_title('Altitude {} m'.format(alt))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Plot scan boundary
    # if params['scan_boundary']:
    #   plot_boundary(ax, tracks, grid, projparams)

    box = get_bounding_box(tracks, date_time, params)
    add_crosshair(tracks, grid, date_time, params, ax, display, box[0], box[1])
    hh.add_tracked_objects(tracks, grid, date_time, params, ax, alt)

    # If focusing on one object, restrict axis limits around object
    if params['uid_ind'] is not None:
        ax.set_xlim(box[2][0], box[2][1])
        ax.set_ylim(box[3][0], box[3][1])

    return True


def vertical_line_cross_section(
        tracks, grid, params={}, alt=None, fig=None, ax=None, date_time=None):

    init_cs = init_cross_section(
        tracks, grid, params, alt, fig, ax, date_time, figsize=(10, 4))
    [
        params, alt, fig, ax, date_time, display, alt_ind, cmap,
        vmin, vmax, projection] = init_cs
    lon, lat, x, y = get_center_coords(tracks, params, date_time)
    lgd_han = []

    tmp_tracks = tracks.tracks.xs(
        (date_time, params['uid_ind'], 0), level=('time', 'uid', 'level'))
    ds = vh.format_pyart(grid)
    vars = ['reflectivity']
    if params['winds']:
        winds_ds = xr.open_dataset(params['winds_fn'])
        ds = xr.merge([ds, winds_ds])
        vars += ['U', 'V', 'W']
    ds = vh.get_line_grid(ds, tmp_tracks['orientation'].iloc[0], vars=vars)

    x_lim = (x + np.array([-40000, 40000])) / 1000
    y_lim = (y + np.array([-40000, 40000])) / 1000

    if params['direction'] == 'parallel':
        ds = ds.sel(y=y, method='nearest').squeeze()
        extent = [
            ds.x[0] / 1000, ds.x[-1] / 1000, ds.z[0] / 1000, ds.z[-1] / 1000]
        cs = ax.imshow(
            ds.reflectivity, vmin=vmin, vmax=vmax, cmap=cmap,
            interpolation='none', origin='lower', extent=extent)
        #
        # [h_low, h_high] = [tx_low, tx_high]
        # if center_ud:
        #     h_draft = drafts_new[:, 0] / 1000
        #     z_draft = grid.z['data'][z0: len(h_draft) + z0] / 1000

        h_lim = x_lim
        t_string = 'Along Line Cross Section'
        x_label = 'Along Line Distance From Origin [km]'

    elif params['direction'] == 'perpendicular':

        if params['line_average']:
            semi_major = tmp_tracks['semi_major'].iloc[0]
            cond = ((ds.x <= x + semi_major * 2500 / 2)
                    & (ds.x >= x - semi_major * 2500 / 2))
            ds = ds.where(cond).dropna(dim='x', how='all').mean(dim='x')
            t_string = 'Line Perpendicular Mean Cross Section'
        else:
            ds = ds.sel(x=x, method='nearest').squeeze()
            t_string = 'Line Perpendicular Cross Section'
        extent = [
            ds.y[0] / 1000, ds.y[-1] / 1000, ds.z[0] / 1000, ds.z[-1] / 1000]
        cs = ax.imshow(
            ds.reflectivity, vmin=vmin, vmax=vmax, cmap=cmap,
            interpolation='none', origin='lower', extent=extent, zorder=1)

        # [h_low, h_high] = [ty_low, ty_high]
        # if center_ud:
        #     h_draft = drafts_new[:,1]/1000
        #     z_draft = grid.z['data'][z0:len(h_draft)+z0]/1000

        h_lim = y_lim
        x_label = 'Line Perpendicular Distance From Origin [km]'

    lgd_so = vh.add_stratiform_offset(ax, tracks, grid, date_time, params)
    lgd_han.append(lgd_so)

    fig.colorbar(cs, ax=ax, label='Reflectivity [DbZ]')
    if params['legend']:
        legend = plt.legend(handles=lgd_han, loc=2)
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor((1, 1, 1, 1))
    # Plot cell tracks
    # if center_ud:
    #     ax.plot(h_draft, z_draft, '-', color=color,
    #             linewidth=1.0)

    # Plot wrf winds if necessary
    # if wrf_winds:
    #     plot_vert_winds_line(
    #         ax, new_wrf, x_draft_new, y_draft_new, direction, quiver=quiver,
    #         semi_major=semi_major, average_along_line=average_along_line)

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


def vertical_cross_section(
        tracks, grid, params={}, alt=None, fig=None, ax=None, date_time=None):

    init_cs = init_cross_section(
        tracks, grid, params, alt, fig, ax, date_time, figsize=(10, 4))
    [
        params, alt, fig, ax, date_time, display, alt_ind, cmap,
        vmin, vmax, projection] = init_cs
    lon, lat, x, y = get_center_coords(tracks, params, date_time)
    lgd_han = []

    # if alt_low is None:
    #     alt_low = f_tracks.params['GS_ALT']
    # if alt_high is None:
    #     alt_high = f_tracks.params['GS_ALT']
    #
    # # Calculate mean height of first and last vertical TINT levels
    # low = f_tracks.params['LEVELS'][0].mean()/1000
    # high = f_tracks.params['LEVELS'][-1].mean()/1000
    #
    # # Restrict to uid
    # cell = f_tracks.tracks.xs(uid, level='uid')
    # cell = cell.reset_index(level=['time'])
    #
    # # Get low and high data
    # n_lvl = f_tracks.params['LEVELS'].shape[0]
    # cell_low = cell.xs(0, level='level')
    # cell_high = cell.xs(n_lvl-1, level='level')
    #
    # # Restrict to specific time
    # cell_low = cell_low.iloc[nframe]
    # cell_high = cell_high.iloc[nframe]
    #
    # # Convert coordinates to those of new grid
    # old_x_low = cell_low['grid_x']
    # old_y_low = cell_low['grid_y']
    # old_low = np.array([old_x_low, old_y_low])
    # new_low = np.transpose(A).dot(old_low)
    # [new_x_low, new_y_low] = new_low.tolist()
    #
    # # Convert coordinates to those of new grid
    # old_x_high = cell_high['grid_x']
    # old_y_high = cell_high['grid_y']
    # old_high = np.array([old_x_high, old_y_high])
    # new_high = np.transpose(A).dot(old_high)
    # [new_x_high, new_y_high] = new_high.tolist()
    #
    # tx_met = new_x_low
    # ty_met = new_y_low
    # tx_low = new_x_low/1000
    # tx_high = new_x_high/1000
    # ty_low = new_y_low/1000
    # ty_high = new_y_high/1000
    #
    x_lim = (x + np.array([-40000, 40000])) / 1000
    y_lim = (y + np.array([-40000, 40000])) / 1000
    #
    # # Get center location
    # if center_ud:
    #     ud = np.array(cell_low['cells'][cell_ind])
    #     x_draft_old = grid.x['data'][ud[:, 2]].data
    #     y_draft_old = grid.y['data'][ud[:, 1]].data
    #
    #     drafts_old = np.array([x_draft_old, y_draft_old])
    #     drafts_new = np.array([np.transpose(A).dot(drafts_old[:, i])
    #                           for i in range(drafts_old.shape[1])])
    #
    #     x_draft = grid.x['data'][ud[0, 2]]
    #     y_draft = grid.y['data'][ud[0, 1]]
    #     ud_old = np.array([x_draft, y_draft])
    #
    #     # Get new coordinates
    #     ud_new = np.transpose(A).dot(ud_old)
    #     [x_draft_new, y_draft_new] = ud_new.tolist()
    #
    #     z0 = get_grid_alt(
    #         f_tracks.record.grid_size, f_tracks.params['CELL_START'])
    # else:
    # x_draft_new = new_x_low
    # y_draft_new = new_y_low

    if params['direction'] == 'lat':
        display.plot_latitude_slice(
            tracks.field, lon=lon, lat=lat, colorbar_flag=True, edges=False,
            vmin=vmin, vmax=vmax, mask_outside=False, cmap=cmap, ax=ax,
            colorbar_label='Reflectivity [DbZ]')
        # [h_low, h_high] = [tx_low, tx_high]
        # if center_ud:
        #     h_draft = grid.x['data'][ud[:, 2]]/1000
        #     z_draft = grid.z['data'][z0:len(h_draft)+z0]/1000
        h_lim = x_lim
        t_string = 'Latitude Cross Section'
        x_label = 'West East Distance From Origin [km]'

    elif params['direction'] == 'lon':
        display.plot_longitude_slice(
            tracks.field, lon=lon, lat=lat, colorbar_flag=True, edges=False,
            vmin=vmin, vmax=vmax, mask_outside=False, cmap=cmap, ax=ax,
            colorbar_label='Reflectivity [DbZ]')
        # [h_low, h_high] = [ty_low, ty_high]
        # if center_ud:
        #     h_draft = grid.y['data'][ud[:, 1]]/1000
        #     z_draft = grid.z['data'][z0:len(h_draft) + z0] / 1000
        h_lim = y_lim
        t_string = 'Longitude Cross Section'
        x_label = 'North South Distance From Origin [km]'

    # if params['uid_ind']:
    #     lgd_so = vh.add_stratiform_offset(ax, tracks, grid, date_time, params)
    #     lgd_han.append(lgd_so)

    if params['legend']:
        legend = plt.legend(handles=lgd_han, loc=2)
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor((1, 1, 1, 1))
    # Plot cell tracks
    # if center_ud:
    #     ax.plot(h_draft, z_draft, '-', color=color,
    #             linewidth=1.0)

    # Plot wrf winds if necessary
    # if wrf_winds:
    #     plot_vert_winds_line(
    #         ax, new_wrf, x_draft_new, y_draft_new, direction, quiver=quiver,
    #         semi_major=semi_major, average_along_line=average_along_line)

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


def two_level_view(tracks, grid, out_dir, date_time=None, alt=None):

    if alt is None:
        alt = f_tracks.params['GS_ALT']
    time_ind = tracks.tracks.index.get_level_values('time')
    if date_time is None:
        date_time = time_ind[0]

    track = tracks.tracks[time_ind == datetime]
    time_ind = track.index.get_level_values('time')

    # Initialise fonts
    init_fonts()

    print('Generating figure for {}.'.format(str(datetime)))
    grid_time = np.datetime64(grid.time['units'][14:])
    if grid_time != date_time:
        msg = 'grid_time {} does not match specified date_time {}. Aborting.'
        msg = msg.format(grid_time, date_time)
        print(msg)

    # Initialise figure
    fig = plt.figure(figsize=(22, 9))
    fig.suptitle('MCS at ' + str(grid_time), fontsize=16)

    print('Plotting scan at {}.'.format(grid_time),
          end='\r', flush=True)

    # Plot frame
    ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    success = plot_tracks_horiz_cross(
        f_tracks, grid, alt_low, fig=fig, ax=ax)

    ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    success = plot_tracks_horiz_cross(
        f_tracks, grid, alt_high, ellipses='strat',
        fig=fig, ax=ax, line_coords=line_coords)

    # Save frame and cleanup
    if success:
        plt.savefig(
            out_dir + '/frame_' + str(counter).zfill(3) + '.png',
            bbox_inches = 'tight', dpi=dpi)
    plt.close()
    del grid, ax
    gc.collect()


def full_view(
        tracks, grids, tmp_dir, dpi=100, vmin=-8, vmax=64, start_datetime=None,
        end_datetime=None, cmap=pyart.graph.cm_colorblind.HomeyerRainbow,
        alt_low=None, alt_high=None, isolated_only=False, tracers=False,
        persist=False, projection=ccrs.PlateCarree(), scan_boundary=False,
        box_rad=.75, line_coords=False, **kwargs):

    # Create a copy of tracks for use by this function
    f_tracks = copy.deepcopy(tracks)

    # Set default arguments if non passed
    if alt_low is None:
        alt_low = f_tracks.params['GS_ALT']
    if alt_high is None:
        alt_high = f_tracks.params['GS_ALT']

    # Restrict tracks data to start and end datetime arguments
    time_ind = f_tracks.tracks.index.get_level_values('time')
    if start_datetime != None:
        cond = (time_ind >= start_datetime)
        f_tracks.tracks = f_tracks.tracks[cond]
        time_ind = f_tracks.tracks.index.get_level_values('time')
    else:
        start_datetime = time_ind[0]
        time_ind = f_tracks.tracks.index.get_level_values('time')
    if end_datetime != None:
        cond = (time_ind <= end_datetime)
        f_tracks.tracks = f_tracks.tracks[cond]
        time_ind = f_tracks.tracks.index.get_level_values('time')
    else:
        end_datetime = time_ind[-1]
        time_ind = f_tracks.tracks.index.get_level_values('time')

    # Initialise fonts
    init_fonts()

    print('Animating from {} to {}.'.format(str(start_datetime),
                                            str(end_datetime)))

    # Loop through all scans/grids provided to function.
    for counter, grid in enumerate(grids):

        grid_time = np.datetime64(grid.time['units'][14:])
        if grid_time > end_datetime:
            del grid
            print('Reached {}. Breaking loop.'.format(str(end_datetime)))
            break
        elif grid_time < start_datetime:
            print('Current grid earlier than {}. Moving to next grid.'.format(
                str(start_datetime)))
            continue

        # Initialise figure
        fig_grid = plt.figure(figsize=(22, 9))
        fig_grid.suptitle('MCS at ' + str(grid_time), fontsize=16)

        print('Plotting scan at {}.'.format(grid_time),
              end='\r', flush=True)

        # Plot frame
        ax = fig_grid.add_subplot(1, 2, 1, projection=projection)
        success = plot_tracks_horiz_cross(
            f_tracks, grid, alt_low, fig=fig_grid, ax=ax, **kwargs)
        ax = fig_grid.add_subplot(1, 2, 2, projection=projection)
        success = plot_tracks_horiz_cross(
            f_tracks, grid, alt_high, fig=fig_grid, ax=ax, **kwargs)

        # Save frame and cleanup
        if success:
            plt.savefig(tmp_dir + '/frame_' + str(counter).zfill(3) + '.png',
                        bbox_inches='tight', dpi=dpi)
        plt.close()
        del grid, ax
        gc.collect()


def plot_obj_vert_cross(
        f_tracks, grid, uid, nframe, fig=None, ax=None, vmin=-8, vmax=64,
        alt_low=None, alt_high=None, scan_boundary=False, center_ud=False,
        cell_ind=0, direction='lat', color='k', wrf_winds=False, quiver=False,
        mp='lin', **kwargs):

    field = f_tracks.field
    projparams = grid.get_projparams()
    display = pyart.graph.GridMapDisplay(grid)
    cmap = pyart.graph.cm_colorblind.HomeyerRainbow

    if alt_low is None:
        alt_low = f_tracks.params['GS_ALT']
    if alt_high is None:
        alt_high = f_tracks.params['GS_ALT']

    # Calculate mean height of first and last vertical TINT levels
    low = f_tracks.params['LEVELS'][0].mean()/1000
    high = f_tracks.params['LEVELS'][-1].mean()/1000

    # Restrict to uid
    cell = f_tracks.tracks.xs(uid, level='uid')
    cell = cell.reset_index(level=['time'])

    # Get low and high data
    n_lvl = f_tracks.params['LEVELS'].shape[0]
    cell_low = cell.xs(0, level='level')
    cell_high = cell.xs(n_lvl-1, level='level')

    # Restrict to specific time
    cell_low = cell_low.iloc[nframe]
    cell_high = cell_high.iloc[nframe]

    # Define box size
    tx_met = cell_low['grid_x']
    ty_met = cell_low['grid_y']
    tx_low = cell_low['grid_x'] / 1000
    tx_high = cell_high['grid_x'] / 1000
    ty_low = cell_low['grid_y'] / 1000
    ty_high = cell_high['grid_y'] / 1000

    xlim = (tx_met + np.array([-75000, 75000])) / 1000
    ylim = (ty_met + np.array([-75000, 75000])) / 1000

    # Get center location
    if center_ud:
        ud = np.array(cell_low['cells'][cell_ind])
        x_draft = grid.x['data'][ud[0, 2]]
        y_draft = grid.y['data'][ud[0, 1]]
        lon, lat = cartesian_to_geographic(
            x_draft, y_draft, projparams)
        z0 = get_grid_alt(
            f_tracks.record.grid_size, f_tracks.params['CELL_START'])
    else:
        lon = cell_low['lon']
        lat = cell_low['lat']
        x_draft = cell_low['grid_x']
        y_draft = cell_low['grid_y']

    if direction == 'lat':
        display.plot_latitude_slice(
            field, lon=lon, lat=lat, title_flag=False, colorbar_flag=False,
            edges=False, vmin=vmin, vmax=vmax, mask_outside=False, cmap=cmap,
            ax=ax)
        [h_low, h_high] = [tx_low, tx_high]
        if center_ud:
            h_draft = grid.x['data'][ud[:, 2]]/1000
            z_draft = grid.z['data'][z0:len(h_draft)+z0]/1000
        h_lim = xlim
        t_string = 'Latitude Cross Section'
        x_label = 'East West Distance From Origin [km]'

    elif direction == 'lon':
        display.plot_longitude_slice(
            field, lon=lon, lat=lat, title_flag=False, colorbar_flag=False,
            edges=False, vmin=vmin, vmax=vmax, mask_outside=False, cmap=cmap,
            ax=ax)
        [h_low, h_high] = [ty_low, ty_high]
        if center_ud:
            h_draft = grid.y['data'][ud[:, 1]]/1000
            z_draft = grid.z['data'][z0:len(h_draft) + z0] / 1000
        h_lim = ylim
        t_string = 'Longitude Cross Section'
        x_label = 'North South Distance From Origin [km]'

    # Plot system tilt
    ax.plot([h_low, h_high], [low, high], '--b', linewidth=2.0)

    # Plot cell tracks
    if center_ud:
        ax.plot(h_draft, z_draft, '-', color=color,
                linewidth=1.0)

    # Plot wrf winds if necessary
    if wrf_winds:
        plot_wrf_winds(
            ax, grid, x_draft, y_draft, direction, quiver=quiver, mp=mp)

    ax.set_xlim(h_lim[0], h_lim[1])
    ax.set_xticks(np.arange(h_lim[0], h_lim[1], 25))
    ax.set_xticklabels(
        np.round((np.arange(h_lim[0], h_lim[1], 25)), 1))

    ax.set_title(t_string)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Distance Above Origin [km]')

    del display


def object_view(
        tracks, grids, tmp_dir, uid=None, dpi=100, vmin=-8, vmax=64,
        start_datetime=None, end_datetime=None, cmap=None, alt_low=None,
        alt_high=None, box_rad=.75, projection=None, center_ud=False,
        cell_ind=None, wrf_winds=False, line_coords=False,
        average_along_line=False, quiver=False, mp='lin', **kwargs):

    if uid is None:
        print("Please specify 'uid' keyword argument.")
        return

    f_tracks = copy.deepcopy(tracks)
    if cmap is None:
        cmap = pyart.graph.cm_colorblind.HomeyerRainbow
    if alt_low is None:
        alt_low = tracks.params['GS_ALT']
    if alt_high is None:
        alt_high = tracks.params['GS_ALT']
    if projection is None:
        projection = ccrs.PlateCarree()

    colors = ['m', 'lime', 'darkorange', 'k', 'b', 'darkgreen', 'yellow']

    cell = f_tracks.tracks.xs(uid, level='uid').xs(0, level='level')
    cell = cell.reset_index(level=['time'])
    nframes = len(cell)
    print('Animating', nframes, 'frames')
    nframe = 0
    pframe = 0

    # Loop through each grid in grids
    for grid in grids:
        # Ensure object exists at current grid
        grid_time = np.datetime64(grid.time['units'][14:])
        if nframe >= nframes:
            info_msg = ('Object died at '
                        + str(cell.iloc[nframe-1].time)
                        + '.\n' + 'Ending loop.')
            print(info_msg)
            del grid
            gc.collect()
            break
        elif cell.iloc[nframe].time > grid_time:
            info_msg = ('Object not yet initiated at '
                        + '{}.\n'.format(grid_time)
                        + 'Moving to next grid.')
            print(info_msg)
            continue
        while cell.iloc[nframe].time < grid_time:

            info_msg = ('Current grid at {}.\n'
                        + 'Object initialises at {}.\n'
                        + 'Moving to next object frame.')
            print(info_msg.format(grid_time, str(cell.iloc[nframe].time)))
            nframe += 1

        print('Plotting frame at {}'.format(grid_time),
              end='\n', flush=True)

        # Initialise fonts
        init_fonts()

        # Don't plot axis
        plt.axis('off')

        if line_coords:
            new_grid, A, angle, semi_major = get_line_grid(f_tracks, grid, uid, nframe)
            if wrf_winds:
                new_wrf = get_line_grid_wrf(grid_time, angle, mp=mp)
            else:
                new_wrf = None

        # Determine whether to plot cells
        if center_ud:
            if cell_ind is None:
                # Plot all cells
                cell_frame = cell.iloc[nframe]
                ud_list = range(len(cell_frame['cells']))
            else:
                ud_list = [cell_ind]
        else:
            ud_list = [cell_ind]

        for j in ud_list:
            fig = plt.figure(figsize=(12, 10))
            if center_ud:
                print('Plotting cell {}.  '.format(str(j)),
                      end='\r', flush=True)
            # Generate title
            if center_ud:
                fig.suptitle('Object ' + uid + ' at '
                             + str(grid_time) + ': Cell '
                             + str(j), fontsize=16, y=1.0)
            else:
                fig.suptitle('Object ' + uid + ' at '
                             + str(grid_time), fontsize=16, y=1.0)

            # Vertical cross section at alt_low
            ax = fig.add_subplot(2, 2, 1, projection=projection)
            success = plot_tracks_horiz_cross(f_tracks, grid, alt_low, fig=fig,
                                    ax=ax, ellipses='conv', legend=False,
                                    uid_ind=uid, center_ud=center_ud, cell_ind=j,
                                    angle=angle, line_coords=line_coords,
                                    wrf_winds=wrf_winds, mp=mp,
                                    **kwargs)

            # Vertical cross section at alt_high
            ax = fig.add_subplot(2, 2, 3, projection=projection)
            success = plot_tracks_horiz_cross(f_tracks, grid, alt_high, fig=fig,
                                    ax=ax, ellipses='strat', legend=False,
                                    uid_ind=uid, center_ud=center_ud, cell_ind=j,
                                    angle=angle, line_coords=line_coords,
                                    wrf_winds=wrf_winds, mp=mp,
                                    **kwargs)

            if center_ud:
                color=colors[np.mod(j,len(colors))]
            else:
                color=None

            # Plot latitude (or cross line) cross section
            ax = fig.add_subplot(2, 2, 2)
            if line_coords:
                plot_obj_line_cross(f_tracks, grid, new_grid, A, uid, nframe, semi_major,
                    fig=fig, ax=ax, alt_low=alt_low, alt_high=alt_high,
                    center_ud=center_ud, cell_ind=j, direction='cross',
                    color=color, wrf_winds=wrf_winds, new_wrf=new_wrf,
                    average_along_line=average_along_line, quiver=quiver,
                    **kwargs)
            else:
                plot_obj_vert_cross(f_tracks, grid, uid, nframe, fig=fig, ax=ax,
                                    alt_low=alt_low, alt_high=alt_high,
                                    center_ud=center_ud, cell_ind=j,
                                    direction='lat', color=color,
                                    wrf_winds=wrf_winds, **kwargs)

            # Plot longitude (or line parallel) cross section
            ax = fig.add_subplot(2, 2, 4)
            if line_coords:
                plot_obj_line_cross(f_tracks, grid, new_grid, A, uid, nframe, semi_major,
                    fig=fig, ax=ax, alt_low=alt_low, alt_high=alt_high,
                    center_ud=center_ud, cell_ind=j, direction='parallel',
                    color=color, wrf_winds=wrf_winds, new_wrf=new_wrf,
                    average_along_line=average_along_line, quiver=quiver,
                    **kwargs)
            else:
                plot_obj_vert_cross(f_tracks, grid, uid, nframe, fig=fig, ax=ax,
                                    alt_low=alt_low, alt_high=alt_high,
                                    cell_ind=j, direction='lon',
                                    center_ud=center_ud, color=color,
                                    wrf_winds=wrf_winds, **kwargs)

            plt.tight_layout()

            # plot and save figure
            if success:
                fig.savefig(tmp_dir + '/frame_' + str(pframe).zfill(3) + '.png',
                            dpi=dpi)
            plt.close()
            pframe += 1
            gc.collect()

        nframe += 1
        del grid, ax, fig
        plt.close()
        gc.collect()
