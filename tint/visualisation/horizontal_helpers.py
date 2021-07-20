import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.lines as mlines
import xarray as xr
from pyart.core.transforms import cartesian_to_geographic
import cartopy.crs as ccrs

from ..grid_utils import get_grid_size, get_grid_alt


def add_tracked_objects(tracks, grid, date_time, params, ax, alt):

    projection = ccrs.PlateCarree()
    tmp_tracks = tracks.tracks.xs(date_time, level='time')
    uids = np.unique(tmp_tracks.index.get_level_values('uid').values)
    lgd_han = []
    for uid in uids:
        tmp_tracks_uid = tmp_tracks.xs(uid, level='uid')

        lon = tmp_tracks_uid.xs(0, level='level')['lon'].iloc[0]
        lat = tmp_tracks_uid.xs(0, level='level')['lat'].iloc[0]
        mergers = list(tmp_tracks_uid.xs(0, level='level')['mergers'].iloc[0])
        label = ", ".join(mergers)

        ax.text(lon-.05, lat+0.05, uid, transform=projection, fontsize=12)
        ax.text(lon+.05, lat-0.05, label, transform=projection, fontsize=9)

        if params['label_splits']:
            parent = list(
                tmp_tracks_uid.xs(0, level='level')['parent'].iloc[0])
            label = ", ".join(parent)
            ax.text(lon+.05, lat+0.1, label, transform=projection, fontsize=9)

        # Plot velocities
        lgd_vel = add_velocities(
            ax, tracks, grid, uid, date_time, var_list=['shift'], c_list=['m'],
            labels=['System Velocity'])
        [lgd_han.append(h) for h in lgd_vel]

        # Plot stratiform offset
        lgd_so = add_stratiform_offset(ax, tracks, grid, uid, date_time)
        lgd_han.append(lgd_so)

        add_ellipses(ax, tracks, grid, uid, date_time, alt)
        add_cells(ax, tracks, grid, uid, date_time, alt)
        if tracks.params['RAIN']:
            add_rain()

        if params['winds']:
            lgd_winds = add_winds(ax, tracks, grid, uid, date_time, alt, params)
            lgd_han.append(lgd_winds)

        if params['legend']:
            legend = plt.legend(handles=lgd_han, loc=2)
            legend.get_frame().set_alpha(None)
            legend.get_frame().set_facecolor((1, 1, 1, 1))


def reduce_tracks(tracks, uid, date_time, alt):
    tests = [
        (alt >= level[0] and alt < level[1])
        for level in tracks.params['LEVELS']]
    level = np.argwhere(tests)[0, 0]
    tmp_tracks = tracks.tracks.xs(
        (date_time, uid, level), level=('time', 'uid', 'level'))
    return tmp_tracks


def add_rain(ax, tracks, grid, uid, date_time):

    projparams = grid.get_projparams()
    projection = ccrs.PlateCarree()
    alt = tracks.params['LEVELS'][0, 0]
    tmp_tracks = reduce_tracks(tracks, uid, date_time, alt=alt)

    rain_ind = tmp_tracks['tot_rain_loc'].iloc[0]
    rain_amount = tmp_tracks['tot_rain'].iloc[0]
    x_rain = grid.x['data'][rain_ind[1]]
    y_rain = grid.y['data'][rain_ind[0]]
    lon_rain, lat_rain = cartesian_to_geographic(
        x_rain, y_rain, projparams)
    ax.plot(
        lon_rain, lat_rain, marker='o', fillstyle='full', color='blue')
    ax.text(
        lon_rain+.025, lat_rain-.025,
        str(int(round(rain_amount)))+' mm',
        transform=projection, fontsize=9)

    rain_ind = tmp_tracks['max_rr_loc'].iloc[0]
    rain_amount = tmp_tracks['max_rr'].iloc[0]
    x_rain = grid.x['data'][rain_ind[1]]
    y_rain = grid.y['data'][rain_ind[0]]
    lon_rain, lat_rain = cartesian_to_geographic(
        x_rain, y_rain, projparams)
    ax.plot(
        lon_rain, lat_rain, marker='o', fillstyle='none', color='blue')
    ax.text(
        lon_rain+.025, lat_rain+.025, str(int(round(rain_amount)))+' mm/h',
        transform=projection, fontsize=9)


def add_ellipses(ax, tracks, grid, uid, date_time, alt):

    projparams = grid.get_projparams()
    tmp_tracks = reduce_tracks(tracks, uid, date_time, alt)

    centroid = np.squeeze(tmp_tracks[['grid_x', 'grid_y']].values)
    orientation = tmp_tracks[['orientation']].values[0]
    major_axis = tmp_tracks[['semi_major']].values[0]
    minor_axis = tmp_tracks[['semi_minor']].values[0]

    # Convert axes into lat/lon units by approximating 1 degree lat or
    # lon = 110 km. This needs to be made more robust!
    [dz, dy, dx] = get_grid_size(grid)
    major_axis = major_axis * dx / 1000 / 110
    minor_axis = minor_axis * dx / 1000 / 110

    lon, lat = cartesian_to_geographic(centroid[0], centroid[1], projparams)

    ell = Ellipse(
        tuple([lon, lat]), major_axis, minor_axis, orientation,
        linewidth=1.5, fill=False)

    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.4)
    ax.add_artist(ell)
    return ell


def add_cells(ax, tracks, grid, uid, date_time, alt, cell_ind=None):
    projparams = grid.get_projparams()
    tmp_tracks = reduce_tracks(tracks, uid, date_time, alt)
    alt_ind = get_grid_alt(get_grid_size(grid), alt)
    colors = ['m', 'lime', 'darkorange', 'k', 'b', 'darkgreen', 'yellow']
    if cell_ind is None:
        cell_list = range(len(tmp_tracks['cells'].values[0]))
    else:
        cell_list = [cell_ind]
    for j in cell_list:
        # Plot location of cell j at alts[i] if it exists
        cell = np.array(tmp_tracks['cells'].values[0][j])
        if max(cell[:, 0]) >= alt_ind:
            # import pdb; pdb.set_trace()
            height_ind = np.argwhere(cell[:, 0] == alt_ind)[0, 0]
            x_ind = np.array(tmp_tracks['cells'].values[0][j])[height_ind, 2]
            y_ind = np.array(tmp_tracks['cells'].values[0][j])[height_ind, 1]
            x_cell = grid.x['data'][x_ind]
            y_cell = grid.y['data'][y_ind]
            lon_cell, lat_cell = cartesian_to_geographic(
                x_cell, y_cell, projparams)
            ax.scatter(
                lon_cell, lat_cell, marker='x', s=30,
                c=colors[np.mod(j, len(colors))], zorder=3)

    return ax


def add_boundary(ax, tracks, grid, projparams):
    b_list = list(tracks.params['BOUNDARY_GRID_CELLS'])
    boundary = np.zeros((117, 117)) * np.nan
    for i in range(len(b_list)):
        boundary[b_list[i][0], b_list[i][1]] = 1
    x_bounds = grid.x['data'][[0, -1]]
    y_bounds = grid.y['data'][[0, -1]]
    lon_b, lat_b = cartesian_to_geographic(x_bounds, y_bounds, projparams)
    ax.imshow(boundary, extent=(lon_b[0], lon_b[1], lat_b[0], lat_b[1]))

    return ax


def add_stratiform_offset(ax, tracks, grid, uid, date_time):
    tmp_tracks = tracks.tracks.xs((date_time, uid), level=('time', 'uid'))
    lon_low = tmp_tracks.xs(0, level='level')['lon'].iloc[0]
    lat_low = tmp_tracks.xs(0, level='level')['lat'].iloc[0]
    num_levels = len(tracks.params['LEVELS'])
    lon_high = tmp_tracks.xs(num_levels-1, level='level')['lon'].iloc[0]
    lat_high = tmp_tracks.xs(num_levels-1, level='level')['lat'].iloc[0]
    ax.plot(
        [lon_low, lon_high], [lat_low, lat_high], '--b', linewidth=2.0)
    lgd_so = mlines.Line2D(
        [], [], color='b', linestyle='--', linewidth=2.0,
        label='Stratiform Offset')
    return lgd_so


def add_velocities(
        ax, tracks, grid, uid, date_time, var_list=None, c_list=None,
        labels=None):

    if var_list is None:
        var_list = ['shift', 'prop', 'shear', 'cl']
    if c_list is None:
        c_list = ['m', 'green', 'red', 'orange']
    if labels is None:
        labels = [
            'System Velocity', 'Propagation', '0-3 km Shear',
            'Mean Cloud-Layer Winds']

    dt = tracks.record.interval.total_seconds()
    projparams = grid.get_projparams()
    tmp_tracks = tracks.tracks.xs(
        (date_time, uid, 0), level=('time', 'uid', 'level'))

    lon = tmp_tracks['lon'].iloc[0]
    lat = tmp_tracks['lat'].iloc[0]
    x = tmp_tracks['grid_x'].iloc[0]
    y = tmp_tracks['grid_y'].iloc[0]

    for i in range(len(var_list)):
        u = tmp_tracks['u_' + var_list[i]].iloc[0]
        v = tmp_tracks['v_' + var_list[i]].iloc[0]
        [new_lon, new_lat] = cartesian_to_geographic(
            x + 4 * u * dt, y + 4 * v * dt, projparams)
        ax.arrow(
            lon, lat, new_lon[0]-lon, new_lat[0]-lat, color=c_list[i],
            head_width=0.024, head_length=0.040)
    lgd_han = []
    for i in range(len(c_list)):
        lgd_line = mlines.Line2D(
            [], [], color=c_list[i], linestyle='-', label=labels[i])
        lgd_han.append(lgd_line)

    return lgd_han


def format_data(
        grid, winds, coords=['longitude', 'latitude', 'z'],
        vars=['U', 'V', 'W', 'reflectivity']):
    # process winds
    return winds


def add_winds(
        ax, tracks, grid, uid, date_time, alt, params,
        horizontal=True, vertical=True):
    # Load WRF winds corresponding to this grid
    winds = xr.open_dataset(params['winds_fn'])
    winds = format_data(grid, winds)
    winds = winds.sel(z=alt, method='nearest')
    winds = winds.squeeze()

    if horizontal:
        q_hdl = ax.quiver(
            winds.longitude[::4, ::4], winds.latitude[::4, ::4],
            winds.U.values[::4, ::4], winds.V.values[::4, ::4])
        ax.quiverkey(
            q_hdl, .9, 1.025, 10, '10 m/s', labelpos='E', coordinates='axes')
        lgd_winds = None
    if vertical:
        ax.contour(
            winds.longitude, winds.latitude, winds.W.values, colors='black',
            linewidths=2, levels=[-2, 2], linestyles=['--', '-'])
        lgd_winds = mlines.Line2D(
            [], [], color='black', linestyle='-', linewidth=2,
            label='2 m/s Vertical Velocity')
    return lgd_winds
