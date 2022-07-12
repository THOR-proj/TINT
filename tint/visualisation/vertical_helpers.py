import numpy as np
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import xarray as xr
from scipy.interpolate import griddata
from pyart.core.transforms import cartesian_to_geographic

from tint.grid_utils import get_grid_alt, get_grid_size


def add_stratiform_offset(ax, tracks, grid, date_time, params):
    angle = None
    tmp_tracks = tracks.tracks.xs(
        (date_time, params['uid_ind']),
        level=('time', 'uid'))
    low = tracks.params['LEVELS'][0].mean() / 1000
    high = tracks.params['LEVELS'][-1].mean() / 1000

    x_low = tmp_tracks.xs(0, level='level')['grid_x'].iloc[0]
    y_low = tmp_tracks.xs(0, level='level')['grid_y'].iloc[0]
    num_levels = len(tracks.params['LEVELS'])
    x_high = tmp_tracks.xs(num_levels-1, level='level')['grid_x'].iloc[0]
    y_high = tmp_tracks.xs(num_levels-1, level='level')['grid_y'].iloc[0]
    [x_low, x_high, y_low, y_high] = [
        coord / 1000 for coord in [x_low, x_high, y_low, y_high]]

    if params['line_coords']:
        A = get_rotation(
            tmp_tracks.xs(0, level='level')['orientation_alt'].iloc[0])
        [x_low, y_low] = np.transpose(A).dot(np.array([x_low, y_low]))
        [x_high, y_high] = np.transpose(A).dot(np.array([x_high, y_high]))

    if params['direction'] in ['lat', 'parallel']:
        [h_low, h_high] = [x_low, x_high]
    elif params['direction'] in ['lon', 'perpendicular']:
        [h_low, h_high] = [y_low, y_high]
        [dz, dh] = [high - low, h_high - h_low]
        angle = np.rad2deg(np.arctan2(dz, dh))

    ax.plot(
        [h_low, h_high], [low, high], '-w', linewidth=2, zorder=3,
        path_effects=[pe.Stroke(linewidth=6, foreground='b'), pe.Normal()])
    lgd_so = mlines.Line2D(
        [], [], color='w', linestyle='-', linewidth=2,
        label='Stratiform Offset',
        path_effects=[pe.Stroke(linewidth=6, foreground='b'), pe.Normal()])
    return lgd_so, angle


def add_cell(ax, tracks, grid, date_time, params):
    tmp_tracks = tracks.tracks.xs(
        (date_time, params['uid_ind'], 0), level=('time', 'uid', 'level'))
    cell = np.array(tmp_tracks['cells'].iloc[0][params['cell_ind']])
    x = grid.x['data'].data[cell[:, 2]]
    y = grid.y['data'].data[cell[:, 1]]

    if params['line_coords']:
        cell_coords = np.array([x, y])
        A = get_rotation(tmp_tracks['orientation_alt'].iloc[0])
        cell_coords = np.array([
            np.transpose(A).dot(cell_coords[:, i])
            for i in range(cell_coords.shape[1])])
        [x, y] = [cell_coords[:, 0], cell_coords[:, 1]]

    z0 = get_grid_alt(tracks.record.grid_size, tracks.params['CELL_START'])
    z = grid.z['data'].data[z0: len(x)+z0]

    if params['direction'] in ['lat', 'parallel']:
        h = x
    elif params['direction'] in ['lon', 'perpendicular']:
        h = y

    colors = ['k']
    color = colors[np.mod(params['cell_ind'], len(colors))]
    ax.plot(
        h / 1000, z / 1000, color='w', linewidth=2,
        path_effects=[pe.Stroke(linewidth=6, foreground=color), pe.Normal()])
    lgd_cell = mlines.Line2D(
        [], [], color='w', linestyle='-', linewidth=2, label='Convective Cell',
        path_effects=[pe.Stroke(linewidth=6, foreground=color), pe.Normal()])
    return lgd_cell


def get_rotation(angle):
    A = np.array([
        [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
        [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]])
    return A


def interp_var(
        x, y, z, var, angle, var_name, min=-210000, max=210000, step=2500):
    # Get rotation matrix
    A = get_rotation(angle)
    xp = np.arange(min, max + step, step)
    yp = np.arange(min, max + step, step)
    Xp, Yp = np.mgrid[
        min: max + step: step, min: max + step: step]
    new_var = np.ones((len(z), len(xp), len(yp))) * np.nan

    points_old = []
    for i in range(len(x)):
        for j in range(len(y)):
            points_old.append(np.array([x[i], y[j]]))

    points_new = []
    for i in range(len(points_old)):
        points_new.append(A.dot(points_old[i]))

    for k in range(len(z)):
        values = []
        for i in range(len(x)):
            for j in range(len(y)):
                values.append(var[k, i, j])
        new_var[k, :, :] = griddata(points_new, values, (Xp, Yp))

    # x is now line parallel, y is now line perpendicular
    new_var = xr.Dataset({
        var_name: (['z', 'y', 'x'],  new_var)},
        coords={'z': z, 'y': yp, 'x': xp})

    return new_var


def get_line_grid(ds, angle, variables=['U', 'V', 'W', 'reflectivity']):
    print('Interpolating onto line coordinates.')
    new_var_list = []
    for var_name in variables:
        var = ds[var_name].values.squeeze()
        x = ds.x.values
        y = ds.y.values
        z = ds.z.values
        new_var = interp_var(x, y, z, var, angle, var_name)
        new_var_list.append(new_var)

    return xr.merge(new_var_list)


def rebase_horizontal_winds(ds, angle):
    print('Calculating horizontal winds in new basis.')
    A = get_rotation(angle)
    for i in range(len(ds.z.values)):
        [U, V] = [ds.U.values[i, :, :], ds.V.values[i, :, :]]
        winds = np.array([U.flatten(), V.flatten()])
        new_winds = np.transpose(A).dot(winds)
        shape = ds.U.values[i, :].shape
        ds.U.values[i, :] = new_winds[0].reshape(shape)
        ds.V.values[i, :] = new_winds[1].reshape(shape)
    return ds


def format_pyart(grid):
    raw = grid.fields['reflectivity']['data'].data
    raw[raw == -9999] = np.nan
    x = grid.x['data'].data
    y = grid.y['data'].data
    z = grid.z['data'].data
    ds = xr.Dataset(
        {'reflectivity': (['z', 'y', 'x'],  raw)},
        coords={'z': z, 'y': y, 'x': x})
    return ds


def get_center_coords(tracks, grid, params, date_time):
    if params['uid_ind'] is not None:
        tmp_tracks = tracks.tracks.xs(
            (date_time, params['uid_ind'], 0), level=('time', 'uid', 'level'))
        if params['center_cell']:
            cell = np.array(tmp_tracks['cells'].iloc[0][params['cell_ind']])
            x = grid.x['data'].data[cell[0, 2]]
            y = grid.y['data'].data[cell[0, 1]]
            projparams = grid.get_projparams()
            lon, lat = cartesian_to_geographic(x, y, projparams)
        else:
            lon = tmp_tracks['lon'].iloc[0]
            lat = tmp_tracks['lat'].iloc[0]
            x = tmp_tracks['grid_x'].iloc[0]
            y = tmp_tracks['grid_y'].iloc[0]
        if params['line_coords']:
            A = get_rotation(tmp_tracks['orientation_alt'].iloc[0])
            [x, y] = np.transpose(A).dot(np.array([x, y]))
    else:
        lon = tracks.radar_info['radar_lon']
        lat = tracks.radar_info['radar_lat']
        [x, y] = [0, 0]

    return lon, lat, x, y


def add_quiver(ds, ax, params):
    if params['direction'] in ['lat', 'parallel']:
        [x, U] = [ds.x, ds.U.values]
    else:
        [x, U] = [ds.y, ds.V.values]
    q_hdl = ax.quiver(
        x[::2] / 1000, ds.z[::2] / 1000, U[::2, ::2],
        ds.W.values[::2, ::2], zorder=2, color='k')
    ax.quiverkey(
        q_hdl, .9, 1.025, 10, '10 m/s', labelpos='E',
        coordinates='axes')


def add_streamplot(ds, ax, params):
    if params['direction'] in ['lat', 'parallel']:
        [x, U] = [ds.x, ds.U.values]
    else:
        [x, U] = [ds.y, ds.V.values]
    if params['line_average']:
        density = 1.5
    else:
        density = 2.5
    ax.streamplot(
        x[::2].values / 1000, ds.z.values[::2] / 1000, U[::2, ::2],
        ds.W.values[::2, ::2], zorder=2, color='k', density=density,
        linewidth=1)
    lgd_wind = mlines.Line2D(
        [], [], color='k', linestyle='-', marker='>', linewidth=1,
        label='Relative Streamlines')
    return lgd_wind


def get_streamline_angles(ds, ax, tracks, grid, date_time, params):
    lon, lat, x, y = get_center_coords(tracks, grid, params, date_time)
    [dz, dy, dx] = get_grid_size(grid)
    tmp_tracks = tracks.tracks.xs(
        (date_time, params['uid_ind'], 0), level=('time', 'uid', 'level'))
    semi_minor = tmp_tracks['semi_minor'].iloc[0]
    cond = ((ds.y <= y + semi_minor * dy / 2)
            & (ds.y >= y - semi_minor * dy / 2))
    tmp_ds = ds.where(cond).dropna(dim='y', how='all')
    w_max = tmp_ds.W.max().values
    w_cond = tmp_ds.W.values > 1
    angles = np.arctan2(
        tmp_ds.W.values[w_cond], tmp_ds.V.values[w_cond]).flatten()
    angles = angles[~np.isnan(angles)]
    angles = np.rad2deg(angles)
    if len(angles) == 0:
        angle = np.nan
        max_count = np.nan
    else:
        hist = np.histogram(angles, bins=72, range=(0, 180))
        labels = hist[1][:-1] + np.diff(hist[1]) / 2
        angle = labels[np.argmax(hist[0])]
        max_count = np.max(hist[0])

    return angle, w_max, len(angles), np.prod(w_cond.shape), max_count


def add_winds(ds, ax, tracks, grid, date_time, params):
    lon, lat, x, y = get_center_coords(tracks, grid, params, date_time)
    angle = None
    [dz, dy, dx] = get_grid_size(grid)
    if params['direction'] in ['lat', 'parallel']:
        tmp_ds = ds.sel(y=y, method='nearest').squeeze()
        if params['streamplot']:
            lgd_wind = add_streamplot(tmp_ds, ax, params)
        else:
            add_quiver(tmp_ds, ax, params)
            lgd_wind = None
    elif params['direction'] in ['lon', 'perpendicular']:
        if params['line_coords']:
            tmp_tracks = tracks.tracks.xs(
                (date_time, params['uid_ind'], 0),
                level=('time', 'uid', 'level'))
            semi_major = tmp_tracks['semi_major'].iloc[0]
            cond = ((ds.x <= x + semi_major * dx / 2)
                    & (ds.x >= x - semi_major * dx / 2))
            tmp_ds = ds.where(cond).dropna(dim='x', how='all').mean(dim='x')
            angle, w_max, n_angles, n_obs, max_count = get_streamline_angles(
                tmp_ds, ax, tracks, grid, date_time, params)
        else:
            tmp_ds = ds.sel(x=x, method='nearest').squeeze()
        if params['streamplot']:
            lgd_wind = add_streamplot(tmp_ds, ax, params)
        else:
            add_quiver(tmp_ds, ax, params)
            lgd_wind = None

    return lgd_wind, angle, w_max, n_angles, n_obs, max_count


def transform_ds(ds, tracks, params, date_time, variables):

    tmp_tracks = tracks.tracks.xs(
        (date_time, params['uid_ind'], 0), level=('time', 'uid', 'level'))
    ds = get_line_grid(
        ds, tmp_tracks['orientation'].iloc[0], variables=variables)
    if params['winds']:
        if params['relative_winds']:
            ds['U'] = ds['U'] - tmp_tracks['u_shift'].values
            ds['V'] = ds['V'] - tmp_tracks['v_shift'].values
        ds = rebase_horizontal_winds(
            ds, tmp_tracks['orientation'].iloc[0])
    return ds, tmp_tracks


def setup_line_coords(ds, tracks, params, date_time, variables):
    if not params['load_line_coords_winds']:
        ds, tmp_tracks = transform_ds(
            ds, tracks, params, date_time, variables)
    else:
        try:
            tmp_tracks = tracks.tracks.xs(
                (date_time, params['uid_ind'], 0),
                level=('time', 'uid', 'level'))
            fn = '{}/{}_{}.nc'.format(
                params['save_dir'], params['uid_ind'], date_time)
            ds = xr.open_dataset(fn)
            print('Transformed data loaded from disk.')
        except FileNotFoundError:
            params['load_line_coords_winds'] = False
            print('Transformed data not found. Calculating.')
            ds, tmp_tracks = transform_ds(
                ds, tracks, params, date_time, variables)
    return ds, tmp_tracks


def setup_perpendicular_coords(ds, x, tmp_tracks, params, dx):
    # Swap 2500 to general size
    # import pdb; pdb.set_trace()
    if params['line_average']:
        semi_major = tmp_tracks['semi_major'].iloc[0]
        cond = ((ds.x <= x + semi_major * dx / 2)
                & (ds.x >= x - semi_major * dx / 2))
        ds_plot = ds.where(cond).dropna(dim='x', how='all').mean(dim='x')
        t_string = 'Line Perpendicular Mean Cross Section'
    else:
        ds_plot = ds.sel(x=x, method='nearest').squeeze()
        t_string = 'Line Perpendicular Cross Section'
    return ds_plot, t_string
