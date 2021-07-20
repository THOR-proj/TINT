import numpy as np
import matplotlib.lines as mlines
import xarray as xr
from scipy.interpolate import griddata


def add_stratiform_offset(ax, tracks, grid, date_time, params):
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
            tmp_tracks.xs(0, level='level')['orientation'].iloc[0])
        [x_low, y_low] = np.transpose(A).dot(np.array([x_low, y_low]))
        [x_high, y_high] = np.transpose(A).dot(np.array([x_high, y_high]))

    if params['direction'] in ['lat', 'parallel']:
        [h_low, h_high] = [x_low, x_high]
    elif params['direction'] in ['lon', 'perpendicular']:
        [h_low, h_high] = [y_low, y_high]

    ax.plot([h_low, h_high], [low, high], '--b', linewidth=2.0, zorder=3)
    lgd_so = mlines.Line2D(
        [], [], color='b', linestyle='--', linewidth=2.0,
        label='Stratiform Offset')
    return lgd_so


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


def get_line_grid(ds, angle, vars=['U', 'V', 'W', 'reflectivity']):
    print('Interpolating onto line coordinates.')
    new_var_list = []

    for var_name in vars:
        var = ds[var_name].values.squeeze()
        x = ds.x.values
        y = ds.y.values
        z = ds.z.values
        new_var = interp_var(x, y, z, var, angle, var_name)
        new_var_list.append(new_var)

    return xr.merge(new_var_list)


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


def plot_wrf_winds(
        ax, grid, x_draft, y_draft, direction, quiver=False, mp='lin'):
    grid_time = np.datetime64(grid.time['units'][14:]).astype('datetime64[s]')
    # Load WRF winds corresponding to this grid
    base = '/g/data/w40/esh563/{0}d04_ref/{0}d04_ref_'.format(mp)
    fn = glob.glob(base + str(grid_time) + '.nc')
    winds = xr.open_dataset(fn[0])
    if direction == 'lat':
        winds = winds.sel(y=y_draft, method='nearest')
    else:
        winds = winds.sel(x=x_draft, method='nearest')
    winds = winds.squeeze()
    U = winds.U
    V = winds.V
    W = winds.W

    x = np.arange(-145000, 145000+2500, 2500) / 1000
    z = np.arange(0, 20500, 500) / 1000

    if quiver:
        if direction == 'lat':
            ax.quiver(x[::2], z[::2], U.values[::2, ::2], W.values[::2, ::2])
        else:
            ax.quiver(x[::2], z[::2], V.values[::2, ::2], W.values[::2, ::2])
    else:
        ax.contour(
            x, z, W, colors='pink', linewidths=1.5, levels=[-2, 2],
            linestyles=['--', '-'])


def plot_vert_winds_line(
        ax, new_wrf, x_draft_new, y_draft_new, direction, quiver=False,
        average_along_line=False, semi_major=None):

    if direction == 'parallel':
        new_wrf = new_wrf.sel(y=y_draft_new, method='nearest')
        x = new_wrf.x / 1000
    else:
        if average_along_line:
            cond = ((new_wrf.x <= x_draft_new + semi_major*2500/2)
                    & (new_wrf.x >= x_draft_new - semi_major*2500/2))
            new_wrf = new_wrf.where(cond).dropna(dim='x', how='all')
            new_wrf = new_wrf.mean(dim='x')
        else:
            new_wrf = new_wrf.sel(x=x_draft_new, method='nearest')
        x = new_wrf.y / 1000
    new_wrf = new_wrf.squeeze()
    U = new_wrf.U
    V = new_wrf.V
    W = new_wrf.W

    z = np.arange(0, 20500, 500)/1000

    if quiver:
        if direction == 'parallel':
            ax.quiver(x[::2], z[::2], U.values[::2, ::2], W.values[::2, ::2])
        else:
            ax.quiver(x[::2], z[::2], V.values[::2, ::2], W.values[::2, ::2])
    else:
        if average_along_line:
            if W.min() > -1:
                levels = [1]
                linestyles = ['-']
            else:
                levels = [-1, 1]
                linestyles = ['--', '-']
        else:
            if W.min() > -2:
                levels = [2]
                linestyles = ['-']
            else:
                levels = [-2, 2]
                linestyles = ['--', '-']
        ax.contour(x, z, W, colors='pink', linewidths=1.5,
                   levels=levels, linestyles=linestyles)
