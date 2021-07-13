import numpy as np
from matplotlib import rcParams
from matplotlib.patches import Ellipse
import matplotlib.lines as mlines
import glob
import xarray as xr
from scipy.interpolate import griddata
from pyart.core.transforms import cartesian_to_geographic


def init_fonts():
    # Initialise fonts
    rcParams.update({'font.family': 'serif'})
    rcParams.update({'font.serif': 'Liberation Serif'})
    rcParams.update({'mathtext.fontset': 'dejavuserif'})
    rcParams.update({'font.size': 12})


def add_ellipses(ax, frame_tracks_ind, projparams):

    centroid = frame_tracks_ind[['grid_x', 'grid_y']].values
    orientation = frame_tracks_ind[['orientation']].values[0]
    major_axis = frame_tracks_ind[['semi_major']].values[0]
    minor_axis = frame_tracks_ind[['semi_minor']].values[0]

    # Convert axes into lat/lon units by approximating 1 degree lat or
    # lon = 110 km.
    major_axis = major_axis*2.5/110
    minor_axis = minor_axis*2.5/110

    lon_e, lat_e = cartesian_to_geographic(
        centroid[0], centroid[1], projparams)

    ell = Ellipse(
        tuple([lon_e, lat_e]), minor_axis, major_axis, orientation,
        linewidth=1.5, fill=False)

    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.4)
    ax.add_artist(ell)
    return ell


def add_cells(
        ax, grid, frame_tracks_ind, hgt_ind, ud_ind, projparams, grid_size,
        cell_ind=None):

    colors = ['m', 'lime', 'darkorange', 'k', 'b', 'darkgreen', 'yellow']
    if cell_ind is None:
        cell_list = range(len(frame_tracks_ind['cells']))
    else:
        cell_list = [cell_ind]
    for j in cell_list:
        # Plot location of cell j at alts[i] if it exists
        ud_height_inds = np.array(frame_tracks_ind['cells'][j])
        if max(ud_height_inds[:, 0]) >= hgt_ind:
            x_ind = np.array(frame_tracks_ind['cells'][j])[ud_ind, 2]
            y_ind = np.array(frame_tracks_ind['cells'][j])[ud_ind, 1]
            x_draft = grid.x['data'][x_ind]
            y_draft = grid.y['data'][y_ind]
            lon_ud, lat_ud = cartesian_to_geographic(
                x_draft, y_draft, projparams)
            ax.scatter(
                lon_ud, lat_ud, marker='x', s=30,
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


def add_velocities(
        ax, frame_tracks_ind, grid, projparams, dt,
        var_list=['shift', 'prop', 'shear', 'cl'],
        c_list=['m', 'green', 'red', 'orange'],
        labels=[
            'System Velocity', 'Propagation', '0-3 km Shear',
            'Mean Cloud-Layer Winds']):
    lon = frame_tracks_ind['lon']
    lat = frame_tracks_ind['lat']
    x = frame_tracks_ind['grid_x']
    y = frame_tracks_ind['grid_y']

    for i in range(len(var_list)):
        u = frame_tracks_ind['u_' + var_list[i]]
        v = frame_tracks_ind['v_' + var_list[i]]
        [new_lon, new_lat] = cartesian_to_geographic(x + 4*u*dt, y + 4*v*dt,
                                                     projparams)
        ax.arrow(lon, lat, new_lon[0]-lon, new_lat[0]-lat,
                 color=c_list[i], head_width=0.024, head_length=0.040)


def create_vel_leg_han(
    c_list=['m', 'green', 'red', 'orange'],
    labels=[
        'System Velocity', 'Propagation',
        'Low-Level Shear', 'Mean Cloud-Layer Winds']):
    lgd_han = []
    for i in range(len(c_list)):
        lgd_line = mlines.Line2D(
            [], [], color=c_list[i], linestyle='-', label=labels[i])
        lgd_han.append(lgd_line)
    return lgd_han


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

    x = np.arange(-145000, 145000+2500, 2500)/1000
    z = np.arange(0, 20500, 500)/1000

    if quiver:
        if direction == 'lat':
            ax.quiver(x[::2], z[::2], U.values[::2, ::2], W.values[::2, ::2])
        else:
            ax.quiver(x[::2], z[::2], V.values[::2, ::2], W.values[::2, ::2])
    else:
        ax.contour(x, z, W, colors='pink', linewidths=1.5,
                   levels=[-2, 2], linestyles=['--', '-'])


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


def plot_horiz_winds(ax, grid, alt, quiver=False, mp='lin'):
    grid_time = np.datetime64(grid.time['units'][14:]).astype('datetime64[s]')
    # Load WRF winds corresponding to this grid
    base = '/g/data/w40/esh563/{0}d04_ref/{0}d04_ref_'.format(mp)
    fn = glob.glob(base + str(grid_time) + '.nc')
    winds = xr.open_dataset(fn[0])
    winds = winds.sel(z=alt, method='nearest')
    winds = winds.squeeze()
    U = winds.U
    V = winds.V

    if quiver:
        ax.quiver(
            U.lon[::4], U.lat[::4], U.values[::4, ::4], V.values[::4, ::4])
    else:
        ax.contour(
            winds.longitude, winds.latitude, winds.W.values, colors='pink',
            linewidths=1.5, levels=[-2, 2], linestyles=['--', '-'])


def get_line_grid_wrf(grid_time, angle, mp='lin'):
    print('Interpolating WRF')
    # Get reflectivity data
    base = '/g/data/w40/esh563/{0}d04_ref/{0}d04_ref_'.format(mp)
    wrf = xr.open_dataset(base + str(grid_time) + '.nc')

    # Get rotation matrix
    A = np.array([
        [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
        [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]])

    new_var_list = []

    for var_name in ['U', 'V', 'W']:

        var = wrf[var_name].values.squeeze()

        x = wrf.x.values
        y = wrf.y.values
        z = wrf.z.values

        xp = np.arange(-210000, 210000+2500, 2500)
        yp = np.arange(-210000, 210000+2500, 2500)

        Xp, Yp = np.mgrid[-210000:210000+2500:2500, -210000:210000+2500:2500]

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

        # x is line parallel, y is line perpendicular
        # Little confusing how relative position of dimensions has changed.
        new_var = xr.Dataset({
            var_name: (['z', 'y', 'x'],  new_var)},
            coords={'z': z, 'y': yp, 'x': xp})

        new_var_list.append(new_var)

    new_wrf = xr.merge(new_var_list)

    return new_wrf


def get_line_grid(f_tracks, grid, uid, nframe):
    print('Interpolating grid')
    # Get raw grid data
    raw = grid.fields['reflectivity']['data'].data
    raw[raw == -9999] = np.nan
    # Get appropriate tracks data
    cell = f_tracks.tracks.xs(uid, level='uid')
    cell = cell.reset_index(level=['time'])
    cell_low = cell.xs(0, level='level')
    cell_low = cell_low.iloc[nframe]
    # Get appropriate transformation angle
    angle = cell_low.orientation
    semi_major = cell_low.semi_major
    # Get rotation matrix
    A = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
                  [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]])

    x = grid.x['data'].data
    y = grid.y['data'].data
    z = grid.z['data'].data

    xp = np.arange(-210000, 210000+2500, 2500)
    yp = np.arange(-210000, 210000+2500, 2500)

    Xp, Yp = np.mgrid[-210000:210000+2500:2500, -210000:210000+2500:2500]

    new_grid = np.ones((len(z), len(xp), len(yp))) * np.nan

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
                values.append(raw[k, i, j])

        new_grid[k, :, :] = griddata(points_new, values, (Xp, Yp))

    # x is line parallel, y is line perpendicular
    # Little confusing how relative position of dimensions has changed.
    new_grid = xr.Dataset(
        {'reflectivity': (['z', 'y', 'x'],  new_grid)},
        coords={'z': z, 'y': yp, 'x': xp})

    return new_grid, A, angle, semi_major
