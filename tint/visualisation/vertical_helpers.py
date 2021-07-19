def get_line_grid(grid_time, angle, mp='lin'):
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
