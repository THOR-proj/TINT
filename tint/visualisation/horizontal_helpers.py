import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import xarray as xr
from pyart.core.transforms import cartesian_to_geographic
import cartopy.crs as ccrs

from tint.grid_utils import get_grid_size, get_grid_alt


def gen_embossed_text(
        x, y, ax, label, transform, fontsize, linewidth, zorder):
    ax.text(
        x, y, label, transform=transform, fontsize=fontsize,
        zorder=zorder, fontweight='bold', color='w',
        path_effects=[
            pe.Stroke(linewidth=linewidth, foreground='k'), pe.Normal()])


def add_tracked_objects(tracks, grid, date_time, params, ax, alt):

    projection = ccrs.PlateCarree()
    tmp_tracks = tracks.tracks.xs(date_time, level='time')
    tmp_class = tracks.tracks_class.xs(date_time, level='time')
    tmp_excl = tracks.exclusions.xs(date_time, level='time')
    if params['uid_ind'] is None:
        uids = np.unique(tmp_tracks.index.get_level_values('uid').values)
    else:
        uids = [params['uid_ind']]
    lgd_han = []

    label_dic = {
        'Front Fed': 'FF', 'Rear Fed': 'RF', 'Parallel Fed (Right)': 'RiF',
        'Parallel Fed (Left)': 'LeF',
        'Ambiguous (Low Velocity)': 'A(LV)',
        'Ambiguous (Low Relative Velocity)': 'A(LRV)',
        'Down-Shear Propagating': 'DSP',
        'Up-Shear Propagating': 'USP',
        'Ambiguous (Low Shear)': 'A(LS)',
        'Ambiguous (Low Relative Velocity)': 'A(LRV)',
        'Down-Shear Tilted': 'DST', 'Up-Shear Tilted': 'UST',
        'Ambiguous (Shear Parallel to Stratiform Offset)': 'A(SP)',
        'Ambiguous (Small Stratiform Offset)': 'A(SO)',
        'Ambiguous (Small Shear)': 'A(SS)',
        'Leading Stratiform': 'LS', 'Trailing Stratiform': 'TS',
        'Parallel Stratiform (Left)': 'LeS',
        'Parallel Stratiform (Right)': 'RiS',
        'Ambiguous (Small Stratiform Offset)': 'A(SO)',
        'Ambiguous (Small Velocity)': 'A(SV)',
        'Ambiguous (On Quadrant Boundary)': 'A(QB)'}

    for uid in uids:
        tmp_tracks_uid = tmp_tracks.xs(uid, level='uid')
        tmp_class_uid = tmp_class.xs(uid, level='uid')
        tmp_excl_uid = tmp_excl.xs(uid, level='uid')

        excluded = tmp_excl_uid[params['exclusions']]
        excluded = excluded.xs(0, level='level').iloc[0]
        excluded = np.any(excluded)

        if not excluded:
            lon = tmp_tracks_uid.xs(0, level='level')['lon'].iloc[0]
            lat = tmp_tracks_uid.xs(0, level='level')['lat'].iloc[0]

            gen_embossed_text(
                lon-.2, lat+0.1, ax, uid, transform=projection, fontsize=16,
                linewidth=3, zorder=5)

            if params['label_mergers']:
                mergers = list(
                    tmp_tracks_uid.xs(0, level='level')['mergers'].iloc[0])
                label = ", ".join(mergers)
                gen_embossed_text(
                    lon+.1, lat-0.1, ax, label, transform=projection,
                    fontsize=18, linewidth=2, zorder=5)

            if params['label_type']:
                label_1 = label_dic[tmp_class_uid.xs(
                    0, level='level')['offset_type'].values[0]]
                label_2 = label_dic[tmp_class_uid.xs(
                    0, level='level')['inflow_type'].values[0]]
                type_fontsize = 16
                gen_embossed_text(
                    lon+.1, lat-0.1, ax, label_1, transform=projection,
                    fontsize=type_fontsize, linewidth=2, zorder=5)
                gen_embossed_text(
                    lon+.1, lat, ax, label_2, transform=projection,
                    fontsize=type_fontsize, linewidth=2, zorder=5)
                non_linear = tmp_excl_uid.xs(0, level='level')['non_linear']
                if non_linear.values[0]:
                    label = 'NL'
                else:
                    label = 'L'
                gen_embossed_text(
                    lon+.1, lat+0.1, ax, label, transform=projection,
                    fontsize=type_fontsize, linewidth=2, zorder=5)

            if params['label_splits']:
                parent = list(
                    tmp_tracks_uid.xs(0, level='level')['parent'].iloc[0])
                label = ", ".join(parent)
                gen_embossed_text(
                    lon+.1, lat+0.15, ax, label, transform=projection,
                    fontsize=12, linewidth=2, zorder=5)

        # Plot velocities
        lgd_vel = add_velocities(
            ax, tracks, grid, uid, date_time, alt, params['system_winds'],
            excluded)

        # Plot stratiform offset
        lgd_so = add_stratiform_offset(
            ax, tracks, grid, uid, date_time, excluded)

        lgd_ellipse = add_ellipses(
            ax, tracks, grid, uid, date_time, alt, excluded)
        # lgd_cell = add_cells(ax, tracks, grid, uid, date_time, alt)

        if params['boundary']:
            add_boundary(ax, tracks, grid)

        if tracks.params['RAIN']:
            add_rain()

    lgd_han.append(lgd_so)
    [lgd_han.append(h) for h in lgd_vel]
    # lgd_han.append(lgd_cell)
    lgd_han.append(lgd_ellipse)

    return lgd_han


def reduce_tracks(tracks, uid, date_time, alt):
    tests = [
        (alt >= level[0] and alt < level[1])
        for level in tracks.params['LEVELS']]
    level = np.argwhere(tests)[0, 0]
    tmp_tracks = tracks.tracks.xs(
        (date_time, uid, level), level=('time', 'uid', 'level'))
    tmp_sys_tracks = tracks.tracks.xs(
        (date_time, uid), level=('time', 'uid'))
    return tmp_tracks, tmp_sys_tracks


def add_rain(ax, tracks, grid, uid, date_time):

    projparams = grid.get_projparams()
    projection = ccrs.PlateCarree()
    alt = tracks.params['LEVELS'][0, 0]
    tmp_tracks = reduce_tracks(tracks, uid, date_time, alt=alt)[0]

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


def add_ellipses(ax, tracks, grid, uid, date_time, alt, excluded=False):

    projparams = grid.get_projparams()
    tmp_tracks = reduce_tracks(tracks, uid, date_time, alt)[0]

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
        linewidth=1.5, fill=False, zorder=3, color='grey', linestyle='--')
    lgd_ellipse = mlines.Line2D(
        [], [], color='grey', linewidth=1.5, label='Best Fit Ellipse',
        linestyle='--')

    ell.set_clip_box(ax.bbox)
    # ell.set_alpha(0.4)
    if not excluded:
        ax.add_artist(ell)
    return lgd_ellipse


def add_cells(ax, tracks, grid, uid, date_time, alt, cell_ind=None):
    projparams = grid.get_projparams()
    tmp_tracks = reduce_tracks(tracks, uid, date_time, alt)[0]
    alt_ind = get_grid_alt(get_grid_size(grid), alt)
    colors = ['k']
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
            color = colors[np.mod(j, len(colors))]
            ax.scatter(
                lon_cell, lat_cell, marker='o', s=8, linewidth=1, c='w',
                zorder=2, path_effects=[
                    pe.Stroke(linewidth=5, foreground=color), pe.Normal()])
    lgd_cell = mlines.Line2D(
        [], [], color='w', marker='o', markersize=2, linewidth=1,
        linestyle='None', label='Convective Cells', path_effects=[
            pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])

    return lgd_cell


def add_boundary(ax, tracks, grid):
    projparams = grid.get_projparams()
    b_list = list(tracks.params['BOUNDARY_GRID_CELLS'])
    boundary = np.zeros((117, 117)) * np.nan
    for i in range(len(b_list)):
        boundary[b_list[i][0], b_list[i][1]] = .75
    x_bounds = grid.x['data'][[0, -1]]
    y_bounds = grid.y['data'][[0, -1]]
    lon_b, lat_b = cartesian_to_geographic(x_bounds, y_bounds, projparams)
    ax.imshow(
        boundary, extent=(lon_b[0], lon_b[1], lat_b[0], lat_b[1]), cmap='gray',
        vmin=0, vmax=1)

    return ax


def add_stratiform_offset(ax, tracks, grid, uid, date_time, excluded):
    tmp_tracks = tracks.tracks.xs((date_time, uid), level=('time', 'uid'))
    lon_low = tmp_tracks.xs(0, level='level')['lon'].iloc[0]
    lat_low = tmp_tracks.xs(0, level='level')['lat'].iloc[0]
    num_levels = len(tracks.params['LEVELS'])
    lon_high = tmp_tracks.xs(num_levels-1, level='level')['lon'].iloc[0]
    lat_high = tmp_tracks.xs(num_levels-1, level='level')['lat'].iloc[0]
    if not excluded:
        ax.plot(
            [lon_low, lon_high], [lat_low, lat_high], '-w', linewidth=2,
            zorder=4, path_effects=[
                pe.Stroke(linewidth=6, foreground='b'), pe.Normal()])
    lgd_so = mlines.Line2D(
        [], [], color='w', linestyle='-', linewidth=2,
        label='Stratiform Offset',
        path_effects=[pe.Stroke(linewidth=6, foreground='b'), pe.Normal()])
    return lgd_so


def add_velocities(
        ax, tracks, grid, uid, date_time, alt, system_winds, excluded):

    level_test = [
        alt >= lvl[0] and alt < lvl[1] for lvl in tracks.params['LEVELS']]
    level_ind = np.where(level_test)[0][0]
    level = tracks.params['LEVELS'][level_ind]
    interval = np.arange(
        level[0], level[1], tracks.record.grid_size[0])
    mid_i = len(interval) // 2
    ambient_mid_alt = int(interval[mid_i])
    ambient_bottom_alt = int(interval[0])
    ambient_top_alt = int(interval[-1])

    colour_dic = {
        'shift': 'm', 'ambient_bottom': 'red', 'ambient_mid': 'red',
        'ambient_top': 'red', 'ambient_mean': 'red', 'relative': 'darkgreen',
        'shear': 'darkblue'}
    label_dic = {
        'shift': 'System Velocity',
        'ambient_bottom': '{} m Winds'.format(ambient_bottom_alt),
        'ambient_mid': '{} m Winds'.format(ambient_mid_alt),
        'ambient_top': '{} m Winds'.format(ambient_top_alt),
        'ambient_mean': '{}-{} m Mean Winds'.format(
            ambient_bottom_alt, ambient_top_alt),
        'relative': 'Relative System Velocity',
        'shear': '{}-{} m Shear'.format(
            ambient_bottom_alt, ambient_top_alt)}

    projparams = grid.get_projparams()
    tmp_tracks = tracks.tracks.xs(
        (date_time, uid, level_ind), level=('time', 'uid', 'level'))
    tmp_tracks_conv = tracks.tracks.xs(
        (date_time, uid, 0), level=('time', 'uid', 'level'))

    lon = tmp_tracks_conv['lon'].iloc[0]
    lat = tmp_tracks_conv['lat'].iloc[0]
    x = tmp_tracks_conv['grid_x'].iloc[0]
    y = tmp_tracks_conv['grid_y'].iloc[0]

    dt = tracks.record.interval.total_seconds()

    lgd_han = []
    for wind in system_winds:
        u = tmp_tracks['u_' + wind].iloc[0]
        v = tmp_tracks['v_' + wind].iloc[0]
        [new_lon, new_lat] = cartesian_to_geographic(
            x + 4 * u * dt, y + 4 * v * dt, projparams)
        if not excluded:
            q_hdl = ax.arrow(
                lon, lat, new_lon[0]-lon, new_lat[0]-lat, color='w', zorder=4,
                head_width=0.016, head_length=0.024, length_includes_head=True,
                path_effects=[
                    pe.Stroke(linewidth=6, foreground=colour_dic[wind]),
                    pe.Normal()])
        lgd_line = mlines.Line2D(
            [], [], color='w', linestyle='-', label=label_dic[wind],
            linewidth=2, path_effects=[
                pe.Stroke(linewidth=6, foreground=colour_dic[wind]),
                pe.Normal()])
        lgd_han.append(lgd_line)

    # Extra factor of 2 - see definition of quiver scale
    scale = 95000 / (4 * dt)
    q_hdl = ax.quiver(
        lon, lat, 0, 0, scale_units='x', scale=scale, zorder=0, linewidth=1)
    ax.quiverkey(
        q_hdl, .9, 1.025, 10, '10 m/s', labelpos='E', coordinates='axes')

    return lgd_han


def format_data(
        grid, winds, coords=['longitude', 'latitude', 'z'],
        vars=['U', 'V', 'W', 'reflectivity']):
    # process winds
    return winds


def add_winds(
        ax, tracks, grid, date_time, alt, params,
        horizontal=True, vertical=False):
    # Load WRF winds corresponding to this grid
    winds = xr.open_dataset(params['winds_fn'])
    winds = format_data(grid, winds)
    winds = winds.sel(z=alt, method='nearest')
    winds = winds.squeeze()

    if horizontal:
        dt = tracks.record.interval.total_seconds()
        scale = 95000 / (4 * dt)
        q_hdl = ax.quiver(
            winds.longitude[4::8, 4::8], winds.latitude[4::8, 4::8],
            winds.U.values[4::8, 4::8], winds.V.values[4::8, 4::8],
            scale_units='x', scale=scale)
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
