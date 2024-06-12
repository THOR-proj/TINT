import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import xarray as xr
from pyart.core.transforms import cartesian_to_geographic
from pyart.core.transforms import geographic_to_cartesian
import cartopy.crs as ccrs
import pandas as pd

from tint.grid_utils import get_grid_size, get_grid_alt


def gen_embossed_text(
        x, y, ax, label, transform, fontsize, linewidth, zorder, color='k'):
    ax.text(
        x, y, label, transform=transform, fontsize=fontsize,
        zorder=zorder, fontweight='bold', color='w',
        path_effects=[
            pe.Stroke(linewidth=linewidth, foreground=color), pe.Normal()])


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
        'Relative Trailing Stratiform': 'RTS',
        'Relative Leading Stratiform': 'RLS',
        'Relative Parallel Stratiform (Left)': 'RLeS',
        'Relative Parallel Stratiform (Right)': 'RRiS',
        'Ambiguous (Low Velocity)': 'A(LV)',
        'Ambiguous (Low Relative Velocity)': 'A(LRV)',
        'Down-Shear Propagating': 'DSP',
        'Up-Shear Propagating': 'USP',
        'Ambiguous (Low Shear)': 'A(LS)',
        'Ambiguous (Low Relative Velocity)': 'A(LRV)',
        'Down-Shear Tilted': 'DST', 'Up-Shear Tilted': 'UST',
        'Ambiguous (Perpendicular Shear)': 'A(SP)',
        'Ambiguous (Small Stratiform Offset)': 'A(SO)',
        'Ambiguous (Small Shear)': 'A(LS)',
        'Leading Stratiform': 'LS', 'Trailing Stratiform': 'TS',
        'Parallel Stratiform (Left)': 'LeS',
        'Parallel Stratiform (Right)': 'RiS',
        'Ambiguous (Small Stratiform Offset)': 'A(SO)',
        'Ambiguous (Small Velocity)': 'A(LV)',
        'Ambiguous (On Quadrant Boundary)': 'A(QB)'}

    label_dic = {
        'Front Fed': 'FF', 'Rear Fed': 'RF', 'Parallel Fed (Right)': 'RiF',
        'Parallel Fed (Left)': 'LeF',
        'Relative Trailing Stratiform': 'RTS',
        'Relative Leading Stratiform': 'RLS',
        'Relative Parallel Stratiform (Left)': 'RLeS',
        'Relative Parallel Stratiform (Right)': 'RRiS',
        'Ambiguous (Low Velocity)': 'A',
        'Ambiguous (Low Relative Velocity)': 'A',
        'Down-Shear Propagating': 'DSP',
        'Up-Shear Propagating': 'USP',
        'Ambiguous (Low Shear)': 'A',
        'Ambiguous (Low Relative Velocity)': 'A',
        'Down-Shear Tilted': 'DST', 'Up-Shear Tilted': 'UST',
        'Ambiguous (Perpendicular Shear)': 'A',
        'Ambiguous (Small Stratiform Offset)': 'A',
        'Ambiguous (Small Shear)': 'A',
        'Leading Stratiform': 'LS', 'Trailing Stratiform': 'TS',
        'Parallel Stratiform (Left)': 'LeS',
        'Parallel Stratiform (Right)': 'RiS',
        'Ambiguous (Small Stratiform Offset)': 'A',
        'Ambiguous (Small Velocity)': 'A',
        'Ambiguous (On Quadrant Boundary)': 'A'}

    emb_lw = 4

    for uid in uids:
        tmp_tracks_uid = tmp_tracks.xs(uid, level='uid')
        tmp_class_uid = tmp_class.xs(uid, level='uid')
        tmp_excl_uid = tmp_excl.xs(uid, level='uid')

        excluded = tmp_excl_uid[params['exclusions']]
        excluded = excluded.xs(0, level='level').iloc[0]
        excluded = np.any(excluded)

        int_border = np.any(
            tmp_excl_uid[[
                'intersect_border', 'intersect_border_convective',
                'small_area', 'large_area']].values)

        if not params['exclude']:
            excluded = False

        if not excluded:
            lon = tmp_tracks_uid.xs(0, level='level')['lon'].iloc[0]
            lat = tmp_tracks_uid.xs(0, level='level')['lat'].iloc[0]

            #gen_embossed_text(
            #    lon-.2, lat+0.2, ax, uid, transform=projection, fontsize=16,
            #    linewidth=emb_lw, zorder=5)

        if not excluded and not int_border:

            if params['label_mergers']:
                mergers = list(
                    tmp_tracks_uid.xs(0, level='level')['mergers'].iloc[0])
                label = ", ".join(mergers)
                gen_embossed_text(
                    lon+.1, lat-0.1, ax, label, transform=projection,
                    fontsize=18, linewidth=emb_lw, zorder=5)

            type_fontsize = 16

            if params['label_type'] == 'velocities':
                label_1 = label_dic[tmp_class_uid.xs(
                    0, level='level')['offset_type'].values[0]]
                label_2 = label_dic[tmp_class_uid.xs(
                    0, level='level')['inflow_type'].values[0]]
                label_3 = label_dic[tmp_class_uid.xs(
                    0, level='level')['rel_offset_type'].values[0]]
                gen_embossed_text(
                    lon-.1, lat-0.3, ax, label_1, transform=projection,
                    fontsize=type_fontsize, linewidth=emb_lw, zorder=5)
                gen_embossed_text(
                    lon-.1, lat-0.2, ax, label_2, transform=projection,
                    fontsize=type_fontsize, linewidth=emb_lw, zorder=5)
                gen_embossed_text(
                    lon-.1, lat-0.4, ax, label_3, transform=projection,
                    fontsize=type_fontsize, linewidth=emb_lw, zorder=5)
            elif params['label_type'] == 'shear':
                label_1 = label_dic[tmp_class_uid.xs(
                    0, level='level')['propagation_type'].values[0]]
                label_2 = label_dic[tmp_class_uid.xs(
                    0, level='level')['tilt_type'].values[0]]
                gen_embossed_text(
                    lon+.1, lat-0.1, ax, label_1, transform=projection,
                    fontsize=type_fontsize, linewidth=emb_lw, zorder=5)
                gen_embossed_text(
                    lon+.1, lat, ax, label_2, transform=projection,
                    fontsize=type_fontsize, linewidth=emb_lw, zorder=5)

                label_1 = label_dic[tmp_class_uid.xs(
                    0, level='level')['offset_type'].values[0]]
                label_2 = label_dic[tmp_class_uid.xs(
                    0, level='level')['inflow_type'].values[0]]
                label_3 = label_dic[tmp_class_uid.xs(
                    0, level='level')['rel_offset_type'].values[0]]
                gen_embossed_text(
                    lon-.1, lat-0.3, ax, label_1, transform=projection,
                    fontsize=type_fontsize, linewidth=emb_lw, zorder=5)
                gen_embossed_text(
                    lon-.1, lat-0.2, ax, label_2, transform=projection,
                    fontsize=type_fontsize, linewidth=emb_lw, zorder=5)
                gen_embossed_text(
                    lon-.1, lat-0.4, ax, label_3, transform=projection,
                    fontsize=type_fontsize, linewidth=emb_lw, zorder=5)
            else:
                label_1 = label_dic[tmp_class_uid.xs(
                    0, level='level')['inflow_type'].values[0]]
                label_2 = label_dic[tmp_class_uid.xs(
                    0, level='level')['offset_type'].values[0]]
                label_3 = label_dic[tmp_class_uid.xs(
                    0, level='level')['tilt_type'].values[0]]
                label_4 = label_dic[tmp_class_uid.xs(
                    0, level='level')['propagation_type'].values[0]]

                gen_embossed_text(
                    lon+.1, lat-.3, ax, label_1, transform=projection,
                    fontsize=type_fontsize, linewidth=emb_lw, zorder=5)
                gen_embossed_text(
                    lon+.35, lat-.3, ax, label_2, transform=projection,
                    fontsize=type_fontsize, linewidth=emb_lw, zorder=5)
                gen_embossed_text(
                    lon+.1, lat-0.425, ax, label_3, transform=projection,
                    fontsize=type_fontsize, linewidth=emb_lw, zorder=5)
                gen_embossed_text(
                    lon+.35, lat-0.425, ax, label_4, transform=projection,
                    fontsize=type_fontsize, linewidth=emb_lw, zorder=5)

            non_linear = tmp_excl_uid.xs(0, level='level')['non_linear']
            if non_linear.values[0]:
                label = 'NL'
            else:
                label = 'L'
            #gen_embossed_text(
            #    lon+.1, lat+0.1, ax, label, transform=projection,
            #    fontsize=type_fontsize, linewidth=emb_lw, zorder=5)

            if params['label_splits']:
                parent = list(
                    tmp_tracks_uid.xs(0, level='level')['parent'].iloc[0])
                label = ", ".join(parent)
                gen_embossed_text(
                    lon+.1, lat+0.15, ax, label, transform=projection,
                    fontsize=12, linewidth=2, zorder=5)

        # Plot stratiform offset
        lgd_so = add_stratiform_offset(
            ax, tracks, grid, uid, date_time, excluded)

        # Plot velocities
        lgd_vel = add_velocities(
            ax, tracks, grid, uid, date_time, alt, params['system_winds'],
            excluded)

        add_ellipses(
            ax, tracks, grid, uid, date_time, alt, params, excluded)
        # lgd_cell = add_cells(ax, tracks, grid, uid, date_time, alt)

        if params['label_cells']:
            lgd_cell = add_cells(
                ax, tracks, grid, uid, date_time, alt, cell_ind=None)

        if tracks.params['RAIN']:
            add_rain()

    add_reports(ax, tracks, uid, grid, date_time, params)

    lgd_han.append(lgd_so)
    [lgd_han.append(h) for h in lgd_vel]
    if params['label_cells']:
        lgd_han.append(lgd_cell)
    #lgd_han.append(lgd_report)
    #lgd_han.append(lgd_m_report)

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


def add_reports(ax, tracks, uid, grid, datetime, params):

    alt_lower = tracks.params['LEVELS'][0, 0]
    alt_upper = tracks.params['LEVELS'][-1, 0]

#    sub_tracks_lower = reduce_tracks(tracks, uid, datetime, alt_lower)[0]
#    sub_tracks_upper = reduce_tracks(tracks, uid, datetime, alt_upper)[0]

    excluded = tracks.exclusions[params['exclusions']]
    excluded = np.any(excluded, 1)
    tmp_tracks = tracks.tracks[np.logical_not(excluded)]

    #import pdb; pdb.set_trace()

    tests = [
        (alt_lower >= level[0] and alt_lower < level[1])
        for level in tracks.params['LEVELS']]
    level = np.argwhere(tests)[0, 0]
    sub_tracks_lower = tmp_tracks.xs(
        (datetime, level), level=('time', 'level'))

    tests = [
        (alt_upper >= level[0] and alt_upper < level[1])
        for level in tracks.params['LEVELS']]
    level = np.argwhere(tests)[0, 0]
    sub_tracks_upper = tmp_tracks.xs(
        (datetime, level), level=('time', 'level'))

    #import pdb; pdb.set_trace()

    storms = pd.read_csv('~/CPOL_analysis/bom_storms_archive.csv')

    storms['Date/Time'] = pd.to_datetime(storms['Date/Time'])
    storms = storms.set_index('Event ID')
    storms = storms.drop(labels=['Comments', 'Unnamed: 9'], axis=1).fillna(-9999)

    midnight_cond = np.logical_or(storms['Date/Time'].dt.hour != 0, storms['Date/Time'].dt.minute != 0)
    storms = storms[midnight_cond]

    storms_lat = storms['Latitude'].values
    storms_lon = storms['Longitude'].values

    lat = tracks.radar_info['radar_lat']
    lon = tracks.radar_info['radar_lon']

    approx_radar_dis = np.sqrt((storms_lat-lat)**2 + (storms_lon-lon)**2)
    loc_cond = (approx_radar_dis < 1.5)
    storms_i = storms.iloc[loc_cond]

    time_cond = np.logical_and(
        storms_i['Date/Time'].dt.year == pd.DatetimeIndex([datetime]).year.values[0],
        storms_i['Date/Time'].dt.month == pd.DatetimeIndex([datetime]).month.values[0])
    storms_i = storms_i[time_cond]

    #import pdb; pdb.set_trace()

    if len(storms_i) == 0:
        print('No reports near this radar this month!')
        return

    semi_minor_lower = sub_tracks_lower['semi_minor'].values*tracks.record.grid_size[1]/2
    semi_major_lower = sub_tracks_lower['semi_major'].values*tracks.record.grid_size[1]/2
    orientation_lower = sub_tracks_lower['orientation'].values
    x_lower = sub_tracks_lower['grid_x'].values
    y_lower = sub_tracks_lower['grid_y'].values

    semi_minor_upper = sub_tracks_upper['semi_minor'].values*tracks.record.grid_size[1]/2
    semi_major_upper = sub_tracks_upper['semi_major'].values*tracks.record.grid_size[1]/2
    orientation_upper = sub_tracks_upper['orientation'].values
    x_upper = sub_tracks_upper['grid_x'].values
    y_upper = sub_tracks_upper['grid_y'].values

    for i in range(len(storms_i)):
        storm = storms_i.iloc[i]

        storm_x, storm_y = geographic_to_cartesian(
            storm['Longitude'], storm['Latitude'],
            grid.get_projparams())

        time_diff = (np.datetime64(storm['Date/Time'])-datetime).astype('timedelta64[s]').astype(float)
        time_check = np.abs(time_diff/3600) <= 1
        if time_check:

            # Note the orientation refers to the rotation of the major axis from the x axis
            loc_a_lower = (
                np.cos(orientation_lower*np.pi/180)*(storm_x-x_lower)
                +np.sin(orientation_lower*np.pi/180)*(storm_y-y_lower))**2/semi_major_lower**2
            loc_b_lower = (
                np.sin(orientation_lower*np.pi/180)*(storm_x-x_lower)
                -np.cos(orientation_lower*np.pi/180)*(storm_y-y_lower))**2/semi_minor_lower**2
            loc_check_lower = (loc_a_lower+loc_b_lower <= 1)

            loc_a_upper = (
                np.cos(orientation_upper*np.pi/180)*(storm_x-x_upper)
                +np.sin(orientation_upper*np.pi/180)*(storm_y-y_upper))**2/semi_major_upper**2
            loc_b_upper = (
                np.sin(orientation_upper*np.pi/180)*(storm_x-x_upper)
                -np.cos(orientation_upper*np.pi/180)*(storm_y-y_upper))**2/semi_minor_upper**2
            loc_check_upper = (loc_a_upper+loc_b_upper <= 1)

            match = np.any(time_check & (loc_check_lower | loc_check_upper))

            if match:
                color='r'
            else:
                color='k'

            report_label = storms['Database'].iloc[0][:2] + ':' + str(storm['ID'])

            ax.scatter(
                storm['Longitude'], storm['Latitude'], marker='X', s=200, linewidth=2,
                facecolor='w', zorder=4, edgecolors=color, transform=ccrs.PlateCarree())
            gen_embossed_text(
                storm['Longitude']+.075, storm['Latitude']+.075, label=report_label,
                transform=ccrs.PlateCarree(), zorder=5, color=color, fontsize=18,
                linewidth=4, ax=ax)

    return


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


def add_ellipses(ax, tracks, grid, uid, date_time, alt, params, excluded=False):

    projparams = grid.get_projparams()
    projection = ccrs.PlateCarree()
    tmp_tracks = reduce_tracks(tracks, uid, date_time, alt)[0]
    tmp_excl = tracks.exclusions.xs(
        (date_time, uid, 0), level=('time', 'uid', 'level'))
    int_border = np.any(
        tmp_excl[[
            'intersect_border', 'intersect_border_convective',
            'small_area', 'large_area']].values)
    if int_border:
        ell_pe = []
        if params['fig_style'] == 'paper':
            ell_c = 'grey'
            ell_w = 2.5
        elif params['fig_style'] == 'presentation':
            ell_c = 'grey'
            ell_w = 2.5
    else:
        if params['fig_style'] == 'paper':
            ell_c = 'black'
            ell_w = 2.5
            ell_shadow = 'grey'
        elif params['fig_style'] == 'presentation':
            ell_c = 'w'
            ell_shadow = 'k'
            ell_w = 2.5
        ell_pe = [pe.SimpleLineShadow(
            shadow_color=ell_shadow, alpha=.9, linewidth=ell_w+2), pe.Normal()]

    centroid = np.squeeze(tmp_tracks[['grid_x', 'grid_y']].values)
    lon, lat = np.squeeze(tmp_tracks[['lon', 'lat']].values)
    orientation = tmp_tracks[['orientation']].values[0]
    theta = orientation*np.pi/180
    #import pdb; pdb.set_trace()
    major_axis = tmp_tracks[['semi_major']].values[0]
    minor_axis = tmp_tracks[['semi_minor']].values[0]

    # Convert axes into lat/lon units by approximating 1 degree lat or
    # lon = 110 km. This needs to be made more robust!

    del_alpha = 2*np.pi/50
    alpha = np.arange(0, 2*np.pi, del_alpha)

    [dz, dy, dx] = get_grid_size(grid)
    semi_major = major_axis*dx/2
    semi_minor = minor_axis*dx/2

    ell_x = semi_major*np.cos(alpha)*np.cos(theta) - semi_minor*np.sin(alpha)*np.sin(theta)+centroid[0]
    ell_y = semi_major*np.cos(alpha)*np.sin(theta) + semi_minor*np.sin(alpha)*np.cos(theta)+centroid[1]

    ell_lon, ell_lat = cartesian_to_geographic(ell_x, ell_y, projparams)

    lgd_ellipse = mlines.Line2D(
        [], [], color=ell_c, linewidth=ell_w, label='Best Fit Ellipse',
        linestyle='--',
        path_effects=ell_pe)

    # ell.set_alpha(0.4)
    if not excluded:
        ax.plot(ell_lon, ell_lat,
            transform=projection, linewidth=ell_w,
            color=ell_c, linestyle='--',
            path_effects=ell_pe, zorder=4)
    return


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
                lon_cell, lat_cell, marker='o', s=100, linewidth=1,
                facecolor="none", edgecolor='k', zorder=2)
    lgd_cell = mlines.Line2D(
        [], [], color='w', marker='o', markersize=10, linewidth=1,
        markeredgecolor='k', linestyle='None', label='Convective Cells')

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
        vmin=0, vmax=1, transform=ccrs.PlateCarree())

    return ax


def add_stratiform_offset(ax, tracks, grid, uid, date_time, excluded):
    tmp_tracks = tracks.tracks.xs((date_time, uid), level=('time', 'uid'))
    lon_low = tmp_tracks.xs(0, level='level')['lon'].iloc[0]
    lat_low = tmp_tracks.xs(0, level='level')['lat'].iloc[0]
    num_levels = len(tracks.params['LEVELS'])
    lon_high = tmp_tracks.xs(num_levels-1, level='level')['lon'].iloc[0]
    lat_high = tmp_tracks.xs(num_levels-1, level='level')['lat'].iloc[0]

    linewidth = 2

    # Check if boundary intersection
    tmp_excl = tracks.exclusions.xs(
        (date_time, uid, 0), level=('time', 'uid', 'level'))
    int_border = np.any(
        tmp_excl[[
            'intersect_border', 'intersect_border_convective',
            'small_area', 'large_area', 'small_offset']].values)

    if not excluded and not int_border:
        ax.plot(
            [lon_low, lon_high], [lat_low, lat_high], '-w', linewidth=linewidth,
            zorder=4, transform=ccrs.PlateCarree(), path_effects=[
                pe.Stroke(linewidth=linewidth+5, foreground='b'), pe.Normal()])
    lgd_so = mlines.Line2D(
        [], [], color='w', linestyle='-', linewidth=linewidth,
        label='Stratiform Offset',
        path_effects=[pe.Stroke(linewidth=linewidth+5, foreground='b'), pe.Normal()])
    return lgd_so


def add_velocities(
        ax, tracks, grid, uid, date_time, alt, system_winds, excluded):

    if tracks.params['INPUT_TYPE'] == 'ACCESS_DATETIMES':
        level_test = [
            alt >= lvl[0] and alt < lvl[1]
            for lvl in tracks.params['LEVELS']]
        level_ind = np.where(level_test)[0][0]
        level = tracks.params['LEVELS'][level_ind]
        interval = tracks.params['WIND_LEVELS'][0]
        ambient_mid_alt = int((interval[0] + interval[1]) // 2)
        ambient_bottom_alt = int(interval[0])
        ambient_top_alt = int(interval[-1])
    else:
        level_test = [
            alt >= lvl[0] and alt < lvl[1]
            for lvl in tracks.params['WIND_LEVELS']]
        level_ind = np.where(level_test)[0][0]
        level = tracks.params['WIND_LEVELS'][level_ind]
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

    #dt = tracks.record.interval.total_seconds()
    dt = tracks.params['DT']*60

    scale = 4
    linewidth = 3

    lgd_han = []
    for wind in system_winds:
        u = tmp_tracks['u_' + wind].iloc[0]
        v = tmp_tracks['v_' + wind].iloc[0]
        [new_lon, new_lat] = cartesian_to_geographic(
            x + scale * u * dt, y + scale * v * dt, projparams)

        tmp_excl = tracks.exclusions.xs(
            (date_time, uid, 0), level=('time', 'uid', 'level'))

        excl_alt = [
            'intersect_border', 'intersect_border_convective',
            'small_area', 'large_area']

        if wind == 'shift':
            excl_alt += ['small_velocity']
        elif wind == 'relative':
            excl_alt += ['small_velocity', 'small_rel_velocity']
        elif wind == 'shear':
            excl_alt += ['small_shear']

        int_border = np.any(tmp_excl[excl_alt].values)

        if not excluded and not int_border:
            ax.arrow(
                lon, lat, new_lon[0]-lon, new_lat[0]-lat, color='w', zorder=5,
                head_width=0.016, head_length=0.024, length_includes_head=True,
                transform=ccrs.PlateCarree(),
                path_effects=[
                    pe.Stroke(linewidth=linewidth+5, foreground=colour_dic[wind]),
                    pe.Normal()])
        lgd_line = mlines.Line2D(
            [], [], color='w', linestyle='-', label=label_dic[wind],
            linewidth=linewidth, path_effects=[
                pe.Stroke(linewidth=linewidth+5, foreground=colour_dic[wind]),
                pe.Normal()])
        lgd_han.append(lgd_line)

    arrow_x = 110e3
    arrow_y = 160e3
    [lon, lat] = cartesian_to_geographic(arrow_x, arrow_y, projparams)

    [new_lon, new_lat] = cartesian_to_geographic(
            arrow_x + scale * 5 * dt, arrow_y, projparams)

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    #import pdb; pdb.set_trace()

    ax.arrow(
        lon[0], lat[0], new_lon[0]-lon[0], new_lat[0]-lat[0], color='w',
        zorder=5, head_width=0.016, head_length=0.024,
        length_includes_head=True,
        transform=ccrs.PlateCarree(), clip_on=False,
        path_effects=[
            pe.Stroke(linewidth=linewidth+5, foreground='red'),
            pe.Normal()])
    ax.text(
        new_lon[0]+.1, new_lat[0]-.025, '5 m/s',
        transform=ccrs.PlateCarree(), fontsize=18)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

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
            scale_units='x', scale=scale, transform=ccrs.PlateCarree())
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
