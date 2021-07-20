"""
tint.objects
============

Functions for managing and recording object properties.

"""
import warnings
import copy

import numpy as np
import pandas as pd
import pyart
from scipy import ndimage
from skimage.measure import regionprops
from scipy.ndimage import center_of_mass

from .grid_utils import get_filtered_frame, get_level_indices
from .rain import update_rain_totals, init_rain_totals, get_object_rain_props
from .rain import update_sys_tracks_rain
from .cells import identify_cells


def get_object_center(obj_id, labeled_image):
    """ Returns index of center pixel of the given object id from labeled
    image. The center is calculated as the median pixel of the object extent;
    it is not a true centroid. """
    obj_index = np.argwhere(labeled_image == obj_id)
    center = np.median(obj_index, axis=0).astype('i')
    return center


def get_obj_extent(labeled_image, obj_label):
    """ Takes in labeled image and finds the radius, area, and center of the
    given object. """
    obj_index = np.argwhere(labeled_image == obj_label)

    xlength = np.max(obj_index[:, 0]) - np.min(obj_index[:, 0]) + 1
    ylength = np.max(obj_index[:, 1]) - np.min(obj_index[:, 1]) + 1
    obj_radius = np.max((xlength, ylength)) / 2
    obj_center = np.round(np.median(obj_index, axis=0), 0)
    obj_area = len(obj_index[:, 0])

    obj_extent = {
        'obj_center': obj_center, 'obj_radius': obj_radius,
        'obj_area': obj_area, 'obj_index': obj_index}
    return obj_extent


def init_current_objects(data_dic, pairs, counter, interval, params):
    """Returns a dictionary for objects with unique ids and their
    corresponding ids in frame and frame_new. This function is called when
    echoes are detected after a period of no echoes. """

    current_objects = {}
    nobj = np.max(data_dic['frame'])

    id1 = np.arange(nobj) + 1
    uid = counter.next_uid(count=nobj)
    id2 = pairs
    obs_num = np.zeros(nobj, dtype='i')

    mergers = [set() for i in range(len(uid))]
    new_mergers = [set() for i in range(len(uid))]
    parents = [set() for i in range(len(uid))]

    if params['RAIN']:
        max_rr, tot_rain = init_rain_totals(data_dic, uid, id1, interval)
        current_objects.update({'max_rr': max_rr, 'tot_rain': tot_rain})

    current_objects.update({
        'id1': id1, 'uid': uid, 'id2': id2, 'mergers': mergers,
        'new_mergers': new_mergers, 'parents': parents, 'obs_num': obs_num})
    current_objects = attach_last_heads(data_dic, current_objects)
    return current_objects, counter


def update_current_objects(
        data_dic, pairs, old_objects, counter,
        old_obj_merge, interval, params, grid_obj):
    """Removes dead objects, updates living objects, and assigns new uids to
    new-born objects. """
    current_objects = {}
    nobj = np.max(data_dic['frame'])
    id1 = np.arange(nobj) + 1
    uid = np.array([], dtype='str')
    obs_num = np.array([], dtype='i')
    mergers = []

    for obj in np.arange(nobj) + 1:
        if obj in old_objects['id2']:
            obj_index = (old_objects['id2'] == obj)
            ind = np.argwhere(obj_index)[0, 0]
            uid = np.append(uid, old_objects['uid'][ind])
            obs_num = np.append(obs_num, old_objects['obs_num'][ind]+1)
        else:
            uid = np.append(uid, counter.next_uid())
            obs_num = np.append(obs_num, 0)

    new_mergers = [set() for i in range(len(uid))]
    mergers = [set() for i in range(len(uid))]
    parents = [set() for i in range(len(uid))]

    for i in range(len(uid)):
        if uid[i] in old_objects['uid']:
            # Check for merger
            old_i = int(np.argwhere(old_objects['uid'] == uid[i]))
            new_mergers[i] = set(old_objects['uid'][old_obj_merge[:, old_i]])
            mergers[i] = new_mergers[i].union(old_objects['mergers'][old_i])
            parents[i] = parents[i].union(old_objects['parents'][old_i])
        else:
            # Check for splits
            recurring = set(old_objects['uid']).intersection(set(uid))
            for j in range(len(recurring)):
                obj_index = (old_objects['uid'] == list(recurring)[j])
                # Check for overlap
                olap = np.any(
                    (data_dic['frame_old'] == old_objects['id1'][obj_index])
                    * ((data_dic['frame'] == id1[i])))
                if olap:
                    parents[i] = parents[i].union(
                        {old_objects['uid'][obj_index][0]})

    if params['RAIN']:
        max_rr, tot_rain = update_rain_totals(
            data_dic, old_objects, uid, interval, grid_obj, params)
        current_objects.update({'max_rr': max_rr, 'tot_rain': tot_rain})

    id2 = pairs

    current_objects.update({
        'id1': id1, 'uid': uid, 'id2': id2, 'obs_num': obs_num,
        'mergers': mergers, 'new_mergers': new_mergers, 'parents': parents})

    current_objects = attach_last_heads(data_dic, current_objects)
    return current_objects, counter


def attach_last_heads(data_dic, current_objects):
    """ Attaches last heading information to current_objects dictionary."""
    nobj = len(current_objects['uid'])
    heads = np.ma.empty((nobj, 2))

    for obj in range(nobj):
        if (
                (current_objects['id1'][obj] > 0)
                and (current_objects['id2'][obj] > 0)):
            center1 = center_of_mass(
                data_dic['refl'], labels=data_dic['frame'],
                index=current_objects['id1'][obj])
            center2 = center_of_mass(
                data_dic['refl_new'], labels=data_dic['frame_new'],
                index=current_objects['id2'][obj])
            heads[obj, :] = np.array(center2) - np.array(center1)
        else:
            heads[obj, :] = np.ma.array([-999, -999], mask=[True, True])

    current_objects['last_heads'] = heads
    return current_objects


def check_isolation(refl, filtered, grid_size, params, level):
    """ Returns list of booleans indicating object isolation. Isolated objects
    are not connected to any other objects by pixels greater than ISO_THRESH,
    and have at most one peak. """
    nobj = np.max(filtered)
    min_size = params['MIN_SIZE'][level] / np.prod(grid_size[1:]/1000)
    iso_filtered = get_filtered_frame(
        refl, min_size, params['ISO_THRESH'][level])
    nobj_iso = np.max(iso_filtered)
    iso = np.empty(nobj, dtype='bool')

    for iso_id in np.arange(nobj_iso) + 1:
        obj_ind = np.where(iso_filtered == iso_id)
        objects = np.unique(filtered[obj_ind])
        objects = objects[objects != 0]
        if len(objects) == 1 and single_max(obj_ind, refl, params):
            iso[objects - 1] = True
        else:
            iso[objects - 1] = False
    return iso


def single_max(obj_ind, refl, params):
    """ Returns True if object has at most one peak. """
    max_proj = np.max(refl, axis=0)
    smooth = ndimage.filters.gaussian_filter(max_proj, params['ISO_SMOOTH'])
    padded = np.pad(smooth, 1, mode='constant')
    obj_ind = [axis + 1 for axis in obj_ind]  # adjust for padding
    maxima = 0
    for pixel in range(len(obj_ind[0])):
        ind_0 = obj_ind[0][pixel]
        ind_1 = obj_ind[1][pixel]
        neighborhood = padded[(ind_0-1):(ind_0+2), (ind_1-1):(ind_1+2)]
        max_ind = np.unravel_index(neighborhood.argmax(), neighborhood.shape)
        if max_ind == (1, 1):
            maxima += 1
            if maxima > 1:
                return False
    return True


def get_object_prop(
        data_dic, grid1, u_shift, v_shift, field, record,
        params, current_objects):
    """Returns dictionary of object properties for all objects found in
    each level of images, where images are the labelled (filtered)
    frames. """
    properties = [
        'id1', 'center', 'com_x', 'com_y', 'grid_x', 'grid_y', 'proj_area',
        'lon', 'lat', 'field_max', 'max_height', 'volume',
        'level', 'touch_border', 'semi_major', 'semi_minor', 'orientation',
        'eccentricity', 'mergers', 'parent', 'cells']
    obj_prop = {p: [] for p in properties}

    nobj = np.max(data_dic['frames'])
    [levels, rows, columns] = data_dic['frames'].shape

    unit_dim = record.grid_size
    if unit_dim[-2] != unit_dim[-1]:
        warnings.warn(
            'x and y grid scales unequal - metrics such as semi_major '
            + 'and semi_minor may be misleading.')
    # These metrics are in km^[1, 2, 3] respectively
    unit_alt = unit_dim[0]/1000
    unit_area = (unit_dim[1]*unit_dim[2])/(1000**2)
    unit_vol = (unit_dim[0]*unit_dim[1]*unit_dim[2])/(1000**3)

    raw3D = grid1.fields[field]['data'].data  # Complete dataset
    z_values = grid1.z['data']/1000

    all_cells = identify_cells(
        raw3D, data_dic['frames'], grid1, record, params,
        data_dic['stein_class'])

    for i in range(levels):

        [z_min, z_max] = get_level_indices(
            grid1, record.grid_size, params['LEVELS'][i, :])
        ski_props = regionprops(
            data_dic['frames'][i], raw3D[z_min], cache=True)

        for obj in np.arange(nobj) + 1:
            obj_prop['id1'].append(obj)
            obj_prop['mergers'].append(current_objects['mergers'][obj-1])
            obj_prop['parent'].append(current_objects['parents'][obj-1])
            obj_prop['level'].append(i)

            # Get objects in images[i], i.e. the frame at i-th level
            obj_index = np.argwhere(data_dic['frames'][i] == obj)

            # Work out how many gridcells touch the border
            obj_ind_list = obj_index.tolist()
            obj_ind_set = set(
                [tuple(obj_ind_list[i]) for i in range(len(obj_ind_list))])
            b_intersect = obj_ind_set.intersection(
                params['BOUNDARY_GRID_CELLS'])
            obj_prop['touch_border'].append(len(b_intersect))

            obj_prop['center'].append(np.median(obj_index, axis=0))
            obj_prop['proj_area'].append(obj_index.shape[0] * unit_area)

            # Append mean object index (centroid) in grid units
            g_y = ski_props[obj-1].centroid[0] * unit_dim[2]
            g_y += grid1.y['data'][0]
            g_x = ski_props[obj-1].centroid[1] * unit_dim[1]
            g_x += grid1.x['data'][0]
            obj_prop['grid_x'].append(np.round(g_x, 1))
            obj_prop['grid_y'].append(np.round(g_y, 1))

            # Append object center of mass (reflectivity weighted
            # centroid) in grid units
            g_y = ski_props[obj-1].weighted_centroid[0] * unit_dim[2]
            g_y += grid1.y['data'][0]
            g_x = ski_props[obj-1].weighted_centroid[1] * unit_dim[1]
            g_x += grid1.x['data'][0]
            obj_prop['com_x'].append(np.round(g_x, 1))
            obj_prop['com_y'].append(np.round(g_y, 1))

            # Append centroid in lat, lon units.
            projparams = grid1.get_projparams()
            lon, lat = pyart.core.transforms.cartesian_to_geographic(
                g_x, g_y, projparams)
            obj_prop['lon'].append(np.round(lon[0], 5))
            obj_prop['lat'].append(np.round(lat[0], 5))

            # Append ellipse properties
            # Note semi_major, semi_minor are stored in `index'
            # units which effectively assumes dx = dy.
            attrs = [
                'major_axis_length', 'minor_axis_length',
                'eccentricity', 'orientation']
            lists = ['semi_major', 'semi_minor', 'eccentricity', 'orientation']
            for j in range(0, len(attrs)):
                try:
                    obj_prop[lists[j]].append(
                        np.round(eval('ski_props[obj-1].' + attrs[j]), 3))
                except AttributeError:
                    obj_prop[lists[j]].append(np.nan)

            # Calculate raw3D slices
            raw3D_i = raw3D[z_min:z_max, :, :]
            obj_slices = [raw3D_i[:, ind[0], ind[1]] for ind in obj_index]

            if params['FIELD_THRESH'][i] == 'convective':
                stein_class_i = data_dic['stein_class'][i]
                stein_class_slices = [
                    stein_class_i[ind[0], ind[1]] for ind in obj_index]
                obj_prop['field_max'].append(np.max(obj_slices))
                filtered_slices = [
                    stein_class_slice == 2
                    for stein_class_slice in stein_class_slices]
            else:
                obj_prop['field_max'].append(np.max(obj_slices))
                filtered_slices = [
                    obj_slice > params['FIELD_THRESH'][i]
                    for obj_slice in obj_slices]

            # Append maximum height
            heights = [np.arange(raw3D_i.shape[0])[ind]
                       for ind in filtered_slices]
            obj_prop['max_height'].append(
                np.max(np.concatenate(heights)) * unit_alt + z_values[z_min])

            # Note volume isn't necessarily consistent with proj_area.
            obj_prop['volume'].append(np.sum(filtered_slices) * unit_vol)

            # Append reflectivity cells based on vertically
            # overlapping reflectivity maxima.
            if i == 0:
                all_cells_0 = [
                    all_cells[i][0].tolist()
                    for i in range(len(all_cells))]
                cell_obj_inds = [
                    i for i in range(len(all_cells_0))
                    if all_cells_0[i][1:] in obj_index.tolist()]
                cells_obj = [all_cells[i] for i in cell_obj_inds]
                obj_prop['cells'].append(cells_obj)
            else:
                obj_prop['cells'].append([])

    if params['RAIN']:
        rain_props = get_object_rain_props(data_dic, current_objects)
        obj_prop.update({
            'tot_rain': rain_props[0], 'max_rr': rain_props[1],
            'tot_rain_loc': rain_props[2], 'max_rr_loc': rain_props[3]})

    obj_prop.update({
        'u_shift': u_shift * levels, 'v_shift': v_shift * levels,
        'orientation': np.round(
            np.rad2deg(np.array(obj_prop['orientation'])), 3)})
    return obj_prop


def write_tracks(old_tracks, record, current_objects, obj_props):
    """ Writes all cell information to tracks dataframe. """
    print('Writing tracks for scan {}.'.format(str(record.scan)))

    nobj = len(current_objects['uid'])
    nlvl = max(obj_props['level'])+1
    scan_num = [record.scan] * nobj * nlvl
    uid = current_objects['uid'].tolist() * nlvl
    da_dic = {'scan': scan_num, 'uid': uid, 'time': record.time}
    da_dic.update(obj_props)
    new_tracks = pd.DataFrame(da_dic)
    new_tracks.set_index(['scan', 'time', 'level', 'uid'], inplace=True)
    new_tracks.sort_index(inplace=True)
    tracks = old_tracks.append(new_tracks)
    return tracks


def smooth(group_df, r=3, n=2):
    group_df_smoothed = copy.deepcopy(group_df)

    # Smooth middle cases.
    if len(group_df) >= (2*n+1):
        group_df_smoothed = group_df.rolling(
            window=(2*n+1), center=True).mean()

    # Deal with end cases.
    new_n = min(n, int(np.ceil(len(group_df) / 2)))
    for k in range(0, new_n):
        # Use a k+1 window on both sides to smooth
        fwd = np.mean(group_df.iloc[:k+2])
        bwd = np.mean(group_df.iloc[-k-2:])
        group_df_smoothed.iloc[k] = fwd
        group_df_smoothed.iloc[-1-k] = bwd

    return np.round(group_df_smoothed, r)


def post_tracks(tracks_obj):
    """ Calculate additional tracks data from final tracks dataframe. """
    print('Calculating additional tracks properties.')

    # Drop last scan so no nans in u_shift etc
    tracks_obj.tracks.drop(
        tracks_obj.tracks.index.max()[0], level='scan', inplace=True)

    tracks_obj.tracks['field_max'] = np.round(
        tracks_obj.tracks['field_max'], 2)

    # Smooth u_shift, v_shift
    tmp_tracks = tracks_obj.tracks[['u_shift', 'v_shift']]
    # Calculate forward difference for first time step
    tmp_tracks = tmp_tracks.groupby(
        level=['uid', 'level'], as_index=False, group_keys=False)
    tracks_obj.tracks[['u_shift', 'v_shift']] = tmp_tracks.apply(
        lambda x: smooth(x))

    # Calculate velocity using centred difference.
    tmp_tracks = tracks_obj.tracks[['grid_x', 'grid_y']]
    tmp_tracks = tmp_tracks.groupby(
        level=['uid', 'level'], as_index=False, group_keys=False)
    tmp_tracks = tmp_tracks.rolling(window=5, center=True)

    dt = tracks_obj.record.interval.total_seconds()
    tmp_tracks = tmp_tracks.apply(
        lambda x: (x.values[4] - x.values[0]) / (4 * dt))
    tmp_tracks = tmp_tracks.rename(columns={'grid_x': 'u', 'grid_y': 'v'})
    tmp_tracks = tmp_tracks.drop(labels=['uid', 'level'], axis=1)
    tracks_obj.tracks = tracks_obj.tracks.merge(
        tmp_tracks, left_index=True, right_index=True)

    # Sort multi-index again as levels will be jumbled after rolling etc.
    tracks_obj.tracks = tracks_obj.tracks.sort_index()

    # Calculate vertical displacement
    tmp_tracks = tracks_obj.tracks[['grid_x', 'grid_y']]
    tmp_tracks = tmp_tracks.groupby(
        level=['uid', 'scan', 'time'], as_index=False, group_keys=False)
    tmp_tracks = tmp_tracks.rolling(window=2, center=False)
    tmp_tracks = tmp_tracks.apply(lambda x: (x.values[1] - x.values[0]))
    tmp_tracks = tmp_tracks.rename(
        columns={'grid_x': 'x_vert_disp', 'grid_y': 'y_vert_disp'})
    tmp_tracks = tmp_tracks.drop(labels=['uid', 'scan', 'time'], axis=1)
    tracks_obj.tracks = tracks_obj.tracks.merge(
        tmp_tracks, left_index=True, right_index=True)
    tracks_obj.tracks = tracks_obj.tracks.sort_index()

    return tracks_obj


def get_system_tracks(tracks_obj):
    """ Calculate system tracks """
    print('Calculating system tracks.')

    # Get position and velocity at tracking level.
    system_tracks = tracks_obj.tracks[
        ['grid_x', 'grid_y', 'com_x', 'com_y', 'lon', 'lat', 'u', 'v',
         'mergers', 'parent', 'u_shift', 'v_shift']]
    system_tracks = system_tracks.xs(
        tracks_obj.params['TRACK_INTERVAL'], level='level')

    # Get number of cells and ellipse fit properties
    # at lowest interval assuming this is first interval in list.
    for prop in [
            'semi_major', 'semi_minor', 'eccentricity',
            'orientation', 'cells']:
        prop_lvl_0 = tracks_obj.tracks[[prop]].xs(0, level='level')
        system_tracks = system_tracks.merge(prop_lvl_0, left_index=True,
                                            right_index=True)

    if tracks_obj.params['RAIN']:
        system_tracks = update_sys_tracks_rain(tracks_obj, system_tracks)

    # Calculate system maximum
    maximum = tracks_obj.tracks[['field_max']]
    maximum = maximum.groupby(level=['scan', 'time', 'uid']).max()
    system_tracks = system_tracks.merge(maximum, left_index=True,
                                        right_index=True)

    # Calculate maximum area
    proj_area = tracks_obj.tracks[['proj_area']]
    proj_area = proj_area.groupby(level=['scan', 'time', 'uid']).max()
    system_tracks = system_tracks.merge(proj_area, left_index=True,
                                        right_index=True)

    # Calculate maximum altitude
    m_alt = tracks_obj.tracks[['max_height']]
    m_alt = m_alt.groupby(level=['scan', 'time', 'uid']).max()
    system_tracks = system_tracks.merge(
        m_alt, left_index=True, right_index=True)

    # Get touch_border for system
    t_border = tracks_obj.tracks[['touch_border']]
    t_border = t_border.groupby(level=['scan', 'time', 'uid']).max()
    system_tracks = system_tracks.merge(t_border, left_index=True,
                                        right_index=True)

    # Calculate total vertical displacement.
    n_lvl = tracks_obj.params['LEVELS'].shape[0]
    pos_0 = tracks_obj.tracks[['grid_x', 'grid_y']].xs(0, level='level')
    pos_1 = tracks_obj.tracks[['grid_x', 'grid_y']].xs(n_lvl-1, level='level')

    pos_0.rename(
        columns={'grid_x': 'x_vert_disp', 'grid_y': 'y_vert_disp'},
        inplace=True)
    pos_1.rename(
        columns={'grid_x': 'x_vert_disp', 'grid_y': 'y_vert_disp'},
        inplace=True)

    system_tracks = system_tracks.merge(
        pos_1-pos_0, left_index=True, right_index=True)

    # Calculate magnitude of vertical displacement.
    tilt_mag = np.round(
        np.sqrt((
            system_tracks['x_vert_disp'] ** 2
            + system_tracks['y_vert_disp'] ** 2)), 3)
    tilt_mag = tilt_mag.rename('tilt_mag')
    system_tracks = system_tracks.merge(
        tilt_mag, left_index=True, right_index=True)

    # Calculate vertical displacement direction.
    vel_dir = np.arctan2(system_tracks['v_shift'], system_tracks['u_shift'])
    vel_dir = np.rad2deg(vel_dir)
    vel_dir = vel_dir.rename('vel_dir')
    vel_dir = np.round(vel_dir, 3)

    tilt_dir = np.arctan2(
        system_tracks['y_vert_disp'], system_tracks['x_vert_disp'])
    tilt_dir = tilt_dir.rename('tilt_dir')
    tilt_dir = np.rad2deg(tilt_dir)
    tilt_dir = np.round(tilt_dir, 3)

    sys_rel_tilt_dir = np.mod(tilt_dir - vel_dir + 180, 360)-180
    sys_rel_tilt_dir = sys_rel_tilt_dir.rename('sys_rel_tilt_dir')
    sys_rel_tilt_dir = np.round(sys_rel_tilt_dir, 3)

    for var in [vel_dir, tilt_dir, sys_rel_tilt_dir]:
        system_tracks = system_tracks.merge(
            var, left_index=True, right_index=True)

    system_tracks = system_tracks.sort_index()
    tracks_obj.system_tracks = system_tracks

    return tracks_obj
