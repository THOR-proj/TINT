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
from skimage.morphology.convex_hull import convex_hull_image
import cv2 as cv

from tint.grid_utils import get_filtered_frame, get_level_indices
from tint.rain import update_rain_totals, init_rain_totals
from tint.rain import get_object_rain_props
from tint.rain import update_sys_tracks_rain
from tint.cells import identify_cells


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


def get_ellipse(data_dic, grid1, record, obj, level_ind, u_shift, v_shift):

    unit_dim = record.grid_size
    hull = convex_hull_image(data_dic['frames'][level_ind] == obj)
    contours = cv.findContours(
        hull.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[0]

    if len(contours[0]) > 6:
        [(x_c, y_c), (a, b), phi] = cv.fitEllipseDirect(contours[0])
    else:
        print('Could not fit ellipse. Retrying with padded contour.')
        new_contour = []
        for r in contours[0]:
            [new_contour.append(r) for i in range(3)]
        new_contour = np.array(new_contour)
        [(x_c, y_c), (a, b), phi] = cv.fitEllipseDirect(new_contour)

    grid_y = y_c * unit_dim[2] + grid1.y['data'][0]
    grid_x = x_c * unit_dim[1] + grid1.x['data'][0]
    if a >= b:
        semi_major = a
        semi_minor = b
        orientation = phi
    else:
        semi_major = b
        semi_minor = a
        orientation = phi - 90
    orientation = orientation % 180
    # Ensure pos y of new coordinates in same half plane as velocity direction
    [n_x, n_y] = [
        np.cos(np.deg2rad(orientation + 90)),
        np.sin(np.deg2rad(orientation + 90))]
    if (u_shift[0] * n_x + v_shift[0] * n_y) < 0:
        orientation += 180

    ecc = np.sqrt(1 - (semi_minor / semi_major) ** 2)
    return grid_x, grid_y, semi_major, semi_minor, ecc, orientation


def get_object_prop(
        data_dic, grid1, u_shift, v_shift, field, record,
        params, current_objects):
    """Returns dictionary of object properties for all objects found in
    each level of images, where images are the labelled (filtered)
    frames. """
    # Note semi_major and semi_minor are actually the major and minor axes
    properties = [
        'center', 'com_x', 'com_y', 'grid_x', 'grid_y', 'proj_area',
        'lon', 'lat', 'field_max', 'max_height', 'volume',
        'level', 'touch_border', 'semi_major', 'semi_minor', 'orientation',
        'eccentricity', #'mergers', 'parent', 'cells',
        'u_ambient_bottom', 'v_ambient_bottom',
        'u_ambient_mid', 'v_ambient_mid',
        'u_ambient_top', 'v_ambient_top',
        'u_ambient_mean', 'v_ambient_mean',
        'u_shift', 'v_shift', 'u_trop_shear', 'v_trop_shear', 'CAPE',
        'trop_height', 'freezing_level', 'av_rh', 'av_ee',
        'av_rh_bfl', 'av_ee_bfl', 'av_rh_afl', 'av_ee_afl']
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

    #all_cells = identify_cells(
    #    raw3D, data_dic['frames'], grid1, record, params,
    #    data_dic['stein_class'])

    for i in range(levels):

        [z_min, z_max] = get_level_indices(
            grid1, record.grid_size, params['LEVELS'][i, :])
        ski_props = regionprops(
            data_dic['frames'][i], raw3D[z_min].astype(float), cache=False)

        for obj in np.arange(nobj) + 1:
            #obj_prop['mergers'].append(current_objects['mergers'][obj-1])
            #obj_prop['parent'].append(current_objects['parents'][obj-1])
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

            # Append object center of mass (reflectivity weighted
            # centroid) in grid units
            g_y = ski_props[obj-1].weighted_centroid[0] * unit_dim[2]
            g_y += grid1.y['data'][0]
            g_x = ski_props[obj-1].weighted_centroid[1] * unit_dim[1]
            g_x += grid1.x['data'][0]
            obj_prop['com_x'].append(np.round(g_x, 1))
            obj_prop['com_y'].append(np.round(g_y, 1))

            ellipse = get_ellipse(
                data_dic, grid1, record, obj, i, u_shift, v_shift)
            props = [
                'grid_x', 'grid_y', 'semi_major', 'semi_minor',
                'eccentricity', 'orientation']
            [obj_prop[props[i]].append(ellipse[i]) for i in range(len(props))]

            # Append centroid in lat, lon units.
            projparams = grid1.get_projparams()
            lon, lat = pyart.core.transforms.cartesian_to_geographic(
                ellipse[0], ellipse[1], projparams)
            obj_prop['lon'].append(np.round(lon[0], 5))
            obj_prop['lat'].append(np.round(lat[0], 5))

            u_shift_obj = np.round(u_shift[obj-1], 5)
            v_shift_obj = np.round(v_shift[obj-1], 5)
            obj_prop['u_shift'].append(u_shift_obj)
            obj_prop['v_shift'].append(v_shift_obj)

            grid_time = np.datetime64(record.time).astype('<M8[m]')

            try:
                # Get CAPE, tropopause height, average relative humidity, trop shear

                CAPE = float(data_dic['ambient_interp']['cape'].sel(
                    longitude=lon[0], latitude=lat[0], time=grid_time,
                    method='nearest').values)
                obj_prop['CAPE'].append(CAPE)
                tp_idx = data_dic['ambient_interp']['tp_idx'].sel(
                    longitude=lon[0], latitude=lat[0], time=grid_time,
                    method='nearest').values
                trop_height = float(data_dic['ambient_interp'].altitude[tp_idx].values)
                obj_prop['trop_height'].append(trop_height)
                t = data_dic['ambient_interp']['t'].sel(
                    longitude=lon[0], latitude=lat[0], time=grid_time,
                    method='nearest').values
                r = data_dic['ambient_interp']['r'].sel(
                    longitude=lon[0], latitude=lat[0], time=grid_time,
                    method='nearest').values
                ee = 100/r-1

                lowest_alt_idx = np.argwhere(t>0).squeeze()[0]

                fl_idx = np.argwhere(t<273.15).squeeze()[0]
                freezing_level = float(data_dic['ambient_interp'].altitude[fl_idx].values)

                # Note if fl_idx == lowest_alt_idx, _bfl means will be nan. Suitable.

                if trop_height > 7000:
                    av_rh = np.nanmean(r[:tp_idx])
                    av_rh_bfl = np.nanmean(r[:fl_idx])
                    av_rh_afl = np.nanmean(r[fl_idx:tp_idx])
                    av_ee = np.nanmean(ee[:tp_idx])
                    av_ee_bfl = np.nanmean(ee[:fl_idx])
                    av_ee_afl = np.nanmean(ee[fl_idx:tp_idx])
                else:
                    av_rh = np.nan
                    av_rh_bfl = np.nan
                    av_rh_afl = np.nan
                    av_ee = np.nan
                    av_ee_bfl = np.nan
                    av_ee_afl = np.nan

                obj_prop['av_rh'] = av_rh
                obj_prop['av_rh_bfl'] = av_rh_bfl
                obj_prop['av_rh_afl'] = av_rh_afl
                obj_prop['av_ee'] = av_ee
                obj_prop['av_ee_bfl'] = av_ee_bfl
                obj_prop['av_ee_afl'] = av_ee_afl

                obj_prop['freezing_level'] = freezing_level

                u = data_dic['ambient_interp']['u'].sel(
                    longitude=lon[0], latitude=lat[0], time=grid_time,
                    method='nearest').values
                v = data_dic['ambient_interp']['v'].sel(
                    longitude=lon[0], latitude=lat[0], time=grid_time,
                    method='nearest').values
                if trop_height > 7000:
                    obj_prop['u_trop_shear'].append(u[tp_idx]-u[2])
                    obj_prop['v_trop_shear'].append(v[tp_idx]-v[2])
                else:
                    obj_prop['u_trop_shear'].append(np.nan)
                    obj_prop['v_trop_shear'].append(np.nan)

            except KeyError:
                print('Missing key - check ERA5 data.')

            if params['AMBIENT'] is not None:

                interval = np.arange(
                    params['WIND_LEVELS'][i, 0],
                    params['WIND_LEVELS'][i, 1]+unit_dim[0], unit_dim[0])
                mid_i = len(interval) // 2
                altitudes = [
                    params['WIND_LEVELS'][i, 0], interval[mid_i],
                    params['WIND_LEVELS'][i, 1]]
                ambients = [
                    data_dic['ambient_interp'].sel(
                        longitude=lon[0], latitude=lat[0], time=grid_time,
                        altitude=alt, method='nearest') for alt in altitudes]
                ambients_u = [np.round(float(a.u.values), 5) for a in ambients]
                ambients_v = [np.round(float(a.v.values), 5) for a in ambients]

                suffix = ['bottom', 'mid', 'top']
                [
                    obj_prop['u_ambient_'+suffix[j]].append(ambients_u[j])
                    for j in range(len(ambients_u))]
                [
                    obj_prop['v_ambient_'+suffix[j]].append(ambients_v[j])
                    for j in range(len(ambients_v))]

                obj_prop['u_ambient_mean'] = np.nanmean(np.array(ambients_u))
                obj_prop['v_ambient_mean'] = np.nanmean(np.array(ambients_v))
                #ambients_u = [
                #    data_dic['ambient_interp']['u'].sel(
                #        longitude=lon[0], latitude=lat[0], time=grid_time,
                #        method='nearest').values
                #ambients_v = [
                #    data_dic['ambient_interp']['v'].sel(
                #        longitude=lon[0], latitude=lat[0], time=grid_time,
                #        method='nearest').values
                #u_ambient_mean =
                #v_ambient_mean =
            else:
                suffix = ['bottom', 'mid', 'top', 'mean']
                [obj_prop['u_ambient_'+s].append(np.nan) for s in suffix]
                [obj_prop['v_ambient_'+s].append(np.nan) for s in suffix]

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
            #if i == 0:
            #    all_cells_0 = [
            #        all_cells[i][0].tolist()
            #        for i in range(len(all_cells))]
            #    cell_obj_inds = [
            #        i for i in range(len(all_cells_0))
            #        if all_cells_0[i][1:] in obj_index.tolist()]
            #    cells_obj = [all_cells[i] for i in cell_obj_inds]
            #    obj_prop['cells'].append(cells_obj)
            #else:
            #    obj_prop['cells'].append([])

    if params['RAIN']:
        rain_props = get_object_rain_props(data_dic, current_objects)
        obj_prop.update({
            'tot_rain': rain_props[0], 'max_rr': rain_props[1],
            'tot_rain_loc': rain_props[2], 'max_rr_loc': rain_props[3]})

    obj_prop.update({'orientation': np.round(obj_prop['orientation'], 3)})
    return obj_prop


def write_tracks(old_tracks, record, current_objects, obj_props):
    """ Writes all cell information to tracks dataframe. """
    print('Writing tracks for scan {}.'.format(str(record.scan)))

    nobj = len(current_objects['uid'])
    nlvl = max(obj_props['level'])+1
    scan_num = [record.scan] * nobj * nlvl
    uid = current_objects['uid'].tolist() * nlvl
    da_dic = {
        'scan': scan_num, 'uid': uid,
        'time': np.datetime64(record.time).astype('<M8[m]')}
    da_dic.update(obj_props)
    new_tracks = pd.DataFrame(da_dic)
    new_tracks.set_index(['scan', 'time', 'uid', 'level'], inplace=True)
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


def fill_end(group_df):
    if len(group_df['u_shift']) > 1:
        group_df['u_shift'].iloc[-1] = group_df['u_shift'].iloc[-2]
        group_df['v_shift'].iloc[-1] = group_df['v_shift'].iloc[-2]
    return group_df


def post_tracks(tracks_obj):
    """ Calculate additional tracks data from final tracks dataframe. """
    print('Calculating additional tracks properties.')

    # Drop last scan so no nans in u_shift etc
    tracks_obj.tracks.drop(
        tracks_obj.tracks.index.max()[0], level='scan', inplace=True)

    tracks_obj.tracks['field_max'] = np.round(
        tracks_obj.tracks['field_max'], 2)

    # Repeat final u_shift values for end of object
    tmp_tracks = tracks_obj.tracks[['u_shift', 'v_shift']]
    tmp_tracks = tmp_tracks.groupby(
        level=['uid', 'level'], as_index=False, group_keys=False)
    tracks_obj.tracks[['u_shift', 'v_shift']] = tmp_tracks.apply(
        lambda x: fill_end(x))

    # Smooth u_shift, v_shift
    tmp_tracks = tracks_obj.tracks[['u_shift', 'v_shift']]
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

    #tracks_obj = calc_mean_ambient_winds(tracks_obj)

    tracks_obj.tracks['u_relative'] = np.round(
        tracks_obj.tracks['u_shift'] - tracks_obj.tracks['u_ambient_mean'], 5)
    tracks_obj.tracks['v_relative'] = np.round(
        tracks_obj.tracks['v_shift'] - tracks_obj.tracks['v_ambient_mean'], 5)
    tracks_obj.tracks['u_shear'] = np.round(
        tracks_obj.tracks['u_ambient_top']
        - tracks_obj.tracks['u_ambient_bottom'], 5)
    tracks_obj.tracks['v_shear'] = np.round(
        tracks_obj.tracks['v_ambient_top']
        - tracks_obj.tracks['v_ambient_bottom'], 5)

    tracks_obj = calc_alt_orientation(tracks_obj)

    return tracks_obj


def classify_tracks(tracks_obj):

    tracks_class = copy.deepcopy(
        tracks_obj.tracks[['grid_x', 'grid_y', 'lon', 'lat']])
    tracks_obj.tracks_class = tracks_class
    tracks_obj = calc_inflow_type(tracks_obj)
    tracks_obj = calc_propagation_type(tracks_obj)
    tracks_obj = calc_tilt_type(tracks_obj)
    tracks_obj = calc_stratiform_type(tracks_obj)
    tracks_obj = calc_relative_stratiform_type(tracks_obj)

    return tracks_obj


def count_consecutive(integers):
    counts = []
    count = 0
    for i in range(len(integers) - 1):
        # Check if the next number is consecutive
        if integers[i] + 1 == integers[i+1]:
            count += 1
        else:
            # If it is not append the count and restart counting
            counts.append(count)
            count = 1
    # Since we stopped the loop one early append the last count
    counts.append(count)
    return max(counts)


def temporal_continuity_check(
        group_df, length=np.timedelta64(30, 'm'),
        dt=np.timedelta64(10, 'm')):
    scans = group_df.index.get_level_values(0).values
    cts_scans = count_consecutive(scans)
    return cts_scans * dt < length


def get_duration_cond(tracks_obj):
    exclusions = [
        'small_area', 'large_area', 'intersect_border',
        'intersect_border_convective', 'small_velocity', 'small_offset']

    excluded = tracks_obj.exclusions[exclusions]
    excluded = excluded.xs(0, level='level')
    excluded = np.any(excluded, 1)

    included = np.logical_not(excluded)
    included = included.where(included == True).dropna()

    duration = np.timedelta64(tracks_obj.params['EXCL_THRESH']['DURATION'])
    dt = np.timedelta64(tracks_obj.params['DT'])

    duration_checks = included.groupby(level='uid').apply(
        lambda g: temporal_continuity_check(g, length=duration, dt=dt))
    uids = duration_checks.reset_index()['uid'].values

    tracks_obj.exclusions['duration_cond'] = [
        True for i in range(len(tracks_obj.exclusions))]

    for uid in uids:
        tracks_obj.exclusions.loc[
            (slice(None), slice(None), uid, slice(None)),
            'duration_cond'] = duration_checks.loc[uid]

    return tracks_obj


def get_simple_duration_cond(tracks_obj):

    exclusions = []

    excluded = tracks_obj.exclusions[exclusions]
    excluded = excluded.xs(0, level='level')
    excluded = np.any(excluded, 1)

    included = np.logical_not(excluded)
    included = included.where(included == True).dropna()

    duration = np.timedelta64(tracks_obj.params['EXCL_THRESH']['DURATION'])
    dt = np.timedelta64(tracks_obj.params['DT'])

    duration_checks = included.groupby(level='uid').apply(
        lambda g: temporal_continuity_check(g, length=duration, dt=dt))
    uids = duration_checks.reset_index()['uid'].values

    tracks_obj.exclusions['simple_duration_cond'] = [
        True for i in range(len(tracks_obj.exclusions))]

    # import pdb; pdb.set_trace()

    for uid in uids:
        tracks_obj.exclusions.loc[
            (slice(None), slice(None), uid, slice(None)),
            'simple_duration_cond'] = duration_checks.loc[uid]

    return tracks_obj


def get_exclusion_categories(tracks_obj):

    #  Ensure multiindex ordered correctly
    tracks_obj.tracks.reset_index(inplace=True)
    tracks_obj.tracks.set_index(['scan', 'time', 'uid', 'level'], inplace=True)

    class_thresh = tracks_obj.params['CLASS_THRESH']
    excl_thresh = tracks_obj.params['EXCL_THRESH']
    n_lvls = len(tracks_obj.params['LEVELS'])
    grid_size = tracks_obj.record.grid_size
    cell_area = grid_size[1] / 1000 * grid_size[2] / 1000

    exclusions = copy.deepcopy(
        tracks_obj.tracks[['grid_x', 'grid_y', 'lon', 'lat']])
    tracks_obj.exclusions = exclusions

    small_area = (
        tracks_obj.system_tracks['proj_area']
        < excl_thresh['SMALL_AREA']).values
    tracks_obj.exclusions['small_area'] = np.repeat(small_area, n_lvls)

    small_conv_area = (
        tracks_obj.tracks['proj_area'].xs(0, level='level')
        < excl_thresh['SMALL_AREA']).values
    tracks_obj.exclusions['small_conv_area'] = np.repeat(
        small_conv_area, n_lvls)

    large_area = (
        tracks_obj.system_tracks['proj_area']
        > excl_thresh['LARGE_AREA']).values
    tracks_obj.exclusions['large_area'] = np.repeat(large_area, n_lvls)

    int_border = (
        tracks_obj.system_tracks['touch_border'] * cell_area
        / tracks_obj.system_tracks['proj_area']) > excl_thresh['BORD_THRESH']
    int_border = int_border.values
    tracks_obj.exclusions['intersect_border'] = np.repeat(int_border, n_lvls)

    tmp_tracks = tracks_obj.tracks[['touch_border', 'proj_area']].xs(
        0, level='level')
    int_border_conv = (
        tmp_tracks['touch_border'] * cell_area
        / tmp_tracks['proj_area']) > excl_thresh['BORD_THRESH']
    int_border_conv = int_border_conv.values
    tracks_obj.exclusions['intersect_border_convective'] = np.repeat(
        int_border_conv, n_lvls)

    velocity_mag = np.sqrt(
        tracks_obj.tracks['u_shift'] ** 2 + tracks_obj.tracks['v_shift'] ** 2)
    small_vel = velocity_mag < class_thresh['VEL_MAG']
    tracks_obj.exclusions['small_velocity'] = small_vel
    rel_velocity_mag = np.sqrt(
        tracks_obj.tracks['u_relative'] ** 2
        + tracks_obj.tracks['v_relative'] ** 2)
    small_rel_vel = rel_velocity_mag < class_thresh['REL_VEL_MAG']
    tracks_obj.exclusions['small_rel_velocity'] = small_rel_vel
    shear_mag = np.sqrt(
        tracks_obj.tracks['u_shear'] ** 2
        + tracks_obj.tracks['v_shear'] ** 2)
    small_shear = shear_mag < class_thresh['SHEAR_MAG']
    tracks_obj.exclusions['small_shear'] = small_shear
    offset_mag = np.sqrt(
        tracks_obj.system_tracks['x_vert_disp'] ** 2
        + tracks_obj.system_tracks['y_vert_disp'] ** 2)
    offset_mag = np.repeat(offset_mag.values, n_lvls)
    small_offset = offset_mag < class_thresh['OFFSET_MAG']
    tracks_obj.exclusions['small_offset'] = small_offset

    semi_major = tracks_obj.tracks.xs(0, level='level')['semi_major']
    semi_minor = tracks_obj.tracks.xs(0, level='level')['semi_minor']
    semi_major_km = semi_major * grid_size[1] / 1000
    length_cond = semi_major_km < excl_thresh['MAJOR_AXIS_LENGTH']
    ratio_cond = semi_major / semi_minor < excl_thresh['AXIS_RATIO']
    linear_cond = np.logical_or(length_cond, ratio_cond).values
    linear_cond = np.repeat(linear_cond, n_lvls)
    tracks_obj.exclusions['non_linear'] = linear_cond

    tracks_obj = get_duration_cond(tracks_obj)
    tracks_obj = get_simple_duration_cond(tracks_obj)

    return tracks_obj


def calc_mean_ambient_winds(tracks_obj):
    u_ambient_mean = copy.deepcopy(tracks_obj.tracks['u_ambient_bottom'])
    u_ambient_mean += tracks_obj.tracks['u_ambient_mid']
    u_ambient_mean += tracks_obj.tracks['u_ambient_top']
    u_ambient_mean = u_ambient_mean/3
    v_ambient_mean = copy.deepcopy(tracks_obj.tracks['v_ambient_bottom'])
    v_ambient_mean += tracks_obj.tracks['v_ambient_mid']
    v_ambient_mean += tracks_obj.tracks['v_ambient_top']
    v_ambient_mean = v_ambient_mean/3
    tracks_obj.tracks['u_ambient_mean'] = np.round(u_ambient_mean, 5)
    tracks_obj.tracks['v_ambient_mean'] = np.round(v_ambient_mean, 5)
    return tracks_obj


def calc_alt_orientation(tracks_obj):
    normal_dir = np.deg2rad(tracks_obj.tracks['orientation'] + 90)
    normal = np.array([np.cos(normal_dir), np.sin(normal_dir)])
    relative_velocity = np.array([
        tracks_obj.tracks['u_relative'],
        tracks_obj.tracks['v_relative']])
    cond = (
        normal[0, :] * relative_velocity[0, :]
        + normal[1, :] * relative_velocity[1, :])
    cond = cond < 0
    phi = tracks_obj.tracks['orientation'] + cond * 180
    # Orientation alt puts the positive y direction in the same half plane
    # as the relative velocity vector
    tracks_obj.tracks['orientation_alt'] = phi

    return tracks_obj


def calc_inflow_type(tracks_obj):

    thresholds = tracks_obj.params['CLASS_THRESH']
    velocity_mag = np.sqrt(
        tracks_obj.tracks['u_shift'] ** 2 + tracks_obj.tracks['v_shift'] ** 2)
    rel_velocity_mag = np.sqrt(
        tracks_obj.tracks['u_relative'] ** 2
        + tracks_obj.tracks['v_relative'] ** 2)

    # Calculate vertical displacement direction.
    vel_dir = np.arctan2(
        tracks_obj.tracks['v_shift'], tracks_obj.tracks['u_shift'])
    vel_dir = np.rad2deg(vel_dir)
    vel_dir = vel_dir.rename('vel_dir')

    rel_vel_dir = np.arctan2(
        tracks_obj.tracks['v_relative'], tracks_obj.tracks['u_relative'])
    rel_vel_dir = rel_vel_dir.rename('rel_vel_dir')
    rel_vel_dir = np.rad2deg(rel_vel_dir)

    inflow_dir = np.mod(vel_dir - rel_vel_dir + 180, 360) - 180
    inflow_dir = inflow_dir.rename('inflow_dir')

    # import pdb; pdb.set_trace()

    inflow_type = np.array(
        ['Front Fed' for i in range(len(inflow_dir))],
        dtype=object)
    inflow_type[np.logical_or(
        inflow_dir.values > 135, inflow_dir.values < -135)] = 'Rear Fed'
    inflow_type[np.logical_and(
        inflow_dir.values > 45,
        inflow_dir.values < 135)] = 'Parallel Fed (Right)'
    inflow_type[np.logical_and(
        inflow_dir.values < -45,
        inflow_dir.values > -135)] = 'Parallel Fed (Left)'
    cond = velocity_mag.values < thresholds['VEL_MAG']
    inflow_type[cond] = 'Ambiguous (Low Velocity)'
    cond = rel_velocity_mag.values < thresholds['REL_VEL_MAG']
    inflow_type[cond] = 'Ambiguous (Low Relative Velocity)'
    tracks_obj.tracks_class['inflow_type'] = inflow_type
    return tracks_obj


def calc_propagation_type(tracks_obj):

    thresholds = tracks_obj.params['CLASS_THRESH']
    rel_velocity_mag = np.sqrt(
        tracks_obj.tracks['u_relative'] ** 2
        + tracks_obj.tracks['v_relative'] ** 2).values
    shear_mag = np.sqrt(
        tracks_obj.tracks['u_shear'] ** 2
        + tracks_obj.tracks['v_shear'] ** 2).values
    propagation_cond = (
        tracks_obj.tracks['u_shear']
        * tracks_obj.tracks['u_relative']
        + tracks_obj.tracks['v_shear']
        * tracks_obj.tracks['v_relative']).values
    propagation_type = np.array(
        ['Down-Shear Propagating' for i in range(len(propagation_cond))],
        dtype=object)
    propagation_type[propagation_cond < 0] = 'Up-Shear Propagating'
    parallel_shear = np.abs(
        propagation_cond / (shear_mag * rel_velocity_mag))
    parallel_shear = parallel_shear < 1 / np.sqrt(2)
    propagation_type[parallel_shear] = 'Ambiguous (Perpendicular Shear)'
    cond = shear_mag < thresholds['SHEAR_MAG']
    propagation_type[cond] = 'Ambiguous (Low Shear)'
    cond = rel_velocity_mag < thresholds['REL_VEL_MAG']
    propagation_type[cond] = 'Ambiguous (Low Relative Velocity)'
    tracks_obj.tracks_class['propagation_type'] = propagation_type

    return tracks_obj


def calc_tilt_type(tracks_obj):

    # Stratiform offset vector
    x_offset = tracks_obj.system_tracks['x_vert_disp']
    y_offset = tracks_obj.system_tracks['y_vert_disp']

    offset_mag = np.sqrt(x_offset ** 2 + y_offset ** 2).values
    n_lvl = len(tracks_obj.params['LEVELS'])
    offset_mag = np.repeat(offset_mag, n_lvl)

    u_shear = tracks_obj.tracks['u_shear']
    v_shear = tracks_obj.tracks['v_shear']
    shear_mag = np.sqrt(u_shear ** 2 + v_shear ** 2).values

    tilt_dir = tracks_obj.system_tracks['shear_rel_tilt_dir'].values
    tilt_dir = np.repeat(tilt_dir, n_lvl)

    theta_e = theta_e = tracks_obj.params['CLASS_THRESH']['ANGLE_BUFFER']

    tilt_type = np.array([
        'Ambiguous (On Quadrant Boundary)'
        for i in range(len(tilt_dir))], dtype=object)
    cond = (-45 + theta_e <= tilt_dir) & (tilt_dir <= 45 - theta_e)
    tilt_type[cond] = 'Down-Shear Tilted'
    cond = (-135 - theta_e >= tilt_dir) | (tilt_dir >= 135 + theta_e)
    tilt_type[cond] = 'Up-Shear Tilted'
    cond = (45 + theta_e <= tilt_dir) & (tilt_dir <= 135 - theta_e)
    tilt_type[cond] = 'Ambiguous (Perpendicular Shear)'
    cond = (-135 + theta_e <= tilt_dir) & (tilt_dir <= -45 - theta_e)
    tilt_type[cond] = 'Ambiguous (Perpendicular Shear)'
    cond = offset_mag < tracks_obj.params['CLASS_THRESH']['OFFSET_MAG']
    tilt_type[cond] = 'Ambiguous (Small Stratiform Offset)'
    cond = shear_mag < tracks_obj.params['CLASS_THRESH']['SHEAR_MAG']
    tilt_type[cond] = 'Ambiguous (Low Shear)'

    tracks_obj.tracks_class['tilt_type'] = tilt_type

    return tracks_obj


def calc_stratiform_type(tracks_obj):

    # Stratiform offset vector
    x_offset = tracks_obj.system_tracks['x_vert_disp']
    y_offset = tracks_obj.system_tracks['y_vert_disp']

    offset_mag = np.sqrt(x_offset ** 2 + y_offset ** 2)
    n_lvl = len(tracks_obj.params['LEVELS'])
    offset_mag = np.repeat(offset_mag.values, n_lvl)

    u_shift = tracks_obj.tracks['u_shift']
    v_shift = tracks_obj.tracks['v_shift']

    vel_mag = np.sqrt(u_shift ** 2 + v_shift ** 2).values

    # Stratiform offset vector
    thresholds = tracks_obj.params['CLASS_THRESH']
    n_lvls = len(tracks_obj.params['LEVELS'])
    theta_e = thresholds['ANGLE_BUFFER']
    rel_tilt_dir = tracks_obj.system_tracks['sys_rel_tilt_dir'].values
    rel_tilt_dir = np.repeat(rel_tilt_dir, n_lvls)
    offset_type = np.array([
        'Ambiguous (On Quadrant Boundary)'
        for i in range(len(rel_tilt_dir))], dtype=object)
    cond = (-45 + theta_e <= rel_tilt_dir) & (rel_tilt_dir <= 45 - theta_e)
    offset_type[cond] = 'Leading Stratiform'
    cond = (-135 - theta_e >= rel_tilt_dir) | (rel_tilt_dir >= 135 + theta_e)
    offset_type[cond] = 'Trailing Stratiform'
    cond = (45 + theta_e <= rel_tilt_dir) & (rel_tilt_dir <= 135 - theta_e)
    offset_type[cond] = 'Parallel Stratiform (Left)'
    cond = (-135 + theta_e <= rel_tilt_dir) & (rel_tilt_dir <= -45 - theta_e)
    offset_type[cond] = 'Parallel Stratiform (Right)'
    cond = offset_mag < tracks_obj.params['CLASS_THRESH']['OFFSET_MAG']
    offset_type[cond] = 'Ambiguous (Small Stratiform Offset)'
    cond = vel_mag < tracks_obj.params['CLASS_THRESH']['VEL_MAG']
    offset_type[cond] = 'Ambiguous (Small Velocity)'

    tracks_obj.tracks_class['offset_type'] = offset_type

    return tracks_obj


def calc_relative_stratiform_type(tracks_obj):
    # Stratiform offset vector
    x_offset = tracks_obj.system_tracks['x_vert_disp']
    y_offset = tracks_obj.system_tracks['y_vert_disp']

    offset_mag = np.sqrt(x_offset ** 2 + y_offset ** 2).values
    n_lvl = len(tracks_obj.params['LEVELS'])
    offset_mag = np.repeat(offset_mag, n_lvl)

    u_shift = tracks_obj.tracks['u_shift']
    v_shift = tracks_obj.tracks['v_shift']
    vel_mag = np.sqrt(u_shift ** 2 + v_shift ** 2).values

    u_relative = tracks_obj.tracks['u_relative']
    v_relative = tracks_obj.tracks['v_relative']
    rel_vel_mag = np.sqrt(u_relative ** 2 + v_relative ** 2).values

    # Stratiform offset vector
    thresholds = tracks_obj.params['CLASS_THRESH']
    n_lvls = len(tracks_obj.params['LEVELS'])
    theta_e = thresholds['ANGLE_BUFFER']
    rel_tilt_dir = tracks_obj.system_tracks['sys_rel_tilt_dir_alt'].values
    rel_tilt_dir = np.repeat(rel_tilt_dir, n_lvls)
    offset_type = np.array([
        'Ambiguous (On Quadrant Boundary)'
        for i in range(len(rel_tilt_dir))], dtype=object)
    cond = (-45 + theta_e <= rel_tilt_dir) & (rel_tilt_dir <= 45 - theta_e)
    offset_type[cond] = 'Relative Leading Stratiform'
    cond = (-135 - theta_e >= rel_tilt_dir) | (rel_tilt_dir >= 135 + theta_e)
    offset_type[cond] = 'Relative Trailing Stratiform'
    cond = (45 + theta_e <= rel_tilt_dir) & (rel_tilt_dir <= 135 - theta_e)
    offset_type[cond] = 'Relative Parallel Stratiform (Left)'
    cond = (-135 + theta_e <= rel_tilt_dir) & (rel_tilt_dir <= -45 - theta_e)
    offset_type[cond] = 'Relative Parallel Stratiform (Right)'
    cond = offset_mag < tracks_obj.params['CLASS_THRESH']['OFFSET_MAG']
    offset_type[cond] = 'Ambiguous (Small Stratiform Offset)'
    cond = vel_mag < tracks_obj.params['CLASS_THRESH']['VEL_MAG']
    offset_type[cond] = 'Ambiguous (Small Velocity)'
    cond = rel_vel_mag < tracks_obj.params['CLASS_THRESH']['REL_VEL_MAG']
    offset_type[cond] = 'Ambiguous (Low Relative Velocity)'

    tracks_obj.tracks_class['rel_offset_type'] = offset_type

    return tracks_obj


def get_system_tracks(tracks_obj):
    """ Calculate system tracks """
    print('Calculating system tracks.')

    # Get position and velocity at tracking level.
    system_tracks = tracks_obj.tracks[
        ['grid_x', 'grid_y', 'com_x', 'com_y', 'lon', 'lat', 'u', 'v',
         'u_shift', 'v_shift', #'mergers', 'parent',
         'u_relative', 'v_relative']]
    system_tracks = system_tracks.xs(
        tracks_obj.params['TRACK_INTERVAL'], level='level')

    # Get number of cells and ellipse fit properties
    # at lowest interval assuming this is first interval in list.
    for prop in [
            'semi_major', 'semi_minor', 'eccentricity',
            'orientation']:#, 'cells']:
        prop_lvl_0 = tracks_obj.tracks[[prop]].xs(0, level='level')
        system_tracks = system_tracks.merge(
            prop_lvl_0, left_index=True, right_index=True)

    if tracks_obj.params['RAIN']:
        system_tracks = update_sys_tracks_rain(tracks_obj, system_tracks)

    # Calculate system maximum
    maximum = tracks_obj.tracks[['field_max']]
    maximum = maximum.groupby(level=['scan', 'time', 'uid']).max()
    system_tracks = system_tracks.merge(
        maximum, left_index=True, right_index=True)

    # Calculate maximum area
    proj_area = tracks_obj.tracks[['proj_area']]
    proj_area = proj_area.groupby(level=['scan', 'time', 'uid']).max()
    system_tracks = system_tracks.merge(
        proj_area, left_index=True, right_index=True)

    # Calculate maximum altitude
    m_alt = tracks_obj.tracks[['max_height']]
    m_alt = m_alt.groupby(level=['scan', 'time', 'uid']).max()
    system_tracks = system_tracks.merge(
        m_alt, left_index=True, right_index=True)

    # Get touch_border for system
    t_border = tracks_obj.tracks[['touch_border']]
    t_border = t_border.groupby(level=['scan', 'time', 'uid']).max()
    system_tracks = system_tracks.merge(
        t_border, left_index=True, right_index=True)

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

    rel_vel_dir = np.arctan2(
        system_tracks['v_relative'], system_tracks['u_relative'])
    rel_vel_dir = np.rad2deg(rel_vel_dir)
    rel_vel_dir = rel_vel_dir.rename('rel_vel_dir')
    rel_vel_dir = np.round(rel_vel_dir, 3)

    u_shear = tracks_obj.tracks['u_shear'].xs(0, level='level').values
    v_shear = tracks_obj.tracks['v_shear'].xs(0, level='level').values
    shear_dir = np.arctan2(v_shear, u_shear)
    shear_dir = np.rad2deg(shear_dir)
    shear_dir = np.round(shear_dir, 3)

    # import pdb; pdb.set_trace()

    tilt_dir = np.arctan2(
        system_tracks['y_vert_disp'], system_tracks['x_vert_disp'])
    tilt_dir = tilt_dir.rename('tilt_dir')
    tilt_dir = np.rad2deg(tilt_dir)
    tilt_dir = np.round(tilt_dir, 3)

    sys_rel_tilt_dir = np.mod(tilt_dir - vel_dir + 180, 360) - 180
    sys_rel_tilt_dir = sys_rel_tilt_dir.rename('sys_rel_tilt_dir')
    sys_rel_tilt_dir = np.round(sys_rel_tilt_dir, 3)

    sys_rel_tilt_dir_alt = np.mod(tilt_dir - rel_vel_dir + 180, 360) - 180
    sys_rel_tilt_dir_alt = sys_rel_tilt_dir_alt.rename('sys_rel_tilt_dir_alt')
    sys_rel_tilt_dir_alt = np.round(sys_rel_tilt_dir_alt, 3)

    shear_rel_tilt_dir = np.mod(tilt_dir - shear_dir + 180, 360) - 180
    shear_rel_tilt_dir = shear_rel_tilt_dir.rename('shear_rel_tilt_dir')
    shear_rel_tilt_dir = np.round(shear_rel_tilt_dir, 3)

    for var in [
            vel_dir, tilt_dir, sys_rel_tilt_dir, sys_rel_tilt_dir_alt,
            shear_rel_tilt_dir]:
        system_tracks = system_tracks.merge(
            var, left_index=True, right_index=True)

    system_tracks = system_tracks.sort_index()
    tracks_obj.system_tracks = system_tracks

    return tracks_obj
