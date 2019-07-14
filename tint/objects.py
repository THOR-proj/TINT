"""
tint.objects
============

Functions for managing and recording object properties.

"""

import numpy as np
import pandas as pd
import pyart
from scipy import ndimage
from skimage.measure import regionprops

from .grid_utils import get_filtered_frame, get_level_indices


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
    obj_radius = np.max((xlength, ylength))/2
    obj_center = np.round(np.median(obj_index, axis=0), 0)
    obj_area = len(obj_index[:, 0])

    obj_extent = {'obj_center': obj_center, 'obj_radius': obj_radius,
                  'obj_area': obj_area, 'obj_index': obj_index}
    return obj_extent


def init_current_objects(first_frame, second_frame, pairs, counter):
    """ Returns a dictionary for objects with unique ids and their
    corresponding ids in frame1 and frame1. This function is called when
    echoes are detected after a period of no echoes. """
    nobj = np.max(first_frame)

    id1 = np.arange(nobj) + 1
    uid = counter.next_uid(count=nobj)
    id2 = pairs
    obs_num = np.zeros(nobj, dtype='i')
    origin = np.array(['-1']*nobj)

    current_objects = {'id1': id1, 'uid': uid, 'id2': id2,
                       'obs_num': obs_num, 'origin': origin}
    current_objects = attach_last_heads(first_frame, second_frame,
                                        current_objects)
    return current_objects, counter


def update_current_objects(frame1, frame2, pairs, old_objects, counter):
    """ Removes dead objects, updates living objects, and assigns new uids to
    new-born objects. """
    nobj = np.max(frame1)
    id1 = np.arange(nobj) + 1
    uid = np.array([], dtype='str')
    obs_num = np.array([], dtype='i')
    origin = np.array([], dtype='str')

    for obj in np.arange(nobj) + 1:
        if obj in old_objects['id2']:
            obj_index = old_objects['id2'] == obj
            uid = np.append(uid, old_objects['uid'][obj_index])
            obs_num = np.append(obs_num, old_objects['obs_num'][obj_index] + 1)
            origin = np.append(origin, old_objects['origin'][obj_index])
        else:
            #  obj_orig = get_origin_uid(obj, frame1, old_objects)
            obj_orig = '-1'
            origin = np.append(origin, obj_orig)
            if obj_orig != '-1':
                uid = np.append(uid, counter.next_cid(obj_orig))
            else:
                uid = np.append(uid, counter.next_uid())
            obs_num = np.append(obs_num, 0)

    id2 = pairs
    current_objects = {'id1': id1, 'uid': uid, 'id2': id2,
                       'obs_num': obs_num, 'origin': origin}
    current_objects = attach_last_heads(frame1, frame2, current_objects)
    return current_objects, counter


def attach_last_heads(frame1, frame2, current_objects):
    """ Attaches last heading information to current_objects dictionary. """
    nobj = len(current_objects['uid'])
    heads = np.ma.empty((nobj, 2))
    for obj in range(nobj):
        if ((current_objects['id1'][obj] > 0) and
                (current_objects['id2'][obj] > 0)):
            center1 = get_object_center(current_objects['id1'][obj], frame1)
            center2 = get_object_center(current_objects['id2'][obj], frame2)
            heads[obj, :] = center2 - center1
        else:
            heads[obj, :] = np.ma.array([-999, -999], mask=[True, True])

    current_objects['last_heads'] = heads
    return current_objects


def check_isolation(raw, filtered, grid_size, params, level):
    """ Returns list of booleans indicating object isolation. Isolated objects
    are not connected to any other objects by pixels greater than ISO_THRESH,
    and have at most one peak. """
    nobj = np.max(filtered)
    min_size = params['MIN_SIZE'][level] / np.prod(grid_size[1:]/1000)
    iso_filtered = get_filtered_frame(raw,
                                      min_size,
                                      params['ISO_THRESH'][level])
    nobj_iso = np.max(iso_filtered)
    iso = np.empty(nobj, dtype='bool')

    for iso_id in np.arange(nobj_iso) + 1:
        obj_ind = np.where(iso_filtered == iso_id)
        objects = np.unique(filtered[obj_ind])
        objects = objects[objects != 0]
        if len(objects) == 1 and single_max(obj_ind, raw, params):
            iso[objects - 1] = True
        else:
            iso[objects - 1] = False
    return iso


def single_max(obj_ind, raw, params):
    """ Returns True if object has at most one peak. """
    max_proj = np.max(raw, axis=0)
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


def get_object_prop(images, cores, grid1, field, record, params):
    """ Returns dictionary of object properties for all objects found in
    each levels of images, where images are the labelled (filtered) 
    frames. """
    id1 = []
    center = []
    grid_x = []
    grid_y = []
    proj_area = []
    longitude = []
    latitude = []
    field_max = []
    max_height = []
    volume = []
    level = []
    isolation = []
    n_cores = []
    semi_major = []
    semi_minor = []
    orientation = []
    eccentricity = []
    nobj = np.max(images)
    levels = images.shape[0]
    
    unit_dim = record.grid_size
    unit_alt = unit_dim[0]/1000
    unit_area = (unit_dim[1]*unit_dim[2])/(1000**2)
    unit_vol = (unit_dim[0]*unit_dim[1]*unit_dim[2])/(1000**3)

    raw3D = grid1.fields[field]['data'].data # Complete dataset
              
    for i in range(levels):

        # Caclulate ellipse fit properties
        ski_props = regionprops(images[i], cache=True)

        for obj in np.arange(nobj) + 1:
            id1.append(obj)        
            
            # Get objects in images[i], i.e. the frame at i-th level
            obj_index = np.argwhere(images[i] == obj)
   
            # 2D frame stats
            level.append(i)
            n_cores.append(len(set(cores[i,obj_index[:,0],obj_index[:,1]])))
            center.append(np.median(obj_index, axis=0))
            # Caclulate mean x and y indices and round to three decimal places
            this_centroid = np.round(np.mean(obj_index, axis=0), 3)
            grid_x.append(this_centroid[1])
            grid_y.append(this_centroid[0])
            proj_area.append(obj_index.shape[0] * unit_area)

            rounded = np.round(this_centroid).astype('i')
            # Convert mean indices to mean position in grid cartesian coords.
            cent_met = np.array([grid1.y['data'][rounded[0]],
                                 grid1.x['data'][rounded[1]]])

            projparams = grid1.get_projparams()
            lon, lat = pyart.core.transforms.cartesian_to_geographic(cent_met[1],
                                                                     cent_met[0],
                                                                     projparams)

            longitude.append(np.round(lon[0], 4))
            latitude.append(np.round(lat[0], 4))

            semi_major.append(ski_props[obj-1].major_axis_length)
            semi_minor.append(ski_props[obj-1].minor_axis_length)
            eccentricity.append(ski_props[obj-1].eccentricity)
            # Negative sign corrects for descending indexing convention 
            # in "y axis". Note orientation given between -pi/2 and pi/2.
            orientation.append(-np.rad2deg(ski_props[obj-1].orientation))
        
            # raw 3D grid stats
            [z_min, z_max] = get_level_indices(
                grid1, record.grid_size, params['LEVELS'][i,:]
            )
            raw3D_i = raw3D[z_min:z_max,:,:]
            obj_slices = [raw3D_i[:, ind[0], ind[1]] for ind in obj_index]
            field_max.append(np.max(obj_slices))
            filtered_slices = [obj_slice > params['FIELD_THRESH'][i]
                               for obj_slice in obj_slices]
            heights = [np.arange(raw3D_i.shape[0])[ind] for ind in filtered_slices]
            max_height.append(np.max(np.concatenate(heights)) * unit_alt)
            
            # Note volume isn't necessarily consistent with proj_area.
            # proj_area is calculated from boolean vertical projection,
            # whereas volume doesn't perform vertical projection.
            volume.append(np.sum(filtered_slices) * unit_vol)

        # cell isolation
        isolation += check_isolation(
            raw3D[z_min:z_max, :,:], images[i].squeeze(), 
            record.grid_size, params, i
        ).tolist()

    objprop = {'id1': id1,
               'center': center,
               'grid_x': grid_x,
               'grid_y': grid_y,
               'proj_area': proj_area,
               'field_max': field_max,
               'max_height': max_height,
               'volume': volume,
               'lon': longitude,
               'lat': latitude,
               'isolated': isolation,
               'level': level,
               'n_cores': n_cores,
               'semi_major': semi_major,
               'semi_minor': semi_minor,
               'eccentricity': eccentricity,
               'orientation': orientation}
    return objprop


def write_tracks(old_tracks, record, current_objects, obj_props):
    """ Writes all cell information to tracks dataframe. """
    print('Writing tracks for scan {}.'.format(str(record.scan)), 
          end='    \r', flush=True)

    nobj = len(current_objects['uid'])
    nlvl = max(obj_props['level'])+1
    scan_num = [record.scan] * nobj * nlvl
    uid = current_objects['uid'].tolist() * nlvl

    new_tracks = pd.DataFrame({
        'scan': scan_num,
        'uid': uid,
        'time': record.time,
        'grid_x': obj_props['grid_x'],
        'grid_y': obj_props['grid_y'],
        'lon': obj_props['lon'],
        'lat': obj_props['lat'],
        'proj_area': obj_props['proj_area'],
        'vol': obj_props['volume'],
        'max': obj_props['field_max'],
        'max_alt': obj_props['max_height'],
        'isolated': obj_props['isolated'],
        'level': obj_props['level'],
        'n_cores': obj_props['n_cores'],
        'semi_major': obj_props['semi_major'],
        'semi_minor': obj_props['semi_minor'],
        'eccentricity': obj_props['eccentricity'],
        'orientation': obj_props['orientation']
    })
   
    new_tracks.set_index(['scan', 'time', 'level', 'uid'], inplace=True)
    new_tracks.sort_index(inplace=True)
    tracks = old_tracks.append(new_tracks)
    return tracks

def post_tracks(tracks_obj):
    """ Calculate additional tracks data from final tracks dataframe. """
    print('Calculating additional tracks properties.', flush=True)
    # Calculate velocity        
    tmp_tracks = tracks_obj.tracks[['grid_x','grid_y']]
    tmp_tracks = tmp_tracks.groupby(
        level=['uid', 'level'], as_index=False, group_keys=False
    )
    tmp_tracks = tmp_tracks.rolling(window=2, center=False)
    # Calculate centred difference.
    dt = tracks_obj.record.interval.total_seconds()
    # Get x,y grid sizes. 
    # Use negative index incase grid two dimensional.     
    dx = tracks_obj.record.grid_size[-2:]
    tmp_tracks = tmp_tracks.apply(lambda x: x[1] - x[0]) * dx / dt
    tmp_tracks = tmp_tracks.rename(
        columns={'grid_x': 'u', 'grid_y': 'v'}
    )
    tracks_obj.tracks = tracks_obj.tracks.merge(
        tmp_tracks, left_index=True, right_index=True
    )
    # Sort multi-index again as levels will be jumbled after rolling etc.
    tracks_obj.tracks = tracks_obj.tracks.sort_index()  

    # Calculate vertical displacement        
    tmp_tracks = tracks_obj.tracks[['grid_x','grid_y']]
    tmp_tracks = tmp_tracks.groupby(
        level=['uid', 'scan', 'time'], as_index=False, group_keys=False
    )
    tmp_tracks = tmp_tracks.rolling(window=2, center=False,)
    tmp_tracks = tmp_tracks.apply(lambda x: x[1] - x[0])
    tmp_tracks = tmp_tracks.rename(
        columns={'grid_x': 'x_vert_disp', 'grid_y': 'y_vert_disp'}
    )
    tracks_obj.tracks = tracks_obj.tracks.merge(
        tmp_tracks, left_index=True, right_index=True
    )
    tracks_obj.tracks = tracks_obj.tracks.sort_index()

    return tracks_obj

def get_system_tracks(tracks_obj):
    """ Calculate system tracks """
    print('Calculating system tracks.', flush=True)
    
    # Get position and velocity at tracking level.        
    system_tracks = tracks_obj.tracks[['grid_x', 'grid_y', 
                                       'lon', 'lat', 'u', 'v']]
    system_tracks = system_tracks.xs(
        tracks_obj.params['TRACK_INTERVAL'], level='level'
    )

    # Get number of cores and ellipse fit properties 
    # at lowest interval assuming this is first
    # interval in list.
    for prop in ['n_cores', 'semi_major', 'semi_minor', 
                 'eccentricity', 'orientation']:
        prop_lvl_0 = tracks_obj.tracks[[prop]].xs(0, level='level')
        system_tracks = system_tracks.merge(prop_lvl_0, left_index=True, 
                                            right_index=True)
    
    # Calculate total tilt.
    tilt = tracks_obj.tracks[['x_vert_disp', 'y_vert_disp']]
    tilt = tilt.sum(level=['scan', 'time', 'uid'])

    system_tracks = system_tracks.merge(tilt, left_index=True, 
                                        right_index=True)
    
    # Calculate magnitude of tilt.
    tilt_mag = np.sqrt((system_tracks['x_vert_disp'] ** 2 
                        + system_tracks['y_vert_disp'] ** 2))
    tilt_mag = tilt_mag.rename('tilt_mag')
    system_tracks = system_tracks.merge(
        tilt_mag, left_index=True, right_index=True
    )
    
    # Calculate tilt direction
    vel_dir = np.arctan2(system_tracks['v'], system_tracks['u'])
    vel_dir = np.rad2deg(vel_dir)
    vel_dir = vel_dir.rename('vel_dir')
    
    tilt_dir = np.arctan2(
        system_tracks['y_vert_disp'], system_tracks['x_vert_disp']
    )
    tilt_dir = tilt_dir.rename('tilt_dir')
    tilt_dir = np.rad2deg(tilt_dir)
        
    sys_rel_tilt_dir = np.mod(tilt_dir - vel_dir + 180, 360)-180
    sys_rel_tilt_dir = sys_rel_tilt_dir.rename('sys_rel_tilt_dir')

    for var in [vel_dir, tilt_dir, sys_rel_tilt_dir]:
        system_tracks = system_tracks.merge(
            var, left_index=True, right_index=True
        )
    
    system_tracks = system_tracks.sort_index()
    tracks_obj.system_tracks = system_tracks 

    return tracks_obj
     
