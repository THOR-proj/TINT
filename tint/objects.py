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
from skimage.feature import peak_local_max
from numba import jit

from .grid_utils import get_filtered_frame, get_level_indices, get_grid_alt
from .steiner import steiner_conv_strat
from scipy.ndimage import center_of_mass

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


def init_current_objects(raw1, raw2, first_frame, second_frame, pairs, counter):
    """ Returns a dictionary for objects with unique ids and their
    corresponding ids in frame1 and frame2. This function is called when
    echoes are detected after a period of no echoes. """
    nobj = np.max(first_frame)

    id1 = np.arange(nobj) + 1
    uid = counter.next_uid(count=nobj)
    id2 = pairs
    obs_num = np.zeros(nobj, dtype='i')
    origin = np.array(['-1']*nobj) # What is origin for?
    # Store uids of objects in frame1 that have merged with objects in
    # frame 2. Store as uids.
    mergers = [set() for i in range(len(uid))] 
    new_mergers = [set() for i in range(len(uid))]

    current_objects = {'id1': id1, 'uid': uid, 'id2': id2, 
                       'mergers': mergers, 'new_mergers':new_mergers,
                       'obs_num': obs_num, 'origin': origin}
    current_objects = attach_last_heads(raw1, raw2, first_frame, second_frame,
                                        current_objects)
    return current_objects, counter


def update_current_objects(raw1, raw2, frame1, frame2, pairs, old_objects, 
                           counter, old_obj_merge):
    """ Removes dead objects, updates living objects, and assigns new uids to
    new-born objects. """
    nobj = np.max(frame1)
    id1 = np.arange(nobj) + 1
    uid = np.array([], dtype='str')
    obs_num = np.array([], dtype='i')
    origin = np.array([], dtype='str')
    mergers = []

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
    
    new_mergers = [set() for i in range(len(uid))]
    mergers = [set() for i in range(len(uid))]
    for i in range(len(uid)):
        if uid[i] in old_objects['uid']:
            old_i = int(np.argwhere(old_objects['uid']==uid[i]))
            new_mergers[i] = set(old_objects['uid'][old_obj_merge[:,old_i]])
            mergers[i]=new_mergers[i].union(old_objects['mergers'][old_i])
        
    current_objects = {'id1': id1, 'uid': uid, 'id2': id2, 
                       'obs_num': obs_num, 'origin': origin,
                       'mergers': mergers, 'new_mergers': new_mergers}

    current_objects = attach_last_heads(raw1, raw2, frame1, frame2, current_objects)
    return current_objects, counter


def attach_last_heads(raw1, raw2, frame1, frame2, current_objects):
    """ Attaches last heading information to current_objects dictionary. """
    nobj = len(current_objects['uid'])
    heads = np.ma.empty((nobj, 2))
    
    for obj in range(nobj):
        if ((current_objects['id1'][obj] > 0) and
                (current_objects['id2'][obj] > 0)):
            center1 = center_of_mass(
                raw1, labels=frame1, index=current_objects['id1'][obj]
            )
            center2 = center_of_mass(
                raw2, labels=frame2, index=current_objects['id2'][obj],
            )
            heads[obj, :] = np.array(center2) - np.array(center1)
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
    
def get_border_indices(obj_ind, b_ind):
    """ Determine the indices that intersect the range boundary of 
    the radar. """
     
    return b_ind
    

def identify_updrafts(raw3D, images, grid1, record, params, sclasses):
    """ Determine "updrafts" by looking for local maxima at each 
    vertical level."""
    
    [dz, dx, dy] = record.grid_size
    z0 = get_grid_alt(record.grid_size, params['LEVELS'][0,0])
    
    if params['FIELD_THRESH'][0] == 'convective':
        sclass = sclasses[0]
    else:
        sclass = steiner_conv_strat(
            raw3D[z0], grid1.x['data'], grid1.y['data'], dx, dy
        )
    
    # Get local maxima
    local_max = []
    
    for k in range(z0, grid1.nz):
        l_max = peak_local_max(
            raw3D[k], threshold_abs = params['UPDRAFT_THRESH']
        )
        l_max = np.insert(
            l_max, 0, np.ones(len(l_max), dtype=int)*k, axis=1
        )
        local_max.append(l_max)
        
    # Find pixels classified as convective by steiner
    conv_ind = np.argwhere(sclass == 2)
    conv_ind_set = set(
        [tuple(conv_ind[i]) for i in range(len(conv_ind))]
    )      
       
    # Define updrafts starting from local maxima at lowest level
    # Append the starting level index - i.e. z0.
    updrafts = [[local_max[0][j]] for j in range(len(local_max[0])) 
                if tuple(local_max[0][j][1:]) in conv_ind_set]
    
    # Find first level with no local_max
    try:
        max_height = [i for i in range(len(local_max))
                      if local_max[i].tolist() == []][0]
    except:
        max_height = len(local_max)
        
    # Create a set to store indices of still 
    # existing updrafts as we check each vertical 
    # level of data.
    current_inds = set(range(len(updrafts)))

    for k in range(1,max_height):

        previous =  np.array([updrafts[i][k-1] for i in current_inds])
        current = local_max[k]
        
        if ((len(previous) == 0) or (len(current)==0)):
            break        

        # Calculate euclidean distance between indices
        mc1, mp1 = np.meshgrid(current[:,1], previous[:,1])
        mc2, mp2 = np.meshgrid(current[:,2], previous[:,2])
        
        match = np.sqrt((mp1-mc1)**2+(mp2-mc2)**2)
       
        next_inds = copy.copy(current_inds)
        minimums = np.argmin(match, axis=1)
        for l in range(match.shape[0]):
            minimum = minimums[l]
            if (match[l,minimum] < 2):
                updrafts[list(current_inds)[l]].append(current[minimum])
            else:
                next_inds = next_inds - set([list(current_inds)[l]])
        # If any indices remain in current - add them to next_inds
        current_inds = next_inds
     
    updrafts = [updrafts[i] for i in range(len(updrafts)) 
                if (len(updrafts[i])>1)]
    
    return updrafts

def get_object_prop(images, cores, grid1, u_shift, v_shift, sclasses,
                    field, record, params, current_objects):
    """ Returns dictionary of object properties for all objects found in
    each level of images, where images are the labelled (filtered) 
    frames. """
    id1 = []
    center = []
    com_x = []
    com_y = []
    grid_x = []
    grid_y = []
    proj_area = []
    longitude = []
    latitude = []
    field_max = []
    max_height = []
    volume = []
    level = []
    #isolation = []
    touch_border = [] # Counts number of pixels touching scan border.
    semi_major = []
    semi_minor = []
    orientation = []
    eccentricity = []
    mergers = []
    updraft_list = []
    
    nobj = np.max(images)
    [levels, rows, columns] = images.shape
    
    unit_dim = record.grid_size
    if unit_dim[-2] != unit_dim[-1]:
        warnings.warn('x and y grid scales unequal - metrics ' +
                      'such as semi_major and semi_minor may be ' +
                      'misleading.')
    # These metrics are in km^[1, 2, 3] respectively
    unit_alt = unit_dim[0]/1000
    unit_area = (unit_dim[1]*unit_dim[2])/(1000**2)
    unit_vol = (unit_dim[0]*unit_dim[1]*unit_dim[2])/(1000**3)

    raw3D = grid1.fields[field]['data'].data # Complete dataset
    z_values = grid1.z['data']/1000
    
    all_updrafts = identify_updrafts(
        raw3D, images, grid1, record, params, sclasses
    )
       
    for i in range(levels):
      
        # Get vertical indices of ith level
        [z_min, z_max] = get_level_indices(
            grid1, record.grid_size, params['LEVELS'][i,:]
        )
        
        # Caclulate ellipse fit properties
        ski_props = regionprops(images[i], raw3D[z_min], cache=True)

        for obj in np.arange(nobj) + 1:
            # Append current object number
            id1.append(obj)
            
            # Append list of objects that have merged with obj
            mergers.append(current_objects['mergers'][obj-1])
            
            # Get objects in images[i], i.e. the frame at i-th level
            obj_index = np.argwhere(images[i] == obj)
   
            # 2D frame stats
            level.append(i)
          
            # Work out how many gridcells touch the border
            obj_ind_list = obj_index.tolist()
            obj_ind_set = set(
                [tuple(obj_ind_list[i]) for i in range(len(obj_ind_list))]
            )
            b_intersect = obj_ind_set.intersection(
                params['BOUNDARY_GRID_CELLS']
            )
            touch_border.append(len(b_intersect))
            
            # Append median object index as measure of center
            center.append(np.median(obj_index, axis=0))
                       
            # Append area of vertical projection.
            proj_area.append(obj_index.shape[0] * unit_area)
            
            # Append mean object index (centroid) in grid units
            # Note, y component is zeroth index of this_centroid, 
            # x component is first index of this_centroid, unit_dim 
            # is list [dz, dx, dy].
            g_y = ski_props[obj-1].centroid[0] * unit_dim[2]
            g_y += grid1.y['data'][0]
            g_x = ski_props[obj-1].centroid[1] * unit_dim[1]
            g_x += grid1.x['data'][0] 
            grid_x.append(np.round(g_x, 1))
            grid_y.append(np.round(g_y, 1))
            
            # Append object center of mass (reflectivity weighted
            # centroid) in grid units
            g_y = ski_props[obj-1].weighted_centroid[0] * unit_dim[2]
            g_y += grid1.y['data'][0]
            g_x = ski_props[obj-1].weighted_centroid[1] * unit_dim[1]
            g_x += grid1.x['data'][0]
            com_x.append(np.round(g_x, 1))
            com_y.append(np.round(g_y, 1))  
            
            # Append centroid in lat, lon units. 
            projparams = grid1.get_projparams()
            lon, lat = pyart.core.transforms.cartesian_to_geographic(
                g_x, g_y, projparams)
            longitude.append(np.round(lon[0], 5))
            latitude.append(np.round(lat[0], 5))
            
            # Append ellipse properties
            # Note semi_major, semi_minor are stored in `index' 
            # units which effectively assumes dx = dy. 
            attrs = ['major_axis_length', 'minor_axis_length', 
                     'eccentricity', 'orientation']
            lists = [semi_major, semi_minor, eccentricity, orientation]
            for j in range(0, len(attrs)):
                try:
                    lists[j].append(
                        np.round(eval('ski_props[obj-1].' + attrs[j]), 3)
                    )
                except:
                    lists[j].append(np.nan)
            
            # Calculate raw3D slices                            
            raw3D_i = raw3D[z_min:z_max,:,:]
            obj_slices = [raw3D_i[:, ind[0], ind[1]] for ind in obj_index]
            
            if params['FIELD_THRESH'][i] == 'convective':
                sclasses_i = sclasses[i]
                sclasses_slices = [sclasses_i[ind[0], ind[1]] 
                                   for ind in obj_index]
                field_max.append(np.max(obj_slices))
                filtered_slices = [sclasses_slice == 2
                                   for sclasses_slice in sclasses_slices]
            else:
                field_max.append(np.max(obj_slices))
                filtered_slices = [obj_slice > params['FIELD_THRESH'][i]
                                   for obj_slice in obj_slices]
    
            # Append maximum height
            heights = [np.arange(raw3D_i.shape[0])[ind] 
                       for ind in filtered_slices]
            max_height.append(np.max(np.concatenate(heights)) * unit_alt 
                              + z_values[z_min])
            
            # Append volume
            # Note volume isn't necessarily consistent with proj_area.
            # proj_area is calculated from boolean vertical projection,
            # whereas volume doesn't perform vertical projection.
            volume.append(np.sum(filtered_slices) * unit_vol)
            
            # Append reflectivity cells based on vertically 
            # overlapping reflectivity maxima. 
            if i==0:
                all_updrafts_0 = [all_updrafts[i][0].tolist() 
                                  for i in range(len(all_updrafts))]
                updraft_obj_inds = [i for i in range(len(all_updrafts_0)) 
                                    if all_updrafts_0[i][1:] 
                                    in obj_index.tolist()]
                updrafts_obj = [all_updrafts[i] for i in updraft_obj_inds]
                updraft_list.append(updrafts_obj)
            else:
                updraft_list.append([])
            
        # cell isolation
        #isolation += check_isolation(
        #    raw3D[z_min:z_max, :,:], images[i].squeeze(), 
        #    record.grid_size, params, i
        #).tolist()
 
    objprop = {
        'id1': id1,
        'center': center,
        'u_shift': u_shift * levels,
        'v_shift': v_shift * levels ,
        'grid_x': grid_x,
        'grid_y': grid_y,
        'com_x': com_x,
        'com_y': com_y,
        'proj_area': proj_area,
        'field_max': field_max,
        'max_height': max_height,
        'volume': volume,
        'lon': longitude,
        'lat': latitude,
        #'isolated': isolation,
        'touch_border': touch_border,
        'level': level,
        'semi_major': semi_major,
        'semi_minor': semi_minor,
        'eccentricity': eccentricity,
        'mergers': mergers,
        # Add orientation
        # Negative sign corrects for descending indexing convention 
        # in "y axis". Orientation given between -pi/2 and pi/2.
        'orientation': np.round(-np.rad2deg(orientation), 3),
        'updrafts': updraft_list,
    }
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
        'com_x': obj_props['com_x'],
        'com_y': obj_props['com_y'],
        'u_shift': obj_props['u_shift'],
        'v_shift': obj_props['v_shift'],
        'lon': obj_props['lon'],
        'lat': obj_props['lat'],
        'proj_area': obj_props['proj_area'],
        'vol': obj_props['volume'],
        'max': obj_props['field_max'],
        'max_alt': obj_props['max_height'],
        #'isolated': obj_props['isolated'],
        'touch_border': obj_props['touch_border'],
        'level': obj_props['level'],
        'semi_major': obj_props['semi_major'],
        'semi_minor': obj_props['semi_minor'],
        'eccentricity': obj_props['eccentricity'],
        'orientation': obj_props['orientation'],
        'mergers': obj_props['mergers'],
        'updrafts': obj_props['updrafts'],
    })
     
    new_tracks.set_index(['scan', 'time', 'level', 'uid'], inplace=True)
    new_tracks.sort_index(inplace=True)
    tracks = old_tracks.append(new_tracks)
    return tracks
    
def smooth(group_df, n=2):
        
    group_df_smoothed = copy.deepcopy(group_df)
      
    # Smooth middle cases.
    if len(group_df) >= (2*n+1):
        group_df_smoothed = group_df.rolling(
            window=(2*n+1), center=True
        ).mean()
    
    # Deal with end cases.
    new_n = min(n, int(np.ceil(len(group_df)/2)))
    for k in range(0,new_n):
        # Use a k+1 window on both sides to smooth
        fwd = np.mean(group_df.iloc[:k+2])
        bwd = np.mean(group_df.iloc[-k-2:])
    
        group_df_smoothed.iloc[k] = fwd
        group_df_smoothed.iloc[-1-k] = bwd
    
    return group_df_smoothed

def post_tracks(tracks_obj):
    """ Calculate additional tracks data from final tracks dataframe. """
    print('Calculating additional tracks properties.', flush=True)
    # Round max - for some reason works here but not in get_obj_props
    # Likely a weird floating point storage issue
    tracks_obj.tracks['max'] = np.round(tracks_obj.tracks['max'], 2)
    
    # Smooth u_shift, v_shift
    tmp_tracks = tracks_obj.tracks[['u_shift','v_shift']]
    # Calculate forward difference for first time step
    tmp_tracks = tmp_tracks.groupby(level=['uid','level'], 
                                   as_index=False, 
                                   group_keys=False)
    tracks_obj.tracks[['u_shift','v_shift']] = tmp_tracks.apply(
        lambda x: smooth(x)
    )
    
    # Calculate velocity using centred difference.    
    tmp_tracks = tracks_obj.tracks[['grid_x','grid_y']]
    tmp_tracks = tmp_tracks.groupby(
        level=['uid', 'level'], as_index=False, group_keys=False
    )
    tmp_tracks = tmp_tracks.rolling(window=5, center=True)
    
    dt = tracks_obj.record.interval.total_seconds()
    tmp_tracks = tmp_tracks.apply(lambda x: np.round(((x[4] - x[0])/(4*dt)), 3))
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
    tmp_tracks = tmp_tracks.rolling(window=2, center=False)
    tmp_tracks = tmp_tracks.apply(lambda x: np.round((x[1] - x[0]), 3))
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
    system_tracks = tracks_obj.tracks[
        ['grid_x', 'grid_y', 'com_x', 'com_y', 'lon', 'lat', 'u', 'v', 
         'mergers', 'u_shift', 'v_shift']
    ]
    system_tracks = system_tracks.xs(
        tracks_obj.params['TRACK_INTERVAL'], level='level'
    )

    # Get number of cores and ellipse fit properties 
    # at lowest interval assuming this is first
    # interval in list.
    for prop in ['semi_major', 'semi_minor', 
                 'eccentricity', 'orientation', 'updrafts'
                 ]:
        prop_lvl_0 = tracks_obj.tracks[[prop]].xs(0, level='level')
        system_tracks = system_tracks.merge(prop_lvl_0, left_index=True, 
                                            right_index=True)

    # Calculate system maximum
    maximum = tracks_obj.tracks[['max']]
    maximum = maximum.max(level=['scan', 'time', 'uid'])
    system_tracks = system_tracks.merge(maximum, left_index=True, 
                                        right_index=True)
                                        
    # Calculate maximum area
    proj_area = tracks_obj.tracks[['proj_area']]
    proj_area = proj_area.max(level=['scan', 'time', 'uid'])
    system_tracks = system_tracks.merge(proj_area, left_index=True, 
                                        right_index=True)
    
    # Calculate maximum altitude
    m_alt = tracks_obj.tracks[['max_alt']]
    m_alt = m_alt.max(level=['scan', 'time', 'uid'])
    system_tracks = system_tracks.merge(m_alt, left_index=True, 
                                        right_index=True)

    # Get touch_border for system
    t_border = tracks_obj.tracks[['touch_border']]
    t_border = t_border.max(level=['scan', 'time', 'uid'])
    system_tracks = system_tracks.merge(t_border, left_index=True, 
                                        right_index=True)

    # Get isolated for system
    #iso = tracks_obj.tracks[['isolated']]
    #iso = iso.prod(level=['scan', 'time', 'uid']).astype('bool')
    #system_tracks = system_tracks.merge(iso, left_index=True, 
    #                                    right_index=True)
    
    # Calculate total vertical displacement.
    n_lvl = tracks_obj.params['LEVELS'].shape[0]
    pos_0 = tracks_obj.tracks[['com_x', 'com_y']].xs(0, level='level')
    pos_1 = tracks_obj.tracks[['grid_x', 'grid_y']].xs(n_lvl-1, level='level') 
    
    pos_0.rename(columns={'com_x': 'x_vert_disp', 
                          'com_y': 'y_vert_disp'},
                 inplace=True)
    pos_1.rename(columns={'grid_x': 'x_vert_disp', 
                          'grid_y': 'y_vert_disp'},
                 inplace=True)

    system_tracks = system_tracks.merge(pos_1-pos_0, left_index=True, 
                                        right_index=True)
    
    # Calculate magnitude of vertical displacement.
    tilt_mag = np.round(np.sqrt((system_tracks['x_vert_disp'] ** 2 
                        + system_tracks['y_vert_disp'] ** 2)), 3)
    tilt_mag = tilt_mag.rename('tilt_mag')
    system_tracks = system_tracks.merge(
        tilt_mag, left_index=True, right_index=True
    )
    
    # Calculate vertical displacement direction.
    vel_dir = np.arctan2(system_tracks['v_shift'], system_tracks['u_shift'])
    vel_dir = np.rad2deg(vel_dir)
    vel_dir = vel_dir.rename('vel_dir')
    vel_dir = np.round(vel_dir, 3)
    
    tilt_dir = np.arctan2(
        system_tracks['y_vert_disp'], system_tracks['x_vert_disp']
    )
    tilt_dir = tilt_dir.rename('tilt_dir')
    tilt_dir = np.rad2deg(tilt_dir)
    tilt_dir = np.round(tilt_dir, 3)    
    
    sys_rel_tilt_dir = np.mod(tilt_dir - vel_dir + 180, 360)-180
    sys_rel_tilt_dir = sys_rel_tilt_dir.rename('sys_rel_tilt_dir')
    sys_rel_tilt_dir = np.round(sys_rel_tilt_dir, 3)

    for var in [vel_dir, tilt_dir, sys_rel_tilt_dir]:
        system_tracks = system_tracks.merge(
            var, left_index=True, right_index=True
        )
    
    system_tracks = system_tracks.sort_index()
    tracks_obj.system_tracks = system_tracks 

    return tracks_obj
     
