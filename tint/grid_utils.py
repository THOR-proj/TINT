"""
tint.grid_utils
===============

Tools for pulling data from pyart grids.


"""

import datetime

import numpy as np
import pandas as pd
from scipy import ndimage
import networkx 
from networkx.algorithms.components.connected import connected_components
from numba import jit
from numba import int32
import copy

from .steiner import steiner_conv_strat


def parse_grid_datetime(grid_obj):
    """ Obtains datetime object from pyart grid_object. """
    dt_string = grid_obj.time['units'].split(' ')[-1]
    date = dt_string[:10]
    time = dt_string[11:19]
    dt = datetime.datetime.strptime(date + ' ' + time, '%Y-%m-%d %H:%M:%S')
    return dt

@jit()
def get_grid_size(grid_obj):
    """ Calculates grid size per dimension given a grid object. """
    z_len = grid_obj.z['data'][-1] - grid_obj.z['data'][0]
    x_len = grid_obj.x['data'][-1] - grid_obj.x['data'][0]
    y_len = grid_obj.y['data'][-1] - grid_obj.y['data'][0]
    z_size = z_len / (grid_obj.z['data'].shape[0] - 1)
    x_size = x_len / (grid_obj.x['data'].shape[0] - 1)
    y_size = y_len / (grid_obj.y['data'].shape[0] - 1)
    return np.array([z_size, y_size, x_size])


def get_radar_info(grid_obj):
    radar_lon = grid_obj.radar_longitude['data'][0]
    radar_lat = grid_obj.radar_latitude['data'][0]
    info = {'radar_lon': radar_lon,
            'radar_lat': radar_lat}
    return info


def get_grid_alt(grid_size, alt_meters=1500):
    """ Returns next z-index above alt_meters. """
    return np.int(np.ceil(alt_meters/grid_size[0]))


def get_vert_projection(grid, thresh=40, z_min=None, z_max=None):
    """ Returns boolean vertical projection from grid. """
    return np.any(grid[z_min:z_max,:,:] > thresh, axis=0)


def get_filtered_frame(grid, min_size, thresh, z_min=None, z_max=None):
    """ Returns a labeled frame from gridded radar data. Smaller objects
    are removed and the rest are labeled. """
    echo_height = get_vert_projection(grid, thresh, z_min, z_max)
    frame = ndimage.label(echo_height)[0]
    return frame
   
def get_filtered_frame_steiner(grid_obj, field, grid_size, min_size, 
                               z_min=None, z_max=None):
    """ Returns a labeled frame from gridded radar data. Smaller objects
    are removed and the rest are labeled. """
    
    masked = grid_obj.fields[field]['data'][z_min]
    masked.data[masked.data == masked.fill_value] = 0
    grid = copy.copy(masked.data)
    grid[grid==0] = np.nan
    
    try:
        sclass = steiner_conv_strat(
            grid, grid_obj.x['data'], grid_obj.y['data'], 
            grid_size[1], grid_size[2])
    except:
        sclass = np.ones(grid.shape)
        print('\nSteiner Scheme Failed.')
            
    sclass[sclass != 2] = 0
    frame = ndimage.label(sclass)[0]

    return frame, sclass

def clear_small_echoes(label_image, min_size):
    """ Takes in binary image and clears objects less than min_size. """
    flat_image = pd.Series(label_image.flatten())
    flat_image = flat_image[flat_image > 0]
    size_table = flat_image.value_counts(sort=False)
    small_objects = size_table.keys()[size_table < min_size]

    for obj in small_objects:
        label_image[label_image == obj] = 0
    label_image = ndimage.label(label_image)
    return label_image[0]
    
def clear_small_echoes_system(images_con, min_sizes):
    """ Takes in binary image and clears objects less than min_size. """
    small_objects = set()
    for i in range(len(min_sizes)):
        flat_image = pd.Series(images_con[i].flatten())
        flat_image = flat_image[flat_image > 0]
        size_table = flat_image.value_counts(sort=False)
        small_objects_i = set(size_table.keys()[size_table < min_sizes[i]])
        small_objects = small_objects.union(small_objects_i)

    for obj in small_objects:
        images_con[images_con == obj] = 0
    
    relabeled_images_con = np.zeros(images_con.shape, dtype=int)
    
    for new_objs, old_objs in enumerate(set(images_con.flatten().tolist())):
        relabeled_images_con[images_con == old_objs] = new_objs
    
    return relabeled_images_con

def get_level_indices(grid_obj, grid_size, levels):
    """ Returns indices corresponding to the inclusive range
    of levels given by a two element array. """
    [z_min, z_max] = [get_grid_alt(grid_size, level) 
                      for level in levels]
    # Correct z_max for if larger than grid height
    if z_max > (grid_obj.nz - 1):
        z_max = None   
    return z_min, z_max

def get_connected_components(frames):
    """ Take the frames at each level interval and calculate connected 
    components."""
    f_maxes = frames.max(axis=(1,2))
    # Relabel frames so that object numbers are unique across frames
    for i in range(1,frames.shape[0]):
            frames[i,frames[i]>0] += np.cumsum(f_maxes[:-1])[i-1]

    # Create graph for cells that overlap at different vertical levels.
    overlap_graph = networkx.Graph()
    total_objs = frames[-1].max()
    overlap_graph.add_nodes_from(set(range(1, total_objs)))

    # Create edges between the objects that overlap vertically.
    for i in range(frames.shape[0]-1):
        # Determine the objects in frame i.
        objects = set(frames[i][frames[i]>0])
        # Determine the objects in frame i+1.
        objects_next = set(frames[i+1][frames[i+1]>0])
        for j in range(len(list(objects))):
            overlap = np.logical_and(frames[i] == list(objects)[j], 
                                     frames[i+1] > 0)
            overlap_objs = set((frames[i+1][overlap]).flatten())
            # If objects overlap, add edge between object j and first 
            # object from overlap set
            if bool(overlap_objs):
                overlap_graph.add_edges_from(
                    [(list(objects)[j], list(overlap_objs)[0])]
                )
                # Add edges between objects in overlap set
                for k in range(0, len(list(overlap_objs))-1):
                    overlap_graph.add_edges_from(
                        [(list(overlap_objs)[k], list(overlap_objs)[k+1])]
                    )  
    
    # Create new objects based on connected components 
    new_objs = list(connected_components(overlap_graph))
    frames_con = np.zeros(frames.shape, dtype=int)
    for i in range(len(new_objs)):
        frames_con[np.isin(frames, list(new_objs[i]))] = i + 1

    # Require that objects be present in all vertical level intervals    
    new_objs = list(set(frames_con[frames_con>0].flatten()))
    object_counter = 1
    for i in range(len(new_objs)):
        if np.all(np.any(frames_con == new_objs[i], axis=(1,2))):
            frames_con[frames_con == new_objs[i]] = object_counter
            object_counter += 1
        else:
            frames_con[frames_con == new_objs[i]] = 0

    return frames_con, frames

def extract_grid_data(grid_obj, field, grid_size, params, rain):
    """ Returns filtered grid frame and raw grid slice at global shift
    altitude. """
    
    masked = grid_obj.fields[field]['data']
    # Note this won't work if fill_value is nan!
    masked.data[(masked.data == masked.fill_value) 
                | np.isnan(masked.data)] = 0
    gs_alt = params['GS_ALT']
    raw = masked.data[get_grid_alt(grid_size, gs_alt), :, :]
    
    if rain:
        masked_rain = grid_obj.fields['radar_estimated_rain_rate']['data']
        # Note this won't work if fill_value is nan!
        masked_rain.data[(masked_rain.data == masked_rain.fill_value) 
                    | np.isnan(masked_rain.data)] = 0
        raw_rain = masked_rain.data[0, :, :]
    else:
        raw_rain=np.nan
    
    n_levels = params['LEVELS'].shape[0]
    frames = np.zeros([n_levels, grid_obj.nx, grid_obj.ny], dtype=int)
    sclasses = [None]*n_levels
    
    min_sizes = params['MIN_SIZE'] / np.prod(grid_size[1:]/1000)

    # Calculate frames for each level interval
    # Count down because we only want to calculate steiner if 
    # absolutely necessary
    for i in range(frames.shape[0]-1, -1, -1):
        
        [z_min, z_max] = get_level_indices(
            grid_obj, grid_size, params['LEVELS'][i,:]
        )
        if np.any(masked.data[z_min:z_max] > 0):
            if params['FIELD_THRESH'][i] == 'convective':
                frames[i], sclass = get_filtered_frame_steiner(
                    grid_obj, field, grid_size, min_sizes[i], z_min, z_max
                )
                sclasses[i] = sclass
            else: 
                frames[i] = get_filtered_frame(
                    masked.data, min_sizes[i],
                    params['FIELD_THRESH'][i],
                    z_min, z_max
                )
        else:
            break           

    # Calculate connected components between frames
    [frames_con, frames] = get_connected_components(frames)
    
    # Clear small echos
    frames_con = clear_small_echoes_system(frames_con, min_sizes)

    return raw, raw_rain, frames_con, frames, sclasses
