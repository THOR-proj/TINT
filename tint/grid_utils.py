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


def parse_grid_datetime(grid_obj):
    """ Obtains datetime object from pyart grid_object. """
    dt_string = grid_obj.time['units'].split(' ')[-1]
    date = dt_string[:10]
    time = dt_string[11:19]
    dt = datetime.datetime.strptime(date + ' ' + time, '%Y-%m-%d %H:%M:%S')
    return dt


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
    labeled_echo = ndimage.label(echo_height)[0]
    # May need to do this elsewhere for system stuff.
    frame = clear_small_echoes(labeled_echo, min_size)
    return frame
    
def get_filtered_frame_steiner(grid_obj, field, z_min=None, z_max=None):
    """ Returns a labeled frame from gridded radar data. Smaller objects
    are removed and the rest are labeled. """
    
    masked = grid_obj.fields[field]['data'][z_min:z_max]
    masked.data[masked.data == masked.fill_value] = 0
    grid = masked.data
    
    [dz, dx, dy] = grid.shape
    sclass = np.zeros(grid.shape, dtype=int32)
       
    for i in range(grid.shape[0]):
        sclass[i] = steiner_conv_strat(
            grid[i], grid_obj.x['data'], grid_obj.y['data'], dx, dy
        )
        
    sclass_proj = np.any(sclass == 2, axis==0)
        
    labeled_echo = ndimage.label(sclass_proj)[0]
    # May need to do this elsewhere for system stuff.
    frame = clear_small_echoes(labeled_echo, min_size)
    return frame

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

@jit(nopython=True)
def steiner_conv_strat(refl, x, y, dx, dy, intense=42, peak_relation=0,
                        area_relation=1, bkg_rad=11000, use_intense=True):
    """
    We perform the Steiner et al. (1995) algorithm for echo classification
    using only the reflectivity field in order to classify each grid point
    as either convective, stratiform or undefined. Grid points are
    classified as follows,
    0 = Undefined
    1 = Stratiform
    2 = Convective
    """
    def convective_radius(ze_bkg, area_relation):
        """
        Given a mean background reflectivity value, we determine via a step
        function what the corresponding convective radius would be.
        Higher background reflectivitives are expected to have larger
        convective influence on surrounding areas, so a larger convective
        radius would be prescribed.
        """
        if area_relation == 0:
            if ze_bkg < 30:
                conv_rad = 1000.
            elif (ze_bkg >= 30) & (ze_bkg < 35.):
                conv_rad = 2000.
            elif (ze_bkg >= 35.) & (ze_bkg < 40.):
                conv_rad = 3000.
            elif (ze_bkg >= 40.) & (ze_bkg < 45.):
                conv_rad = 4000.
            else:
                conv_rad = 5000.

        if area_relation == 1:
            if ze_bkg < 25:
                conv_rad = 1000.
            elif (ze_bkg >= 25) & (ze_bkg < 30.):
                conv_rad = 2000.
            elif (ze_bkg >= 30.) & (ze_bkg < 35.):
                conv_rad = 3000.
            elif (ze_bkg >= 35.) & (ze_bkg < 40.):
                conv_rad = 4000.
            else:
                conv_rad = 5000.

        if area_relation == 2:
            if ze_bkg < 20:
                conv_rad = 1000.
            elif (ze_bkg >= 20) & (ze_bkg < 25.):
                conv_rad = 2000.
            elif (ze_bkg >= 25.) & (ze_bkg < 30.):
                conv_rad = 3000.
            elif (ze_bkg >= 30.) & (ze_bkg < 35.):
                conv_rad = 4000.
            else:
                conv_rad = 5000.

        if area_relation == 3:
            if ze_bkg < 40:
                conv_rad = 0.
            elif (ze_bkg >= 40) & (ze_bkg < 45.):
                conv_rad = 1000.
            elif (ze_bkg >= 45.) & (ze_bkg < 50.):
                conv_rad = 2000.
            elif (ze_bkg >= 50.) & (ze_bkg < 55.):
                conv_rad = 6000.
            else:
                conv_rad = 8000.

        return conv_rad

    def peakedness(ze_bkg, peak_relation):
        """
        Given a background reflectivity value, we determine what the necessary
        peakedness (or difference) has to be between a grid point's
        reflectivity and the background reflectivity in order for that grid
        point to be labeled convective.
        """
        if peak_relation == 0:
            if ze_bkg < 0.:
                peak = 10.
            elif (ze_bkg >= 0.) and (ze_bkg < 42.43):
                peak = 10. - ze_bkg ** 2 / 180.
            else:
                peak = 0.

        elif peak_relation == 1:
            if ze_bkg < 0.:
                peak = 14.
            elif (ze_bkg >= 0.) and (ze_bkg < 42.43):
                peak = 14. - ze_bkg ** 2 / 180.
            else:
                peak = 4.

        return peak

    sclass = np.zeros(refl.shape, dtype=int32)
    ny, nx = refl.shape

    for i in range(0, nx):
        # Get stencil of x grid points within the background radius
        imin = np.max(np.array([1, (i - bkg_rad / dx)], dtype=int32))
        imax = np.min(np.array([nx, (i + bkg_rad / dx)], dtype=int32))

        for j in range(0, ny):
            # First make sure that the current grid point has not already been
            # classified. This can happen when grid points within the
            # convective radius of a previous grid point have also been
            # classified.
            if ~np.isnan(refl[j, i]) & (sclass[j, i] == 0):
                # Get stencil of y grid points within the background radius
                jmin = np.max(np.array([1, (j - bkg_rad / dy)], dtype=int32))
                jmax = np.min(np.array([ny, (j + bkg_rad / dy)], dtype=int32))

                n = 0
                sum_ze = 0

                # Calculate the mean background reflectivity for the current
                # grid point, which will be used to determine the convective
                # radius and the required peakedness.

                for l in range(imin, imax):
                    for m in range(jmin, jmax):
                        if not np.isnan(refl[m, l]):
                            rad = np.sqrt(
                                (x[l] - x[i]) ** 2 + (y[m] - y[j]) ** 2)

                        # The mean background reflectivity will first be
                        # computed in linear units, i.e. mm^6/m^3, then
                        # converted to decibel units.
                            if rad <= bkg_rad:
                                n += 1
                                sum_ze += 10. ** (refl[m, l] / 10.)

                if n == 0:
                    ze_bkg = np.inf
                else:
                    ze_bkg = 10.0 * np.log10(sum_ze / n)

                # Now get the corresponding convective radius knowing the mean
                # background reflectivity.
                conv_rad = convective_radius(ze_bkg, area_relation)

                # Now we want to investigate the points surrounding the current
                # grid point that are within the convective radius, and whether
                # they too are convective, stratiform or undefined.

                # Get stencil of x and y grid points within the convective
                # radius.
                lmin = np.max(
                    np.array([1, int(i - conv_rad / dx)], dtype=int32))
                lmax = np.min(
                    np.array([nx, int(i + conv_rad / dx)], dtype=int32))
                mmin = np.max(
                    np.array([1, int(j - conv_rad / dy)], dtype=int32))
                mmax = np.min(
                    np.array([ny, int(j + conv_rad / dy)], dtype=int32))

                if use_intense and (refl[j, i] >= intense):
                    sclass[j, i] = 2

                    for l in range(lmin, lmax):
                        for m in range(mmin, mmax):
                            if not np.isnan(refl[m, l]):
                                rad = np.sqrt(
                                    (x[l] - x[i]) ** 2
                                    + (y[m] - y[j]) ** 2)

                                if rad <= conv_rad:
                                    sclass[m, l] = 2

                else:
                    peak = peakedness(ze_bkg, peak_relation)

                    if refl[j, i] - ze_bkg >= peak:
                        sclass[j, i] = 2

                        for l in range(imin, imax):
                            for m in range(jmin, jmax):
                                if not np.isnan(refl[m, l]):
                                    rad = np.sqrt(
                                        (x[l] - x[i]) ** 2
                                        + (y[m] - y[j]) ** 2)

                                    if rad <= conv_rad:
                                        sclass[m, l] = 2

                    else:
                        # If by now the current grid point has not been
                        # classified as convective by either the intensity
                        # criteria or the peakedness criteria, then it must be
                        # stratiform.
                        sclass[j, i] = 1

    return sclass

def extract_grid_data(grid_obj, field, grid_size, params):
    """ Returns filtered grid frame and raw grid slice at global shift
    altitude. """
    
    masked = grid_obj.fields[field]['data']
    masked.data[masked.data == masked.fill_value] = 0
    gs_alt = params['GS_ALT']
    raw = masked.data[get_grid_alt(grid_size, gs_alt), :, :]
    
    n_levels = params['LEVELS'].shape[0]
    frames = np.zeros([n_levels, grid_obj.nx, grid_obj.ny], dtype=int)

    # Calculate frames for each level interval
    for i in range(0, frames.shape[0]):
        min_size = params['MIN_SIZE'][i] / np.prod(grid_size[1:]/1000)
        [z_min, z_max] = get_level_indices(
            grid_obj, grid_size, params['LEVELS'][i,:]
        ) 
        frames[i] = get_filtered_frame(
            masked.data, min_size,
            params['FIELD_THRESH'][i],
            z_min, z_max
        )   

    # Calculate connected components between frames
    [frames_con, frames] = get_connected_components(frames)

    return raw, frames_con, frames
