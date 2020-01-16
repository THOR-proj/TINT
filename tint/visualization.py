"""
tint.visualization
==================

Visualization tools for tracks objects.

"""
import warnings
warnings.filterwarnings('ignore')

import gc
import os
import pandas as pd
import numpy as np
import shutil
import tempfile
import matplotlib as mpl
from IPython.display import display, Image
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse
from matplotlib.patches import FancyArrow
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import copy
import glob
import xarray as xr
from scipy.interpolate import griddata

import pyart
from pyart.core.transforms import cartesian_to_geographic
from pyart.core.transforms import geographic_to_cartesian

from .grid_utils import get_grid_alt
from .visualization_aux import *

class Tracer(object):
    colors = ['m', 'r', 'lime', 'darkorange', 'k', 'b', 'darkgreen', 'yellow']
    colors.reverse()

    def __init__(self, tobj, persist):
        self.tobj = tobj
        self.persist = persist
        self.color_stack = self.colors * 10
        self.cell_color = pd.Series()
        self.history = None
        self.current = None

    def update(self, nframe):
        self.history = self.tobj.tracks.loc[:nframe]
        self.current = self.tobj.tracks.loc[nframe]
        if not self.persist:
            mergers = set.union(*self.current['mergers'].values.tolist())
            dead_cells = [key for key in self.cell_color.keys()
                          if (key
                          not in self.current.index.get_level_values('uid')
                          and key not in mergers)]
            self.color_stack.extend(self.cell_color[dead_cells])
            self.cell_color.drop(dead_cells, inplace=True)

    def _check_uid(self, uid):
        mergers = set.union(*self.current['mergers'].values.tolist())
        if ((uid not in self.cell_color.keys()) and (uid not in mergers)):
            try:
                self.cell_color[uid] = self.color_stack.pop()
            except IndexError:
                self.color_stack += self.colors * 5
                self.cell_color[uid] = self.color_stack.pop()

    def plot(self, ax):
        mergers = set.union(*self.current['mergers'].values.tolist())
        for uid, group in self.history.groupby(level='uid'):
            self._check_uid(uid)
            tracer = group[['lon', 'lat']]
            if (self.persist
                or (uid in self.current.reset_index(level=['time']).index)
                or (uid in mergers)):
                ax.plot(tracer.lon, tracer.lat, self.cell_color[uid])                             
                      
                      
def plot_tracks_horiz_cross(f_tobj, grid, alt, vmin=-8, vmax=64,
                            cmap=pyart.graph.cm_colorblind.HomeyerRainbow, 
                            fig=None, ax=None, 
                            projection=ccrs.PlateCarree(), 
                            scan_boundary=False,
                            tracers=False, ellipses='conv', legend=True,
                            uid_ind=None, center_ud=False, updraft_ind=None, 
                            box_rad=.75, wrf_winds=False, line_coords=False,
                            angle=None, mp='lin', **kwargs):       
                                                      
    # Initialise fig and ax if not passed as arguments
    if fig is None:
        init_fonts()
        fig = plt.figure(figsize=(18, 7.5))
    if ax is None:
        ax = fig.add_subplot(1, 1, 1, projection=projection)
    if tracers:
        tracer = Tracer(f_tobj, persist)
                
    # Initialise dataframes for convective and stratiform regions
    n_lvl = f_tobj.params['LEVELS'].shape[0]
    tracks_low = f_tobj.tracks.xs(0, level='level')
    tracks_high = f_tobj.tracks.xs(n_lvl-1, level='level') 
    
    # Restrict tracks to time of grid
    time_ind = f_tobj.tracks.index.get_level_values('time')
    time_ind = np.array(time_ind).astype('datetime64[m]')
    # Below perhaps not necessary!
    grid_time = np.datetime64(grid.time['units'][14:]).astype('datetime64[m]')
    scan_ind = f_tobj.tracks.index.get_level_values('scan')
    nframe = scan_ind[time_ind == grid_time][0]
    frame_tracks_low = tracks_low.loc[nframe].reset_index(level=['time'])
    frame_tracks_high = tracks_high.loc[nframe].reset_index(level=['time'])
    display = pyart.graph.GridMapDisplay(grid)
    projparams = grid.get_projparams()
    grid_size = f_tobj.grid_size
    transform = projection._as_mpl_transform(ax)
    hgt_ind = get_grid_alt(grid_size, alt)
    up_start_ind = f_tobj.params['UPDRAFT_START']
    up_start_ind = get_grid_alt(grid_size, up_start_ind)
    ud_hgt_ind = hgt_ind - up_start_ind
    
    # Plot tracers if necessary
    if tracers:
        tracer.update(nframe)
        tracer.plot(ax)
        
    if uid_ind is not None:
        # Drop other objects from frame_tracks
        drop_set = set(frame_tracks_low.index) - {uid_ind}
        frame_tracks_low.drop(drop_set, inplace=True)
        frame_tracks_high.drop(drop_set, inplace=True)
        
        # Define bounding box of object
        lat_box = frame_tracks_low['lat'].iloc[0]
        lon_box = frame_tracks_low['lon'].iloc[0]
        box = np.array([-1*box_rad, box_rad])
        lvxlim = (lon_box) + box
        lvylim = (lat_box) + box
        
        if center_ud and (updraft_ind is not None):
            ud = frame_tracks_low.iloc[0]['updrafts'][updraft_ind]
            x_ud = grid.x['data'][np.array(ud)[0,2]]
            y_ud = grid.y['data'][np.array(ud)[0,1]]
            projparams = grid.get_projparams()
            lon_ud, lat_ud = cartesian_to_geographic(x_ud, y_ud, projparams)
            # Set crosshair to be at updraft location
            [lon_ch, lat_ch] = [lon_ud, lat_ud]
        else:
            # Set crosshair to be at object location
            [lon_ch, lat_ch] = [lon_box, lat_box]
    else:
        # Set crosshair to be at radar position
        radar_lon = f_tobj.radar_info['radar_lon']
        radar_lat = f_tobj.radar_info['radar_lat']
        [lon_ch, lat_ch] = [radar_lon, radar_lat]
        
    # Plot reflectivity and crosshairs
    if line_coords:
        # Plot along line cross hair
        m = np.tan(np.deg2rad(angle))
        c = lat_ch - m*lon_ch
        lat_ch_0 = m*kwargs['lon_lines'][0] + c
        lat_ch_1 = m*kwargs['lon_lines'][-1] + c
        ax.plot([kwargs['lon_lines'][0], kwargs['lon_lines'][-1]], 
                [lat_ch_0, lat_ch_1], 
                '--r', linewidth=2.0)
        
        # Plot across line cross hair
        m = - 1/m
        c = lat_ch - m*lon_ch
        lat_ch_0 = m*kwargs['lon_lines'][0] + c
        lat_ch_1 = m*kwargs['lon_lines'][-1] + c
        ax.plot([kwargs['lon_lines'][0], kwargs['lon_lines'][-1]], 
                [lat_ch_0, lat_ch_1], 
                '--r', linewidth=2.0)
    else:
        display.plot_crosshairs(lon=lon_ch, lat=lat_ch)
    
    # Plot grid
    display.plot_grid(f_tobj.field, level=hgt_ind,
                      vmin=vmin, vmax=vmax, mask_outside=False,
                      cmap=cmap, transform=projection, ax=ax, **kwargs)           
    # Set labels
    ax.set_title('Altitude {} m'.format(alt))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Plot scan boundary
    if scan_boundary:        
        plot_boundary(ax, f_tobj, grid, projparams)

    # Return if no objects exist at current grid time
    if grid_time not in time_ind: 
        del display                     
        return
    
    # Plot objects
    for ind, uid in enumerate(frame_tracks_low.index):
                                                                        
        # Plot object labels 
        lon_low = frame_tracks_low['lon'].iloc[ind]
        lat_low = frame_tracks_low['lat'].iloc[ind]
        mergers = list(frame_tracks_low['mergers'].iloc[ind])
        mergers_str = ", ".join(mergers)
        
        ax.text(lon_low-.05, lat_low+0.05, uid, 
                transform=projection, fontsize=12)
        ax.text(lon_low+.05, lat_low-0.05, mergers_str, 
                transform=projection, fontsize=9)
                
        split_label = False
        if split_label:
            parent = list(frame_tracks_low['parent'].iloc[ind])
            parent_str = ", ".join(parent)
            ax.text(lon_low+.05, lat_low+0.1, parent_str, 
                    transform=projection, fontsize=9)
        
        # Plot velocities 
        dt = f_tobj.record.interval.total_seconds()
        add_velocities(ax, frame_tracks_low.iloc[ind], grid, 
                       projparams, dt,
                       var_list=['shift'], c_list=['m'],
                       labels=['System Velocity'])
        lgd_han = create_vel_leg_han(c_list=['m'], labels=['System Velocity'])
        
        # Plot stratiform offset
        lon_high = frame_tracks_high['lon'].iloc[ind]
        lat_high = frame_tracks_high['lat'].iloc[ind]
        ax.plot([lon_low, lon_high], [lat_low, lat_high], 
                '--b', linewidth=2.0,)
        lgd_strat = mlines.Line2D([], [], color='b', linestyle='--', 
                                  linewidth=2.0, label='Stratiform Offset')
        lgd_han.append(lgd_strat)
                                                 
        # Plot ellipses if required
        if ellipses=='conv':
            add_ellipses(ax, frame_tracks_low.iloc[ind], projparams)                    
        if ellipses=='strat':
            add_ellipses(ax, frame_tracks_high.iloc[ind], projparams)                   
                            
        # Plot reflectivity cells
        if updraft_ind is None:                         
            add_updrafts(ax, grid, frame_tracks_low.iloc[ind], 
                          hgt_ind, ud_hgt_ind, projparams, grid_size)
        else: 
            add_updrafts(ax, grid, frame_tracks_low.iloc[ind], 
                         hgt_ind, ud_hgt_ind, projparams, grid_size, 
                         updraft_ind=updraft_ind)
        # Plot WRF winds if necessary              
        if wrf_winds:
            plot_horiz_winds(ax, grid, alt, mp=mp)
            lgd_winds = mlines.Line2D([], [], color='pink', linestyle='-', 
                                      linewidth=1.5,
                                      label='2 m/s Vertical Velocity')
            lgd_han.append(lgd_winds)
                      
        if legend:
            plt.legend(handles=lgd_han)

    # If focusing on one object, restrict axis limits around object                  
    if uid_ind is not None:
        ax.set_xlim(lvxlim[0], lvxlim[1])
        ax.set_ylim(lvylim[0], lvylim[1])
                                                      

def full_domain(tobj, grids, tmp_dir, dpi=100, vmin=-8, vmax=64,
                start_datetime = None, end_datetime = None,
                cmap=pyart.graph.cm_colorblind.HomeyerRainbow, 
                alt_low=None, alt_high=None, isolated_only=False,
                tracers=False, persist=False, projection=ccrs.PlateCarree(), 
                scan_boundary=False, box_rad=.75, line_coords=False, 
                **kwargs):
                              
    # Create a copy of tobj for use by this function
    f_tobj = copy.deepcopy(tobj)

    # Set default arguments if non passed
    if alt_low is None:
        alt_low = f_tobj.params['GS_ALT']
    if alt_high is None:
        alt_high = f_tobj.params['GS_ALT']
    if tracers:
        tracer = Tracer(f_tobj, persist)
     
    # Restrict tracks data to start and end datetime arguments
    time_ind = f_tobj.tracks.index.get_level_values('time')
    if start_datetime != None:
        cond = (time_ind >= start_datetime)
        f_tobj.tracks = f_tobj.tracks[cond]
        time_ind = f_tobj.tracks.index.get_level_values('time')
    else:
        start_datetime = time_ind[0]
        time_ind = f_tobj.tracks.index.get_level_values('time')    
    if end_datetime != None:
        cond = (time_ind <= end_datetime)
        f_tobj.tracks = f_tobj.tracks[cond]
        time_ind = f_tobj.tracks.index.get_level_values('time')
    else:
        end_datetime = time_ind[-1]
        time_ind = f_tobj.tracks.index.get_level_values('time')
                    
    # Initialise fonts
    init_fonts()    

    print('Animating from {} to {}.'.format(str(start_datetime), 
                                            str(end_datetime)))
    
    # Loop through all scans/grids provided to function.
    for counter, grid in enumerate(grids):
        
        grid_time = np.datetime64(grid.time['units'][14:])
        if grid_time > end_datetime:
            del grid
            print('\nReached {}.\n'.format(str(end_datetime)) 
                  + 'Breaking loop.', flush=True)
            break
        elif grid_time < start_datetime:
            print('\nCurrent grid earlier than {}.\n'
                  + 'Moving to next grid.'.format(str(start_datetime)))
            continue
        
        # Initialise figure
        fig_grid = plt.figure(figsize=(22, 9))
        fig_grid.suptitle('MCS at ' + str(grid_time), fontsize=16)
                
        print('Plotting scan at {}.'.format(grid_time), 
              end='\r', flush=True)
        
        # Plot frame
        ax = fig_grid.add_subplot(1, 2, 1, projection=projection)
        plot_tracks_horiz_cross(f_tobj, grid, alt_low, fig=fig_grid, 
                                ax=ax, ellipses='conv', legend=True, 
                                line_coords=line_coords, **kwargs)
        ax = fig_grid.add_subplot(1, 2, 2, projection=projection)
        plot_tracks_horiz_cross(f_tobj, grid, alt_high, ellipses='strat', 
                                legend=True, fig=fig_grid, ax=ax, 
                                line_coords=line_coords, **kwargs)
                          
        # Save frame and cleanup
        plt.savefig(tmp_dir + '/frame_' + str(counter).zfill(3) + '.png',
                    bbox_inches = 'tight', dpi=dpi)
        plt.close()
        del grid, ax
        gc.collect()
                        
                
def plot_obj_line_cross(f_tobj, grid, new_grid, A, uid, nframe, semi_major,
                        new_wrf=None,
                        fig=None, ax=None,
                        vmin=-8, vmax=64, alt_low=None, alt_high=None,
                        cmap=pyart.graph.cm_colorblind.HomeyerRainbow, 
                        projection=ccrs.PlateCarree(), 
                        scan_boundary=False, center_ud=False,
                        updraft_ind=0, direction='cross', color='k',
                        wrf_winds=False, average_along_line=False, 
                        quiver=False, mp='lin', **kwargs):
                        
    field = f_tobj.field
    grid_size = f_tobj.grid_size
    projparams = grid.get_projparams()
    display = pyart.graph.GridMapDisplay(grid)

    if alt_low is None:
        alt_low = f_tobj.params['GS_ALT']
    if alt_high is None:
        alt_high = f_tobj.params['GS_ALT']

    # Calculate mean height of first and last vertical TINT levels
    low = f_tobj.params['LEVELS'][0].mean()/1000
    high = f_tobj.params['LEVELS'][-1].mean()/1000
        
    # Restrict to uid
    cell = f_tobj.tracks.xs(uid, level='uid')
    cell = cell.reset_index(level=['time'])
        
    # Get low and high data
    n_lvl = f_tobj.params['LEVELS'].shape[0]
    cell_low = cell.xs(0, level='level')
    cell_high = cell.xs(n_lvl-1, level='level')
    
    # Restrict to specific time
    cell_low = cell_low.iloc[nframe]
    cell_high = cell_high.iloc[nframe]
    
    # Convert coordinates to those of new grid
    old_x_low = cell_low['grid_x']
    old_y_low = cell_low['grid_y']
    old_low = np.array([old_x_low, old_y_low])
    new_low = np.transpose(A).dot(old_low)
    [new_x_low, new_y_low] = new_low.tolist()
    
    # Convert coordinates to those of new grid
    old_x_high = cell_high['grid_x']
    old_y_high = cell_high['grid_y']
    old_high = np.array([old_x_high, old_y_high])
    new_high = np.transpose(A).dot(old_high)
    [new_x_high, new_y_high] = new_high.tolist()
    
    tx_met = new_x_low
    ty_met = new_y_low
    tx_low = new_x_low/1000
    tx_high = new_x_high/1000
    ty_low = new_y_low/1000
    ty_high = new_y_high/1000
    
    xlim = (tx_met + np.array([-75000, 75000]))/1000
    ylim = (ty_met + np.array([-75000, 75000]))/1000 
    
    # Get center location
    if center_ud:
        ud = np.array(cell_low['updrafts'][updraft_ind])
        x_draft_old = grid.x['data'][ud[:,2]].data
        y_draft_old = grid.y['data'][ud[:,1]].data
        
        drafts_old = np.array([x_draft_old, y_draft_old])
        drafts_new = np.array([np.transpose(A).dot(drafts_old[:,i])
                              for i in range(drafts_old.shape[1])])       

        x_draft = grid.x['data'][ud[0,2]]
        y_draft = grid.y['data'][ud[0,1]]
        ud_old = np.array([x_draft, y_draft])
        
        # Get new coordinates
        ud_new = np.transpose(A).dot(ud_old)
        [x_draft_new, y_draft_new] = ud_new.tolist()
        
        z0 = get_grid_alt(f_tobj.record.grid_size, 
                          f_tobj.params['UPDRAFT_START'])
    else:
        x_draft_new = new_x_low
        y_draft_new = new_y_low        
        
    if direction == 'parallel':
         
        new_grid = new_grid.sel(y=y_draft_new, method='nearest').squeeze()
        
        extent = [new_grid.x[0]/1000, new_grid.x[-1]/1000, 
                  new_grid.z[0]/1000, new_grid.z[-1]/1000] 
        ax.imshow(new_grid.reflectivity, vmin=vmin, vmax=vmax, cmap=cmap, 
                  interpolation='none', origin='lower',
                  extent=extent, aspect='auto')

        [h_low, h_high] = [tx_low, tx_high]
        if center_ud:
            h_draft = drafts_new[:,0]/1000
            z_draft = grid.z['data'][z0:len(h_draft)+z0]/1000

        h_lim = xlim
        t_string = 'Along Line Cross Section'
        x_label = 'Along Line Distance From Origin [km]' 
                                                              
    
    elif direction == 'cross':
        
        if average_along_line:
            cond = ((new_grid.x <= x_draft_new + semi_major*2500/2) 
                    & (new_grid.x >= x_draft_new - semi_major*2500/2))
            new_grid = new_grid.where(cond).dropna(dim='x', how='all')
            new_grid = new_grid.mean(dim='x')
        else:
            new_grid = new_grid.sel(x=x_draft_new, method='nearest').squeeze()
        
        extent = [new_grid.y[0]/1000, new_grid.y[-1]/1000, 
                  new_grid.z[0]/1000, new_grid.z[-1]/1000] 
        ax.imshow(new_grid.reflectivity, vmin=vmin, vmax=vmax, cmap=cmap, 
                  interpolation='none', origin='lower',
                  extent=extent, aspect='auto')
        
        [h_low, h_high] = [ty_low, ty_high]
        if center_ud:
            h_draft = drafts_new[:,1]/1000
            z_draft = grid.z['data'][z0:len(h_draft)+z0]/1000
        
        h_lim = ylim
        if average_along_line:
            t_string = 'Line Perpendicular Mean Cross Section'
        else:
            t_string = 'Line Perpendicular Cross Section'
        x_label = 'Line Perpendicular Distance From Origin [km]'
                                          
    # Plot system tilt
    ax.plot([h_low, h_high], [low, high], '--b', linewidth=2.0)
    
    # Plot updraft tracks
    if center_ud:
        ax.plot(h_draft, z_draft, '-', color=color,
                linewidth=1.0)
                
    # Plot wrf winds if necessary
    if wrf_winds:
        plot_vert_winds_line(ax, new_wrf, x_draft_new, y_draft_new, direction, 
                             quiver=quiver, semi_major=semi_major,
                             average_along_line=average_along_line)

    ax.set_xlim(h_lim[0], h_lim[1])
    ax.set_xticks(np.arange(h_lim[0], h_lim[1], 25))
    ax.set_xticklabels(
        np.round((np.arange(h_lim[0], h_lim[1], 25)), 1)
    )

    ax.set_title(t_string)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Distance Above Origin [km]')
    
    del display                       
                                        
                
def plot_obj_vert_cross(f_tobj, grid, uid, nframe, fig=None, ax=None,
                        vmin=-8, vmax=64, alt_low=None, alt_high=None,
                        cmap=pyart.graph.cm_colorblind.HomeyerRainbow, 
                        projection=ccrs.PlateCarree(), 
                        scan_boundary=False, center_ud=False,
                        updraft_ind=0, direction='lat', color='k',
                        wrf_winds=False, quiver=False, mp='lin', **kwargs):
          
    field = f_tobj.field
    grid_size = f_tobj.grid_size
    projparams = grid.get_projparams()
    display = pyart.graph.GridMapDisplay(grid) 

    if alt_low is None:
        alt_low = f_tobj.params['GS_ALT']
    if alt_high is None:
        alt_high = f_tobj.params['GS_ALT']

    # Calculate mean height of first and last vertical TINT levels
    low = f_tobj.params['LEVELS'][0].mean()/1000
    high = f_tobj.params['LEVELS'][-1].mean()/1000
        
    # Restrict to uid
    cell = f_tobj.tracks.xs(uid, level='uid')
    cell = cell.reset_index(level=['time'])
        
    # Get low and high data
    n_lvl = f_tobj.params['LEVELS'].shape[0]
    cell_low = cell.xs(0, level='level')
    cell_high = cell.xs(n_lvl-1, level='level')
    
    # Restrict to specific time
    cell_low = cell_low.iloc[nframe]
    cell_high = cell_high.iloc[nframe]
    
    # Define box size
    tx_met = cell_low['grid_x']
    ty_met = cell_low['grid_y']
    tx_low = cell_low['grid_x']/1000
    tx_high = cell_high['grid_x']/1000
    ty_low = cell_low['grid_y']/1000
    ty_high = cell_high['grid_y']/1000
    
    xlim = (tx_met + np.array([-75000, 75000]))/1000
    ylim = (ty_met + np.array([-75000, 75000]))/1000
    
    # Get center location
    if center_ud:
        ud = np.array(cell_low['updrafts'][updraft_ind])
        x_draft = grid.x['data'][ud[0,2]]        
        y_draft = grid.y['data'][ud[0,1]]
        lon, lat = cartesian_to_geographic(
            x_draft, y_draft, projparams
        )
        z0 = get_grid_alt(f_tobj.record.grid_size, 
                          f_tobj.params['UPDRAFT_START'])
    else:
        lon = cell_low['lon']
        lat = cell_low['lat']
        x_draft = cell_low['grid_x']
        y_draft = cell_low['grid_y']
        
    if direction == 'lat':
        display.plot_latitude_slice(field, lon=lon, lat=lat,
                                    title_flag=False, colorbar_flag=False, 
                                    edges=False, vmin=vmin, vmax=vmax, 
                                    mask_outside=False, cmap=cmap, ax=ax)
        [h_low, h_high] = [tx_low, tx_high]
        if center_ud:
            h_draft = grid.x['data'][ud[:,2]]/1000
            z_draft = grid.z['data'][z0:len(h_draft)+z0]/1000
        h_lim = xlim
        t_string = 'Latitude Cross Section'
        x_label = 'East West Distance From Origin [km]'                                                               
    
    elif direction == 'lon':
        display.plot_longitude_slice(field, lon=lon, lat=lat,
                                     title_flag=False, colorbar_flag=False, 
                                     edges=False, vmin=vmin, vmax=vmax, 
                                     mask_outside=False, cmap=cmap, ax=ax)
        [h_low, h_high] = [ty_low, ty_high]
        if center_ud:
            h_draft = grid.y['data'][ud[:,1]]/1000
            z_draft = grid.z['data'][z0:len(h_draft)+z0]/1000
        h_lim = ylim
        t_string = 'Longitude Cross Section'
        x_label = 'North South Distance From Origin [km]'
                                          
    # Plot system tilt
    ax.plot([h_low, h_high], [low, high], '--b', linewidth=2.0)
    
    # Plot updraft tracks
    if center_ud:
        ax.plot(h_draft, z_draft, '-', color=color,
                linewidth=1.0)
                
    # Plot wrf winds if necessary
    if wrf_winds:
        plot_wrf_winds(ax, grid, x_draft, y_draft, direction, quiver=quiver, mp=mp)

    ax.set_xlim(h_lim[0], h_lim[1])
    ax.set_xticks(np.arange(h_lim[0], h_lim[1], 25))
    ax.set_xticklabels(
        np.round((np.arange(h_lim[0], h_lim[1], 25)), 1)
    )

    ax.set_title(t_string)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Distance Above Origin [km]')
    
    del display
                
        
def updraft_view(tobj, grids, tmp_dir, uid=None, dpi=100, 
                 vmin=-8, vmax=64, start_datetime=None, end_datetime=None,
                 cmap=None, alt_low=None, alt_high=None, 
                 box_rad=.75, projection=None, center_ud=False, 
                 updraft_ind=None, wrf_winds=False, line_coords = False,
                 average_along_line=False, quiver=False, mp='lin',
                 **kwargs):

    if uid is None:
        print("Please specify 'uid' keyword argument.")
        return
        
    f_tobj = copy.deepcopy(tobj)
    if cmap is None:
        cmap = pyart.graph.cm_colorblind.HomeyerRainbow
    if alt_low is None:
        alt_low = tobj.params['GS_ALT']
    if alt_high is None:
        alt_high = tobj.params['GS_ALT']
    if projection is None:
        projection = ccrs.PlateCarree()
   
    colors = ['m', 'lime', 'darkorange', 'k', 'b', 'darkgreen', 'yellow']

    cell = f_tobj.tracks.xs(uid, level='uid').xs(0, level='level')
    cell = cell.reset_index(level=['time'])
    nframes = len(cell)
    print('Animating', nframes, 'frames')
    nframe = 0
    pframe = 0
        
    # Loop through each grid in grids
    for grid in grids:
        # Ensure object exists at current grid
        grid_time = np.datetime64(grid.time['units'][14:])        
        if nframe >= nframes:
            info_msg = ('Object died at ' 
                        + str(cell.iloc[nframe-1].time)
                        + '.\n' + 'Ending loop.')
            print(info_msg)
            del grid
            gc.collect()
            break        
        elif cell.iloc[nframe].time > grid_time:
            info_msg = ('Object not yet initiated at '
                        + '{}.\n'.format(grid_time) 
                        + 'Moving to next grid.') 
            print(info_msg)
            continue
        while cell.iloc[nframe].time < grid_time:
            
            info_msg = ('Current grid at {}.\n'
                        + 'Object initialises at {}.\n'
                        + 'Moving to next object frame.') 
            print(info_msg.format(grid_time, str(cell.iloc[nframe].time)))
            nframe += 1
             
        print('Plotting frame at {}'.format(grid_time),
              end='\n', flush=True)

        # Initialise fonts
        init_fonts()

        # Don't plot axis
        plt.axis('off')
        
        if line_coords:
            new_grid, A, angle, semi_major = get_line_grid(f_tobj, grid, uid, nframe)
            if wrf_winds:
                new_wrf = get_line_grid_wrf(grid_time, angle, mp=mp)
            else:
                new_wrf = None
        
        # Determine whether to plot updrafts
        if center_ud:
            if updraft_ind is None:
                # Plot all updrafts
                cell_frame = cell.iloc[nframe]
                ud_list = range(len(cell_frame['updrafts']))
            else:
                ud_list = [updraft_ind]
        else:
            ud_list = [updraft_ind]
        
        for j in ud_list:
            fig = plt.figure(figsize=(12, 10))
            if center_ud:
                print('Plotting updraft {}.  '.format(str(j)),
                      end='\r', flush=True)        
            # Generate title
            if center_ud:
                fig.suptitle('Object ' + uid + ' at ' 
                             + str(grid_time) + ': Updraft ' 
                             + str(j), fontsize=16, y=1.0)
            else:
                fig.suptitle('Object ' + uid + ' at ' 
                             + str(grid_time), fontsize=16, y=1.0)
                         
            # Vertical cross section at alt_low
            ax = fig.add_subplot(2, 2, 1, projection=projection)
            plot_tracks_horiz_cross(f_tobj, grid, alt_low, fig=fig, 
                                    ax=ax, ellipses='conv', legend=False, 
                                    uid_ind=uid, center_ud=center_ud, updraft_ind=j,
                                    angle=angle, line_coords=line_coords, 
                                    wrf_winds=wrf_winds, mp=mp,
                                    **kwargs)
                                    
            # Vertical cross section at alt_high
            ax = fig.add_subplot(2, 2, 3, projection=projection)
            plot_tracks_horiz_cross(f_tobj, grid, alt_high, fig=fig, 
                                    ax=ax, ellipses='strat', legend=False, 
                                    uid_ind=uid, center_ud=center_ud, updraft_ind=j,
                                    angle=angle, line_coords=line_coords, 
                                    wrf_winds=wrf_winds, mp=mp,
                                    **kwargs)     
            
            if center_ud:
                color=colors[np.mod(j,len(colors))]
            else:
                color=None
                                                     
            # Plot latitude (or cross line) cross section 
            ax = fig.add_subplot(2, 2, 2)
            if line_coords:
                plot_obj_line_cross(f_tobj, grid, new_grid, A, uid, nframe, semi_major,
                    fig=fig, ax=ax, alt_low=alt_low, alt_high=alt_high,
                    center_ud=center_ud, updraft_ind=j, direction='cross', 
                    color=color, wrf_winds=wrf_winds, new_wrf=new_wrf, 
                    average_along_line=average_along_line, quiver=quiver,
                    **kwargs)
            else:
                plot_obj_vert_cross(f_tobj, grid, uid, nframe, fig=fig, ax=ax,
                                    alt_low=alt_low, alt_high=alt_high,
                                    center_ud=center_ud, updraft_ind=j, 
                                    direction='lat', color=color, 
                                    wrf_winds=wrf_winds, **kwargs)
                                
            # Plot longitude (or line parallel) cross section
            ax = fig.add_subplot(2, 2, 4)
            if line_coords:
                plot_obj_line_cross(f_tobj, grid, new_grid, A, uid, nframe, semi_major,
                    fig=fig, ax=ax, alt_low=alt_low, alt_high=alt_high,
                    center_ud=center_ud, updraft_ind=j, direction='parallel', 
                    color=color, wrf_winds=wrf_winds, new_wrf=new_wrf, 
                    average_along_line=average_along_line, quiver=quiver, 
                    **kwargs)
            else:
                plot_obj_vert_cross(f_tobj, grid, uid, nframe, fig=fig, ax=ax,
                                    alt_low=alt_low, alt_high=alt_high,
                                    updraft_ind=j, direction='lon', 
                                    center_ud=center_ud, color=color, 
                                    wrf_winds=wrf_winds, **kwargs)                   
                              
            plt.tight_layout()

            # plot and save figure
            fig.savefig(tmp_dir + '/frame_' + str(pframe).zfill(3) + '.png', 
                        dpi=dpi)
            plt.close()
            pframe += 1
            gc.collect()
            
        nframe += 1
        del grid, ax, fig
        plt.close()
        gc.collect()


def make_mp4_from_frames(tmp_dir, dest_dir, basename, fps):
    os.chdir(tmp_dir)
    os.system(" ffmpeg -framerate " + str(fps)
              + " -pattern_type glob -i '*.png'"
              + " -movflags faststart -pix_fmt yuv420p -vf"
              + " 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -y "
              + basename + '.mp4')
    try:
        shutil.move(basename + '.mp4', dest_dir)
    except FileNotFoundError:
        print('Make sure ffmpeg is installed properly.')

def make_gif_from_frames(tmp_dir, dest_dir, basename, fps):
    print('\nCreating GIF - may take a few minutes.')
    os.chdir(tmp_dir)
    delay = round(100/fps)
    
    command = "convert -delay {} frame_*.png -loop 0 {}.gif"
    os.system(command.format(str(delay), basename))
    try:
        shutil.move(basename + '.gif', dest_dir)
    except FileNotFoundError:
        print('Make sure Image Magick is installed properly.')


def animate(tobj, grids, outfile_name, style='full', fps=2, 
            start_datetime = None, end_datetime = None, 
            keep_frames=False, dpi=100, **kwargs):
    """
    Creates gif animation of tracked cells.

    Parameters
    ----------
    tobj : Cell_tracks
        The Cell_tracks object to be visualized.
    grids : iterable
        An iterable containing all of the grids used to generate tobj
    outfile_name : str
        The name of the output file to be produced.
    alt : float
        The altitude to be plotted in meters.
    vmin, vmax : float
        Limit values for the colormap.
    arrows : bool
        If True, draws arrow showing corrected shift for each object. Only used
        in 'full' style.
    isolation : bool
        If True, only annotates uids for isolated objects. Only used in 'full'
        style.
    uid : str
        The uid of the object to be viewed from a lagrangian persepective. Only
        used when style is 'lagrangian'.
    fps : int
        Frames per second for output gif.

    """

    styles = {'full': full_domain,
              'updraft': updraft_view}
    anim_func = styles[style]

    dest_dir = os.path.dirname(outfile_name)
    basename = os.path.basename(outfile_name)
    if len(dest_dir) == 0:
        dest_dir = os.getcwd()

    if os.path.exists(basename + '.mp4'):
        print('Filename already exists.')
        return

    tmp_dir = tempfile.mkdtemp()

    try:
        anim_func(tobj, grids, tmp_dir, dpi=dpi,
                  start_datetime=start_datetime, 
                  end_datetime=end_datetime,
                  **kwargs)
        if len(os.listdir(tmp_dir)) == 0:
            print('Grid generator is empty.')
            return
        make_gif_from_frames(tmp_dir, dest_dir, basename, fps)
        if keep_frames:
            frame_dir = os.path.join(dest_dir, basename + '_frames')
            shutil.copytree(tmp_dir, frame_dir)
            os.chdir(dest_dir)
    finally:
        shutil.rmtree(tmp_dir)

def embed_mp4_as_gif(filename):
    """ Makes a temporary gif version of an mp4 using ffmpeg for embedding in
    IPython. Intended for use in Jupyter notebooks. """
    if not os.path.exists(filename):
        print('file does not exist.')
        return

    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    newfile = tempfile.NamedTemporaryFile()
    newname = newfile.name + '.gif'
    if len(dirname) != 0:
        os.chdir(dirname)

    os.system('ffmpeg -i ' + basename + ' ' + newname)

    try:
        with open(newname, 'rb') as f:
            display(Image(f.read(), format='png'))
    finally:
        os.remove(newname)
