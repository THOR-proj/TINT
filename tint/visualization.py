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
                               
                
def add_ellipses(ax, frame_tracks_ind, projparams):

    centroid = frame_tracks_ind[['grid_x', 'grid_y']].values
    orientation = frame_tracks_ind[['orientation']].values[0]
    major_axis = frame_tracks_ind[['semi_major']].values[0]
    minor_axis = frame_tracks_ind[['semi_minor']].values[0]
                                
    # Convert axes into lat/lon units by approximating 1 degree lat or
    # lon = 110 km.
    major_axis = major_axis*2.5/110
    minor_axis = minor_axis*2.5/110
    
    lon_e, lat_e = cartesian_to_geographic(
        centroid[0], centroid[1], projparams
    )

    ell = Ellipse(tuple([lon_e, lat_e]), major_axis, minor_axis, orientation,
                  linewidth=1.5, fill=False)

    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.4)
    ax.add_artist(ell)
    return ell
    
    
def add_updrafts(ax, grid, frame_tracks_ind, hgt_ind, ud_ind, projparams, grid_size,
                 updraft_ind=None):
    
    colors = ['m', 'r', 'lime', 'darkorange', 'k', 'b', 'darkgreen', 'yellow']
    if updraft_ind is None:
        updraft_list = range(len(frame_tracks_ind['updrafts']))
    else:
        updraft_list = [updraft_ind]
    for j in updraft_list:
        # Plot location of updraft j at alts[i] if it exists          
        if len(np.array(frame_tracks_ind['updrafts'][j])) > hgt_ind:
            x_ind = np.array(frame_tracks_ind['updrafts'][j])[ud_ind,2]
            y_ind = np.array(frame_tracks_ind['updrafts'][j])[ud_ind,1]
            x_draft = grid.x['data'][x_ind]         
            y_draft = grid.y['data'][y_ind]
            lon_ud, lat_ud = cartesian_to_geographic(
                x_draft, y_draft, projparams
            )
            ax.scatter(lon_ud, lat_ud, marker='x', s=25, 
                       c=colors[np.mod(j,len(colors))], zorder=3)
                       
    return ax
                       
                       
def add_boundary(ax, tobj, grid, projparams):
    b_list = list(tobj.params['BOUNDARY_GRID_CELLS'])
    boundary = np.zeros((117,117))*np.nan
    for i in range(len(b_list)):
        boundary[b_list[i][0], b_list[i][1]]=1
    x_bounds = grid.x['data'][[0,-1]]
    y_bounds = grid.y['data'][[0,-1]]      
    lon_b, lat_b = cartesian_to_geographic(x_bounds, y_bounds, projparams)
    ax.imshow(boundary, extent=(lon_b[0], lon_b[1], lat_b[0], lat_b[1]))
    
    return ax
    
    
def add_velocities(ax, frame_tracks_ind, grid, projparams, dt, 
                    var_list=['shift', 'prop', 'shear', 'cl'], 
                    c_list=['m', 'green', 'red', 'orange'],
                    labels=['System Velocity', 'Propagation',
                            '0-3 km Shear', 'Mean Cloud-Layer Winds']):
    lon = frame_tracks_ind['lon']
    lat = frame_tracks_ind['lat']    
    x = frame_tracks_ind['grid_x']
    y = frame_tracks_ind['grid_y']
    
    for i in range(len(var_list)):
        u = frame_tracks_ind['u_' + var_list[i]]
        v = frame_tracks_ind['v_' + var_list[i]]     
        [new_lon, new_lat] = cartesian_to_geographic(x + 4*u*dt, y + 4*v*dt, 
                                                     projparams)
        ax.arrow(lon, lat, new_lon[0]-lon, new_lat[0]-lat, 
                 color=c_list[i], head_width=0.024, head_length=0.040)      

    
def create_vel_leg_han(c_list=['m', 'green', 'red', 'orange'],
                      labels=['System Velocity', 'Propagation',
                      'Low-Level Shear', 'Mean Cloud-Layer Winds']):
    lgd_han = []
    for i in range(len(c_list)):
        lgd_line = mlines.Line2D([], [], color=c_list[i], linestyle='-', 
                                 label=labels[i])
        lgd_han.append(lgd_line)
    return lgd_han
    
    
def plot_wrf_winds(ax, grid, x_draft, y_draft, direction, 
                   projparams, quiver=False):
    grid_time = np.datetime64(grid.time['units'][14:]).astype('datetime64[s]')
    # Load WRF winds corresponding to this grid
    base = '/g/data/w40/esh563/lind04_ref/lind04_ref_'
    fn = glob.glob(base + str(grid_time) + '.nc')
    winds = xr.open_dataset(fn[0])
    if direction=='lat':
        winds = winds.sel(y=y_draft, method='nearest')
    else:
        winds = winds.sel(x=x_draft, method='nearest')
    winds = winds.squeeze()
    U = winds.U
    V = winds.V
    W = winds.W
    
    x = np.arange(-145000, 145000+2500, 2500)/1000
    z = np.arange(0, 20500, 500)/1000
    
    if quiver:
        if direction=='lat':
            ax.quiver(x[::2], z[::2], U.values[::2,::2], W.values[::2,::2])
        else:
            ax.quiver(x[::2], z[::2], V.values[::2,::2], W.values[::2,::2])
    else:
        ax.contour(x, z, W, colors='pink', linewidths=1.5, 
                   levels=[-2, 2], linestyles=['--', '-'])
        
                      
def plot_horiz_winds(ax, grid, alt, quiver=False):
    grid_time = np.datetime64(grid.time['units'][14:]).astype('datetime64[s]')
    # Load WRF winds corresponding to this grid
    base = '/g/data/w40/esh563/lind04_ref/lind04_ref_'
    fn = glob.glob(base + str(grid_time) + '.nc')
    winds = xr.open_dataset(fn[0])
    winds = winds.sel(z=alt, method='nearest')
    winds = winds.squeeze()
    U = winds.U
    V = winds.V
    
    if quiver:
        ax.quiver(U.lon[::4], U.lat[::4], 
                  U.values[::4,::4], V.values[::4,::4])
    else:
        ax.contour(winds.longitude, winds.latitude, winds.W.values, 
                   colors='pink', linewidths=1.5, 
                   levels=[-2, 2], linestyles=['--', '-'])
    
    
def init_fonts():
    # Initialise fonts
    rcParams.update({'font.family' : 'serif'})
    rcParams.update({'font.serif': 'Liberation Serif'})
    rcParams.update({'mathtext.fontset' : 'dejavuserif'}) 
    rcParams.update({'font.size': 12})        
        

def plot_tracks_horiz_cross(f_tobj, grid, alt, vmin=-8, vmax=64,
                            cmap=pyart.graph.cm_colorblind.HomeyerRainbow, 
                            fig=None, ax=None, 
                            projection=ccrs.PlateCarree(), 
                            scan_boundary=False,
                            tracers=False, ellipses='conv', legend=True,
                            uid_ind=None, center_ud=False, updraft_ind=None, 
                            box_rad=.75, wrf_winds=False, **kwargs):       
                                                      
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

    display.plot_crosshairs(lon=lon_ch, lat=lat_ch)
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
                transform=projection, fontsize=10)
        
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
            plot_horiz_winds(ax, grid, alt)
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
                scan_boundary=False, box_rad=.75, **kwargs):
                              
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
                                ax=ax, ellipses='conv', legend=True, **kwargs)
        ax = fig_grid.add_subplot(1, 2, 2, projection=projection)
        plot_tracks_horiz_cross(f_tobj, grid, alt_high, ellipses='strat', 
                                legend=True, fig=fig_grid, ax=ax, **kwargs)
                          
        # Save frame and cleanup
        plt.savefig(tmp_dir + '/frame_' + str(counter).zfill(3) + '.png',
                    bbox_inches = 'tight', dpi=dpi)
        plt.close()
        del grid, ax
        gc.collect()
        
        
def get_line_cross_wrf():    
    # Get reflectivity data
    base = '/g/data/w40/esh563/lind04_ref/lind04_ref_'
    test = xr.open_dataset(base + str(grid_time) + '.nc')
    test['reflectivity'].isel(time=0).isel(z=1).squeeze().transpose()
    # Get appropriate transformation angle
    angle = f_tobj.system_tracks.xs(np.datetime64('2006-02-09T10:00'), level='time').orientation.values[0]
    # Get transform formula (should be basic rotation matrix right?)             
        
                
def get_line_grid(f_tobj, grid, uid, nframe):
    # Get raw grid data
    raw = grid.fields['reflectivity']['data'].data
    # Get appropriate tracks data
    cell = f_tobj.tracks.xs(uid, level='uid')
    cell = cell.reset_index(level=['time'])
    cell_low = cell.xs(0, level='level')
    cell_low = cell_low.iloc[nframe]
    # Get appropriate transformation angle
    angle = cell_low.orientation.values[0]
    # Get rotation matrix
    A = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))], 
                  [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]])              

    x = grid.x['data'].data
    y = grid.y['data'].data
    z = grid.z['data'].data

    xp = np.arange(-210000,210000+2500,2500)
    yp = np.arange(-210000,210000+2500,2500)

    Xp, Yp = np.mgrid[-210000:210000+2500:2500, -210000:210000+2500:2500]

    new_grid = np.ones((len(z), len(yp), len(xp))) * np.nan

    points_old = []
    for j in range(len(y)):
        for i in range(len(x)):
            points_old.append(np.array([y[j], x[i]]))

    points_new = []
    for i in range(len(points_old)):
        points_new.append(np.transpose(A.dot(np.transpose(points_old[i]))))

    for k in range(len(z)):
        values = []
        for j in range(len(y)):
            for i in range(len(x)):
                values.append(raw[k,j,i])

        new_grid[k,:,:] = griddata(points_new, values, (Xp, Yp))
            
    return new_grid, A
                
                
def plot_obj_line_cross(f_tobj, grid, uid, nframe, fig=None, ax=None,
                        vmin=-8, vmax=64, alt_low=None, alt_high=None,
                        cmap=pyart.graph.cm_colorblind.HomeyerRainbow, 
                        projection=ccrs.PlateCarree(), 
                        scan_boundary=False, center_ud=False,
                        updraft_ind=0, direction='lat', color='k',
                        wrf_winds=False, **kwargs):
                        
                        
    # Create new grid (seperate function) x should be across line, y along line
    # Get formula for converting old coordinates to new coordinates. Easier in Cartesian.

    # Take new grid and plot either across line or along line vertical cross sections
                        
                                        
                
def plot_obj_vert_cross(f_tobj, grid, uid, nframe, fig=None, ax=None,
                        vmin=-8, vmax=64, alt_low=None, alt_high=None,
                        cmap=pyart.graph.cm_colorblind.HomeyerRainbow, 
                        projection=ccrs.PlateCarree(), 
                        scan_boundary=False, center_ud=False,
                        updraft_ind=0, direction='lat', color='k',
                        wrf_winds=False, **kwargs):
          
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
        plot_wrf_winds(ax, grid, x_draft, y_draft, direction, projparams)

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
                 vmin=-8, vmax=64,
                 start_datetime=None, end_datetime=None,
                 cmap=None, alt_low=None, alt_high=None, 
                 box_rad=.75, projection=None, center_ud=False, 
                 updraft_ind=None, wrf_winds=False, **kwargs):

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
   
    colors = ['m', 'r', 'lime', 'darkorange', 'k', 'b', 'darkgreen', 'yellow']

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
                                    **kwargs)
                                    
            # Vertical cross section at alt_high
            ax = fig.add_subplot(2, 2, 3, projection=projection)
            plot_tracks_horiz_cross(f_tobj, grid, alt_high, fig=fig, 
                                    ax=ax, ellipses='strat', legend=False, 
                                    uid_ind=uid, center_ud=center_ud, updraft_ind=j,
                                    **kwargs)     
            
            if center_ud:
                color=colors[np.mod(j,len(colors))]
            else:
                color=None
                                                     
            # Latitude Cross Section
            ax = fig.add_subplot(2, 2, 2)
            plot_obj_vert_cross(f_tobj, grid, uid, nframe, fig=fig, ax=ax,
                                alt_low=alt_low, alt_high=alt_high,
                                center_ud=center_ud, updraft_ind=j, 
                                direction='lat', color=color, 
                                wrf_winds=wrf_winds, **kwargs,)
                                
            ax = fig.add_subplot(2, 2, 4)
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
