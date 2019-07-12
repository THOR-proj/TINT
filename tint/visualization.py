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
import cartopy.crs as ccrs
import copy

import pyart

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
            dead_cells = [key for key in self.cell_color.keys()
                          if key
                          not in self.current.index.get_level_values('uid')]
            self.color_stack.extend(self.cell_color[dead_cells])
            self.cell_color.drop(dead_cells, inplace=True)

    def _check_uid(self, uid):
        if uid not in self.cell_color.keys():
            try:
                self.cell_color[uid] = self.color_stack.pop()
            except IndexError:
                self.color_stack += self.colors * 5
                self.cell_color[uid] = self.color_stack.pop()

    def plot(self, ax):
        for uid, group in self.history.groupby(level='uid'):
            self._check_uid(uid)
            tracer = group[['lon', 'lat']]
            if (self.persist or 
                (uid in self.current.reset_index(level=['time']).index)):
                ax.plot(tracer.lon, tracer.lat, self.cell_color[uid])

def full_domain(tobj, grids, tmp_dir, dpi=100, vmin=-8, vmax=64,
                start_datetime = None, end_datetime = None,
                cmap=None, alt=None, isolated_only=False,
                tracers=False, persist=False,
                projection=None, **kwargs):

    # Create a copy of tobj for use by this function
    f_tobj = copy.deepcopy(tobj)

    # Consider just the track data at the tracking interval.
    f_tobj.tracks = f_tobj.tracks.xs(f_tobj.params['TRACK_INTERVAL'], 
                                 level='level')

    grid_size = tobj.grid_size
    if cmap is None:
        cmap = pyart.graph.cm_colorblind.HomeyerRainbow
    if alt is None:
        alt = f_tobj.params['GS_ALT']
    if projection is None:
        projection=ccrs.PlateCarree()
    if tracers:
        tracer = Tracer(f_tobj, persist)

    radar_lon = f_tobj.radar_info['radar_lon']
    radar_lat = f_tobj.radar_info['radar_lat']
    lon = np.arange(round(radar_lon-5,2),round(radar_lon+5,2), 1)
    lat = np.arange(round(radar_lat-5,2),round(radar_lat+5,2), 1)

    time_ind = f_tobj.tracks.index.get_level_values('time')    
    
    # Restrict tracks data to start and end datetime arguments
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
    
    # Create index of scan numbers so scans can be easily
    # retrieved for a given time.
    scan_ind = f_tobj.tracks.index.get_level_values('scan')     

    print('Animating from {} to {}.'.format(str(start_datetime), 
                                            str(end_datetime)))

    for counter, grid in enumerate(grids):
        
        grid_time = np.datetime64(grid.metadata['start_time'])

        if grid_time > end_datetime:
            del grid
            print('\nReached {}.\n'.format(str(end_datetime)) 
                  + 'Breaking loop.', flush=True)
            break
        elif grid_time < start_datetime:
            print('\nCurrent grid earlier than {}.\n'
                  + 'Moving to next grid.'.format(str(start_datetime)))
            continue
                
        fig_grid = plt.figure(figsize=(10, 8))
        print('Plotting scan at {}.'.format(grid_time), 
              end='\r', flush=True)
        
        display = pyart.graph.GridMapDisplay(grid)
        ax = fig_grid.add_subplot(111, projection=projection)
        transform = projection._as_mpl_transform(ax)
        display.plot_crosshairs(lon=radar_lon, lat=radar_lat)
        display.plot_grid(tobj.field, level=get_grid_alt(grid_size, alt),
                          vmin=vmin, vmax=vmax, mask_outside=False,
                          cmap=cmap, transform=projection, ax=ax, **kwargs)

        if grid_time in time_ind:

            nframe = scan_ind[time_ind == grid_time][0]
            frame_tracks = f_tobj.tracks.loc[nframe]
            frame_tracks = frame_tracks.reset_index(level=['time'])
            
            if tracers:
                tracer.update(nframe)
                tracer.plot(ax)

            for ind, uid in enumerate(frame_tracks.index):
                if isolated_only and not frame_tracks['isolated'].iloc[ind]:
                    continue
                x = frame_tracks['lon'].iloc[ind]
                y = frame_tracks['lat'].iloc[ind]
                ax.text(x, y, uid, transform=projection, fontsize=20)

        plt.savefig(tmp_dir + '/frame_' + str(counter).zfill(3) + '.png',
                    bbox_inches = 'tight', dpi=dpi)
        plt.close()
        del grid, display, ax
        gc.collect()


def lagrangian_view(tobj, grids, tmp_dir, uid=None, dpi=100, vmin=-8, vmax=64,
                    start_datetime=None, end_datetime=None,
                    cmap=None, alt=None, box_rad=.2, projection=None):

    if uid is None:
        print("Please specify 'uid' keyword argument.")
        return
    stepsize = 0.05
    title_font = 10
    axes_font = 8
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8

    field = tobj.field
    grid_size = tobj.grid_size

    if cmap is None:
        cmap = pyart.graph.cm_colorblind.HomeyerRainbow
    if alt is None:
        alt = tobj.params['GS_ALT']
    if projection is None:
        projection = ccrs.PlateCarree()
        
    cell = tobj.tracks.xs(uid, level='uid')
    cell = cell.xs(tobj.params['TRACK_INTERVAL'] ,level='level')
    cell = cell.reset_index(level=['time'])

    nframes = len(cell)
    print('Animating', nframes, 'frames')
    nframe = 0
        
    for grid in grids:
        grid_time = grid.metadata['start_time']        
        if nframe >= nframes:
            info_msg = ('Object died at ' 
                        + str(cell.iloc[nframe-1].time)
                        + '.\n' + 'Ending loop.')
            print(info_msg)
            del grid
            gc.collect()
            break        
        elif cell.iloc[nframe].time > np.datetime64(
            grid.metadata['start_time']
        ):
            info_msg = ('Object not yet initiated at '
                        + '{}.\n'.format(grid_time) 
                        + 'Moving to next grid.') 
            print(info_msg)
            continue
        while cell.iloc[nframe].time < np.datetime64(grid_time):
            
            info_msg = ('Current grid at {}.\n'
                        + 'Object initialises at {}.\n'
                        + 'Moving to next object frame.') 
            print(info_msg.format(grid_data, str(cell.iloc[nframe].time)))
            nframe += 1
             
        print('Plotting frame at {}'.format(grid_time),
              end='\r', flush=True)

        row = cell.iloc[nframe]
        display = pyart.graph.GridMapDisplay(grid)

        # Box Size
        tx = np.int(np.round(row['grid_x']))
        ty = np.int(np.round(row['grid_y']))
        tx_met = grid.x['data'][tx]
        ty_met = grid.y['data'][ty]
        lat = row['lat']
        lon = row['lon']
        box_rad_met = box_rad 
        box = np.array([-1*box_rad_met, box_rad_met])
        

        lvxlim = (lon) + box
        lvylim = (lat) + box
        xlim = (tx_met + np.array([-25000, 25000]))/1000
        ylim = (ty_met + np.array([-25000, 25000]))/1000

        fig = plt.figure(figsize=(14, 10.5))

        fig.suptitle('Cell ' + uid + ' Scan ' + str(nframe), fontsize=22)
        plt.axis('off')

        # Lagrangian View
        ax = fig.add_subplot(3, 2, (1, 3), projection=projection)

        display.plot_grid(field, level=get_grid_alt(grid_size, alt),
                          vmin=vmin, vmax=vmax, mask_outside=False,
                          cmap=cmap, colorbar_flag=False,
                          ax=ax, projection=projection)

        display.plot_crosshairs(lon=lon, lat=lat, linestyle='--', 
                                color='k', linewidth=3, ax=ax)

        ax.set_xlim(lvxlim[0], lvxlim[1])
        ax.set_ylim(lvylim[0], lvylim[1])

        ax.set_xticks(np.arange(lvxlim[0], lvxlim[1], stepsize))
        ax.set_yticks(np.arange(lvylim[0], lvylim[1], stepsize))

        ax.set_title('Top-Down View', fontsize=title_font)
        ax.set_xlabel('Longitude of grid cell center\n [degree_E]',
                       fontsize=axes_font)
        ax.set_ylabel('Latitude of grid cell center\n [degree_N]',
                       fontsize=axes_font)

        # Latitude Cross Section
        ax = fig.add_subplot(3, 2, 2)
        display.plot_latitude_slice(field, lon=lon, lat=lat,
                                    title_flag=False,
                                    colorbar_flag=False, edges=False,
                                    vmin=vmin, vmax=vmax, mask_outside=False,
                                    cmap=cmap,
                                    ax=ax)

        ax.set_xlim(xlim[0], xlim[1])
        ax.set_xticks(np.arange(xlim[0], xlim[1], 6))
        ax.set_xticklabels(np.round((np.arange(xlim[0], xlim[1], 6)),
                                     2))

        ax.set_title('Latitude Cross Section', fontsize=title_font)
        ax.set_xlabel('East West Distance From Origin (km)' + '\n',
                       fontsize=axes_font)
        ax.set_ylabel('Distance Above Origin (km)', fontsize=axes_font)
        ax.set_aspect(aspect=1.3)

        # Longitude Cross Section
        ax = fig.add_subplot(3, 2, 4)
        display.plot_longitude_slice(field, lon=lon, lat=lat,
                                     title_flag=False,
                                     colorbar_flag=False, edges=False,
                                     vmin=vmin, vmax=vmax, mask_outside=False,
                                     cmap=cmap,
                                     ax=ax)
        ax.set_xlim(ylim[0], ylim[1])
        ax.set_xticks(np.arange(ylim[0], ylim[1], 6))
        ax.set_xticklabels(np.round(np.arange(ylim[0], ylim[1], 6), 2))

        ax.set_title('Longitudinal Cross Section', fontsize=title_font)
        ax.set_xlabel('North South Distance From Origin (km)',
                       fontsize=axes_font)
        ax.set_ylabel('Distance Above Origin (km)', fontsize=axes_font)
        ax.set_aspect(aspect=1.3)

        # Time Series Statistic
        max_field = cell['max']
        plttime = cell['time']

        # Plot
        ax = fig.add_subplot(3, 2, (5, 6))
        ax.plot(plttime, max_field, color='b', linewidth=3)
        ax.axvline(x=plttime.iloc[nframe], linewidth=4, color='r')
        ax.set_title('Time Series', fontsize=title_font)
        ax.set_xlabel('Time (UTC) \n Lagrangian Viewer Time',
                       fontsize=axes_font)
        ax.set_ylabel('Maximum ' + field, fontsize=axes_font)

        # plot and save figure
        fig.savefig(tmp_dir + '/frame_' + str(nframe).zfill(3) + '.png', 
                    dpi=dpi)
        plt.close()
        nframe += 1
        del grid, display
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
    print('\nCreating GIF - may take a few minutes.', flush=True)
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
              'lagrangian': lagrangian_view}
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
