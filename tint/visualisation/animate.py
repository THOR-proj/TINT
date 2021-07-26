import os
import tempfile
import shutil
from IPython.display import display, Image

from .figures import full_view, object_view


def animate(
        tracks, grids, outfile_name, style='full', fps=2, start_datetime=None,
        end_datetime=None, keep_frames=False, dpi=100, **kwargs):
    """Creates gif animation of tracked objects. """

    styles = {
        'full': full_view, 'object': object_view}
    anim_func = styles[style]

    dest_dir = os.path.dirname(outfile_name)
    basename = os.path.basename(outfile_name)
    if len(dest_dir) == 0:
        dest_dir = os.getcwd()
    tmp_dir = tempfile.mkdtemp()

    try:
        anim_func(
            tracks, grids, tmp_dir, dpi=dpi, start_datetime=start_datetime,
            end_datetime=end_datetime, **kwargs)
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


def make_mp4_from_frames(tmp_dir, dest_dir, basename, fps):
    os.chdir(tmp_dir)
    os.system(
        " ffmpeg -framerate " + str(fps)
        + " -pattern_type glob -i '*.png'"
        + " -movflags faststart -pix_fmt yuv420p -vf"
        + " 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -y "
        + basename + '.mp4')
    try:
        shutil.move(basename + '.mp4', dest_dir)
    except FileNotFoundError:
        print('Make sure ffmpeg is installed properly.')


def make_gif_from_frames(tmp_dir, dest_dir, basename, fps):

    print('Creating GIF - may take a few minutes.')
    os.chdir(tmp_dir)
    delay = round(100/fps)
    command = "convert -delay {} frame_*.png -loop 0 {}.gif"
    os.system(command.format(str(delay), basename))
    try:
        shutil.move(basename + '.gif', dest_dir)
    except FileNotFoundError:
        print('Make sure Image Magick is installed properly.')
