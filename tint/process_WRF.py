from pyart.core.transforms import cartesian_to_geographic
import numpy as np
import glob
import xarray as xr
from tint.grid_utils import parse_grid_datetime
from tint.process_ERA5 import flexible_round, get_datetime_components


def get_WRF_ds(
        start_datetime, end_datetime,
        time_delta=np.timedelta64(10, 'm'),
        base_dir='/g/data/w40/esh563/lind02'):
    s_comps = get_datetime_components(start_datetime)
    [s_year, s_month, s_day, s_hour, s_minute] = s_comps
    e_comps = get_datetime_components(end_datetime)
    [e_year, e_month, e_day, e_hour, e_minute] = e_comps

    start_day = np.datetime64('{}-{}-{}'.format(
        str(s_year).zfill(4), str(s_month).zfill(2), str(s_day).zfill(2)))
    end_day = np.datetime64('{}-{}-{}'.format(
        str(e_year).zfill(4), str(e_month).zfill(2), str(e_day).zfill(2)))
    end_day = end_day + np.timedelta64(1, 'D')

    files = sorted(glob.glob(base_dir + '*.nc'))
    files_subset = []
    for f in files:
        f_datetime = 'T'.join(f.split('.')[-2].split('_')[-2:])
        f_datetime = np.datetime64(f_datetime)
        if f_datetime >= start_day and f_datetime <= end_day:
            files_subset.append(f)

    return xr.open_mfdataset(files_subset)


def interp_WRF_ds(ds_all, grid, timedelta=np.timedelta64(10, 'm')):
    lon, lat = cartesian_to_geographic(
        grid.x['data'].data, grid.y['data'].data, grid.get_projparams())
    alt = grid.z['data'].data

    min_lon = flexible_round(min(lon), prec=1, base=.2, method=np.floor)
    max_lon = flexible_round(max(lon), prec=1, base=.2, method=np.ceil)
    min_lat = flexible_round(min(lat), prec=2, base=.25, method=np.floor)
    max_lat = flexible_round(max(lat), prec=2, base=.25, method=np.ceil)

    grid_time = parse_grid_datetime(grid)
    rounded_hour = int(flexible_round(grid_time.hour, base=2, method=np.floor))

    start_time = np.datetime64(grid_time.replace(
        hour=rounded_hour, minute=0, second=0))
    end_time = start_time + np.timedelta64(2, 'h')
    t_max = ds_all.time.values.max()
    end_time = np.min([end_time, t_max])

    ds = ds_all.loc[dict(
        latitude=slice(min_lat-.25, max_lat-.25),
        longitude=slice(min_lon-.2, max_lon+.2),
        time=slice(start_time, end_time))]
    ds = ds.rename({'bottom_top': 'altitude'})
    ds['phi'] = ds['phi'] / 9.80665
    altitude = ds['phi'].mean(['longitude', 'latitude', 'time'])
    ds = ds.assign_coords({'altitude': altitude})
    ds = ds.drop_vars('phi')
    ds = ds.loc[dict(altitude=slice(0, 22000))]
    times = np.arange(start_time, end_time, timedelta)
    if len(times) == 0:
        ds = ds.interp(longitude=lon, latitude=lat, altitude=alt)
    else:
        ds = ds.interp(longitude=lon, latitude=lat, altitude=alt, time=times)

    return ds


def init_WRF(grid, params):
    grid_datetime = parse_grid_datetime(grid)
    grid_datetime = np.datetime64(grid_datetime.replace(second=0))
    print('Getting WRF metadata.')
    WRF_all = get_WRF_ds(
        grid_datetime, grid_datetime,
        base_dir=params['AMBIENT_BASE_DIR'])
    print('Getting Intepolated WRF for next hour.')
    WRF_interp = interp_WRF_ds(
        WRF_all, grid, timedelta=np.timedelta64(10, 'm'))
    WRF_interp.load()
    return WRF_all, WRF_interp


def update_WRF(grid, params, WRF_all, WRF_interp):
    grid_datetime = parse_grid_datetime(grid)
    grid_datetime = np.datetime64(grid_datetime.replace(second=0))
    t_min = WRF_all.time.values.min()
    t_max = WRF_all.time.values.max()
    if not (grid_datetime >= t_min and grid_datetime <= t_max):
        print('Getting WRF metadata.')
        WRF_all = get_WRF_ds(
            grid_datetime, grid_datetime, base_dir=params['AMBIENT_BASE_DIR'])
    if grid_datetime not in WRF_interp.time.values:
        print('Getting Intepolated WRF for the next two hours.')
        WRF_interp = interp_WRF_ds(
            WRF_all, grid, timedelta=np.timedelta64(10, 'm'))
        WRF_interp.load()
    return WRF_all, WRF_interp
