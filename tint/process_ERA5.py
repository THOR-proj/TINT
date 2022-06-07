from pyart.core.transforms import cartesian_to_geographic
import xarray as xr
import numpy as np
import pandas as pd
import calendar
from tint.grid_utils import parse_grid_datetime


def get_datetime_components(date_time):
    dt_index = pd.DatetimeIndex([date_time])
    components = []
    for block in ['year', 'month', 'day', 'hour', 'minute']:
        components.append(
            eval('dt_index.{}.values[0]'.format(block)))
    return components


def get_ERA5_daterange(year, month):
    last_day = calendar.monthrange(year, month)[1]
    date_range = '{}{}{}-{}{}{}'.format(
        year, str(month).zfill(2), '01',
        year, str(month).zfill(2), last_day)
    return date_range


def collect_ERA5_files(base_dir, year, range_str, files=[]):
    for variable in ['z', 'u', 'v']:
        fn = '{}/{}/{}/{}_era5_oper_pl_{}.nc'.format(
            base_dir, variable, year, variable, range_str)
        files.append(fn)
    return files


def get_ERA5_ds(
        start_datetime, end_datetime,
        time_delta=np.timedelta64(10, 'm'),
        base_dir='/g/data/rt52/era5/pressure-levels/reanalysis/',
        base_timestep=1):
    s_comps = get_datetime_components(start_datetime)
    [s_year, s_month, s_day, s_hour, s_minute] = s_comps
    e_comps = get_datetime_components(end_datetime)
    [e_year, e_month, e_day, e_hour, e_minute] = e_comps

    if not (e_day == 1 and e_hour == 0 and e_minute == 0):
        end_datetime = end_datetime + time_delta
        e_comps = get_datetime_components(end_datetime)
        [e_year, e_month, e_day, e_hour, e_minute] = e_comps

    start_month = np.datetime64('{}-{}'.format(
        str(s_year).zfill(2), str(s_month).zfill(2)))
    end_month = np.datetime64('{}-{}'.format(
        str(e_year).zfill(2), str(e_month).zfill(2)))
    month_range = np.arange(
        start_month, end_month+np.timedelta64(1, 'M'),
        np.timedelta64(1, 'M'))

    files = []
    for month_dt in month_range:
        year, month = get_datetime_components(month_dt)[:2]
        range_str = get_ERA5_daterange(year, month)
        files = collect_ERA5_files(
            base_dir, year, range_str, files=files)

    era5_all = xr.open_mfdataset(
        files, chunks={'longitude': 180})
    if base_timestep != 1:
        start_time = era5_all.time.values[0]
        end_time = era5_all.time.values[-1]
        timestep = np.timedelta64(base_timestep, 'h')
        times = np.arange(start_time, end_time, timestep)
        era5_all = era5_all.sel(time=times)
        print('Restricting ERA5 data to {} hour timestep'.format(
            base_timestep))

    return era5_all


def flexible_round(x, prec=2, base=.05, method=round):
    return round(base * method(float(x) / base), prec)


def interp_ERA_ds(ds_all, grid, params, timedelta=np.timedelta64(10, 'm')):
    lon, lat = cartesian_to_geographic(
        grid.x['data'].data, grid.y['data'].data, grid.get_projparams())
    alt = grid.z['data'].data

    min_lon = flexible_round(min(lon), prec=1, base=.2, method=np.floor)
    max_lon = flexible_round(max(lon), prec=1, base=.2, method=np.ceil)
    min_lat = flexible_round(min(lat), prec=2, base=.25, method=np.floor)
    max_lat = flexible_round(max(lat), prec=2, base=.25, method=np.ceil)

    grid_time = parse_grid_datetime(grid)

    if params['AMBIENT_TIMESTEP'] == 1:
        start_time = np.datetime64(grid_time.replace(minute=0, second=0))
        end_time = start_time + np.timedelta64(1, 'h')
    else:
        components = get_datetime_components(grid_time)
        new_hour = components[3] // params['AMBIENT_TIMESTEP']
        new_hour = new_hour * params['AMBIENT_TIMESTEP']
        dt_string = '{:04d}-{:02d}-{:02d}T{:02d}:00:00'.format(
            components[0], components[1], components[2], new_hour)
        start_time = np.datetime64(dt_string)
        end_time = start_time + np.timedelta64(params['AMBIENT_TIMESTEP'], 'h')

    ds = ds_all.loc[dict(
        latitude=slice(max_lat+.25, min_lat-.25),
        longitude=slice(min_lon-.2, max_lon+.2),
        time=slice(start_time, end_time))]

    filename = params['SAVE_DIR'] + '/tmp_{:04d}{:02d}.nc'.format(
        components[0], components[1])
    ds.to_netcdf(filename)
    ds = xr.open_dataset(filename)

    ds['z'] = ds['z'] / 9.80665
    altitude = ds['z'].mean(['longitude', 'latitude', 'time'])
    ds = ds.assign_coords({'level': altitude})
    ds = ds.rename({'level': 'altitude'})
    ds = ds.drop_vars('z')
    ds = ds.loc[dict(altitude=slice(22000, 0))]
    times = np.arange(start_time, end_time, timedelta)
    ds = ds.interp(longitude=lon, latitude=lat, altitude=alt, time=times)

    # if params['AMBIENT_TIMESTEP'] != 1:
    #     start_time = np.datetime64(grid_time.replace(minute=0, second=0))
    #     end_time = start_time + np.timedelta64(1, 'h')
    #     ds = ds.loc[dict(time=slice(start_time, end_time))]

    return ds


def init_ERA5(grid, params):
    grid_datetime = parse_grid_datetime(grid)
    grid_datetime = np.datetime64(grid_datetime.replace(second=0))
    print('Getting ERA5 metadata.')
    ERA5_all = get_ERA5_ds(
        grid_datetime,
        grid_datetime+np.timedelta64(params['AMBIENT_TIMESTEP'], 'h'),
        base_dir=params['AMBIENT_BASE_DIR'],
        base_timestep=params['AMBIENT_TIMESTEP'])
    print('Getting interpolated ERA5 for next {} hours.'.format(
        params['AMBIENT_TIMESTEP']))
    ERA5_interp = interp_ERA_ds(
        ERA5_all, grid, params, timedelta=np.timedelta64(10, 'm'))
    ERA5_interp.load()
    return ERA5_all, ERA5_interp


def update_ERA5(grid, params, ERA5_all, ERA5_interp):
    grid_datetime = parse_grid_datetime(grid)
    grid_datetime = np.datetime64(grid_datetime.replace(second=0))
    grid_datetime_hour = parse_grid_datetime(grid)
    grid_datetime_hour = np.datetime64(
        grid_datetime_hour.replace(second=0, minute=0))
    t_min = ERA5_all.time.values.min()
    t_max = ERA5_all.time.values.max()
    if not (
            grid_datetime >= t_min
            and grid_datetime_hour + np.timedelta64(1, 'h') <= t_max):
        print('Getting ERA5 metadata.')
        ERA5_all = get_ERA5_ds(
            grid_datetime_hour,
            grid_datetime_hour+np.timedelta64(params['AMBIENT_TIMESTEP'], 'h'),
            base_dir=params['AMBIENT_BASE_DIR'],
            base_timestep=params['AMBIENT_TIMESTEP'])
    if grid_datetime not in ERA5_interp.time.values:
        print('Getting interpolated ERA5 for next {} hours.'.format(
            params['AMBIENT_TIMESTEP']))
        ERA5_interp = interp_ERA_ds(
            ERA5_all, grid, params, timedelta=np.timedelta64(10, 'm'))
        ERA5_interp.load()
    return ERA5_all, ERA5_interp
