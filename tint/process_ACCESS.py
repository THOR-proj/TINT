from pyart.core.transforms import cartesian_to_geographic
import xarray as xr
import numpy as np
import pandas as pd
import pyart
import copy


def flexible_round(x, prec=2, base=.05, method=round):
    return round(base * method(float(x) / base), prec)


def get_reference_grid(path, format='ODIM'):

    print('Creating a reference grid.')

    if format == 'ODIM':
        my_radar = pyart.aux_io.read_odim_h5(
            path, file_field_names=False)

        grid = pyart.map.grid_from_radars(
            my_radar, grid_shape=(41, 121, 121),
            grid_limits=(
                (0., 20000,), (-150000., 150000.), (-150000, 150000.)))

        x = grid.x['data']
        y = grid.y['data']
        X, Y = np.meshgrid(x, y)
        mask_cond = np.sqrt(X**2 + Y**2) > 152500

        grid.fields['reflectivity']['data'].data[
            grid.fields['reflectivity']['data'].data < 0] = np.nan
        grid.fields['reflectivity']['data'].mask[:, mask_cond] = True

        grid.fields = {'reflectivity': grid.fields['reflectivity']}
    else:
        grid = pyart.io.read_grid(path)
        grid.fields = {'reflectivity': grid.fields['corrected_reflectivity']}

    return grid


def get_datetime_components(date_time):
    dt_index = pd.DatetimeIndex([date_time])
    components = []
    for block in ['year', 'month', 'day', 'hour', 'minute']:
        components.append(
            eval('dt_index.{}.values[0]'.format(block)))
    return components


def format_ACCESS_C(datetime, reference_grid, gadi=False):

    if gadi:
        base_dir = '/g/data/wr45/'
    else:
        base_dir = 'https://dapds00.nci.org.au/thredds/dodsC/wr45/'
    base_dir += 'ops_aps3/access-dn/1/'

    missing_data = False

    components = get_datetime_components(datetime-np.timedelta64(1, 'D'))

    year = components[0]
    month = components[1]
    day = components[2]

    date = '{:04d}{:02d}{:02d}'.format(year, month, day)

    try:
        maxcol_refl = xr.open_dataset(
            base_dir + date + '/1200/fcmm/sfc/maxcol_refl.nc')['maxcol_refl']
        radar_refl_1km = xr.open_dataset(
            base_dir + date + '/1200/fcmm/sfc/radar_refl_1km.nc')[
                'radar_refl_1km']
    except OSError:
        print('No reflectivity data at {}'.format(datetime))
        ref_date = '{:04d}{:02d}{:02d}'.format(2021, 12, 10)
        maxcol_refl = xr.open_dataset(
            base_dir + ref_date + '/1200/fcmm/sfc/maxcol_refl.nc')[
                'maxcol_refl']
        radar_refl_1km = xr.open_dataset(
            base_dir + ref_date + '/1200/fcmm/sfc/radar_refl_1km.nc')[
                'radar_refl_1km']
        # import pdb; pdb.set_trace()
        fill_start = np.datetime64(
            '{:04d}-{:02d}-{:02d}T12:00'.format(year, month, day))
        fill_times = np.arange(
            fill_start,
            fill_start + np.timedelta64(36, 'h') + np.timedelta64(10, 'm'),
            np.timedelta64(10, 'm'))
        maxcol_refl['time'] = fill_times
        radar_refl_1km['time'] = fill_times
        missing_data = True

    # Remember we are taking the +12 forecast at 12:00 UTC on the previous day
    start = np.datetime64(str(datetime)[:10])
    end = start + np.timedelta64(1, 'D')

    maxcol_refl = maxcol_refl.sel(time=slice(start, end))
    radar_refl_1km = radar_refl_1km.sel(time=slice(start, end))

    if len(maxcol_refl.time.values) < 24*6:
        print('Missing times from ACCESS-C reflectivity.')

    x = reference_grid.x['data'].data
    y = reference_grid.y['data'].data

    lon, lat = cartesian_to_geographic(x, y, reference_grid.get_projparams())

    min_lon = flexible_round(min(lon), prec=1, base=.2, method=np.floor)
    max_lon = flexible_round(max(lon), prec=1, base=.2, method=np.ceil)
    min_lat = flexible_round(min(lat), prec=2, base=.25, method=np.floor)
    max_lat = flexible_round(max(lat), prec=2, base=.25, method=np.ceil)

    maxcol_refl = maxcol_refl.loc[dict(
        lat=slice(max_lat+.25, min_lat-.25),
        lon=slice(min_lon-.2, max_lon+.2))]
    radar_refl_1km = radar_refl_1km.loc[dict(
        lat=slice(max_lat+.25, min_lat-.25),
        lon=slice(min_lon-.2, max_lon+.2))]

    if missing_data:
        maxcol_refl[:] = np.nan
        radar_refl_1km[:] = np.nan

    ACCESS_refl = xr.concat([radar_refl_1km, maxcol_refl], dim='z')
    ACCESS_refl = ACCESS_refl.assign_coords(z=('z', [0, 1]))
    ACCESS_refl = ACCESS_refl.interp(lon=lon, lat=lat)
    ACCESS_refl = ACCESS_refl.assign_coords(lon=x, lat=y)
    ACCESS_refl = ACCESS_refl.rename({'lat': 'y', 'lon': 'x'})

    return ACCESS_refl


def ACCESS_to_pyart(ACCESS_refl, datetime, reference_grid):
    try:
        data = ACCESS_refl.sel(time=datetime)
    except KeyError:
        print('{} observation missing from ACCESS-C reflectivity.'.format(
            datetime))
        data = ACCESS_refl.isel(time=0)
        data['time'] = datetime
        data[:] = np.nan

    pseudo_grid = copy.deepcopy(reference_grid)
    pseudo_grid.z['data'] = np.array([0, 1])
    pseudo_grid.nz = 2

    ACCESS_time = pseudo_grid.time['units'][:14]
    ACCESS_time += datetime.astype(str).split('.')[0] + 'Z'

    pseudo_grid.time['units'] = ACCESS_time
    pseudo_grid.time['data'] = np.array([0.0], dtype=np.float32)

    mask = np.zeros(ACCESS_refl.isel(time=0).shape).astype(bool)
    data = data.values
    mask[data < 0] = True
    x = reference_grid.x['data']
    y = reference_grid.y['data']
    X, Y = np.meshgrid(x, y)
    mask_cond = np.sqrt(X**2 + Y**2) > 152500
    mask[:, mask_cond] = True

    data[data < 0] = np.nan
    data[:, mask_cond] = np.nan

    masked_data = np.ma.masked_array(data, mask=mask)

    pseudo_grid.fields['reflectivity']['data'] = masked_data
    return pseudo_grid


def init_ACCESS_C(datetime, reference_grid, gadi=False):

    ACCESS_refl = format_ACCESS_C(datetime, reference_grid, gadi=False)
    pseudo_grid = ACCESS_to_pyart(ACCESS_refl, datetime, reference_grid)

    return ACCESS_refl, pseudo_grid


def update_ACCESS_C(datetime, ACCESS_refl, reference_grid, gadi=False):

    if (ACCESS_refl is None) or (datetime not in ACCESS_refl.time):
        ACCESS_refl = format_ACCESS_C(datetime, reference_grid, gadi=False)
        print('Interpolating ACCESS-C reflectivity at {}.'.format(datetime))

    pseudo_grid = ACCESS_to_pyart(ACCESS_refl, datetime, reference_grid)
    print('Updating ACCESS-C pseudo pyart reflectivity grid.')

    return ACCESS_refl, pseudo_grid


def format_ACCESS_G(datetime, reference_grid, gadi=False):

    # datetime = np.datetime64('2021-01-15T09:50')

    if gadi:
        base_dir = '/g/data/wr45/'
    else:
        base_dir = 'https://dapds00.nci.org.au/thredds/dodsC/wr45/'
    base_dir += 'ops_aps3/access-g/1/'

    components = get_datetime_components(datetime)
    year = components[0]
    month = components[1]
    day = components[2]
    date = '{:04d}{:02d}{:02d}'.format(year, month, day)
    hour = '/{:02d}00'.format(components[3])

    topog = xr.open_dataset(base_dir + date + hour + '/an/sfc/topog.nc')
    wnd_ucmp = xr.open_dataset(base_dir + date + hour + '/an/ml/wnd_ucmp.nc')
    wnd_vcmp = xr.open_dataset(base_dir + date + hour + '/an/ml/wnd_vcmp.nc')

    x = reference_grid.x['data'].data
    y = reference_grid.y['data'].data

    lon, lat = cartesian_to_geographic(x, y, reference_grid.get_projparams())

    min_lon = flexible_round(min(lon), prec=1, base=.2, method=np.floor)
    max_lon = flexible_round(max(lon), prec=1, base=.2, method=np.ceil)
    min_lat = flexible_round(min(lat), prec=2, base=.25, method=np.floor)
    max_lat = flexible_round(max(lat), prec=2, base=.25, method=np.ceil)

    wnd_ucmp = wnd_ucmp.loc[dict(
        lat=slice(max_lat+.25, min_lat-.25),
        lon=slice(min_lon-.2, max_lon+.2))]
    wnd_vcmp = wnd_vcmp.loc[dict(
        lat=slice(max_lat+.25, min_lat-.25),
        lon=slice(min_lon-.2, max_lon+.2))]
    topog = topog.loc[dict(
        lat=slice(max_lat+.25, min_lat-.25),
        lon=slice(min_lon-.2, max_lon+.2))]

    z = wnd_ucmp.A_rho + wnd_ucmp.B_rho * topog['topog']

    dims = z.squeeze().shape
    new_z = np.arange(0, 20000+500, 500)
    new_u = np.zeros([len(new_z), dims[1], dims[2]])
    new_v = np.zeros([len(new_z), dims[1], dims[2]])

    u = wnd_ucmp['wnd_ucmp'].squeeze().values
    v = wnd_vcmp['wnd_vcmp'].squeeze().values
    for i in range(z.shape[2]):
        for j in range(z.shape[3]):
            new_u[:, i, j] = np.interp(
                new_z, z.squeeze()[:, i, j], u[:, i, j],
                left=np.nan, right=np.nan)
            new_v[:, i, j] = np.interp(
                new_z, z.squeeze()[:, i, j], v[:, i, j],
                left=np.nan, right=np.nan)

    latitude = wnd_vcmp['wnd_vcmp'].lat.values
    longitude = wnd_vcmp['wnd_vcmp'].lon.values
    u = xr.DataArray(
        new_u, coords={
            'altitude': new_z, 'latitude': latitude,
            'longitude': longitude},
        dims=['altitude', 'latitude', 'longitude'])
    v = xr.DataArray(
        new_v, coords={
            'altitude': new_z, 'latitude': latitude, 'longitude': longitude},
        dims=['altitude', 'latitude', 'longitude'])
    winds = xr.Dataset({'u': u, 'v': v})
    winds['time'] = datetime

    return winds


def interp_ACCESS_G(datetime, reference_grid, gadi=False):

    components = get_datetime_components(datetime)
    year = components[0]
    month = components[1]
    day = components[2]
    hour = components[3]

    start_hour = int(np.floor(hour/6)*6)
    start_datetime = '{:04d}-{:02d}-{:02d}T{:02d}:00'
    start_datetime = start_datetime.format(year, month, day, start_hour)
    start_datetime = np.datetime64(start_datetime)
    end_datetime = start_datetime + np.timedelta64(6, 'h')

    winds_1 = format_ACCESS_G(start_datetime, reference_grid, gadi)
    winds_2 = format_ACCESS_G(end_datetime, reference_grid, gadi)

    winds = xr.concat([winds_1, winds_2], dim='time')

    x = reference_grid.x['data'].data
    y = reference_grid.y['data'].data
    lon, lat = cartesian_to_geographic(x, y, reference_grid.get_projparams())

    times = winds.time
    times = np.arange(
        times[0].values, times[1].values, np.timedelta64(10, 'm'))
    winds = winds.interp(longitude=lon, latitude=lat, time=times)

    return winds


def init_ACCESS_G(datetime, reference_grid, gadi=False):

    winds = interp_ACCESS_G(datetime, reference_grid, gadi)

    print('Getting ACCESS-G ambient winds at {}.'.format(datetime))

    return winds


def update_ACCESS_G(winds, reference_grid, datetime, gadi=False):

    if datetime not in winds.time:
        print('Updating ACCESS-G ambient winds at {}.'.format(datetime))
        winds = init_ACCESS_G(datetime, reference_grid, gadi)

    return winds
