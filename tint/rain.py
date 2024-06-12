import numpy as np
import xarray as xr


def save_rain(acc_rain_list, acc_rain_uid_list, grid_obj):

    acc_rain = np.stack(acc_rain_list, axis=0)
    acc_rain_uid = np.array(acc_rain_uid_list)
    if len(acc_rain_uid_list) > 1:
        acc_rain_uid = np.squeeze(acc_rain_uid)

    x = grid_obj.x['data'].data
    y = grid_obj.y['data'].data
    acc_rain_da = xr.DataArray(
        acc_rain, coords=[acc_rain_uid, y, x], dims=['uid', 'y', 'x'])
    acc_rain_da.attrs = {
        'long_name': 'Accumulated Rainfall',
        'units': 'mm',
        'standard_name': 'Accumulated Rainfall',
        'description': (
            'Derived from rainfall rate algorithm based on '
            + 'Thompson et al. 2016, integrated in time.')}
    acc_rain_da.to_netcdf(
        '/g/data/w40/esh563/CPOL_analysis/accumulated_rainfalls/'
        + 'acc_rain.nc')


def update_rain_totals(data_dic, old_objects, uid, interval, grid_obj, params):
    nobj = np.max(data_dic['frame'])
    id1 = np.arange(nobj) + 1
    max_rr = []
    tot_rain = []

    for obj in np.arange(nobj) + 1:
        cond = (np.any(data_dic['frames'] == id1[obj-1], axis=0))
        obj_max_rr = np.zeros(data_dic['rain'].shape, dtype=np.float32)
        obj_max_rr[cond] = data_dic['rain'][cond]
        max_rr.append(obj_max_rr)
        if obj in old_objects['id2']:
            obj_index = (old_objects['id2'] == obj)
            ind = np.argwhere(obj_index)[0, 0]
            obj_tot_rain = old_objects['tot_rain'][ind]
            obj_tot_rain[cond] += data_dic['rain'][cond]/3600*interval
            tot_rain.append(obj_tot_rain)
        else:
            obj_tot_rain = np.zeros(data_dic['rain'].shape)
            obj_tot_rain[cond] = data_dic['rain'][cond]/3600*interval
            tot_rain.append(obj_tot_rain)

    # call save rain here
    if params['SAVE_RAIN']:
        acc_rain_list = []
        acc_rain_uid_list = []
        if data_dic['refl_new'] is None:
            dead_uids = old_objects['uid']
        else:
            dead_uids = set(old_objects['uid']) - set(uid)
        for u in dead_uids:
            u_ind = int(np.argwhere(old_objects['uid'] == u))
            acc_rain_list.append(old_objects['tot_rain'][u_ind])
            acc_rain_uid_list.append(u)
        if acc_rain_list:
            save_rain(acc_rain_list, acc_rain_uid_list, grid_obj)

    return max_rr, tot_rain


def init_rain_totals(data_dic, uid, id1, interval):
    max_rr = [
        np.zeros(data_dic['rain'].shape, dtype=np.float32)
        for i in range(len(uid))]
    tot_rain = [
        np.zeros(data_dic['rain'].shape, dtype=np.float32)
        for i in range(len(uid))]

    for i in range(len(uid)):
        cond = (np.any(data_dic['frames'] == id1[i], axis=0))
        max_rr[i][cond] = data_dic['rain'][cond]
        tot_rain[i][cond] = data_dic['rain'][cond]/3600*interval
    return max_rr, tot_rain


def get_object_rain_props(data_dic, current_objects):
    nobj = np.max(data_dic['frames'])
    [levels, rows, columns] = data_dic['frames'].shape
    [tot_rain, max_rr, tot_rain_loc, max_rr_loc] = [[] for i in range(4)]

    for i in range(levels):
        for obj in np.arange(nobj) + 1:
            obj_tot_rain = current_objects['tot_rain'][obj-1]
            obj_max_rr = current_objects['max_rr'][obj-1]

            tot_rain_loc_obj = list(
                np.unravel_index(
                    np.argmax(obj_tot_rain), obj_tot_rain.shape))
            tot_rain_loc.append(tot_rain_loc_obj)

            max_rr_loc_obj = list(
                np.unravel_index(
                    np.argmax(obj_max_rr), obj_max_rr.shape))
            max_rr_loc.append(max_rr_loc_obj)

            max_rr.append(np.max(obj_max_rr))
            tot_rain.append(np.max(obj_tot_rain))

    return tot_rain, max_rr, tot_rain_loc, max_rr_loc


def update_sys_tracks_rain(tracks_obj, system_tracks):
    for prop in ['tot_rain', 'tot_rain_loc', 'max_rr', 'max_rr_loc']:
        prop_lvl_0 = tracks_obj.tracks[[prop]].xs(0, level='level')
        system_tracks = system_tracks.merge(
            prop_lvl_0, left_index=True, right_index=True)
    return system_tracks
