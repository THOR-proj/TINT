from skimage.feature import peak_local_max
import numpy as np
import copy

from .grid_utils import get_grid_alt
from .steiner import steiner_conv_strat


def identify_cells(raw3D, images, grid1, record, params, stein_class):
    """Identify convective cells by looking for local maxima at each
    vertical level. """
    [dz, dx, dy] = record.grid_size
    z0 = get_grid_alt(record.grid_size, params['LEVELS'][0, 0])
    if params['FIELD_THRESH'][0] == 'convective':
        sclass = stein_class[0]
    else:
        sclass = steiner_conv_strat(
            raw3D[z0], grid1.x['data'], grid1.y['data'], dx, dy)

    local_max = []
    for k in range(z0, grid1.nz):
        l_max = peak_local_max(
            raw3D[k], threshold_abs=params['CELL_THRESH'])
        l_max = np.insert(
            l_max, 0, np.ones(len(l_max), dtype=int)*k, axis=1)
        local_max.append(l_max)
    # Find pixels classified as convective by steiner
    conv_ind = np.argwhere(sclass == 2)
    conv_ind_set = set(
        [tuple(conv_ind[i]) for i in range(len(conv_ind))])
    cells = [
        [local_max[0][j]] for j in range(len(local_max[0]))
        if tuple(local_max[0][j][1:]) in conv_ind_set]
    # Find first level with no local_max
    try:
        max_height = [
            i for i in range(len(local_max))
            if local_max[i].tolist() == []][0]
    except:
        max_height = len(local_max)

    current_inds = set(range(len(cells)))
    for k in range(1, max_height):
        previous = np.array([cells[i][k-1] for i in current_inds])
        current = local_max[k]
        if ((len(previous) == 0) or (len(current) == 0)):
            break
        mc1, mp1 = np.meshgrid(current[:, 1], previous[:, 1])
        mc2, mp2 = np.meshgrid(current[:, 2], previous[:, 2])
        match = np.sqrt((mp1 - mc1) ** 2 + (mp2 - mc2) ** 2)
        next_inds = copy.copy(current_inds)
        minimums = np.argmin(match, axis=1)
        for m in range(match.shape[0]):
            minimum = minimums[m]
            if (match[m, minimum] < 2):
                cells[list(current_inds)[m]].append(current[minimum])
            else:
                next_inds = next_inds - set([list(current_inds)[m]])
        current_inds = next_inds
    cells = [
        cells[i] for i in range(len(cells)) if (len(cells[i]) > 1)]

    return cells
