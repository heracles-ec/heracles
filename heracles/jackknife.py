'''module for jackknife estimation'''

import numpy as np
import healpy as hp

from .io import read_mask
from .kmeans_radec import kmeans_sample, find_nearest


def get_radec(indx, nside):
    '''return an array of RA and Dec for pixel indices'''
    return np.transpose(hp.pix2ang(nside, indx, lonlat=True))


def iterate_kmeans(params, mask_radec, maxiterac=350):
    '''runs the k-means code iteratively'''

    _tol_ini = 1.0e-2
    _ncut_tol = 1

    # number of centers starts at maximum
    ncen = params.nClustersMax

    print("\n\tIterating over number of regions and tolerance separation:")
    conv = False
    while conv is False:
        if ncen < params.nClustersMin:
            break

        # runs the kmeans with the masks RA and DEC as training
        km = kmeans_sample(mask_radec, ncen, maxiter=maxiterac, tol=_tol_ini, verbose=0)

        # gets the area for each cluster region
        clusters_area = hp.nside2pixarea(params.nside, degrees=True) * np.bincount(km.labels)

        # sums the areas that are smaller than the given area tolerance:
        area2cut = np.sum(clusters_area[clusters_area < params.areaTol])

        # number of reagions to be cut
        # ncut_index = np.where(clusters_area < params.areaTol)[0]
        ncut = len(clusters_area[clusters_area < params.areaTol])

        if (area2cut <= params.areaTol) & (ncut <= _ncut_tol):
            print("\n\t\t>> Finished! <<")
            print(f"\n\t\tTolerance used: {_tol_ini}\n\t\tNcenters = {ncen}")
            print(f"\t\tNumber of regions larger than lmin/2: {ncut} \n\t\tSmallest area region: {area2cut}")
            print("------------------------------------------------------------------")
            conv = True
        else:
            print(f"\n\t\tNumber of regions larger than lmin/2: {ncut} \n\t\tSmallest area region: {area2cut}")
            _tol_ini = _tol_ini/5
            ncen = ncen - 1

    return km, ncen, ncut


def find_mask_regions(params, mask, km_centers):
    '''label all pixels in the mask'''

    indx = np.where(mask != 0)[0]

    mask_radec = get_radec(indx, params.nside)

    km_mask_labels = find_nearest(mask_radec, km_centers)

    mask_labels = np.int64(-1 * np.ones(hp.nside2npix(params.nside), dtype=np.int64))

    mask_labels[indx] = km_mask_labels.astype(np.int64)

    return mask_labels


def kmeans_split(params):
    '''split the input mask into jackknife regions'''

    assert params.nside >= params.nsideKmeans, "Nside for the kmeans mask is larger than given mask!"

    assert params.nClustersMax >= params.nClustersMin, "Min number of clusters larger than the Max!"

    mask = read_mask(params.mask_name, params.nside)

    con = False

    indx = np.where(mask != 0)[0]

    maskradec = get_radec(indx, params.nside)

    max_iter = 0
    while con is False and (max_iter <= 350):

        km, ncen, ncut = iterate_kmeans(params, maskradec)

        if (ncen - ncut) >= params.nClustersMin:
            print(f"\nConverged with {ncen - ncut} jack-knife regions!")
            con = True
        else:
            print(f" [!!!] Found {ncen}, will re-start the K-Means iterations [!!!]")
            max_iter = max_iter + 1
    print("------------------------------------------------------------------")

    # get the label for each pixel of the mask
    mask_labels = find_mask_regions(params, mask, km.centers)

    return ncen, mask_labels
