#
# adapted from https://github.com/esheldon/kmeans_radec/
#
# kmeans_radec - kmeans on the sphere
#
# Copyright (C) 2021 Erin S. Sheldon
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
"""
k means on the sphere

Adapted from this stack overflow answer

http://stackoverflow.com/questions/5529625/is-it-possible-to-specify-your-own-distance-function-using-scikit-learn-k-means

"""
import random
import numpy as np
from numpy import deg2rad, rad2deg, pi, sin, cos, arccos, arctan2, newaxis, sqrt


_TOL_DEF = 1.0e-5
_MAXITER_DEF = 100
_VERBOSE_DEF = 1


class KMeans:
    """
    A class to perform K-means on the input ra,dec using spherical distances

    parameters
    ----------
    centers_guess: array
        [ncen, 2] starting guesses, where the two dimensions are ra and dec
        respectively.  e.g. ra centers are centers_guess[:,0] and the dec
        centers are centers_guess[:,1].
        Can reset later with set_centers()
    tol: float, optional
        The relative change in the average distance to
        centers, signifies convergence
    verbose: int, optional
        How verbose.  0 silent, 1 minimal starting info, 2 prints running
        distances
    method: string, optional
        method = 'fast' will use numba to accelerate the code.  Default
        is method = 'slow'

    attributes after running
    ------------------------
    .converged: bool
        True if converged
    .centers: array
        the found centers
    .labels: array
        [N] array of labels
    .distances: array
        Distance from each point to each center
    .X: array
        The data that was processed

    example
    -------
    import kmeans_radec
    from kmeans_radec import KMeans

    cen_guess = numpy.zeros( (ncen, 2) )
    cen_guess[:,0] = ra_guesses
    cen_guess[:,1] = dec_guesses
    km = KMeans(cen_guess)
    km.run(X, maxiter = 100)

    # did it converge?
    if not km.converged:
        # did not converge.  This might be ok, but if we want
        # to run more we can
        km.run(X, maxiter = maxiter)

        # or we could try a different set of center guesses...
        km.set_centers(cen_guess2)
        km.run(X, maxiter = 100)

    # results are saved in attributes
    print(km.centers, km.labels, km.distances)
    print("copy of centers:",km.get_centers())

    # once we have our centers, we can identify to which cluster
    # a *different* set of points belong.  This could be a set
    # of random points we want to associate with the same regions

    labels = km.find_nearest(X2)

    # you can save the centers and load them into a KMeans
    # object at a later time
    km = KMeans(centers)
    labels = km.find_nearest(X)
    """
    def __init__(self, centers,
                 tol=_TOL_DEF,
                 verbose=_VERBOSE_DEF):

        self.set_centers(centers)

        self.tol = float(tol)
        self.verbose = verbose
        self.converged = False

    def run(self, x, maxiter=_MAXITER_DEF):
        """
        run k means, either until convergence is reached or the indicated
        number of iterations are performed

        parameters
        ----------
        X: array
            [N, 2] array for ra,dec points
        maxiter: int, optional
            Max number of iterations to run.
        """

        centers = self.get_centers()
        _check_dims(x, self.centers)

        n, _ = x.shape
        ncen, _ = centers.shape

        if self.verbose:
            tup = (x.shape, centers.shape, self.tol, maxiter)
            print("X %s  centers %s  tol = %.2g  maxiter = %d" % tup)

        xx, xy, xz = radec2xyz(x[:, 0], x[:, 1])

        self.converged = False
        allx = np.arange(n)
        prevdist = 0
        for jiter in range(1, maxiter+1):

            # npoints x ncenters
            d = cdist_radec(x, centers)

            # X -> nearest centre
            labels = d.argmin(axis=1)

            distances = d[allx, labels]
            # median ?
            avdist = distances.mean()
            if self.verbose >= 2:
                print("    av |X - nearest centre| = %.4g" % avdist)

            self.converged = (1 - self.tol) * prevdist <= avdist <= prevdist
            if self.converged:
                break

            if jiter == maxiter:
                break

            prevdist = avdist
            # (1 pass in C)
            for jc in range(ncen):
                c, = np.where(labels == jc)
                if len(c) > 0:
                    centers[jc] = get_mean_center(xx[c], xy[c], xz[c])

        if self.verbose:
            print(jiter, "iterations  cluster "
                  "sizes:", np.bincount(labels))

        self.centers = centers
        self.labels = labels
        self.distances = distances

        if self.verbose >= 2:
            self._print_info()

    def set_centers(self, centers):
        """
        set starting centers

        parameters
        ----------
        centers: array
            [Ncen,2] array of centers ra,dec
        """
        centers = np.asanyarray(centers)

        # this will evolve during the run
        self.centers = centers.copy()

    def get_centers(self):
        """
        get a copy of the centers
        """

        centers = self.centers
        if centers is None:
            raise ValueError("you must set centers first")

        return centers.copy()

    def find_nearest(self, x):
        """
        find the nearest centers to the input points
        """
        return find_nearest(x, self.centers)

    def _print_info(self):
        ncen = self.centers.size
        r50 = np.zeros(ncen)
        r90 = np.zeros(ncen)

        distances = self.distances
        labels = self.labels

        for j in range(ncen):
            dist = distances[labels == j]
            if len(dist) > 0:
                r50[j], r90[j] = np.percentile(dist, (50, 90))
        self.r50 = r50.copy()
        self.r90 = r90.copy()
        print("kmeans: cluster 50 % radius", r50.astype(int))
        print("kmeans: cluster 90 % radius", r90.astype(int))
        # scale L1 / dim, L2 / sqrt(dim) ?


def kmeans_sample(x, ncen, nsample=None, maxiter=_MAXITER_DEF, **kw):
    """
    2-pass kmeans, fast for large N

    - kmeans a smaller random sample from X
    - take starting guesses for the centers from a random sample
      of the input points
    - full kmeans, starting from the centers from pass 1

    parameters
    ----------
    x: array
        [N, 2] array of ra,dec points
    ncen: int
        Number of centers
    nsample: int, optional
        Number of samples to use on first pass, default
        max( 2*sqrt(N), 10*ncen )
    tol: float, optional
        The relative change in the average distance to
        centers, signifies convergence
    verbose: int, optional
        How verbose.  0 silent, 1 minimal starting info, 2 prints running
        distances

    returns
    -------
    A KMeans object, with attributes .centers, .labels, .distances etc.

    .converged: bool
        True if converged
    .centers: array
        The array of centers, [ncen, ra, dec]
    .labels: array
        The index of the center closest to each input point [N]
    .distances: array
        The distance to the closest center for each poit [N]
    """

    n, _ = x.shape
    if nsample is None:
        nsample = max(2*np.sqrt(n), 10*ncen)

    # smaller random sample to start with
    xsample = random_sample(x, int(nsample))

    # choose random sample as centers
    pass1centers = random_sample(x, int(ncen))

    km = KMeans(pass1centers, **kw)
    km.run(xsample, maxiter=maxiter)

    # now a full run with these centers
    sample_centers = km.get_centers()

    km = KMeans(sample_centers, **kw)
    km.run(x, maxiter=maxiter)

    return km


_PIOVER2 = np.pi*0.5


def cdist_radec(a1, a2):
    """
    use broadcasting to get all distance pairs

    a represents [N,2] for ra,dec points
    """

    ra1 = a1[:, 0]
    dec1 = a1[:, 1]
    ra2 = a2[:, 0]
    dec2 = a2[:, 1]

    ra1 = ra1[:, newaxis]
    dec1 = dec1[:, newaxis]

    phi1 = deg2rad(ra1)
    theta1 = _PIOVER2 - deg2rad(dec1)
    phi2 = deg2rad(ra2)
    theta2 = _PIOVER2 - deg2rad(dec2)

    sintheta = sin(theta1)
    x1 = sintheta * cos(phi1)
    y1 = sintheta * sin(phi1)
    z1 = cos(theta1)

    sintheta = sin(theta2)
    x2 = sintheta * cos(phi2)
    y2 = sintheta * sin(phi2)
    z2 = cos(theta2)

    costheta = x1*x2 + y1*y2 + z1*z2

    costheta = np.clip(costheta, -1.0, 1.0)
    theta = arccos(costheta)
    return theta


def random_sample(x, n):
    """
    random.sample of the rows of X
    """
    sampleix = random.sample(range(x.shape[0]), int(n))
    return x[sampleix]


def find_nearest(x, centers):
    """
    find the nearest center for each input point

    parameters
    ----------
    X: array
        [N,2] points array of ra,dec points
    centers: array
        [ncen,2] center points

    returns
    -------
    labels: array
        The index of the nearest center for each input point
    """
    _check_dims(x, centers)
    d = cdist_radec(x, centers)  # |X| x |centers|
    return d.argmin(axis=1)


def _check_dims(x, centers):
    """
    check the dims are compatible
    """
    _, dim = x.shape
    ncen, cdim = centers.shape
    if dim != cdim:
        tup = (x.shape, centers.shape)
        raise ValueError("X %s and centers %s must have the same "
                         "number of columns" % tup)


def get_mean_center(x, y, z):
    """
    parameters
    ----------
    x: array
    y: array
    z: array

    returns
    -------
    ramean, decmean
    """

    xmean = x.mean()
    ymean = y.mean()
    zmean = z.mean()

    rmean = sqrt(xmean ** 2 + ymean ** 2 + zmean ** 2)

    thetamean = arccos(zmean / rmean)
    phimean = arctan2(ymean, xmean)

    ramean = rad2deg(phimean)
    decmean = rad2deg(pi/2.0 - thetamean)

    ramean = atbound1(ramean, 0.0, 360.0)

    return ramean, decmean


def radec2xyz(ra, dec):
    phi = deg2rad(ra)
    theta = _PIOVER2 - deg2rad(dec)

    sintheta = sin(theta)
    x = sintheta * cos(phi)
    y = sintheta * sin(phi)
    z = cos(theta)

    return x, y, z


def atbound1(longitude_in, minval, maxval):

    longitude = longitude_in
    while longitude < minval:
        longitude += 360.0

    while longitude > maxval:
        longitude -= 360.0

    return longitude
