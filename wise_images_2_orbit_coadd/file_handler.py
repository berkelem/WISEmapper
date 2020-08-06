"""
:author: Matthew Berkeley
:date: Jun 9 2019

Module defining objects for FITS files and Healpix files.

Main classes
------------

FITSFile : For working with FITS files

HealpixMap : For working with Healpix maps

WISEMap : For working with full-sky WISE maps

ZodiMap : For working with full-sky Zodiacal light maps

"""

import logging
from astropy.io import fits
from astropy import wcs
import healpy as hp
from healpy.rotator import Rotator
import os
import numpy as np

logger = logging.getLogger(__name__)


class File:
    """Base class for all file types"""

    def __init__(self, filename):
        self.filename = filename
        self.basename = os.path.basename(self.filename)


class FITSFile(File):
    """
    Class for working with FITS files.

    Parameters
    ----------
    :param filename: str
        Name of FITS file, including full path

    Attributes
    ----------
    header : Header of FITS file
    data : Data contained in FITS file
    """

    def __init__(self, filename):

        super().__init__(filename)
        self._check_file_exists()
        self.header = None
        self.data = None

    def _check_file_exists(self):
        """
        Tests for existence of file. If file is present in gunzip form ('.gz' suffix) that is taken into
        consideration
        """
        if os.path.exists(self.filename):
            return
        elif os.path.exists(f'{self.filename}.gz'):
            self.filename = f'{self.filename}.gz'
            self.basename = f'{self.basename}.gz'
            return
        else:
            raise IOError(f'File {self.filename} not found')

    def read_header(self):
        """Read FITS header"""
        self.header = fits.open(self.filename)[0].header
        self.header.rename_keyword('RADECSYS', 'RADESYS')

    def wcs2px(self):
        """Read file header and return a 1-dim array of wcs pixel coordinates."""
        x_dim = self.header['NAXIS1']
        y_dim = self.header['NAXIS2']
        coord_array = [(x, y) for y in range(y_dim) for x in range(x_dim)]
        wcs_file = wcs.WCS(self.header).wcs_pix2world(coord_array, 0,
                                                      ra_dec_order=True)
        return wcs_file

    def read_data(self):
        """Read data in FITS file"""
        self.data = fits.getdata(self.filename)


class HealpixMap(File):
    """
    Class for working with Healpix files.

    Parameters
    ----------
    :param filename: str
        Name of Healpix file, including full path

    Methods
    -------
    read_data :
        Read data from Healpix file
    write_data :
        Write data to Healpix file
    set_resolution :
        Set Healpix map resolution by providing NSIDE parameter
    rotate_map :
        Rotate Healpix pixel numbering by applying a coordinate system transformation
    save_map :
        Save map to file

    Attributes
    ----------
    mapdata : numpy.array
        1-D array containing data for Healpix pixels
    nside : int
        Healpix NSIDE parameter
    npix : int
        Number of pixels in Healpix map
    theta : numpy.array
        co-latitude (in radians)
    phi : numpy.array
        longitude (in radians)
    """

    def __init__(self, filename):

        super().__init__(filename)
        self.mapdata = None
        self.nside = None
        self.npix = None
        self.theta = None
        self.phi = None

    def read_data(self):
        """Read data from Healpix file"""
        self.mapdata = hp.fitsfunc.read_map(self.filename, verbose=False)
        self.npix = len(self.mapdata)
        self.nside = hp.npix2nside(self.npix)

    def write_data(self, coord='G', clobber=False):
        """
        Write data to Healpix file

        Parameters
        ----------
        :param coord: str
            One of 'G' (galactic, default), 'C' (celestial), 'E' (ecliptic)
        :param clobber: bool
            Overwrite existing file with the same name
        """
        hp.fitsfunc.write_map(self.filename, self.mapdata, coord=coord, overwrite=clobber)

    def set_resolution(self, nside):
        """
        Set Healpix map resolution by providing NSIDE parameter

        Parameters
        ----------
        :param nside: int
            Any value of the form 2^n, where n is a positive integer
        """
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        if self.mapdata is not None:
            self.mapdata = hp.ud_grade(self.mapdata, nside)
        else:
            self.mapdata = np.zeros(self.npix, dtype=float)

    def rotate_map(self, old_coord='G', new_coord='E'):
        """
        Rotate Healpix pixel numbering by applying a coordinate system transformation

        Parameters
        ----------
        :param old_coord: str
            One of 'G' (galactic, default), 'C' (celestial), 'E' (ecliptic)
        :param new_coord: str
            One of 'G' (galactic), 'C' (celestial), 'E' (ecliptic, default)
        """
        self.theta, self.phi = hp.pix2ang(self.nside, np.arange(self.npix))
        r = Rotator(coord=[new_coord, old_coord])  # Transforms galactic to ecliptic coordinates
        theta_rot, phi_rot = r(self.theta, self.phi)  # Apply the conversion
        rot_pixorder = hp.ang2pix(hp.npix2nside(self.npix), theta_rot, phi_rot)
        self.mapdata = self.mapdata[rot_pixorder]
        self.theta = theta_rot
        self.phi = phi_rot
        return

    def save_map(self, coord="G"):
        """
        Save map to file

        Parameters
        ----------
        :param coord: str
            One of 'G' (galactic, default), 'C' (celestial), 'E' (ecliptic)
        """
        hp.fitsfunc.write_map(self.filename, self.mapdata, coord=coord, overwrite=True)
        return


class WISEMap(HealpixMap):
    """
    Class for working with WISE full sky maps

    Parameters
    ----------
    :param filename: str
        Name of Healpix file, including full path
    :param band: int
        Any of [1,2,3,4]

    Methods
    -------
    ind2wcs :
        Convert pixel index into theta, phi coordinates corresponding to co-latitude and longitude in radians
    wcs2ind :
        Convert RA, Dec values into pixel indices on the Healpix grid.

    Attributes
    ----------
    band : int
        One of [1,2,3,4]
    mapdata : numpy.array
        1-D array containing data for Healpix pixels
    timedata : numpy.array
        1-D array containing timestamps for each pixel
    nside : int
        Healpix NSIDE parameter
    npix : int
        Number of pixels in Healpix map
    theta : numpy.array
        co-latitude (in radians)
    phi : numpy.array
        longitude (in radians)
    """

    r = Rotator(coord=['C', 'G'])

    def __init__(self, filename, band):
        super().__init__(filename)
        self.band = band
        self.nside = 256
        super().set_resolution(self.nside)
        self.timedata = np.zeros_like(self.mapdata)

    def ind2wcs(self, index):
        theta, phi = hp.pixelfunc.pix2ang(self.nside, index)
        return np.degrees(phi), np.degrees(np.pi / 2. - theta)

    def wcs2ind(self, ra_arr, dec_arr, nest=False):
        """
        Define theta and phi in galactic coords, then rotate to celestial coords to determine the correct healpix pixel
        using ang2pix.
        hp.ang2pix only converts from celestial lon lat coords to pixels.
        :param ra_arr: numpy.array
            Right ascension values
        :param dec_arr: numpy.array
            Declination values
        :param nest: bool
            Use NEST pixel ordering instead of RING (default is RING)
        :return: numpy.array
            Array containing the pixel indices for the given coordinates
        """
        theta = [np.radians(-float(dec) + 90.) for dec in dec_arr]
        phi = [np.radians(float(ra)) for ra in ra_arr]
        theta_c, phi_c = self.r(theta, phi)
        return hp.pixelfunc.ang2pix(self.nside, theta_c, phi_c, nest=nest)


class ZodiMap(WISEMap):
    """
    Class for working with Zodiacal light full sky maps

    Parameters
    ----------
    :param filename: str
        Name of Healpix file, including full path
    :param band: int
        Any of [1,2,3,4]

    Attributes
    ----------
    band : int
        One of [1,2,3,4]
    mapdata : numpy.array
        1-D array containing data for Healpix pixels
    nside : int
        Healpix NSIDE parameter
    npix : int
        Number of pixels in Healpix map
    theta : numpy.array
        co-latitude (in radians)
    phi : numpy.array
        longitude (in radians)
    """

    def __init__(self, filename, band):
        super().__init__(filename, band)
