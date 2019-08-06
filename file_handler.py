# -*- coding: utf-8 -*-

"""
Created on Sun Jun 9 2019

@author: Matthew Berkeley
"""

import logging
from astropy.io import fits
from astropy import wcs
import healpy as hp
from healpy.rotator import Rotator
import os
import numpy as np

logger = logging.getLogger(__name__)

class File(object):

    def __init__(self, filename):
        self.filename = filename
        self.basename = self.filename.split('/')[-1]


class FITSFile(File):

    def __init__(self, filename):

        super().__init__(filename)
        self._check_file_exists()
        self.header = None

    def _check_file_exists(self):
        if os.path.exists(self.filename):
            return
        elif os.path.exists(f'{self.filename}.gz'):
            self.filename = f'{self.filename}.gz'
            self.basename = f'{self.basename}.gz'
            return
        else:
            raise IOError(f'File {self.filename} not found')

    def read_header(self):
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


class HealpixMap(File):

    def __init__(self, filename):

        super().__init__(filename)
        self.mapdata = None
        self.nside = None
        self.npix = None
        self.theta = None
        self.phi = None

    def read_data(self):
        self.mapdata = hp.fitsfunc.read_map(self.filename, verbose=False)
        self.npix = len(self.mapdata)
        self.nside = hp.npix2nside(self.npix)

    def write_data(self, coord='G'):
        hp.fitsfunc.write_map(self.filename, self.mapdata, coord=coord)

    def set_resolution(self, nside):
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        if self.mapdata is not None:
            self.mapdata = hp.ud_grade(self.mapdata, nside)
        else:
            self.mapdata = np.zeros(self.npix, dtype=float)

    def rotate_map(self, old_coord='G', new_coord='E'):
        self.theta, self.phi = hp.pix2ang(self.nside, np.arange(self.npix))
        r = Rotator(coord=[new_coord, old_coord])  # Transforms galactic to ecliptic coordinates
        theta_rot, phi_rot = r(self.theta, self.phi)  # Apply the conversion
        rot_pixorder = hp.ang2pix(hp.npix2nside(self.npix), theta_rot, phi_rot)
        self.mapdata = self.mapdata[rot_pixorder]
        self.theta = theta_rot
        self.phi = phi_rot
        return


class WISEMap(HealpixMap):

    def __init__(self, filename, band):
        super().__init__(filename)
        self.band = band


class ZodiMap(WISEMap):

    def __init__(self, filename, band):
        super().__init__(filename, band)
