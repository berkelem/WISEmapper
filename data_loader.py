from file_handler import FITSFile
import numpy as np
import numpy.ma as ma
from astropy.io import fits
from functools import reduce


class WISEDataLoader(object):

    def __init__(self, filename, mjd_obs=None):
        self.int_filename = filename
        self.msk_filename = filename.replace('-int-', '-msk-')
        self.unc_filename = filename.replace('-int-', '-unc-')
        self.mask = None
        self.wcs_coords = None
        self.mjd_obs = mjd_obs

    def load_data(self):
        self._read_file()
        self._mask_data()

    def _read_file(self):
        self.int_file = FITSFile(self.int_filename)
        self.msk_file = FITSFile(self.msk_filename)
        self.unc_file = FITSFile(self.unc_filename)

        self.int_data = fits.getdata(self.int_file.filename).astype(float).flatten()
        self.msk_data = fits.getdata(self.msk_file.filename).astype(bool).flatten()
        self.unc_data = fits.getdata(self.unc_file.filename).astype(float).flatten()

    def _mask_data(self):
        nan_mask = np.isnan(self.int_data)
        nan_mask_unc = np.isnan(self.unc_data)
        neg_mask = ma.masked_less(ma.array(self.int_data, mask=nan_mask), 0.0).mask
        self.mask = reduce(np.add, [self.msk_data, neg_mask, nan_mask, nan_mask_unc])
        self.int_data = ma.array(self.int_data, mask=self.mask)
        self.unc_data = ma.array(self.unc_data, mask=self.mask)
        if self.mjd_obs:
            self.time_data = ma.array(np.ones_like(self.int_data)*float(self.mjd_obs), mask=self.mask)
        else:
            self.time_data = None

    def load_coords(self):
        self.int_file.read_header()
        self.wcs_coords = self.int_file.wcs2px()
        self.wcs_coords = self.wcs_coords[~self.mask]
