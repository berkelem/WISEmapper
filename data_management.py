from file_handler import FITSFile
import numpy as np
import numpy.ma as ma
from astropy.io import fits
from functools import reduce


class WISEDataLoader(object):

    def __init__(self, filename):
        self.int_filename = filename
        self.msk_filename = filename.replace('-int-', '-msk-')
        self.unc_filename = filename.replace('-int-', '-unc-')
        self.mask = None
        self.wcs_coords = None

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

    def load_coords(self):
        self.int_file.read_header()
        self.wcs_coords = self.int_file.wcs2px()
        self.wcs_coords = self.wcs_coords[~self.mask]

class FileBatcher:
    """Collection of pixels to be mapped (can be facet or orbit)"""
    def __init__(self, filename):
        self.filename = filename
        self.filtered_data = None
        self.timestamps_df = None
        self.groups = None

    def load_dataframe(self):

        self.filtered_data = pd.read_csv(self.filename,
                                         names=["band", "crval1", "crval2", "ra1", "dec1", "ra2", "dec2", "ra3", "dec3",
                                                "ra4", "dec4", "magzp", "magzpunc", "modeint", "scan_id", "scangrp",
                                                "frame_num", "date_obs_date", "date_obs_time", "mjd_obs", "dtanneal",
                                                "utanneal_date", "utanneal_time", "exptime", "qa_status", "qual_frame",
                                                "debgain", "febgain", "moon_sep", "saa_sep", "qual_scan",
                                                "full_filepath"],
                                         dtype={"band": np.int32, "crval1": np.float64, "crval2": np.float64,
                                                "ra1": np.float64, "dec1": np.float64, "ra2": np.float64,
                                                "dec2": np.float64, "ra3": np.float64, "dec3": np.float64,
                                                "ra4": np.float64, "dec4": np.float64, "magzp": np.float64,
                                                "magzpunc": np.float64, "modeint": np.float64, "scan_id": np.object_,
                                                "scangrp": np.object_, "frame_num": np.object_,
                                                "date_obs_date": np.object_, "date_obs_time": np.object_,
                                                "mjd_obs": np.float64, "dtanneal": np.float64,
                                                "utanneal_date": np.object_, "utanneal_time": np.object_,
                                                "exptime": np.float64, "qa_status": np.object_, "qual_frame": np.int32,
                                                "debgain": np.float64, "febgain": np.float64, "moon_sep": np.float64,
                                                "saa_sep": np.float64, "qual_scan": np.int32,
                                                "full_filepath": np.object_},
                                         skiprows=1
                                         )
        return

    def filter_timestamps(self):
        self.timestamps_df = self.filtered_data[["date_obs_date", "date_obs_time", "mjd_obs", "scan_id", "scangrp",
                                                "full_filepath"]].copy().sort_values("mjd_obs")
        return

    def get_orbit(self):
        self.filter_timestamps()
        self.groups = self.timestamps_df.groupby('scan_id')
        return

    def get_day(self):
        self.filter_timestamps()
        self.groups = self.timestamps_df.groupby(self.timestamps_df['mjd_obs'].apply(lambda x: round(x)))
        return

    def group_days(self):
        self.load_dataframe()
        self._group_files(groupby='day')

    def group_orbits(self):
        self.load_dataframe()
        self._group_files(groupby='orbit')

    def _group_files(self, groupby='day'):
        # Use orbit parameters to filter WISE images
        if groupby == 'day':
            self.get_day()
        elif groupby == 'orbit':
            self.get_orbit()
        # filegroup_generator = self.filelist_generator()

    def filelist_generator(self):
        grp_num = 0
        for name, group in self.groups:
            yield group["full_filepath"], group["mjd_obs"], grp_num
            grp_num += 1

    def clean_files(self):
        # Run files through CNN model. Good files undergo outlier removal.
        pass
