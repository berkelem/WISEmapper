import numpy as np
import healpy as hp
import pandas as pd
from file_handler import HealpixMap, WISEMap
from data_loader import WISEDataLoader
from healpy.rotator import Rotator
from calibration import ZodiCalibrator
from functools import reduce
import os
import pickle
from scipy import stats



class FullSkyMap(HealpixMap):

    r = Rotator(coord=['C', 'G'])

    def __init__(self, filename, nside):
        super().__init__(filename)
        super().set_resolution(nside)

    def ind2wcs(self, index):
        theta, phi = hp.pixelfunc.pix2ang(self.nside, index)
        return np.degrees(phi), np.degrees(np.pi / 2. - theta)

    def wcs2ind(self, ra_arr, dec_arr, nest=False):
        """
        Define theta and phi in galactic coords, then rotate to celestial coords to determine the correct healpix pixel
        using ang2pix.
        hp.ang2pix only converts from celestial lon lat coords to pixels.
        :param ra_arr:
        :param dec_arr:
        :param nest:
        :return:
        """
        theta = [np.radians(-float(dec) + 90.) for dec in dec_arr]
        phi = [np.radians(float(ra)) for ra in ra_arr]
        theta_c, phi_c = self.r(theta, phi)
        return hp.pixelfunc.ang2pix(self.nside, theta_c, phi_c, nest=nest)

    def save_map(self):
        hp.fitsfunc.write_map(self.filename, self.mapdata, coord='G', overwrite=True)
        return


class MapMaker:
    """Class managing how WISE tiles fill the FullSkyMap object"""

    def __init__(self, band, n, nside=256):
        self.band = band
        self.path = f"/home/users/mberkeley/wisemapper/data/output_maps/w{self.band}/"
        self.label = n
        self.mapname = f'fsm_w{self.band}_day_{self.label}.fits'
        self.uncname = self.mapname.replace('day', 'unc_day')
        self.fsm = FullSkyMap(self.mapname, nside)
        self.unc_fsm = FullSkyMap(self.uncname, nside)
        # self.pix_count = np.zeros_like(self.fsm.mapdata, dtype=int)
        self.calibrator = ZodiCalibrator(band)
        self.numerator_cumul = np.zeros_like(self.fsm.mapdata)
        self.denominator_cumul = np.zeros_like(self.fsm.mapdata)


    def add_image(self, filename):
        self._load_image(filename)
        self._place_image()
        return

    def _load_image(self, filename):
        # Function to get image and coordinate data from File and map to the Healpix grid
        self.data_loader = WISEDataLoader(filename)
        self.data_loader.load_data()
        self.data_loader.load_coords()
        return

    def _place_image(self):
        int_data = self.data_loader.int_data.compressed()
        unc_data = self.data_loader.unc_data.compressed()
        coords = self.data_loader.wcs_coords
        ra, dec = coords.T
        hp_inds = self.fsm.wcs2ind(ra, dec)
        self._fill_map(hp_inds, int_data, unc_data)

        return

    def _fill_map(self, inds, ints, uncs):
        data_grouped, uncs_grouped = self._groupby(inds, ints, uncs)
        numerator, denominator = zip(*np.array([self._calc_hp_pixel(data_grouped[i], uncs_grouped[i])
                                                if len(data_grouped[i]) > 0 else (0, 0)
                                                for i in range(len(data_grouped))
                                                ]))
        self.numerator_cumul[:len(numerator)] += numerator
        self.denominator_cumul[:len(denominator)] += denominator
        return

    @staticmethod
    def _groupby(inds, data, uncs):
        # Get argsort indices, to be used to sort arrays in the next steps
        sidx = inds.argsort()
        data_sorted = data[sidx]
        inds_sorted = inds[sidx]
        uncs_sorted = uncs[sidx]

        # Get the group limit indices (start, stop of groups)
        cut_idx = np.flatnonzero(np.r_[True, inds_sorted[1:] != inds_sorted[:-1], True])

        # Create cut indices for all unique IDs in b
        n = inds_sorted[-1]+2
        cut_idxe = np.full(n, cut_idx[-1], dtype=int)

        insert_idx = inds_sorted[cut_idx[:-1]]
        cut_idxe[insert_idx] = cut_idx[:-1]
        cut_idxe = np.minimum.accumulate(cut_idxe[::-1])[::-1]

        # Split input array with those start, stop ones
        data_out = np.array([data_sorted[i:j] for i, j in zip(cut_idxe[:-1], cut_idxe[1:])])
        uncs_out = np.array([uncs_sorted[i:j] for i, j in zip(cut_idxe[:-1], cut_idxe[1:])])

        return data_out, uncs_out

    @staticmethod
    def _calc_hp_pixel(data, unc):
        numerator = np.sum(data/unc**2)
        denominator = np.sum(1/unc**2)
        return numerator, denominator

    def normalize(self):
        self.fsm.mapdata = np.divide(self.numerator_cumul, self.denominator_cumul, out=np.zeros_like(self.fsm.mapdata),
                                     where=self.denominator_cumul > 0)
        self.unc_fsm.mapdata = np.sqrt(np.divide(np.ones_like(self.denominator_cumul), self.denominator_cumul,
                                                 out=np.zeros_like(self.unc_fsm.mapdata),
                                                 where=self.denominator_cumul > 0))
        return

    def unpack_multiproc_data(self, alldata):
        numerators, denominators = zip(*alldata)
        self.numerator_cumul = reduce(np.add, numerators)
        self.denominator_cumul = reduce(np.add, denominators)
        return

    def normalize_multiproc_data(self, alldata):
        vals, counts = zip(*alldata)
        allvals = reduce(np.add, vals)
        allcounts = reduce(np.add, counts)
        self.fsm.mapdata = np.divide(allvals, allcounts, out=np.zeros_like(allvals), where=(allcounts > 0))

    def calibrate(self):
        self.fsm.mapdata, self.unc_fsm.mapdata = self.calibrator.calibrate(self.fsm.mapdata, self.unc_fsm.mapdata)
        self.calibrator.plot(self.label, self.path)
        popt = self.calibrator.popt
        with open(f'popt_w{self.band}.pkl', 'rb') as f1:
            gain, offset = pickle.load(f1)

        gain[self.label] = popt[0]
        offset[self.label] = popt[1]
        with open(f'popt_w{self.band}.pkl', 'wb') as f2:
            pickle.dump([gain, offset], f2, pickle.HIGHEST_PROTOCOL)

    def save_map(self):
        hp.fitsfunc.write_map(self.path + self.mapname, self.fsm.mapdata,
                              coord='G', overwrite=True)
        hp.fitsfunc.write_map(self.path + self.uncname, self.unc_fsm.mapdata,
                              coord='G', overwrite=True)
        return


class FileBatcher:
    """Collection of pixels to be mapped (can be facet or orbit)"""
    def __init__(self, filename):
        self.filename = filename
        self.filtered_data = None
        self.timestamps_df = None
        self.groups = None

    # def __iter__(self):
    #     self.load_dataframe()
    #     self._group_files()
    #     return self
    #
    # def __next__(self):
    #     return self.filelist_generator()

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

    def _group_files(self, groupby='day'):
        # Use orbit parameters to filter WISE images
        if groupby == 'day':
            self.get_day()
        elif groupby == 'orbit':
            self.get_orbit()
        # filegroup_generator = self.filelist_generator()

    def filelist_generator(self):
        for name, group in self.groups:
            yield group["full_filepath"]

    def clean_files(self):
        # Run files through CNN model. Good files undergo outlier removal.
        pass


class MapCombiner:

    def __init__(self, band, nside=256):
        self.band = band
        self.path = f"/home/users/mberkeley/wisemapper/data/output_maps/w{band}/"
        self.mapname = f'fsm_w{band}_all.fits'
        self.uncname = self.mapname.replace('.fits', '_unc.fits')
        self.fsm = FullSkyMap(self.mapname, nside)
        self.unc_fsm = FullSkyMap(self.uncname, nside)
        self.numerator = np.zeros_like(self.fsm.mapdata)
        self.denominator = np.zeros_like(self.fsm.mapdata)

    def add_days(self, day_start, day_end):
        for i in range(day_start, day_end):
            filename = f"{self.path}fsm_w{self.band}_day_{i}.fits"
            if not os.path.exists(filename):
                print(f'Skipping file {filename} as it does not exist')
                continue
            day_map = WISEMap(filename, self.band)
            day_map.read_data()
            day_uncmap = WISEMap(filename.replace("day", "unc_day"), self.band)
            day_uncmap.read_data()
            day_uncmap.mapdata = np.sqrt(day_uncmap.mapdata)

            self.numerator += np.divide(day_map.mapdata, day_uncmap.mapdata**2, out=np.zeros_like(day_map.mapdata),
                                        where=day_uncmap.mapdata != 0.0)
            self.denominator += np.divide(np.ones_like(day_map.mapdata), day_uncmap.mapdata**2,
                                          out=np.zeros_like(day_map.mapdata), where=day_uncmap.mapdata != 0.0)

    def normalize(self):
        self.fsm.mapdata = np.divide(self.numerator, self.denominator, out=np.zeros_like(self.numerator),
                                     where=self.numerator != 0.0)
        self.unc_fsm.mapdata = np.sqrt(np.divide(np.ones_like(self.numerator), self.denominator,
                                                 out=np.zeros_like(self.numerator), where=self.denominator != 0.0))

    def save_file(self):
        self.fsm.save_map()
        self.unc_fsm.save_map()

    @staticmethod
    def mask_outliers(data, threshold=1):
        z = np.abs(stats.zscore(data[data > 0.0]))
        mask = z > threshold
        return mask

    def clean_add(self, day_start, day_end):
        map_datas = []
        map_uncs = []
        for i in range(day_start, day_end):
            filename = f"{self.path}fsm_w{self.band}_day_{i}.fits"
            if not os.path.exists(filename):
                print(f'Skipping file {filename} as it does not exist')
                continue
            day_map = WISEMap(filename, self.band)
            day_map.read_data()
            map_datas.append(day_map.mapdata)
            day_uncmap = WISEMap(filename.replace("day", "unc_day"), self.band)
            day_uncmap.read_data()
            day_uncmap.mapdata = np.sqrt(day_uncmap.mapdata)
            map_uncs.append(day_uncmap.mapdata)

        px_vals = np.array(map_datas).T
        unc_vals = np.array(map_uncs).T
        for p, px in enumerate(px_vals):
            if len(px[px > 0.0]) > 2:
                mask = self.mask_outliers(px)
                good_vals = px[px > 0.0][~mask]
                good_unc_vals = unc_vals[p][px > 0.0][~mask]
            else:
                good_vals = px[px > 0.0]
                good_unc_vals = unc_vals[p][px > 0.0]

            if len(good_unc_vals) > 0:
                numerator = np.sum(good_vals/good_unc_vals**2)
                denominator = np.sum(1/good_unc_vals**2)
                self.fsm.mapdata[p] = numerator/denominator
                self.unc_fsm.mapdata[p] = 1/np.sqrt(denominator)
