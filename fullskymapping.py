import numpy as np
import healpy as hp
import pandas as pd
from file_handler import HealpixMap, WISEMap
from data_loader import WISEDataLoader
from healpy.rotator import Rotator
from calibration import ZodiCalibrator, SmoothFit
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

    def save_map(self, coord="G"):
        hp.fitsfunc.write_map(self.filename, self.mapdata, coord=coord, overwrite=True)
        return


class MapMaker:
    """Class managing how WISE tiles fill the FullSkyMap object"""

    def __init__(self, band, n, nside=256):
        self.band = band
        self.path = f"/home/users/mberkeley/wisemapper/data/output_maps/w{self.band}/"
        self.label = n
        self.mapname = f'fsm_w{self.band}_orbit_{self.label}.fits'
        self.uncname = self.mapname.replace('orbit', 'unc_orbit')
        self.csv_name = f"band_w{self.band}_orbit_{self.label}_pixel_timestamps.csv"
        self.fsm = FullSkyMap(self.mapname, nside)
        self.unc_fsm = FullSkyMap(self.uncname, nside)
        # self.pix_count = np.zeros_like(self.fsm.mapdata, dtype=int)
        self.calibrator = ZodiCalibrator(band)
        self.numerator_cumul = np.zeros_like(self.fsm.mapdata)
        self.denominator_cumul = np.zeros_like(self.fsm.mapdata)
        self.time_numerator_cumul = np.zeros_like(self.fsm.mapdata)
        self.time_denominator_cumul = np.zeros_like(self.fsm.mapdata)


    def add_image(self, args):
        filename, mjd_obs = args
        self._load_image(filename)
        self._place_image(mjd_obs)
        return

    def _load_image(self, filename):
        # Function to get image and coordinate data from File and map to the Healpix grid
        self.data_loader = WISEDataLoader(filename)
        self.data_loader.load_data()
        self.data_loader.load_coords()
        return

    def _place_image(self, mjd_obs):
        int_data = self.data_loader.int_data.compressed()
        unc_data = self.data_loader.unc_data.compressed()
        coords = self.data_loader.wcs_coords
        ra, dec = coords.T
        hp_inds = self.fsm.wcs2ind(ra, dec)
        self._fill_map(hp_inds, int_data, unc_data, mjd_obs)

        return

    def _fill_map(self, inds, ints, uncs, mjd_obs):
        data_grouped, uncs_grouped = self._groupby(inds, ints, uncs)
        numerator, denominator, t_numerator, t_denominator = zip(*np.array([self._calc_hp_pixel(data_grouped[i], uncs_grouped[i], mjd_obs)
            if len(data_grouped[i]) > 0 else (0, 0, 0, 0)
            for i in range(len(data_grouped))
        ]))
        self.numerator_cumul[:len(numerator)] += numerator
        self.denominator_cumul[:len(denominator)] += denominator
        self.time_numerator_cumul[:len(t_numerator)] += t_numerator
        self.time_denominator_cumul[:len(t_denominator)] += t_denominator

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
    def _calc_hp_pixel(data, unc, time):
        numerator = np.sum(data/unc**2)
        denominator = np.sum(1/unc**2)
        N = len(data)

        time_numerator = N*time*denominator
        time_denominator = N*denominator
        return numerator, denominator, time_numerator, time_denominator

    def normalize(self):
        self.fsm.mapdata = np.divide(self.numerator_cumul, self.denominator_cumul, out=np.zeros_like(self.fsm.mapdata),
                                     where=self.denominator_cumul > 0)
        self.unc_fsm.mapdata = np.sqrt(np.divide(np.ones_like(self.denominator_cumul), self.denominator_cumul,
                                                 out=np.zeros_like(self.unc_fsm.mapdata),
                                                 where=self.denominator_cumul > 0))
        self.fsm.timedata = np.divide(self.time_numerator_cumul, self.time_denominator_cumul,
                                      out=np.zeros_like(self.fsm.mapdata),
                                      where=self.time_denominator_cumul > 0)
        return

    def unpack_multiproc_data(self, alldata):
        numerators, denominators, time_numerators, time_denominators = zip(*alldata)
        self.numerator_cumul = reduce(np.add, numerators)
        self.denominator_cumul = reduce(np.add, denominators)
        self.time_numerator_cumul = reduce(np.add, time_numerators)
        self.time_denominator_cumul = reduce(np.add, time_denominators)
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
        popt_file = f'popt_w{self.band}.pkl'
        if os.path.exists(popt_file):
            with open(popt_file, 'rb') as f1:
                gain, offset = pickle.load(f1)
        else:
            gain = np.zeros(6323, dtype=float)
            offset = np.zeros(6323, dtype=float)

        gain[self.label] = popt[0]
        offset[self.label] = popt[1]
        with open(popt_file, 'wb') as f2:
            pickle.dump([gain, offset], f2, pickle.HIGHEST_PROTOCOL)

    def save_map(self):
        hp.fitsfunc.write_map(self.path + self.mapname, self.fsm.mapdata,
                              coord='G', overwrite=True)
        hp.fitsfunc.write_map(self.path + self.uncname, self.unc_fsm.mapdata,
                              coord='G', overwrite=True)
        self.save_csv()
        return

    def save_csv(self):
        pixel_inds = np.arange(len(self.fsm.mapdata), dtype=int)
        nonzero_inds = self.unc_fsm.mapdata != 0.0

        data_to_save = np.vstack((pixel_inds[nonzero_inds], self.fsm.mapdata[nonzero_inds],
                                  self.unc_fsm.mapdata[nonzero_inds], self.fsm.timedata[nonzero_inds]))

        np.savetxt(os.path.join(self.path, self.csv_name), data_to_save, delimiter=',', header="hp_pixel_index,pixel_value,pixel_unc,pixel_mjd_obs")



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
        for name, group in self.groups:
            yield group["full_filepath"], group["mjd_obs"]

    def clean_files(self):
        # Run files through CNN model. Good files undergo outlier removal.
        pass


class MapCombiner:

    def __init__(self, band, nside=256):
        self.band = band
        self.path = f"/home/users/mberkeley/wisemapper/data/output_maps/w{band}/"
        self.mapname = f'fsm_w{band}_all_interp.fits'
        self.uncname = self.mapname.replace('.fits', '_unc.fits')
        self.fsm = FullSkyMap(self.mapname, nside)
        self.unc_fsm = FullSkyMap(self.uncname, nside)
        self.numerator = np.zeros_like(self.fsm.mapdata)
        self.denominator = np.zeros_like(self.fsm.mapdata)


    def add_days(self, day_start, day_end, smooth_window=11):

        smooth_fitter = SmoothFit(self.band, "/home/users/mberkeley/wisemapper/scripts")
        gain, offset = smooth_fitter.load_fit_vals()

        bad_fit_days = [26, 27, 28, 30, 46, 47, 48, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                        67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 85, 86, 87, 88, 95, 98, 99, 100,
                        102, 108, 112, 113, 114, 115, 116, 117, 125, 126, 127, 128, 129, 130, 131, 132, 136, 146,
                        147, 150, 158, 176, 177]
        bad_fit_days2 = [0, 23, 24, 25, 36, 37, 38, 39, 41, 42, 49, 84, 89, 90, 91, 92, 93, 94, 96, 97, 101, 103, 104,
                         105,
                         106, 118, 120, 121, 122, 123, 124, 133, 136, 138, 139, 140, 141, 142, 143, 144, 145, 151, 180,
                         194, 199, 200, 201, 202, 203, 204, 205]

        all_bad_days = bad_fit_days + bad_fit_days2
        import numpy.ma as ma
        mask = np.zeros_like(gain)
        mask[bad_fit_days] = 1
        mask[bad_fit_days2] = 1
        days = np.arange(len(gain))
        days_masked = ma.array(days, mask=mask)
        masked_gain = ma.array(gain, mask=mask)
        masked_offset = ma.array(offset, mask=mask)

        interp_gain = np.interp(all_bad_days, days_masked.compressed(), masked_gain.compressed())
        gain[all_bad_days] = interp_gain

        interp_offset = np.interp(all_bad_days, days_masked.compressed(), masked_offset.compressed())
        offset[all_bad_days] = interp_offset

        smooth_gain = smooth_fitter.mean_filter(gain, smooth_window)
        smooth_offset = smooth_fitter.mean_filter(offset, smooth_window)
        smooth_fitter.plot_smoothfit_vals(gain, smooth_gain, offset, smooth_offset)

        map_datas = []
        map_uncs = []
        day_maps = []
        for i in range(day_start, day_end):
            filename = f"{self.path}fsm_w{self.band}_day_{i}.fits"
            if not os.path.exists(filename):
                print(f'Skipping file {filename} as it does not exist')
                continue
            day_map = WISEMap(filename, self.band)
            day_maps.append(day_map)
            day_map.read_data()

            orig_gain = gain[i-day_start]
            orig_offset = offset[i-day_start]
            adj_gain = smooth_gain[i-day_start]
            adj_offset = smooth_offset[i-day_start]
            data = day_maps[i-day_start].mapdata

            adj_data = smooth_fitter.adjust_calibration(data, orig_gain, adj_gain, orig_offset, adj_offset)
            map_datas.append(adj_data)

            day_uncmap = WISEMap(filename.replace("day", "unc_day"), self.band)
            day_uncmap.read_data()
            day_uncmap.mapdata = np.sqrt(np.abs(day_uncmap.mapdata)) * np.abs(adj_gain / orig_gain)
            # unc data originally saved as variance, so have to take sqrt. Also adjust for smooth fit by applying
            # gain adjustment factor.
            map_uncs.append(day_uncmap.mapdata)

            # self.numerator += np.divide(adj_data, day_uncmap.mapdata**2, out=np.zeros_like(adj_data),
            #                             where=day_uncmap.mapdata != 0.0)
            # self.denominator += np.divide(np.ones_like(adj_data), day_uncmap.mapdata**2,
            #                               out=np.zeros_like(adj_data), where=day_uncmap.mapdata != 0.0)
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
                numerator = np.sum(good_vals / good_unc_vals ** 2)
                denominator = np.sum(1 / good_unc_vals ** 2)
                self.fsm.mapdata[p] = numerator / denominator
                self.unc_fsm.mapdata[p] = 1 / np.sqrt(denominator)
        return

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
