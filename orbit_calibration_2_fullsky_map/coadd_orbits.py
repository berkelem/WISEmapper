from wise_images_2_orbit_coadd.file_handler import WISEMap, ZodiMap, HealpixMap
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
import pickle
from wise_images_2_orbit_coadd.fullskymapping import BaseMapper
import os
from collections import OrderedDict

class Orbit(BaseMapper):

    coadd_map = None
    orbit_file_path = ""
    zodi_file_path = ""

    def __init__(self, orbit_num, band, mask, nside):

        super().__init__(band, orbit_num, self.orbit_file_path)
        self.orbit_num = orbit_num
        self.band = band
        self.nside = nside
        self.filename = os.path.join(self.orbit_file_path, self.orbit_csv_name)
        self.zodi_filename = os.path.join(self.zodi_file_path, f"zodi_map_cal_W{self.band}_{self.orbit_num}.fits")
        self.mask = mask
        self.mask_inds = np.arange(len(self.mask))[self.mask.astype(bool)]
        self.outlier_inds = np.array([])

        self.orbit_data = None
        self.orbit_uncs = None
        self.pixel_inds = None
        self.orbit_mjd_obs = None
        self.zodi_data = None
        self.mean_mjd_obs = None
        self.std_mjd_obs = None

        self.smooth_gain = None
        self.smooth_offset = None

        self.orbit_data_masked = None
        self.orbit_uncs_masked = None
        self.zodi_data_masked = None

        self.gain = 1.0
        self.offset = 0.0

        self.cal_data_clean = None
        self.cal_uncs_clean = None
        self.zs_data_clean = None


    def load_orbit_data(self):
        all_orbit_data = pd.read_csv(self.filename)
        self.orbit_data = np.array(all_orbit_data["pixel_value"])
        self.orbit_uncs = np.array(all_orbit_data["pixel_unc"])
        self.pixel_inds = np.array(all_orbit_data["hp_pixel_index"])
        self.orbit_mjd_obs = np.array(all_orbit_data["pixel_mjd_obs"])
        self.mean_mjd_obs = np.mean(self.orbit_mjd_obs)
        self.std_mjd_obs = np.std(self.orbit_mjd_obs)

        return

    # @staticmethod
    # def update_param(orig_param, smooth_param):
    #     first_val = np.where(smooth_param == list(filter(lambda x: x!=1.0, smooth_param))[0])[0][0]
    #     last_val = np.where(smooth_param == list(filter(lambda x: x!=1.0, smooth_param))[-1])[0][-1]
    #     orig_param[first_val:last_val+1] = smooth_param[first_val:last_val+1]
    #     if first_val != 0 and orig_param[first_val-1] != orig_param[first_val]:
    #         orig_param[:first_val] = orig_param[first_val]
    #     if last_val+1 != len(orig_param) and orig_param[last_val + 1] != orig_param[last_val]:
    #         orig_param[last_val + 1:] = orig_param[last_val]
    #
    #     return orig_param

    def load_zodi_orbit_data(self):
        zodi_orbit = ZodiMap(self.zodi_filename, self.band)
        zodi_orbit.read_data()
        pixels = np.zeros_like(zodi_orbit.mapdata)
        pixels[self.pixel_inds] = 1.0
        self.zodi_data = zodi_orbit.mapdata[pixels.astype(bool)]
        if not all(self.zodi_data.astype(bool)):
            print(f"Orbit {self.orbit_num} mismatch with zodi: zeros in zodi orbit")
        elif any(zodi_orbit.mapdata[~pixels.astype(bool)]):
            print(f"Orbit {self.orbit_num} mismatch with zodi: nonzeros outside zodi orbit")
        return

    def apply_mask(self):
        self.pixel_inds_clean = np.array(
            [self.pixel_inds[i] for i in range(len(self.pixel_inds)) if i not in self.outlier_inds], dtype=int)
        self.orbit_data_clean = np.array(
            [self.orbit_data[i] for i in range(len(self.orbit_data)) if i not in self.outlier_inds])
        self.orbit_uncs_clean = np.array(
            [self.orbit_uncs[i] for i in range(len(self.orbit_uncs)) if i not in self.outlier_inds])
        self.orbit_mjd_clean = np.array(
            [self.orbit_mjd_obs[i] for i in range(len(self.orbit_mjd_obs)) if i not in self.outlier_inds])
        self.zodi_data_clean = np.array(
            [self.zodi_data[i] for i in range(len(self.zodi_data)) if i not in self.outlier_inds])

        self.entries_to_mask = [i for i in range(len(self.pixel_inds)) if self.pixel_inds[i] in self.mask_inds]
        self.pixel_inds_clean_masked = np.array([self.pixel_inds[i] for i in range(len(self.pixel_inds)) if
                                                 i not in self.entries_to_mask and i not in self.outlier_inds],
                                                dtype=int)
        self.orbit_data_clean_masked = np.array([self.orbit_data[i] for i in range(len(self.orbit_data)) if
                                                 i not in self.entries_to_mask and i not in self.outlier_inds])
        self.orbit_uncs_clean_masked = np.array([self.orbit_uncs[i] for i in range(len(self.orbit_uncs)) if
                                                 i not in self.entries_to_mask and i not in self.outlier_inds])
        self.orbit_mjd_clean_masked = np.array([self.orbit_mjd_obs[i] for i in range(len(self.orbit_mjd_obs)) if
                                                i not in self.entries_to_mask and i not in self.outlier_inds])
        self.zodi_data_clean_masked = np.array([self.zodi_data[i] for i in range(len(self.zodi_data)) if
                                                i not in self.entries_to_mask and i not in self.outlier_inds])

        return

    def fit(self):
        orbit_data_to_fit_clean_masked = self.orbit_data_clean_masked
        if self.coadd_map is not None:
            self.clean_data()
            prev_itermap_clean_masked = self.coadd_map[self.pixel_inds_clean_masked]
            t_gal_clean_masked = prev_itermap_clean_masked * self.gain
            orbit_data_to_fit_clean_masked = self.orbit_data_clean_masked - t_gal_clean_masked

        orbit_fitter = IterativeFitter(self.zodi_data_clean_masked, orbit_data_to_fit_clean_masked,
                                       self.orbit_uncs_clean_masked)
        self.gain, self.offset = orbit_fitter.iterate_fit(1)
        return

    def apply_fit(self):
        self.cal_data_clean = (self.orbit_data_clean - self.offset) / self.gain
        self.cal_data_clean_masked = (self.orbit_data_clean_masked - self.offset) / self.gain
        self.cal_uncs_clean = self.orbit_uncs_clean / abs(self.gain)
        self.cal_uncs_clean_masked = self.orbit_uncs_clean_masked / abs(self.gain)

        self.zs_data_clean = self.cal_data_clean - self.zodi_data_clean
        self.zs_data_clean[self.zs_data_clean < 0.0] = 0.0
        self.zs_data_clean_masked = self.cal_data_clean_masked - self.zodi_data_clean_masked
        self.zs_data_clean_masked[self.zs_data_clean_masked < 0.0] = 0.0

    def apply_spline_fit(self, gain_spline, offset_spline):
        gains = gain_spline(self.orbit_mjd_clean_masked)
        offsets = offset_spline(self.orbit_mjd_clean_masked)
        self.cal_data_clean_masked = (self.orbit_data_clean_masked - offsets) / gains
        self.cal_uncs_clean_masked = self.orbit_uncs_clean_masked / abs(gains)

        self.zs_data_clean_masked = self.cal_data_clean_masked - self.zodi_data_clean_masked
        # self.zs_data[self.zs_data < 0.0] = 0.0

        # self.zs_data_clean_masked = np.array(
        #     [self.zs_data[i] for i in range(len(self.zs_data)) if i not in self.entries_to_mask])

        return

    def clean_data(self):
        z = np.abs(stats.zscore(self.zs_data_clean_masked))
        mask = z > 1
        inds_to_mask = self.pixel_inds_clean_masked[mask]
        self.outlier_inds = np.append(self.outlier_inds, inds_to_mask)
        self.apply_mask()
        return

    def plot_fit(self, output_path):
        theta, phi = hp.pix2ang(self.nside, self.pixel_inds_clean_masked, lonlat=True)
        plt.plot(phi, self.cal_data_clean_masked, 'r.', ms=0.5)
        plt.plot(phi, self.zodi_data_clean_masked, 'b.', ms=0.5)
        plt.xlabel("Latitude (degrees)")
        plt.ylabel("MJy/sr")
        plt.savefig(os.path.join(output_path, "orbit_{}_fit.png".format(self.orbit_num)))
        plt.close()


class Coadder:

    def __init__(self, band, moon_stripe_file, gain_pickle_file, offset_pickle_file, fsm_map_file, orbit_file_path,
                 zodi_file_path, output_path=None):
        self.band = band
        self.gain_pickle_file = gain_pickle_file
        self.offset_pickle_file = offset_pickle_file
        self.fsm_map_file = fsm_map_file
        self.unc_fsm_map_file = "{}_{}.{}".format(fsm_map_file.splitext()[0], "unc", fsm_map_file.splitext()[1])
        self.output_path = output_path if output_path else os.getcwd()
        setattr(Orbit, "orbit_file_path", orbit_file_path)
        setattr(Orbit, "zodi_file_path", zodi_file_path)

        self.iter = 0

        self.moon_stripe_mask = HealpixMap(moon_stripe_file)
        self.moon_stripe_mask.read_data()
        self.moon_stripe_inds = \
            np.arange(len(self.moon_stripe_mask.mapdata))[self.moon_stripe_mask.mapdata.astype(bool)]

        self.full_mask = self.moon_stripe_mask.mapdata.astype(bool)
        self.npix = self.moon_stripe_mask.npix
        self.nside = self.moon_stripe_mask.nside

        self.numerator_masked = np.zeros(self.npix)
        self.denominator_masked = np.zeros_like(self.numerator_masked)

        self.all_data = [[] for _ in range(self.npix)]
        self.all_uncs = [[] for _ in range(self.npix)]

        self.gains = []
        self.offsets = []
        self.orbit_sizes = []

        self.fsm_masked = None
        self.unc_fsm_masked = None

        self.month_timestamps = OrderedDict([("Jan", 55197), ("Feb", 55228), ("Mar", 55256), ("Apr", 55287),
                                             ("May", 55317), ("Jun", 55348), ("Jul", 55378), ("Aug", 55409)])

        self.all_orbits = []


    def load_pickle_vals(self):
        with open(self.gain_pickle_file, "rb") as gain_spline_file:
            self.gain_spline = pickle.load(gain_spline_file)

        with open(self.offset_pickle_file, "rb") as offset_spline_file:
            self.offset_spline = pickle.load(offset_spline_file)

    def _set_output_filenames(self):
        self.fsm_masked = WISEMap(self.fsm_map_file, self.band)
        self.unc_fsm_masked = WISEMap(self.unc_fsm_map_file, self.band)

    def _filter_timestamps(self, month, mjd_obs):
        if not month in self.month_timestamps:
            print("Unrecognized time period. Please specify one of ['all', 'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', "
                  "'Aug']. Proceeding with 'all'.")
            return True
        else:
            months = list(self.month_timestamps.keys())
            if months.index(month) == len(months) - 1:
                if mjd_obs >= self.month_timestamps[month]:
                    return True
                else:
                    return False
            elif months.index(month) == 0:
                if mjd_obs < self.month_timestamps[months[months.index(month) + 1]]:
                    return True
                else:
                    return False
            else:
                if (mjd_obs < self.month_timestamps[months[months.index(month) + 1]] and
                        mjd_obs >= self.month_timestamps[month]):
                    return True
                else:
                    return False


    def run_iterative_fit(self, num_orbits, iterations, month="all", plot=False):

        for it in range(iterations):
            self.numerator_masked = np.zeros(self.npix)
            self.denominator_masked = np.zeros_like(self.numerator_masked)

            for i in range(num_orbits):
                print(f"Iteration {it}; Fitting orbit {i}")
                if it == 0:
                    orbit = Orbit(i, self.band, self.full_mask, self.nside)
                    self.all_orbits.append(orbit)
                    orbit.load_orbit_data()
                    orbit.load_zodi_orbit_data()
                    orbit.apply_mask()
                else:
                    orbit = self.all_orbits[i]

                if month == "all":
                    pass
                else:
                    if not self._filter_timestamps(month, orbit.mean_mjd_obs):
                        print(f"Skipping orbit {i}")
                        continue

                orbit.fit()
                orbit.apply_fit()

                self.add_orbit(orbit)
                if plot and i % 15 == 0.0:
                    orbit.plot_fit(self.output_path)

            self.save_fit_params_to_file(it)

            self.clean_data()
            self.compile_map()

            self.normalize()
            self.save_maps()

            setattr(Orbit, "coadd_map", self.fsm_masked.mapdata)
            self.iter += 1

    def save_fit_params_to_file(self, it):
        print("Saving data for iteration {}".format(it))
        all_gains = np.array([orb.gain for orb in self.all_orbits])
        all_offsets = np.array([orb.offset for orb in self.all_orbits])
        all_mjd_vals = np.array([orb.orbit_mjd_obs for orb in self.all_orbits])
        with open(os.path.join(self.output_path, "fitvals_iter_{}.pkl".format(it)), "wb") as f:
            pickle.dump([all_gains, all_offsets, all_mjd_vals], f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def add_calibrated_orbits(self):
        for i in range(num_orbits):
            print(f"Adding orbit {i}")
            self._set_output_filenames()
            orbit = Orbit(i, self.band, self.full_mask)
            orbit.load_orbit_data()
            if month == "all":
                pass
            else:
                if not self._filter_timestamps(month, orbit.mean_mjd_obs):
                    continue
            orbit.load_zodi_orbit_data()
            orbit.apply_mask()
            orbit.apply_spline_fit(self.gain_spline, self.offset_spline)
            self.add_orbit(orbit)
            if i % 15 == 0.0:
                orbit.plot_fit()

        self.clean_data()
        self.compile_map()
        self.normalize()
        self.save_maps()

        # setattr(Orbit, "coadd_map_unmasked", self.fsm_unmasked.mapdata)
        # setattr(Orbit, "coadd_map_masked", self.fsm_masked.mapdata)
        # self.iter += 1

        # all_gains = np.array([orb.gain for orb in all_orbits])
        # all_offsets = np.array([orb.offset for orb in all_orbits])
        # all_mean_time = np.array([orb.mean_mjd_obs for orb in all_orbits])
        # import pickle
        # with open("fitvals.pkl", "wb") as f:
        #     pickle.dump([all_gains, all_offsets, all_mean_time], f, protocol=pickle.HIGHEST_PROTOCOL)

    # def smooth_fit_params(self, orbit1, orbit2):
    #     t1 = orbit1.orbit_mjd_obs
    #     t2 = orbit2.orbit_mjd_obs
    #     g1 = orbit1.gain
    #     g2 = orbit2.gain
    #     f1 = orbit1.offset
    #     f2 = orbit2.offset
    #
    #     mt1 = np.median(t1)
    #     mt2 = np.median(t2)
    #     t1_fit = t1[int(len(t1) / 2):]
    #     t2_fit = t2[:int(len(t2) / 2)]
    #
    #     t = np.zeros(len(t1_fit) + len(t2_fit), dtype=float)
    #
    #     t[:len(t1_fit)] = t1_fit
    #     t[len(t1_fit):] = t2_fit
    #
    #     A = np.sin((np.pi / 2) * ((t - mt2) / (mt1 - mt2))) ** 2
    #     B = np.cos((np.pi / 2) * ((t - mt2) / (mt1 - mt2))) ** 2
    #
    #     G = A * g1 + B * g2
    #     F = A * f1 + B * f2
    #
    #     G1 = np.ones_like(t1)
    #     G2 = np.ones_like(t2)
    #     F1 = np.zeros_like(t1)
    #     F2 = np.zeros_like(t2)
    #
    #     G1[int(len(t1) / 2):] = G[:len(t1_fit)]
    #     G2[:int(len(t2) / 2)] = G[len(t1_fit):]
    #     F1[int(len(t1) / 2):] = F[:len(t1_fit)]
    #     F2[:int(len(t2) / 2)] = F[len(t1_fit):]
    #
    #     return G1, G2, F1, F2


    def simple_plot(self, x_data, y_data, x_label, y_label, filename):
        plt.plot(x_data, y_data, 'r.')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(filename)
        plt.close()

    def plot_all_fits(self):
        plt.plot(range(len(self.gains)), self.gains, 'r.')
        plt.xlabel("orbit iteration")
        plt.ylabel("gain")
        plt.savefig("/home/users/mberkeley/wisemapper/data/output_maps/w3/fitted_gains.png")
        plt.close()
        plt.plot(range(len(self.offsets)), self.offsets, 'r.')
        plt.xlabel("orbit iteration")
        plt.ylabel("offset")
        plt.savefig("/home/users/mberkeley/wisemapper/data/output_maps/w3/fitted_offsets.png")
        plt.close()

    def add_orbit(self, orbit):
        if len(orbit.orbit_uncs_clean_masked[orbit.orbit_uncs_clean_masked != 0.0]) > 0 and orbit.gain != 0.0:
            orbit_data = orbit.zs_data_clean_masked
            orbit_uncs = orbit.cal_uncs_clean_masked
            orbit_pixels = orbit.pixel_inds_clean_masked
            for p, px in enumerate(orbit_pixels):
                self.all_data[px].append(orbit_data[p])
                self.all_uncs[px].append(orbit_uncs[p])

    def clean_data(self):
        for p, px_list in enumerate(self.all_data):
            if len(px_list) <= 1:
                continue
            unc_list = self.all_uncs[p]
            z = np.abs(stats.zscore(px_list))
            mask = z > 1
            inds_to_mask = np.arange(len(px_list), dtype=int)[mask]
            for ind in inds_to_mask[::-1]:
                px_list.pop(ind)
                unc_list.pop(ind)

    def compile_map(self):
        for p, px_list in enumerate(self.all_data):
            unc_list = self.all_uncs[p]
            self.numerator_masked[p] += sum(np.array([px_list[i]/(unc_list[i]**2) for i in range(len(px_list))]))
            self.denominator_masked[p] += sum(np.array([1/(unc_list[i]**2) for i in range(len(px_list))]))




    # def add_orbit_masked(self, orbit):
    #
    #     if len(orbit.orbit_uncs_clean_masked[orbit.orbit_uncs_clean_masked!=0.0]) > 0 and orbit.gain!=0.0:
    #         self.numerator_masked[orbit.pixel_inds_clean_masked] += np.divide(orbit.zs_data_clean_masked, np.square(orbit.cal_uncs_clean_masked), where=orbit.cal_uncs_clean_masked != 0.0, out=np.zeros_like(orbit.cal_uncs_clean_masked))
    #         self.denominator_masked[orbit.pixel_inds_clean_masked] += np.divide(1, np.square(orbit.cal_uncs_clean_masked), where=orbit.cal_uncs_clean_masked != 0.0, out=np.zeros_like(orbit.cal_uncs_clean_masked))
    #     return
    #
    # def add_orbit_unmasked(self, orbit):
    #
    #     if len(orbit.orbit_uncs_clean[orbit.orbit_uncs_clean!=0.0]) > 0 and orbit.gain!=0.0:
    #         self.numerator_unmasked[orbit.pixel_inds_clean] += np.divide(orbit.zs_data_clean, np.square(orbit.cal_uncs_clean), where=orbit.cal_uncs_clean != 0.0, out=np.zeros_like(orbit.cal_uncs_clean))
    #         self.denominator_unmasked[orbit.pixel_inds_clean] += np.divide(1, np.square(orbit.cal_uncs_clean), where=orbit.cal_uncs_clean != 0.0, out=np.zeros_like(orbit.cal_uncs_clean))
    #     return

    def normalize(self):
        self.fsm_masked.mapdata = np.divide(self.numerator_masked, self.denominator_masked, where=self.denominator_masked != 0.0, out=np.zeros_like(self.denominator_masked))
        self.unc_fsm_masked.mapdata = np.divide(1, self.denominator_masked, where=self.denominator_masked != 0.0, out=np.zeros_like(self.denominator_masked))
        self.fsm_unmasked.mapdata = np.divide(self.numerator_unmasked, self.denominator_unmasked,
                                            where=self.denominator_unmasked != 0.0,
                                            out=np.zeros_like(self.denominator_unmasked))
        self.unc_fsm_unmasked.mapdata = np.divide(1, self.denominator_unmasked, where=self.denominator_unmasked != 0.0,
                                                out=np.zeros_like(self.denominator_unmasked))

    def save_maps(self):
        self.fsm_masked.save_map()
        self.unc_fsm_masked.save_map()
        self.fsm_unmasked.save_map()
        self.unc_fsm_unmasked.save_map()


    # def load_zodi_orbit(self, orbit_num, pixel_inds):
    #     filename = f"/home/users/jguerraa/AME/cal_files/W3/zodi_map_cal_W{self.band}_{orbit_num}.fits"
    #     zodi_orbit = WISEMap(filename, self.band)
    #     zodi_orbit.read_data()
    #     pixels = np.zeros_like(zodi_orbit.mapdata)
    #     pixels[pixel_inds] = 1.0
    #     zodi_data = zodi_orbit.mapdata[pixels.astype(bool)]
    #     if not all(zodi_data.astype(bool)):
    #         print(f"Orbit {orbit_num} mismatch with zodi: zeros in zodi orbit")
    #     elif any(zodi_orbit.mapdata[~pixels.astype(bool)]):
    #         print(f"Orbit {orbit_num} mismatch with zodi: nonzeros outside zodi orbit")
    #     return zodi_data

    # def load_orbit_data(self, orbit_num):
    #     filename = f"/home/users/mberkeley/wisemapper/data/output_maps/w{self.band}/csv_files/band_w{self.band}_orbit_{orbit_num}_pixel_timestamps.csv"
    #     all_orbit_data = pd.read_csv(filename)
    #     orbit_data = np.array(all_orbit_data["pixel_value"])
    #     orbit_uncs = np.array(all_orbit_data["pixel_unc"])
    #     pixel_inds = np.array(all_orbit_data["hp_pixel_index"])
    #     return orbit_data, orbit_uncs, pixel_inds

    # @staticmethod
    # def weighted_mean_filter(array, weights, size):
    #     output = []
    #     step = int(size / 2)
    #     for p, px in enumerate(array):
    #         min_ind = max(0, p - step)
    #         max_ind = min(len(array), p + step)
    #         window = array[min_ind:max_ind].copy()
    #         weights_window = weights[min_ind:max_ind].copy()
    #         weights_window = weights_window / np.sum(weights_window)
    #         weighted_mean = np.average(window, weights=weights_window)
    #         output.append(weighted_mean)
    #     return np.array(output)


class IterativeFitter:

    def __init__(self, zodi_data, raw_data, raw_uncs):
        self.zodi_data = zodi_data
        self.raw_data = raw_data
        self.raw_uncs = raw_uncs

    @staticmethod
    def chi_sq(params, x_data, y_data, sigma):
        residual = x_data - ((y_data * params[0]) + params[1])
        weighted_residual = residual / (np.mean(sigma) ** 2)
        chi_sq = (np.sum(weighted_residual ** 2) / len(x_data)) if len(x_data) > 0 else 0.0
        return chi_sq

    def fit_to_zodi(self, orbit_data, zodi_data, orbit_uncs):
        init_gain = 1.0
        init_offset = 0.0
        popt = minimize(self.chi_sq, [init_gain, init_offset], args=(orbit_data, zodi_data, orbit_uncs),
                        method='Nelder-Mead').x
        gain, offset = popt
        return gain, offset

    def iterate_fit(self, n):
        i = 0
        data_to_fit = self.raw_data
        uncs_to_fit = self.raw_uncs
        if len(data_to_fit) > 0:
            while i < n:
                gain, offset = self.fit_to_zodi(data_to_fit, self.zodi_data, uncs_to_fit)
                # print("Gain:", gain)
                # print("Offset:", offset)
                data_to_fit = self.adjust_data(gain, offset, data_to_fit)
                i += 1
        else:
            gain = offset = 0.0
        return gain, offset


    def adjust_data(self, gain, offset, data):
        residual = ((data - offset)/gain) - self.zodi_data
        new_data = data - gain*residual
        return new_data

