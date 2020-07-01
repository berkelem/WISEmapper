from file_handler import WISEMap, HealpixMap
import numpy as np
from scipy.optimize import minimize
from scipy import stats
from fullskymapping import FullSkyMap
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
import pickle

class Orbit:

    coadd_map = None

    def __init__(self, orbit_num, band, mask):
        self.orbit_num = orbit_num
        self.band = band
        self.filename = f"/home/users/mberkeley/wisemapper/data/output_maps/w{self.band}/csv_files/" \
            f"band_w{self.band}_orbit_{self.orbit_num}_pixel_timestamps.csv"
        self.zodi_filename = f"/home/users/jguerraa/AME/cal_files/W3/zodi_map_cal_W{self.band}_{self.orbit_num}.fits"
        self.mask = mask
        self.mask_inds = np.arange(len(self.mask))[self.mask.astype(bool)]

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

        self.cal_data = None
        self.cal_uncs = None
        self.zs_data = None


    def load_orbit_data(self):
        all_orbit_data = pd.read_csv(self.filename)
        self.orbit_data = all_orbit_data["pixel_value"]
        self.orbit_uncs = all_orbit_data["pixel_unc"]
        self.pixel_inds = all_orbit_data["hp_pixel_index"]
        self.orbit_mjd_obs = all_orbit_data["pixel_mjd_obs"]
        self.mean_mjd_obs = np.mean(self.orbit_mjd_obs)
        self.std_mjd_obs = np.std(self.orbit_mjd_obs)

        # self.smooth_gain = np.ones_like(self.orbit_mjd_obs, dtype=float)
        # self.smooth_offset = np.zeros_like(self.orbit_mjd_obs)
        return

    @staticmethod
    def update_param(orig_param, smooth_param):
        first_val = np.where(smooth_param == list(filter(lambda x: x!=1.0, smooth_param))[0])[0][0]
        last_val = np.where(smooth_param == list(filter(lambda x: x!=1.0, smooth_param))[-1])[0][-1]
        orig_param[first_val:last_val+1] = smooth_param[first_val:last_val+1]
        if first_val != 0 and orig_param[first_val-1] != orig_param[first_val]:
            orig_param[:first_val] = orig_param[first_val]
        if last_val+1 != len(orig_param) and orig_param[last_val + 1] != orig_param[last_val]:
            orig_param[last_val + 1:] = orig_param[last_val]

        return orig_param

    def load_zodi_orbit_data(self):
        zodi_orbit = WISEMap(self.zodi_filename, self.band)
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
        self.entries_to_mask = [i for i in range(len(self.pixel_inds)) if self.pixel_inds[i] in self.mask_inds]
        self.pixel_inds_masked = np.array([self.pixel_inds[i] for i in range(len(self.pixel_inds)) if i not in self.entries_to_mask])
        self.orbit_data_masked = np.array([self.orbit_data[i] for i in range(len(self.orbit_data)) if i not in self.entries_to_mask])
        self.orbit_uncs_masked = np.array([self.orbit_uncs[i] for i in range(len(self.orbit_uncs)) if i not in self.entries_to_mask])
        self.orbit_mjd_masked = np.array([self.orbit_mjd_obs[i] for i in range(len(self.orbit_mjd_obs)) if i not in self.entries_to_mask])
        self.zodi_data_masked = np.array([self.zodi_data[i] for i in range(len(self.zodi_data)) if i not in self.entries_to_mask])
        return

    def fit(self):
        if self.coadd_map is not None:
            prev_itermap = self.coadd_map[self.pixel_inds]
            t_gal = prev_itermap * self.gain #self.smooth_gain

            t_gal_masked = np.array([t_gal[i] for i in range(len(t_gal)) if i not in self.entries_to_mask])

            self.orbit_data_masked -= t_gal_masked

            self.clean_data()

        orbit_fitter = IterativeFitter(self.zodi_data_masked, self.orbit_data_masked, self.orbit_uncs_masked)
        self.gain, self.offset = orbit_fitter.iterate_fit(1)

    def apply_fit(self):
        self.cal_data = (self.orbit_data - self.offset) / self.gain #self.smooth_offset) / self.smooth_gain
        self.cal_uncs = self.orbit_uncs / abs(self.gain) #np.abs(self.smooth_gain)

        self.zs_data = self.cal_data - self.zodi_data
        self.zs_data[self.zs_data < 0.0] = 0.0

        self.zs_data_masked = np.array([self.zs_data[i] for i in range(len(self.zs_data)) if i not in self.entries_to_mask])

    def apply_spline_fit(self, gain_spline, offset_spline):
        gains = gain_spline(self.orbit_mjd_obs)
        offsets = offset_spline(self.orbit_mjd_obs)
        self.cal_data = (self.orbit_data - offsets) / gains
        self.cal_uncs = self.orbit_uncs / abs(gains)

        self.zs_data = self.cal_data - self.zodi_data
        # self.zs_data[self.zs_data < 0.0] = 0.0

        self.zs_data_masked = np.array(
            [self.zs_data[i] for i in range(len(self.zs_data)) if i not in self.entries_to_mask])

        return

    def clean_data(self):
        z = np.abs(stats.zscore(self.zs_data_masked))
        mask = z > 1
        inds_to_mask = self.pixel_inds_masked[mask]
        self.mask_inds = np.append(self.mask_inds, inds_to_mask)
        self.apply_mask()
        return


class Coadder:

    def __init__(self, band):
        self.band = band
        self.iter = 0
        self.nside = 256
        self.npix = hp.nside2npix(self.nside)

        self.moon_stripe_mask = HealpixMap("/home/users/mberkeley/wisemapper/data/masks/stripe_mask_G.fits")
        self.moon_stripe_mask.read_data()
        self.moon_stripe_inds = np.arange(len(self.moon_stripe_mask.mapdata))[self.moon_stripe_mask.mapdata.astype(bool)]

        self.galaxy_mask = self.mask_galaxy()
        self.galaxy_mask_inds = np.arange(len(self.galaxy_mask))[self.galaxy_mask]

        # self.south_pole_mask = HealpixMap("/home/users/mberkeley/wisemapper/data/masks/south_pole_mask.fits")
        # self.south_pole_mask.read_data()
        # self.south_pole_mask_inds = np.arange(len(self.south_pole_mask.mapdata))[self.south_pole_mask.mapdata.astype(bool)]

        self.full_mask = self.moon_stripe_mask.mapdata.astype(bool) | self.galaxy_mask.astype(bool) #| ~self.south_pole_mask.mapdata.astype(bool)

        with open("gain_spline.pkl", "rb") as gain_spline_file:
            self.gain_spline = pickle.load(gain_spline_file)

        with open("offset_spline.pkl", "rb") as offset_spline_file:
            self.offset_spline = pickle.load(offset_spline_file)

        self.numerator = np.zeros(self.npix)
        self.denominator = np.zeros_like(self.numerator)

        self.gains = []
        self.offsets = []
        self.orbit_sizes = []

        self.all_gains = []
        self.all_offsets = []

        self.fsm = None
        self.unc_fsm = None

    def set_output_filenames(self):
        self.fsm = FullSkyMap(
            f"/home/users/mberkeley/wisemapper/data/output_maps/w3/fullskymap_band3_iter_{self.iter}.fits", self.nside)
        self.unc_fsm = FullSkyMap(
            f"/home/users/mberkeley/wisemapper/data/output_maps/w3/fullskymap_unc_band3_iter_{self.iter}.fits", self.nside)

    def mask_galaxy(self):
        """
        Remove 20% of the sky around the galactic plane where zodi is not the dominant foreground.
        :return:
        """
        theta, _ = hp.pix2ang(256, np.arange(self.npix))
        mask = (np.pi * 0.4 < theta) & (theta < 0.6 * np.pi)
        galaxy_mask = np.zeros_like(theta)
        galaxy_mask[mask] = 1.0
        return galaxy_mask.astype(bool)

    def run(self):
        num_orbits = 6323
        iterations = 1
        all_orbits = []

        for it in range(iterations):

            for i in range(num_orbits):
                print(f"Iteration {it}; Fitting orbit {i}")
                self.set_output_filenames()
                if it == 0:
                    orbit = Orbit(i, self.band, self.full_mask)
                    all_orbits.append(orbit)
                    orbit.load_orbit_data()
                    orbit.load_zodi_orbit_data()
                    orbit.apply_mask()
                else:
                    orbit = all_orbits[i]
                # orbit.fit()
                orbit.apply_spline_fit(self.gain_spline, self.offset_spline)
                self.add_orbit(orbit)

            # all_gains = np.array([orb.gain for orb in all_orbits])
            # all_offsets = np.array([orb.offset for orb in all_orbits])
            # all_orbit_sizes = np.array([len(orb.orbit_data) for orb in all_orbits])
            #
            # smoothed_gains = self.weighted_mean_filter(all_gains, all_orbit_sizes, 25)
            # smoothed_offsets = self.weighted_mean_filter(all_offsets, all_orbit_sizes, 25)
            #
            # l1, l2 = plt.plot(range(len(all_gains)), all_gains, 'r.', range(len(smoothed_gains)), smoothed_gains, 'b.')
            # plt.xlabel("Orbit number")
            # plt.ylabel("Gain")
            # plt.legend((l1, l2), ("Original fit", "w median filtered"))
            # plt.savefig("all_gains_iter_{}.png".format(it))
            # plt.close()
            #
            # l1, l2 = plt.plot(range(len(all_offsets)), all_offsets, 'r.', range(len(smoothed_offsets)), smoothed_offsets, 'b.')
            # plt.xlabel("Orbit number")
            # plt.ylabel("Offset")
            # plt.legend((l1, l2), ("Original fit", "w median filtered"))
            # plt.savefig("all_offsets_iter_{}.png".format(it))
            # plt.close()
            #
            # for i in range(num_orbits-1):
            #     orbit1 = all_orbits[i]
            #     orbit2 = all_orbits[i+1]
            #     orbit1.gain = smoothed_gains[i]
            #     orbit1.offset = smoothed_offsets[i]
            #     orbit2.gain = smoothed_gains[i+1]
            #     orbit2.offset = smoothed_offsets[i+1]
            #     sg1, sg2, sb1, sb2 = self.smooth_fit_params(orbit1, orbit2)
            #     orbit1.update_param(orbit1.smooth_gain, sg1)
            #     orbit2.update_param(orbit2.smooth_gain, sg2)
            #     orbit1.update_param(orbit1.smooth_offset, sb1)
            #     orbit2.update_param(orbit2.smooth_offset, sb2)

            # for i in range(num_orbits):
            #     orbit = all_orbits[i]
            #     orbit.apply_fit()
            #     self.add_orbit(orbit)

            self.normalize()
            self.save_maps()

            setattr(Orbit, "coadd_map", self.fsm.mapdata)
            self.iter += 1

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

        if len(orbit.orbit_uncs[orbit.orbit_uncs!=0.0]) > 0 and orbit.gain!=0.0:
            self.numerator[orbit.pixel_inds] += np.divide(orbit.zs_data, np.square(orbit.cal_uncs), where=orbit.cal_uncs != 0.0, out=np.zeros_like(orbit.cal_uncs))
            self.denominator[orbit.pixel_inds] += np.divide(1, np.square(orbit.cal_uncs), where=orbit.cal_uncs != 0.0, out=np.zeros_like(orbit.cal_uncs))
        return

    def normalize(self):
        self.fsm.mapdata = np.divide(self.numerator, self.denominator, where=self.denominator != 0.0, out=np.zeros_like(self.denominator))
        self.unc_fsm.mapdata = np.divide(1, self.denominator, where=self.denominator != 0.0, out=np.zeros_like(self.denominator))

    def save_maps(self):
        self.fsm.save_map()
        self.unc_fsm.save_map()


    def load_zodi_orbit(self, orbit_num, pixel_inds):
        filename = f"/home/users/jguerraa/AME/cal_files/W3/zodi_map_cal_W{self.band}_{orbit_num}.fits"
        zodi_orbit = WISEMap(filename, self.band)
        zodi_orbit.read_data()
        pixels = np.zeros_like(zodi_orbit.mapdata)
        pixels[pixel_inds] = 1.0
        zodi_data = zodi_orbit.mapdata[pixels.astype(bool)]
        if not all(zodi_data.astype(bool)):
            print(f"Orbit {orbit_num} mismatch with zodi: zeros in zodi orbit")
        elif any(zodi_orbit.mapdata[~pixels.astype(bool)]):
            print(f"Orbit {orbit_num} mismatch with zodi: nonzeros outside zodi orbit")
        return zodi_data

    def load_orbit_data(self, orbit_num):
        filename = f"/home/users/mberkeley/wisemapper/data/output_maps/w{self.band}/csv_files/band_w{self.band}_orbit_{orbit_num}_pixel_timestamps.csv"
        all_orbit_data = pd.read_csv(filename)
        orbit_data = all_orbit_data["pixel_value"]
        orbit_uncs = all_orbit_data["pixel_unc"]
        pixel_inds = all_orbit_data["hp_pixel_index"]
        return orbit_data, orbit_uncs, pixel_inds

    @staticmethod
    def weighted_mean_filter(array, weights, size):
        output = []
        step = int(size / 2)
        for p, px in enumerate(array):
            min_ind = max(0, p - step)
            max_ind = min(len(array), p + step)
            window = array[min_ind:max_ind].copy()
            weights_window = weights[min_ind:max_ind].copy()
            weights_window = weights_window / np.sum(weights_window)
            weighted_mean = np.average(window, weights=weights_window)
            output.append(weighted_mean)
        return np.array(output)


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


if __name__ == "__main__":
    coadd_map = Coadder(3)
    coadd_map.run()
