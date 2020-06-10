from file_handler import WISEMap, HealpixMap
import numpy as np
from scipy.optimize import minimize
from fullskymapping import FullSkyMap
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt

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
        self.zodi_data = None

        self.orbit_data_masked = None
        self.orbit_uncs_masked = None
        self.zodi_data_masked = None

        self.gain = 1.0
        self.offset = 0.0


    def load_orbit_data(self):
        all_orbit_data = pd.read_csv(self.filename)
        self.orbit_data = all_orbit_data["pixel_value"]
        self.orbit_uncs = all_orbit_data["pixel_unc"]
        self.pixel_inds = all_orbit_data["hp_pixel_index"]
        return

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
        self.zodi_data_masked = np.array([self.zodi_data[i] for i in range(len(self.zodi_data)) if i not in self.entries_to_mask])
        return

    def fit(self):
        if self.coadd_map:
            prev_itermap = self.coadd_map[self.pixel_inds]
            prev_itermap_masked = np.array([prev_itermap[i] for i in range(len(prev_itermap)) if i not in self.entries_to_mask])

            t_gal = prev_itermap_masked * self.gain
            self.orbit_data_masked -= t_gal

        orbit_fitter = IterativeFitter(self.zodi_data_masked, self.orbit_data_masked, self.orbit_uncs_masked)
        self.gain, self.offset = orbit_fitter.iterate_fit(1)


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

        self.south_pole_mask = HealpixMap("/home/users/mberkeley/wisemapper/data/masks/south_pole_mask.fits")
        self.south_pole_mask.read_data()
        self.south_pole_mask_inds = np.arange(len(self.south_pole_mask.mapdata))[self.south_pole_mask.mapdata.astype(bool)]

        self.full_mask = self.moon_stripe_mask.mapdata.astype(bool) | self.galaxy_mask.astype(bool) | ~self.south_pole_mask.mapdata.astype(bool)

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
        num_orbits = 10
        iterations = 10
        all_orbits = []

        for it in range(iterations):

            for i in range(num_orbits):
                print(f"Fitting orbit {i}")
                self.set_output_filenames()
                orbit = Orbit(i, self.band, self.full_mask)
                all_orbits.append(orbit)
                orbit.load_orbit_data()
                orbit.load_zodi_orbit_data()
                orbit.apply_mask()
                orbit.fit()
                print(f"Adding orbit {j}")
                self.add_orbit(orbit)

            self.normalize()
            self.save_maps()

            setattr(Orbit, "coadd_map", self.fsm)
            self.iter += 1


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

    def fit_orbit(self, k):
        if self.iter == 0:
            self.fit_initial_orbit(k)
        else:
            self.fit_adjusted_orbit(k)

    def fit_initial_orbit(self, orbit_num):
        orbit_data, orbit_uncs, pixel_inds = self.load_orbit_data(orbit_num)
        test1 = [i for i in range(len(pixel_inds)) if pixel_inds[i] in self.moon_stripe_inds]
        test2 = [i for i in range(len(pixel_inds)) if pixel_inds[i] in self.moon_stripe_inds or pixel_inds[i] in self.galaxy_mask_inds]
        test3 = [i for i in range(len(pixel_inds)) if pixel_inds[i] in self.moon_stripe_inds or pixel_inds[i] in self.galaxy_mask_inds or pixel_inds[i] not in self.south_pole_mask_inds]
        entries_to_mask = [i for i in range(len(pixel_inds)) if pixel_inds[i] in self.moon_stripe_inds or pixel_inds[i] in self.galaxy_mask_inds or pixel_inds[i] not in self.south_pole_mask_inds]
        orbit_data_masked = np.array([orbit_data[i] for i in range(len(orbit_data)) if i not in entries_to_mask])
        orbit_uncs_masked = np.array([orbit_uncs[i] for i in range(len(orbit_uncs)) if i not in entries_to_mask])

        zodi_data = self.load_zodi_orbit(orbit_num, pixel_inds)
        zodi_data_masked = np.array([zodi_data[i] for i in range(len(zodi_data)) if i not in entries_to_mask])

        orbit_fitter = IterativeFitter(zodi_data_masked, orbit_data_masked, orbit_uncs_masked)
        gain, offset = orbit_fitter.iterate_fit(1)

        l1, l2 = plt.plot(np.arange(len(orbit_data_masked)), (orbit_data_masked - offset)/gain, 'r.', np.arange(len(orbit_data_masked)), zodi_data_masked, 'b.')
        plt.xlabel("pixel id")
        plt.ylabel("signal")
        plt.legend((l1, l2), ("Calibrated data", "zodi template"))
        plt.savefig(f"orbit_{orbit_num}_fit_{self.iter}.png")
        plt.close()

        diff = zodi_data_masked - (orbit_data_masked - offset)/gain
        plt.plot(np.arange(len(orbit_data_masked)), diff, 'r.')
        plt.xlabel("pixel id")
        plt.ylabel("signal")
        plt.savefig(f"orbit_{orbit_num}_diff_{self.iter}.png")
        plt.close()

        zs = (orbit_data - offset)/gain - zodi_data
        plt.plot(np.arange(len(orbit_data)), zs, 'r.')
        plt.xlabel("pixel id")
        plt.ylabel("signal")
        plt.savefig(f"orbit_{orbit_num}_zs_{self.iter}.png")
        plt.close()

        self.gains[orbit_num] = gain
        self.offsets[orbit_num] = offset
        self.orbit_sizes[orbit_num] = len(zodi_data_masked)

    def fit_adjusted_orbit(self, orbit_num):
        orbit_data, orbit_uncs, pixel_inds = self.load_orbit_data(orbit_num)
        entries_to_mask = [i for i in range(len(pixel_inds)) if
                           pixel_inds[i] in self.moon_stripe_inds or pixel_inds[i] in self.galaxy_mask_inds or pixel_inds[i] not in self.south_pole_mask_inds]
        orbit_data_masked = np.array([orbit_data[i] for i in range(len(orbit_data)) if i not in entries_to_mask])
        orbit_uncs_masked = np.array([orbit_uncs[i] for i in range(len(orbit_uncs)) if i not in entries_to_mask])

        zodi_data = self.load_zodi_orbit(orbit_num, pixel_inds)
        zodi_data_masked = np.array([zodi_data[i] for i in range(len(zodi_data)) if i not in entries_to_mask])

        prev_itermap = self.fsm_prev.mapdata[pixel_inds]
        prev_itermap_masked = np.array([prev_itermap[i] for i in range(len(prev_itermap)) if i not in entries_to_mask])

        t_gal = prev_itermap_masked * self.gains[orbit_num]
        orbit_data_adj = orbit_data_masked - t_gal

        orbit_fitter = IterativeFitter(zodi_data_masked, orbit_data_adj, orbit_uncs_masked)
        gain, offset = orbit_fitter.iterate_fit(1)

        l1, l2 = plt.plot(np.arange(len(orbit_data_masked)), orbit_data_masked, 'r.',
                          np.arange(len(orbit_data_masked)), orbit_data_adj, 'b.')
        plt.xlabel("pixel id")
        plt.ylabel("signal")
        plt.legend((l1, l2), ("Original data", "Adjusted data"))
        plt.savefig(f"orbit_{orbit_num}_adj_{self.iter}.png")
        plt.close()

        l1, l2 = plt.plot(np.arange(len(orbit_data_adj)), (orbit_data_adj - offset) / gain, 'r.',
                          np.arange(len(orbit_data_adj)), zodi_data_masked, 'b.')
        plt.xlabel("pixel id")
        plt.ylabel("signal")
        plt.legend((l1, l2), ("Calibrated data", "zodi template"))
        plt.savefig(f"orbit_{orbit_num}_fit_{self.iter}.png")
        plt.close()

        diff = zodi_data_masked - (orbit_data_adj - offset) / gain
        plt.plot(np.arange(len(orbit_data_adj)), diff, 'r.')
        plt.xlabel("pixel id")
        plt.ylabel("signal")
        plt.savefig(f"orbit_{orbit_num}_diff_{self.iter}.png")
        plt.close()

        plt.plot(np.arange(len(orbit_data_adj)), t_gal, 'r.')
        plt.xlabel("pixel id")
        plt.ylabel("signal")
        plt.savefig(f"orbit_{orbit_num}_tgal_{self.iter}.png")
        plt.close()

        zs = (orbit_data - offset) / gain - zodi_data
        plt.plot(np.arange(len(orbit_data)), zs, 'r.')
        plt.xlabel("pixel id")
        plt.ylabel("signal")
        plt.savefig(f"orbit_{orbit_num}_zs_{self.iter}.png")
        plt.close()

        self.gains[orbit_num] = gain
        self.offsets[orbit_num] = offset
        self.orbit_sizes[orbit_num] = len(zodi_data_masked)


    def add_orbit(self, orbit):

        if len(orbit.orbit_uncs[orbit.orbit_uncs!=0.0]) > 0 and orbit.gain!=0.0:

            cal_data = (orbit.orbit_data_masked - orbit.offset)/orbit.gain
            cal_uncs = orbit.orbit_uncs_masked / abs(orbit.gain)

            zs_data = cal_data - orbit.zodi_data_masked
            zs_data[zs_data < 0.0] = 0.0

            self.numerator[orbit.pixel_inds_masked] += np.divide(zs_data, np.square(cal_uncs), where=cal_uncs != 0.0, out=np.zeros_like(cal_uncs))
            self.denominator[orbit.pixel_inds_masked] += np.divide(1, np.square(cal_uncs), where=cal_uncs != 0.0, out=np.zeros_like(cal_uncs))
        return

    @staticmethod
    def plot_fit(i, orbit_data, zodi_data):
        plt.plot(np.arange(len(orbit_data)), orbit_data, 'r.', np.arange(len(orbit_data)), zodi_data, 'b.')
        plt.savefig(f"/home/users/mberkeley/wisemapper/data/output_maps/w3/calibration_fit_orbit_{i}.png")
        plt.close()

    @staticmethod
    def plot_fit_improvement(i, orbit_data, zodi_data, gain1, offset1, gain2, offset2):
        cal_data1 = (orbit_data - offset1)/gain1
        cal_data2 = (orbit_data - offset2)/gain2
        l1, = plt.plot(np.arange(len(orbit_data)), zodi_data, 'k.', ms=0.7, alpha=1)
        l2, = plt.plot(np.arange(len(orbit_data)), cal_data1, 'r.', ms=0.7, alpha=1)
        l3, = plt.plot( np.arange(len(orbit_data)), cal_data2, 'b.', ms=0.7, alpha=1)
        plt.legend((l1,l2,l3), ("zodi template", "1 iteration", "10 iterations"), markerscale=10)
        plt.title("Orbit {}: gain ratio: {}; offset ratio: {}".format(i, gain2/gain1, offset2/offset1))
        plt.savefig(f"/home/users/mberkeley/wisemapper/data/output_maps/w3/calibration_iterfit_orbit_{i}.png")
        plt.close()

    def smooth_fit_params(self, window_size):
        smooth_gains = self.weighted_mean_filter(self.gains, self.orbit_sizes, window_size)
        smooth_offsets = self.weighted_mean_filter(self.offsets, self.orbit_sizes, window_size)
        return smooth_gains, smooth_offsets

    @staticmethod
    def weighted_mean_filter_wraparound(array, weights, size):
        output = []
        for p, px in enumerate(array):
            window = np.ma.zeros(size)
            weights_window = np.zeros(size)
            step = int(size / 2)
            if p - step < 0:
                undershoot = step - p
                window[:undershoot] = array[-undershoot:]
                window[undershoot:step] = array[:p]
                weights_window[:undershoot] = weights[-undershoot:]
                weights_window[undershoot:step] = weights[:p]
            else:
                window[:step] = array[p - step:p]
                weights_window[:step] = weights[p - step:p]

            if p + step + 1 > len(array):
                overshoot = p + step + 1 - len(array)
                array_roll = np.roll(array, overshoot)
                weights_roll = np.roll(weights, overshoot)
                window[step:] = array_roll[-(size - step):]
                weights_window[step:] = weights_roll[-(size - step):]
            else:
                window[step:] = array[p:p + step + 1]
                weights_window[step:] = weights[p:p + step + 1]

            weights_window /= np.sum(weights_window)
            weighted_mean = np.average(window, weights=weights_window)
            output.append(weighted_mean)
        return np.array(output)

    @staticmethod
    def weighted_mean_filter(array, weights, size):
        output = []
        step = int(size / 2)
        for p, px in enumerate(array):
            min_ind = max(0, p - step)
            max_ind = min(len(array), p + step)
            window = array[min_ind:max_ind].copy()
            weights_window = weights[min_ind:max_ind].copy()
            weights_window /= np.sum(weights_window)
            weighted_mean = np.average(window, weights=weights_window)
            output.append(weighted_mean)
        return np.array(output)


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
