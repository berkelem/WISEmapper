from file_handler import WISEMap, HealpixMap
import numpy as np
from scipy.optimize import minimize
from fullskymapping import FullSkyMap
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt

class Coadder:

    def __init__(self, band):
        self.band = band
        self.fsm = FullSkyMap(
            f"/home/users/mberkeley/wisemapper/data/output_maps/w3/fullskymap_sim.fits", 256)
        self.unc_fsm = FullSkyMap(
            f"/home/users/mberkeley/wisemapper/data/output_maps/w3/fullskymap_unc_sim.fits", 256)

        self.moon_stripe_mask = HealpixMap("/home/users/mberkeley/wisemapper/data/masks/stripe_mask_G.fits")
        self.moon_stripe_mask.read_data()
        self.moon_stripe_inds = np.arange(len(self.moon_stripe_mask.mapdata))[self.moon_stripe_mask.mapdata.astype(bool)]

        self.galaxy_mask = self.mask_galaxy()
        self.galaxy_mask_inds = np.arange(len(self.galaxy_mask))[self.galaxy_mask]

        self.numerator = np.zeros_like(self.fsm.mapdata)
        self.denominator = np.zeros_like(self.fsm.mapdata)

    def mask_galaxy(self):
        """
        Remove 20% of the sky around the galactic plane where zodi is not the dominant foreground.
        :return:
        """
        npix = self.fsm.npix
        theta, _ = hp.pix2ang(256, np.arange(npix))
        mask = (np.pi * 0.4 < theta) & (theta < 0.6 * np.pi)
        galaxy_mask = np.zeros_like(theta)
        galaxy_mask[mask] = 1.0
        return galaxy_mask.astype(bool)

    def run(self):
        for i in range(1):
            print(f"Adding orbit {i}")
            self.add_file(i)
        self.normalize()
        self.save_maps()

    def add_file(self, orbit_num):
        orbit_data, orbit_uncs, pixel_inds = self.load_orbit_data(orbit_num)
        entries_to_mask = np.array([i for i in range(len(pixel_inds)) if
                           pixel_inds[i] in self.moon_stripe_inds or pixel_inds[i] in self.galaxy_mask_inds])
        good_pixels = np.array([pixel_inds[p] for p in range(len(pixel_inds)) if p not in entries_to_mask])

        zodi_data = self.load_zodi_orbit(orbit_num, pixel_inds)
        zodi_data_masked = np.array([zodi_data[i] for i in range(len(zodi_data)) if i not in entries_to_mask])

        sim_galaxy = 10 * np.sin(good_pixels/len(good_pixels))
        plt.plot(range(len(sim_galaxy)), sim_galaxy, 'r.')
        plt.xlabel("pixel id")
        plt.ylabel("signal")
        plt.title("Simulated galaxy signal")
        plt.savefig("sim_galaxy.png")
        plt.close()

        sim_data = zodi_data_masked + sim_galaxy
        plt.plot(range(len(sim_galaxy)), sim_data, 'r.')
        plt.xlabel("pixel id")
        plt.ylabel("signal")
        plt.title("Simulated sky signal (galaxy + zodi)")
        plt.savefig("sim_data.png")
        plt.close()


        sim_uncs = np.ones_like(sim_data)*0.001

        orbit_fitter = IterativeFitter(zodi_data_masked, sim_data, sim_uncs)
        gain, offset = orbit_fitter.iterate_fit(10)

        cal_data = (sim_data - offset)/gain
        cal_uncs = sim_uncs / abs(gain)

        zs_data = cal_data - zodi_data

        self.numerator[pixel_inds] += np.divide(zs_data, np.square(cal_uncs), where=cal_uncs != 0.0, out=np.zeros_like(cal_uncs))
        self.denominator[pixel_inds] += np.divide(1, np.square(cal_uncs), where=cal_uncs != 0.0, out=np.zeros_like(cal_uncs))

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

    def normalize(self):
        self.fsm.mapdata = np.divide(self.numerator, self.denominator, where=self.denominator != 0.0, out=np.zeros_like(self.denominator))
        self.unc_fsm.mapdata = np.divide(1, self.denominator, where=self.denominator != 0.0, out=np.zeros_like(self.denominator))

    def save_maps(self):
        self.fsm.save_map()
        self.unc_fsm.save_map()



class IterativeFitter:

    def __init__(self, zodi_data, raw_data, raw_uncs):
        self.zodi_data = zodi_data
        self.raw_data = raw_data
        self.raw_uncs = raw_uncs

    @staticmethod
    def chi_sq(params, x_data, y_data, sigma):
        residual = x_data - ((y_data*params[0]) + params[1])
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
        while i < n:
            plt.plot(range(len(data_to_fit)), data_to_fit, 'r.')
            plt.xlabel("pixel id")
            plt.ylabel("signal")
            plt.title(f"adjusted sky signal; iter {i}")
            plt.savefig(f"adjusted_signal_iter_{i}.png")
            plt.close()
            gain, offset = self.fit_to_zodi(data_to_fit, self.zodi_data, uncs_to_fit)
            self.plot_fit(i, data_to_fit, self.zodi_data, gain, offset)
            print("Gain:", gain)
            print("Offset:", offset)
            data_to_fit = self.adjust_data(gain, offset, data_to_fit, i)
            i += 1
        return gain, offset

    @staticmethod
    def plot_fit(i, raw_data, zodi_data, gain, offset):

        plt.plot(np.arange(len(zodi_data)), zodi_data*gain + offset, 'b.', np.arange(len(raw_data)), raw_data, 'r.')
        plt.savefig(f"/home/users/mberkeley/wisemapper/data/output_maps/w3/calibration_fit_simdata_iter_{i}.png")
        plt.close()


    def adjust_data(self, gain, offset, data, i):
        residual = data - (self.zodi_data*gain + offset)
        plt.plot(range(len(residual)), residual, 'r.')
        plt.xlabel("pixel id")
        plt.ylabel("residual")
        plt.title("Raw data - (zodi * gain + offset)")
        plt.savefig(f"residual_iter_{i}.png")
        plt.close()
        new_data = self.raw_data - residual
        return new_data

if __name__ == "__main__":
    coadd_map = Coadder(3)
    coadd_map.run()