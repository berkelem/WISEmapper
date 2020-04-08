from file_handler import WISEMap
import numpy as np
from scipy.optimize import minimize
from fullskymapping import FullSkyMap
import pandas as pd

class Coadder:

    def __init__(self, band):
        self.band = band
        self.fsm = FullSkyMap(
            f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/fullskymap_band3.fits", 256)
        self.unc_fsm = FullSkyMap(
            f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/fullskymap_unc_band3.fits", 256)
        self.numerator = np.zeros_like(self.fsm.mapdata)
        self.denominator = np.zeros_like(self.fsm.mapdata)

    def run(self):
        for i in range(6323):
            self.add_file(i)
        self.normalize()
        self.save_maps()

    def add_file(self, orbit_num):
        orbit_data, orbit_uncs, pixel_inds = self.load_orbit_data(orbit_num)
        zodi_data = self.load_zodi_orbit(orbit_num, pixel_inds)
        gain, offset = self.fit_to_zodi(orbit_data, zodi_data, orbit_uncs)

        cal_data = gain * orbit_data + offset
        cal_uncs = abs(gain) * orbit_uncs

        self.numerator[pixel_inds] += np.divide(cal_data, np.square(cal_uncs), where=cal_uncs != 0.0, out=np.zeros_like(cal_uncs))
        self.denominator[pixel_inds] += np.divide(1, np.square(cal_uncs), where=cal_uncs != 0.0, out=np.zeros_like(cal_uncs))

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
        zodi_data = zodi_orbit.mapdata[pixel_inds]
        return zodi_data

    def load_orbit_data(self, orbit_num):
        filename = f"/home/users/mberkeley/wisemapper/data/output_maps/w{self.band}/csv_files/band_w{self.band}_orbit_{orbit_num}_pixel_timestamps.csv"
        all_orbit_data = pd.read_csv(filename)
        orbit_data = all_orbit_data["pixel_value"]
        orbit_uncs = all_orbit_data["pixel_unc"]
        pixel_inds = all_orbit_data["hp_pixel_index"]
        return orbit_data, orbit_uncs, pixel_inds

    @staticmethod
    def chi_sq(params, x_data, y_data, sigma):
        residual = (y_data - (params[0] * x_data + params[1]))
        weighted_residual = residual / (np.mean(sigma) ** 2)
        chi_sq = (np.ma.sum(weighted_residual ** 2) / len(x_data)) if len(x_data) > 0 else 0.0
        return chi_sq

    def fit_to_zodi(self, orbit_data, zodi_data, orbit_uncs):
        init_gain = 1.0
        init_offset = 0.0
        popt = minimize(self.chi_sq, [init_gain, init_offset], args=(orbit_data, zodi_data, orbit_uncs), method='Nelder-Mead').x
        gain, offset = popt
        return gain, offset