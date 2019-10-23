from file_handler import ZodiMap
import healpy as hp
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import math
import os
import pickle

class ZodiCalibrator:

    zodi_maps = {1: 'kelsall_model_wise_scan_lam_3.5_v3.fits',
                 2: 'kelsall_model_wise_scan_lam_4.9_v3.fits',
                 3: 'kelsall_model_wise_scan_lam_12_v3.fits',
                 4: 'kelsall_model_wise_scan_lam_25_v3.fits'}

    def __init__(self, band):
        self.band = band
        self.kelsall_map = ZodiMap(f'/home/users/mberkeley/wisemapper/data/kelsall_maps/{self.zodi_maps[self.band]}', self.band)
        self.kelsall_map.read_data()
        self.nside = None

    def calibrate(self, raw_map, unc_map):
        """
        This is the main function call that takes a raw map and calibrates it to the Kelsall zodi map.
        :param raw_map: Raw WISE data in HEALpix form.
        :param unc_map: The corresponding Kelsall zodi map.
        :return:
        """
        # Ensure the Zodi map has the same resolution as the WISE map
        self.nside = hp.npix2nside(len(raw_map))
        self.kelsall_map.set_resolution(self.nside)

        # Create a mask to remove nonzero pixels and pixels in the galactic plane
        nonzero_mask = raw_map != 0.0
        galaxy_mask = self.mask_galaxy()
        full_mask = nonzero_mask & galaxy_mask
        raw_vals = raw_map[full_mask]
        unc_vals = unc_map[full_mask]
        cal_vals = self.kelsall_map.mapdata[full_mask]

        # Remove outliers using a z-score cutoff
        clean_mask = ~ self.clean_with_z_score(raw_vals, cal_vals, threshold=1)
        self.raw_vals = raw_vals[clean_mask]
        self.unc_vals = unc_vals[clean_mask]
        self.cal_vals = cal_vals[clean_mask]

        # Perform least squares fit
        self.popt = self.fit()

        # Apply fit parameters to raw map and return as a calibrated map
        calib_map = np.zeros_like(raw_map)
        calib_uncmap = np.zeros_like(raw_map)
        calib_map[nonzero_mask] = raw_map[nonzero_mask]*self.popt[0] + self.popt[1]
        calib_uncmap[nonzero_mask] = unc_map[nonzero_mask]*abs(self.popt[0])
        return calib_map, calib_uncmap

    def clean_with_z_score(self, raw_vals, cal_vals, threshold=3):
        data = cal_vals/raw_vals
        z = np.abs(stats.zscore(data))
        mask = z > threshold
        return mask


    def mask_galaxy(self):
        """
        Remove 20% of the sky around the galactic plane where zodi is not the dominant foreground.
        :return:
        """
        npix = hp.nside2npix(self.nside)
        theta, _ = hp.pix2ang(self.nside, np.arange(npix))
        mask = (np.pi * 0.4 < theta) & (theta < 0.6 * np.pi)
        galaxy_mask = np.ones_like(theta)
        galaxy_mask[mask] = 0.0
        return galaxy_mask.astype(bool)

    @staticmethod
    def line(x, m, c):
        return m * x + c

    def fit(self):
        """
        Straightforward least squares curve fitting.
        :return:
        """
        x, y = self.raw_vals, self.cal_vals
        popt, _ = curve_fit(self.line, x, y, sigma=self.unc_vals)
        return popt

    # The following methods were just different variations on the fit that I used when fitting entire days.
    # They shouldn't be necessary for individual orbits.
    @staticmethod
    def ceiling(data, window=5):
        ceildata = np.array(
            [max(data[max(0, i - math.floor(window / 2)):min(len(data), i + math.ceil(window / 2))]) for i in
             range(len(data))])
        return ceildata

    def binmax(self, nbins=3):
        minind = np.where(self.cal_vals == min(self.cal_vals))
        maxind = np.where(self.cal_vals == max(self.cal_vals))
        minrange = float(self.raw_vals[minind][0])
        maxrange = float(self.raw_vals[maxind][0])
        bin_size = int(round((maxrange - minrange) / nbins))
        nbins = int(round((maxrange - minrange) / bin_size))
        pts_x = []
        pts_y = []
        for n in range(nbins):
            binmask = ((minrange + n * bin_size) <= self.raw_vals) & (self.raw_vals < (minrange + (n + 1) * bin_size))
            if len(self.cal_vals[binmask]) > 0:
                sample_point_ind = np.where(self.cal_vals == max(self.cal_vals[binmask]))
                pt_x = self.raw_vals[sample_point_ind]
                pt_y = self.cal_vals[sample_point_ind]
                pts_x.append(float(pt_x[0]))
                pts_y.append(float(pt_y[0]))

        pts_x = np.array(pts_x)
        pts_y = np.array(pts_y)
        return np.array(pts_x), np.array(pts_y)

    def plot(self, label, path):
        plt.plot(self.raw_vals, self.cal_vals, 'r.', self.raw_vals, self.line(self.raw_vals, *self.popt), 'b')
        plt.xlabel('Raw values (DN)')
        plt.ylabel('Calibrated values (MJy/sr)')
        plt.title(f'Calibration fit for day {label}')
        plt.savefig(f'{path}band{self.band}_orbit{label}.png')
        plt.close()


class SmoothFit:

    #  1) load each individual day file and get number of nonzero pixels for weight;
    #  2) implement smoothing algorithm;
    #  3) Obtain new gain/offset values for each day;
    #  4) Apply reversal and recalibration to each day;
    #  5) Recombine all days using new cleaning method (z <= 1)

    def __init__(self, band, basepath="./"):
        self.band = band
        self.basepath = basepath
        self.popt_file = os.path.join(self.basepath, f"popt_w{self.band}.pkl")
        self.bad_fit_days = [26, 27, 28, 30, 46, 47, 48, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                             67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 85, 86, 87, 88, 95, 98, 99, 100,
                             102, 108, 112, 113, 114, 115, 116, 117, 125, 126, 127, 128, 129, 130, 131, 132, 136, 146,
                             147, 150, 158, 176, 177]


    def load_fit_vals(self):
        with open(self.popt_file, "rb") as f:
            gain, offset = pickle.load(f)
        return gain, offset

    def plot_smoothfit_vals(self, orig_gain, adj_gain, orig_offset, adj_offset):
        plt.plot(orig_gain, 'r.', adj_gain, 'b')
        plt.xlabel("Day")
        plt.ylabel("Fitted gain")
        plt.title("Mean filter; gain smoothing")
        plt.savefig(os.path.join(self.basepath, f"w{self.band}_gain_mean_filter_interp.png"))
        plt.close()

        plt.plot(orig_offset, 'r.', adj_offset, 'b')
        plt.xlabel("Day")
        plt.ylabel("Fitted gain")
        plt.title("Mean filter; offset smoothing")
        plt.savefig(os.path.join(self.basepath, f"w{self.band}_offset_mean_filter_interp.png"))
        plt.close()
        return

    @staticmethod
    def median_filter(array, size):
        output = []
        for p, px in enumerate(array):
            window = np.zeros(size)
            step = int(size / 2)
            if p - step < 0:
                undershoot = step - p
                window[:undershoot] = array[-undershoot:]
                window[undershoot:step] = array[:p]
            else:
                window[:step] = array[p - step:p]

            if p + step + 1 > len(array):
                overshoot = p + step + 1 - len(array)
                array_roll = np.roll(array, overshoot)
                window[step:] = array_roll[-(size - step):]
            else:
                window[step:] = array[p:p + step + 1]

            window_median = np.median(window)
            output.append(window_median)
        return np.array(output)

    @staticmethod
    def mean_filter(array, size):
        output = []
        for p, px in enumerate(array):
            window = np.ma.zeros(size)
            step = int(size / 2)
            if p - step < 0:
                undershoot = step - p
                window[:undershoot] = array[-undershoot:]
                window[undershoot:step] = array[:p]
            else:
                window[:step] = array[p - step:p]

            if p + step + 1 > len(array):
                overshoot = p + step + 1 - len(array)
                array_roll = np.roll(array, overshoot)
                window[step:] = array_roll[-(size - step):]
            else:
                window[step:] = array[p:p + step + 1]

            window_mean = np.ma.mean(window)
            output.append(window_mean)
        return np.array(output)


    @staticmethod
    def weighted_median_filter(array, weights, size):
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
            weighted_median_index = np.searchsorted(np.cumsum(weights_window), 0.5)
            weighted_median = window[weighted_median_index]
            output.append(weighted_median)
        return np.array(output)

    @staticmethod
    def weighted_mean_filter(array, weights, size):
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
    def adjust_calibration(data, orig_gain, adj_gain, orig_offset, adj_offset):
        """Apply adjustment across entire map and restore original zeros afterwards"""
        zeros = data == 0.0
        adj_data = (((data) - orig_offset)/orig_gain) * adj_gain + adj_offset
        adj_data[zeros] = 0.0
        return adj_data
