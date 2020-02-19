from file_handler import ZodiMap, HealpixMap
import healpy as hp
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit, minimize
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
        self.moon_stripe_mask = HealpixMap("/home/users/mberkeley/wisemapper/data/masks/stripe_mask_G.fits")
        self.moon_stripe_mask.read_data()
        self.pole_region = HealpixMap("/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/mask_map_70.fits")
        self.pole_region.read_data()
        self.pole_region_mask = self.pole_region.mapdata.astype(bool)
        self.nside = None

    def mask_stripes(self, map_data, map_unc):
        nonzero_mask = map_data != 0.0
        moon_mask = ~self.moon_stripe_mask.mapdata.astype(bool)
        full_mask = nonzero_mask & moon_mask
        map_data_filtered = np.ma.array(map_data, mask=full_mask, fill_value=0.0).filled()
        map_unc_filtered = np.ma.array(map_unc, mask=full_mask, fill_value=0.0).filled()
        return map_data_filtered, map_unc_filtered

    def mask_except_poles(self, map_data, map_unc):
        nonzero_mask = map_data != 0.0
        moon_mask = ~self.moon_stripe_mask.mapdata.astype(bool)
        pole_mask = self.pole_region_mask
        full_mask = nonzero_mask & moon_mask & pole_mask
        map_data_filtered = np.ma.array(map_data, mask=full_mask, fill_value=0.0).filled()
        map_unc_filtered = np.ma.array(map_unc, mask=full_mask, fill_value=0.0).filled()
        return map_data_filtered, map_unc_filtered

    def offset_fit_iteration(self, all_orbits_data, all_orbits_unc, gain):
        moon_mask = ~self.moon_stripe_mask.mapdata.astype(bool)
        pole_mask = self.pole_region_mask
        combined_region_mask = moon_mask & pole_mask
        selected_region_data = all_orbits_data[combined_region_mask]
        selected_region_unc = all_orbits_unc[combined_region_mask]
        calibrated_data = gain * selected_region_data + offset

    @staticmethod
    def calibrated_variance(offset, data, gain):
        cal_data = gain * data + offset
        variance = np.var(cal_data, axis=0)
        return variance

    def fit_variance(self, init_offset, data, gain):
        popt = minimize(self.calibrated_variance, init_offset, args=(data, gain), method='Nelder-Mead').x
        return popt

    def calibrate(self, all_orbits_data, all_orbits_unc):
        self.nside = hp.npix2nside(all_orbits_data.shape[1])
        self.kelsall_map.set_resolution(self.nside)
        fitted_offset = self.fit_variance(0.0, all_orbits_data[:, 0], 1.0)


    def calibrate_stripe(self, raw_map, unc_map, id=None):
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
        # hp.fitsfunc.write_map("nonzero_mask.fits", nonzero_mask, coord="G", overwrite=True)
        # galaxy_mask = self.mask_galaxy()
        # hp.fitsfunc.write_map("g_mask.fits", galaxy_mask, coord="G", overwrite=True)
        moon_mask = ~self.moon_stripe_mask.mapdata.astype(bool)
        # hp.fitsfunc.write_map("m_mask.fits", moon_mask, coord="G", overwrite=True)
        pole_mask = self.pole_region_mask
        # hp.fitsfunc.write_map("p_mask.fits", pole_mask, coord="G", overwrite=True)
        sky_mask_for_gain = nonzero_mask & moon_mask

        full_mask_for_offset = nonzero_mask & moon_mask & pole_mask
        # hp.fitsfunc.write_map("fullmask.fits", full_mask, coord="G", overwrite=True)
        raw_vals_for_gain = raw_map[sky_mask_for_gain]
        unc_vals_for_gain = unc_map[sky_mask_for_gain]
        cal_vals_for_gain = self.kelsall_map.mapdata[sky_mask_for_gain]

        raw_vals_for_offset = raw_map[full_mask_for_offset]
        unc_vals_for_offset = unc_map[full_mask_for_offset]
        cal_vals_for_offset = self.kelsall_map.mapdata[full_mask_for_offset]

        # raw_vals = raw_map[full_mask]
        # unc_vals = unc_map[full_mask]
        # cal_vals = self.kelsall_map.mapdata[full_mask]

        # Remove outliers using a z-score cutoff
        clean_mask_for_gain = ~ self.clean_with_z_score(raw_vals_for_gain, cal_vals_for_gain, threshold=3)
        clean_mask_for_offset = ~ self.clean_with_z_score(raw_vals_for_offset, cal_vals_for_offset, threshold=3)
        self.raw_vals_for_gain = raw_vals_for_gain[clean_mask_for_gain]
        self.unc_vals_for_gain = unc_vals_for_gain[clean_mask_for_gain]
        self.cal_vals_for_gain = cal_vals_for_gain[clean_mask_for_gain]

        self.raw_vals_for_offset = raw_vals_for_offset[clean_mask_for_offset]
        self.unc_vals_for_offset = unc_vals_for_offset[clean_mask_for_offset]
        self.cal_vals_for_offset = cal_vals_for_offset[clean_mask_for_offset]

        # self.popt = self.fit()
        # mean_gain, mean_offset = self.popt
        gains = []
        offsets = []
        offset = 0
        gain = 1.0
        i = 0
        while i < 150:
            # Perform least squares fit
            gain = self.fit_gain(self.raw_vals_for_gain, self.unc_vals_for_gain, self.cal_vals_for_gain, gain, offset)
            gains.append(gain)

            t_gal_offset = (self.raw_vals_for_offset - offset) / gain - self.cal_vals_for_offset
            # t_gal_gain = (self.raw_vals_for_gain - offset) / gain - self.cal_vals_for_gain
            self.raw_vals_for_offset -= gain * t_gal_offset
            # self.raw_vals_for_gain -= gain * t_gal_gain

            offset = self.fit_offset(self.raw_vals_for_offset, self.unc_vals_for_offset, self.cal_vals_for_offset, offset, gain)
            offsets.append(offset)
            # Residuals as galactic signal
            t_gal_gain = (self.raw_vals_for_gain - offset)/gain - self.cal_vals_for_gain
            self.raw_vals_for_gain -= gain*t_gal_gain

            gain = self.fit_gain(self.raw_vals_for_gain, self.unc_vals_for_gain, self.cal_vals_for_gain, gain, offset)
            gains.append(gain)

            t_gal_offset = (self.raw_vals_for_offset - offset)/gain - self.cal_vals_for_offset
            # t_gal_gain = (self.raw_vals_for_gain - offset) / gain - self.cal_vals_for_gain
            self.raw_vals_for_offset -= gain*t_gal_offset
            # self.raw_vals_for_gain -= gain * t_gal_gain

            i += 1

        mean_gain, mean_offset = self.plot_fit_vals(gains, offsets, id)
        return [mean_gain, mean_offset], len(self.raw_vals_for_offset)

        # # Apply fit parameters to raw map and return as a calibrated map
        # calib_map = np.zeros_like(raw_map)
        # calib_uncmap = np.zeros_like(raw_map)
        # calib_map[nonzero_mask] = raw_map[nonzero_mask]*self.popt[0] + self.popt[1]
        # calib_uncmap[nonzero_mask] = unc_map[nonzero_mask]*abs(self.popt[0])
        # return calib_map, calib_uncmap

    def plot_fit_vals(self, gains, offsets, id):

        mean_gain = np.mean(gains[50:])
        mean_offset = np.mean(offsets[50:])

        plt.plot(gains, 'r.', np.ones_like(gains)*mean_gain, 'b')
        plt.ylabel("Gain")
        plt.xlabel("Iteration")
        plt.savefig(f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/cal_param_plots/gain_convergence_orbit_{id}.png")
        plt.close()

        plt.plot(offsets, 'b.', np.ones_like(offsets)*mean_offset, 'r')
        plt.ylabel("Offset")
        plt.xlabel("Iteration")
        plt.savefig(f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/cal_param_plots/offset_convergence_orbit_{id}.png")
        plt.close()
        return mean_gain, mean_offset

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

    @staticmethod
    def chi_sq(params, x_data, y_data, sigma):
        '''Calculate chi_sq'''
        residual = (y_data - (params[0] * x_data + params[1]))
        weighted_residual = residual/sigma
        chi_sq = np.sum(weighted_residual ** 2) / len(x_data)
        return chi_sq

    @staticmethod
    def chi_sq_gain(param, x_data, y_data, sigma, offset):
        residual = (y_data - (param*x_data + offset))
        weighted_residual = residual / sigma
        chi_sq = (np.sum(weighted_residual ** 2) / len(x_data)) if len(x_data) > 0 else 0.0
        return chi_sq

    @staticmethod
    def chi_sq_offset(param, x_data, y_data, sigma, gain):
        residual = (y_data - (gain * x_data + param))
        weighted_residual = residual / sigma
        chi_sq = (np.sum(weighted_residual ** 2) / len(x_data)) if len(x_data) > 0 else 0.0
        return chi_sq

    def fit_gain(self, raw_vals, unc_vals, cal_vals, x0, offset):
        popt = minimize(self.chi_sq_gain, x0, args=(raw_vals, cal_vals, unc_vals, offset), method='Nelder-Mead').x
        return popt

    def fit_offset(self, raw_vals, unc_vals, cal_vals, x0, gain):
        popt = minimize(self.chi_sq_offset, x0, args=(raw_vals, cal_vals, unc_vals, gain), method='Nelder-Mead').x
        return popt

    def fit(self, raw_vals, unc_vals, cal_vals):
        """
        Straightforward least squares curve fitting.
        :return:
        """
        x0 = [1.0, 0.0]
        y, x = raw_vals, cal_vals
        popt = minimize(self.chi_sq, x0, args=(x, y, unc_vals), method='Nelder-Mead').x
        # popt, _ = curve_fit(self.line, x, y, sigma=self.unc_vals)
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
