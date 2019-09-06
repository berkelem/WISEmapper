from file_handler import ZodiMap
import healpy as hp
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math

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
        self.nside = hp.npix2nside(len(raw_map))
        self.kelsall_map.set_resolution(self.nside)
        nonzero_mask = raw_map != 0.0
        galaxy_mask = self.mask_galaxy()
        full_mask = nonzero_mask & galaxy_mask
        raw_vals = raw_map[full_mask]
        cal_vals = self.kelsall_map.mapdata[full_mask]
        clean_mask = ~ self.clean_with_z_score(raw_vals, cal_vals, threshold=1)
        self.raw_vals = raw_vals[clean_mask]
        self.cal_vals = cal_vals[clean_mask]
        self.popt = self.fit()
        calib_map = np.zeros_like(raw_map)
        calib_uncmap = np.zeros_like(raw_map)
        calib_map[nonzero_mask] = raw_map[nonzero_mask]*self.popt[0] + self.popt[1]
        calib_uncmap[nonzero_mask] = unc_map[nonzero_mask]*self.popt[0]
        return calib_map, calib_uncmap

    def clean_with_z_score(self, raw_vals, cal_vals, threshold=3):
        data = cal_vals/raw_vals
        z = np.abs(stats.zscore(data))
        mask = z > threshold
        return mask


    def mask_galaxy(self):
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
        # sort_order = np.argsort(self.raw_vals)
        # ceil_data = self.ceiling(self.cal_vals[sort_order], window=50)
        x, y = self.binmax()
        popt, _ = curve_fit(type(self).line, x, y)
        return popt

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
        plt.savefig(f'{path}band{self.band}_day{label}.png')
        plt.close()

