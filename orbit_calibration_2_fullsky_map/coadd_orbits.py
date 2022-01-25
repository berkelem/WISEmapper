"""
:author: Matthew Berkeley
:date: Jun 2020

Module managing the iterative calibration approach, modified from the approach used by WMAP.

Main classes
------------

Orbit :
    Class containing information on individual WISE scans. Includes methods for calibrating to the corresponding
    zodiacal light scan.

IterativeFitter :
    Class used to perform the chi-squared minimization fit on the WISE data using the zodiacal light as a template.

Coadd :
    Class managing the creation of a calibrated full-sky coadd map.
"""

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
from healpy.rotator import Rotator


class Orbit(BaseMapper):
    """
    Class containing information on individual WISE scans. Includes methods for calibrating to the corresponding
    zodiacal light scan.

    Parameters
    ----------
    :param orbit_num: int
        Orbit number label
    :param band: int
        WISE Band number (one of [1,2,3,4])
    :param mask: numpy.array
        Boolean array with True values indicating pixels to mask
    :param nside: int
        Healpix NSIDE parameter

    Methods
    -------
    apply_fit :
        Apply calibration using current values for gain and offset
    apply_mask :
        Remove all masked pixels along with any pixels flagged as outliers
    apply_spline_fit :
        Apply calibration using gain and offset values drawn from the spline fits.
    fit :
        Fit gain and offset values to minimize the chi_sq difference between the WISE orbit coadd and the corresponding
        zodiacal light orbit generated using the Kelsall model
    load_orbit_data :
        Load all orbit data from previously-created csv files. These files contain pixel intensity values,
        uncertainty values, Healpix pixel index and pixel mjd_obs timestamps for every Healpix pixel in each orbit
        coadd.
    load_zodi_orbit_data :
        Load zodiacal light files made based on the Kelsall model. These files are sparse full-sky maps, with
        non-zero pixels corresponding spatially to an individual WISE coadd.
    plot_fit :
        Plot calibrated data along with the zodiacal light template with galactic latitude on the x-axis
    reset_outliers :
        Restore values that were removed for the purposes of fitting

    Attributes
    ----------
    coadd_map (class attr) : numpy.array
        Array containing the zodi-subtracted coadd of the previous iteration
    orbit_file_path (class attr) : str
        Path to orbit csv files
    zodi_file_path (class attr) : str
        Path to zodiacal light scan files
    orbit_num : int/str
        Orbit/scan index number (0-6322)
    orbit_mjd_obs : numpy.array
        Array containing mjd_obs timestamp values for all orbit pixels
    pixel_inds_clean_masked : numpy.array
        Healpix pixels containing non-zero values after masked and outlier pixels are removed
    cal_uncs_clean_masked : numpy.array
        Calibrated uncertainty values for each non-zero Healpix pixel after masked and outlier pixels are removed
    zs_data_clean_masked : numpy.array
        Calibrated zodi-subtracted pixel values for each non-zero Healpix pixel after masked and outlier pixels are
        removed
    gain : float
        Fitted gain value (initialized to 1.0)
    offset : float
        Fitted offset value (initialized to 0.0)

    """

    coadd_map = None
    orbit_file_path = ""
    zodi_file_path = ""

    theta_ecl = None
    phi_ecl = None
    theta_gal = None
    phi_gal = None

    def __init__(self, orbit_num, band, mask, nside):

        super().__init__(band, orbit_num, self.orbit_file_path)
        self.orbit_num = orbit_num
        self._band = band
        self._nside = nside
        self._filename = os.path.join(self.orbit_file_path, self.orbit_csv_name)
        self._zodi_filename = os.path.join(
            self.zodi_file_path, f"zodi_map_cal_W{self._band}_{self.orbit_num}.fits"
        )
        self._mask = mask
        self._mask_inds = np.arange(len(self._mask))[self._mask.astype(bool)] if (self._mask is not None) else []
        self._outlier_inds = np.array([])
        self.mask_rule = None

        self._orbit_data = None
        self._orbit_uncs = None
        self._pixel_inds = None
        self.orbit_mjd_obs = None
        self._zodi_data = None
        self.mean_mjd_obs = None

        self.pixel_inds_clean_masked = None
        self._orbit_data_clean_masked = None
        self._orbit_uncs_clean_masked = None
        self._orbit_mjd_clean_masked = None
        self._zodi_data_clean_masked = None

        self.gain = 1.0
        self.offset = 0.0

        self._cal_data_clean_masked = None
        self.cal_uncs_clean_masked = None
        self.zs_data_clean_masked = None

    @staticmethod
    def rotate_data(old_coord, new_coord, data, pix_inds, nside):
        """
        Rotate Healpix pixel numbering by applying a coordinate system transformation

        Parameters
        ----------
        :param old_coord: str
            One of 'G' (galactic, default), 'C' (celestial), 'E' (ecliptic)
        :param new_coord: str
            One of 'G' (galactic), 'C' (celestial), 'E' (ecliptic, default)
        """
        npix = hp.nside2npix(nside)
        map_arr = np.zeros(npix)
        map_arr[pix_inds] = data
        px_region_map = np.zeros(npix)
        px_region_map[pix_inds] = 1
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        r = Rotator(coord=[new_coord, old_coord])  # Transforms galactic to ecliptic coordinates
        theta_rot, phi_rot = r(theta, phi)  # Apply the conversion
        rot_pixorder = hp.ang2pix(hp.npix2nside(npix), theta_rot, phi_rot)
        rot_data = map_arr[rot_pixorder]
        px_region_map_rot = px_region_map[rot_pixorder]
        rot_data_pix_inds = np.arange(npix)[px_region_map_rot.astype(bool)]
        return rot_data[px_region_map_rot.astype(bool)], rot_data_pix_inds, theta_rot, phi_rot

    def apply_fit(self):
        """Apply calibration using current values for gain and offset."""
        self._cal_data_clean_masked = (
                                              self._orbit_data_clean_masked - self.offset
                                      ) / self.gain
        self.cal_uncs_clean_masked = self._orbit_uncs_clean_masked / abs(self.gain)

        self.zs_data_clean_masked = (
                self._cal_data_clean_masked - self._zodi_data_clean_masked
        )
        self.zs_data_clean_masked[self.zs_data_clean_masked < 0.0] = 0.0

    def apply_mask(self):
        """Remove all masked pixels along with any pixels flagged as outliers"""

        # self.mask_ecliptic_crossover()

        mask = np.ones_like(self._pixel_inds, dtype=bool)

        entries_to_mask = [
            i
            for i in range(len(self._pixel_inds))
            if self._pixel_inds[i] in self._mask_inds]

        if self.rule == "phi_neg":
            moon_stripe_to_mask = [
                i
                for i in range(len(self._pixel_inds))
                if self.phi_ecl[i] < 0]
            entries_to_mask = list(set(entries_to_mask + moon_stripe_to_mask))
        elif self.rule == "phi_pos":
            moon_stripe_to_mask = [
                i
                for i in range(len(self._pixel_inds))
                if self.phi_ecl[i] > 0]
            entries_to_mask = list(set(entries_to_mask + moon_stripe_to_mask))

        mask[entries_to_mask] = False

        self.pixel_inds_clean_masked = self._pixel_inds[mask]
        # self.pixel_inds_clean_masked = np.array(
        #     [
        #         self._pixel_inds[i]
        #         for i in range(len(self._pixel_inds))
        #         if i not in entries_to_mask and i not in self._outlier_inds
        #     ],
        #     dtype=int,
        # )
        self._orbit_data_clean_masked = self._orbit_data[mask]
        # self._orbit_data_clean_masked = np.array(
        #     [
        #         self._orbit_data[i]
        #         for i in range(len(self._orbit_data))
        #         if i not in entries_to_mask and i not in self._outlier_inds
        #     ]
        # )
        self._orbit_uncs_clean_masked = self._orbit_uncs[mask]
        # self._orbit_uncs_clean_masked = np.array(
        #     [
        #         self._orbit_uncs[i]
        #         for i in range(len(self._orbit_uncs))
        #         if i not in entries_to_mask and i not in self._outlier_inds
        #     ]
        # )
        self._orbit_mjd_clean_masked = self.orbit_mjd_obs[mask]
        # self._orbit_mjd_clean_masked = np.array(
        #     [
        #         self.orbit_mjd_obs[i]
        #         for i in range(len(self.orbit_mjd_obs))
        #         if i not in entries_to_mask and i not in self._outlier_inds
        #     ]
        # )
        self._zodi_data_clean_masked = self._zodi_data[mask]
        # self._zodi_data_clean_masked = np.array(
        #     [
        #         self._zodi_data[i]
        #         for i in range(len(self._zodi_data))
        #         if i not in entries_to_mask and i not in self._outlier_inds
        #     ]
        # )
        self._theta_ecl_clean_masked = self.theta_ecl[mask]
        self._theta_gal_clean_masked = self.theta_gal[mask]

        # self._theta_clean_masked = np.array([self.theta[i] for i in range(len(self.theta)) if i not in entries_to_mask and i not in self._outlier_inds])

        self._phi_gal_clean_masked = self.phi_gal[mask]
        self._phi_ecl_clean_masked = self.phi_ecl[mask]

        self.galaxy_mask = ((self._theta_gal_clean_masked < 80) | (self._theta_gal_clean_masked > 100)) & (
                    (self._theta_ecl_clean_masked > 50) & (self._theta_ecl_clean_masked < 130))

        return

    def calc_rsq(self):

        correlation_matrix = np.corrcoef(self._cal_data_clean_masked[self.galaxy_mask], self._zodi_data_clean_masked[self.galaxy_mask])
        correlation_xy = correlation_matrix[0, 1]
        self.r_squared = correlation_xy ** 2
        return

    def apply_spline_fit(self, gain_spline, offset_spline):
        """
        Apply calibration using gain and offset values drawn from the spline fits.

        Parameters
        ----------
        :param gain_spline: scipy UnivariateSpline object
            Spline fitted to all of the gain fits after N iterations
        :param offset_spline: scipy UnivariateSpline object
            Spline fitted to all of the offset fits after N iterations
        """
        gains = gain_spline(self._orbit_mjd_clean_masked)
        # offsets = offset_spline(self._theta_ecl_clean_masked, self._orbit_mjd_clean_masked)
        offsets = offset_spline(self._orbit_mjd_clean_masked)
        self._cal_data_clean_masked = (self._orbit_data_clean_masked - offsets) / gains
        self.cal_uncs_clean_masked = self._orbit_uncs_clean_masked / abs(gains)

        self.zs_data_clean_masked = (
                self._cal_data_clean_masked - self._zodi_data_clean_masked
        )


        diff_spline = self.plot_diff()
        self._cal_data_clean_masked -= diff_spline
        # self.calc_rsq()
        self.zs_data_clean_masked = (
                self._cal_data_clean_masked - self._zodi_data_clean_masked
        )

        return

    def fit(self):
        """
        Fit gain and offset values to minimize the chi_sq difference between the WISE orbit coadd and the corresponding
        zodiacal light orbit generated using the Kelsall model

        The fit is based on the map made during the previous iteration of the full-sky coadd. This coadd is the
        result of subtracting the modelled zodiacal light from the calibrated WISE data of the previous iteration. It
        therefore serves as a proxy for galactic signal. This galactic signal is subtracted from the WISE data before
        fitting.
        """
        orbit_data_to_fit_clean_masked = self._orbit_data_clean_masked
        if self.coadd_map is not None:
            self._clean_data()
            prev_itermap_clean_masked = self.coadd_map[self.pixel_inds_clean_masked]
            t_gal_clean_masked = prev_itermap_clean_masked * self.gain
            orbit_data_to_fit_clean_masked = (
                    self._orbit_data_clean_masked - t_gal_clean_masked
            )


        orbit_fitter = IterativeFitter(
            self._zodi_data_clean_masked[self.galaxy_mask],
            orbit_data_to_fit_clean_masked[self.galaxy_mask],
            self._orbit_uncs_clean_masked[self.galaxy_mask],
            self._theta_ecl_clean_masked[self.galaxy_mask],
            self._phi_ecl_clean_masked[self.galaxy_mask],
        )
        self.gain, self.offset, self.segmented_offsets = orbit_fitter.iterate_fit(1)
        return

    def mask_ecliptic_crossover(self):
        rot_data, rot_pix_inds, theta_rot, phi_rot = self.rotate_data("G", "E", self._orbit_data,
                                                                      self._pixel_inds, self._nside)
        px_theta, px_phi = hp.pix2ang(self._nside, rot_pix_inds, lonlat=True)
        crossover_pixels = (-5 < px_phi) & (px_phi < 5)
        rerot_data, rerot_pix_inds, theta_rerot, phi_rerot = self.rotate_data("E", "G", rot_data[crossover_pixels],
                                                                              rot_pix_inds[crossover_pixels],
                                                                              self._nside)
        self._mask_inds = np.append(self._mask_inds, rerot_pix_inds)

    def load_orbit_data(self):
        """
        Load all orbit data from previously-created csv files. These files contain pixel intensity values,
        uncertainty values, Healpix pixel index and pixel mjd_obs timestamps for every Healpix pixel in each orbit
        coadd.
        """
        npix = hp.nside2npix(self._nside)
        r = Rotator(coord=["G", "E"])  # Transforms galactic to ecliptic coordinates
        if type(self).theta_gal is None:
            theta_gal, phi_gal = hp.pix2ang(self._nside, np.arange(npix))

            theta_ecl, phi_ecl = r(theta_gal, phi_gal)  # Apply the conversion

            type(self).theta_ecl = theta_ecl * 180 / np.pi
            type(self).phi_ecl = phi_ecl * 180 / np.pi
            type(self).theta_gal = theta_gal * 180 / np.pi
            type(self).phi_gal = phi_gal * 180 / np.pi

        all_orbit_data = pd.read_csv(self._filename)
        self._orbit_data = np.array(all_orbit_data["pixel_value"])
        self._orbit_uncs = np.array(all_orbit_data["pixel_unc"])
        self._pixel_inds = np.array(all_orbit_data["hp_pixel_index"])
        self.orbit_mjd_obs = np.array(all_orbit_data["pixel_mjd_obs"])
        self.mean_mjd_obs = np.mean(self.orbit_mjd_obs)

        self.theta_ecl = type(self).theta_ecl[self._pixel_inds]
        self.phi_ecl = type(self).phi_ecl[self._pixel_inds]
        self.theta_gal = type(self).theta_gal[self._pixel_inds]
        self.phi_gal = type(self).phi_gal[self._pixel_inds]

        # map_template = np.zeros(npix)
        # map_template[self._pixel_inds] = self._orbit_data
        # data_test = HealpixMap("data_test.fits")
        # data_test.mapdata = map_template
        # data_test.save_map("G")
        #
        # map_template = np.zeros(npix)
        # map_template[self._pixel_inds] = self.theta
        # theta_test = HealpixMap("theta_test.fits")
        # theta_test.mapdata = map_template
        # theta_test.save_map("G")
        #
        # map_template = np.zeros(npix)
        # map_template[self._pixel_inds] = self.phi
        # phi_test = HealpixMap("phi_test.fits")
        # phi_test.mapdata = map_template
        # phi_test.save_map("G")

        # rot_data, rot_pix_inds, theta_rot, phi_rot = self.rotate_data("G", "E", self._orbit_data,
        #                                                               self._pixel_inds, self._nside)

        return

    def load_zodi_orbit_data(self):
        """
        Load zodiacal light files made based on the Kelsall model. These files are sparse full-sky maps, with
        non-zero pixels corresponding spatially to an individual WISE coadd.
        """
        zodi_orbit = ZodiMap(self._zodi_filename, self._band)
        zodi_orbit.read_data()
        pixels = np.zeros_like(zodi_orbit.mapdata)
        pixels[self._pixel_inds] = 1.0
        self._zodi_data = zodi_orbit.mapdata[pixels.astype(bool)]
        if not all(self._zodi_data.astype(bool)):
            print(f"Orbit {self.orbit_num} mismatch with zodi: zeros in zodi orbit")
        elif any(zodi_orbit.mapdata[~pixels.astype(bool)]):
            print(
                f"Orbit {self.orbit_num} mismatch with zodi: nonzeros outside zodi orbit"
            )
        return

    def plot_fit(self, output_path=os.getcwd(), iteration=None, label=None):
        """Plot calibrated data along with the zodiacal light template with galactic latitude on the x-axis"""
        # theta, phi = hp.pix2ang(self._nside, self.pixel_inds_clean_masked, lonlat=True)
        plt.plot(self._theta_gal_clean_masked[self.galaxy_mask], self._cal_data_clean_masked[self.galaxy_mask], "r.", ms=0.5, alpha=0.5,
                 label="Calibrated data")
        plt.plot(self._theta_gal_clean_masked, self._zodi_data_clean_masked, "b.", ms=0.5, alpha=0.5,
                 label="Zodi model")
        plt.legend()
        plt.title("Orbit {}; r^2={}".format(self.orbit_num, self.r_squared))
        plt.xlabel("Latitude (degrees)")
        plt.ylabel("MJy/sr")
        outfile_name = (
            "orbit_{}_fit_iter_{}{}.png".format(self.orbit_num, iteration, label)
            if iteration
            else "orbit_{}_fit{}.png".format(self.orbit_num, label)
        )
        plt.savefig(os.path.join(output_path, outfile_name))
        plt.close()

    def plot_diff(self, output_path=os.getcwd()):
        diff_data = self._cal_data_clean_masked - self._zodi_data_clean_masked
        t_data = self._orbit_mjd_clean_masked
        ang_data = self._theta_gal_clean_masked
        mask = np.ones_like(ang_data, dtype=bool)
        mask[(ang_data > 80) & (ang_data < 100)] = False
        self.calc_rsq()

        from mpl_toolkits.mplot3d import axes3d
        from scipy.interpolate import Rbf

        str_month_dict = OrderedDict([(55197, "Jan"), (55228, "Feb"), (55256, "Mar"), (55287, "Apr"),
                                      (55317, "May"), (55348, "Jun"), (55378, "Jul"), (55409, "Aug")])
        min_time = min(self._orbit_mjd_clean_masked)
        max_time = max(self._orbit_mjd_clean_masked)
        month_start_times = list(str_month_dict.keys())

        start_month_ind = month_start_times.index(min(month_start_times, key=lambda x: abs(x - min_time)))
        start_month_ind = start_month_ind if month_start_times[start_month_ind] < min_time else start_month_ind - 1

        end_month_ind = month_start_times.index(min(month_start_times, key=lambda x: abs(x - max_time)))
        end_month_ind = end_month_ind if month_start_times[end_month_ind] > max_time else end_month_ind + 1
        y_ticks = month_start_times[start_month_ind:end_month_ind + 1]

        x = t_data[mask]
        y = ang_data[mask]
        z = diff_data[mask]

        if len(z) < 10:
            return np.zeros_like(self._cal_data_clean_masked)

        spline = Rbf(x, y, z, function='linear', smooth=100)

        # x_grid = np.linspace(min(x), max(x))
        # y_grid = np.linspace(min(y), max(y))
        # B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')
        # Z = spline(B1, B2)
        # # for ang in range(60, 300, 60):
        # fig = plt.figure(figsize=(10, 6))
        # ax = axes3d.Axes3D(fig)
        # ax.plot_wireframe(B1, B2, Z)
        # ax.plot_surface(B1, B2, Z, alpha=0.2)
        # ax.scatter3D(x, y, z, c='r')
        # ax.set_yticks(y_ticks)
        # ax.set_yticklabels([str_month_dict[y] for y in y_ticks], rotation=45)
        # # ax.view_init(elev=10., azim=ang)
        # plt.savefig("3d_offset_spline_orbit_{}.png".format(self.orbit_num))
        # plt.close()

        start_spline = spline(np.ones_like(t_data) * min(t_data), ang_data)
        end_spline = spline(np.ones_like(t_data) * max(t_data), ang_data)
        mean_spline = np.mean([start_spline, end_spline], axis=0)
        plt.plot(self._theta_gal_clean_masked, self._cal_data_clean_masked - self._zodi_data_clean_masked, "r.", ms=0.5,
                 label="diff")
        plt.plot(self._theta_gal_clean_masked, mean_spline, "b--", label="mean spline")
        # plt.plot(self._theta_gal_clean_masked, end_spline, "g--", label="end spline")
        plt.legend()
        plt.title("Orbit {}: R^2 {}".format(self.orbit_num, self.r_squared))
        plt.xlabel("Latitude (degrees)")
        plt.ylabel("MJy/sr")
        outfile_name = (
            "orbit_{}_diff_theta.png".format(self.orbit_num)
        )
        plt.savefig(os.path.join(output_path, outfile_name))
        plt.close()
        return mean_spline

    def reset_outliers(self):
        """Restore values that were removed for the purposes of fitting"""
        self._outlier_inds = np.array([])

    def _clean_data(self):
        """
        Remove pixels for which the zodi-subtracted data is an outlier (z-score > 1), as these are likely to be
        pixels with a high signal from galactic emission
        """
        z = np.abs(stats.zscore(self.zs_data_clean_masked))
        mask = z > 1
        inds_to_mask = self.pixel_inds_clean_masked[mask]
        self._outlier_inds = np.append(self._outlier_inds, inds_to_mask)
        self.apply_mask()
        return

    def save_orbit_map(self, label):
        orbit_map = WISEMap("orbit_{}_{}.fits".format(self.orbit_num, label), self.band)
        orbit_map.mapdata[self.pixel_inds_clean_masked] = self.zs_data_clean_masked
        orbit_map.save_map()
        return


class IterativeFitter:
    """
    Class used to perform the chi-squared minimization fit on the WISE data using the zodiacal light as a template.

    Parameters
    ----------
    zodi_data : numpy.array
        Array containing the zodiacal light data
    raw_data : numpy.array
        Array containing the uncalibrated WISE data
    raw_uncs : numpy.array
        Array containing the uncalibrated WISE uncertainty data

    Methods
    -------
    iterate_fit :
        Perform an iterative fit on the data. The iteration procedure is described in the method docstring
    """

    def __init__(self, zodi_data, raw_data, raw_uncs, theta, phi):
        self.zodi_data = zodi_data
        self.raw_data = raw_data
        self.raw_uncs = raw_uncs
        self.theta = theta
        self.phi = phi

    def iterate_fit(self, n):
        """
        The aim of the fitting procedure is to calibrate the WISE data to the zodiacal light modelled using the Kelsall
        model.
        Following the initial fit, the residual is considered to be a proxy for galactic signal. This is subtracted
        from the data before attempting a second fit, and so on for n iterations. The gain and offset values converge
        as the iterations continue.

        Parameters
        ----------

        :param n: int
            Number of iterations
        :return gain, offset: (float, float)
            Fitted gain and offset values
        """
        i = 0
        data_to_fit = self.raw_data
        uncs_to_fit = self.raw_uncs
        gain = offset = 0.0
        if len(data_to_fit) > 0:
            while i < n:
                gain, offset = self._fit_to_zodi(
                    data_to_fit, self.zodi_data, uncs_to_fit
                )
                # offset_spline = self.fit_offset_spline(data_to_fit, gain, offset)
                segmented_offsets = self._segmented_fit(data_to_fit, uncs_to_fit, gain)
                data_to_fit = self._adjust_data(gain, offset, data_to_fit)
                i += 1
        else:
            gain = offset = 0.0
            segmented_offsets = [0.0] * 24
        return gain, offset, segmented_offsets

    # def fit_offset_spline(self, gain, offset):
    #     cal_data = (self.raw_data - offset) / gain
    #     zs_data = (cal_data - self.zodi_data)
    #     plt.plot(self.theta, zs_data, "r.", ms=0.5)
    #     plt.title("Orbit {}".format(self.orbit_num))
    #     plt.xlabel("Latitude (degrees)")
    #     plt.ylabel("MJy/sr")
    #     outfile_name = (
    #         "orbit_{}_diff_theta.png".format(self.orbit_num)
    #     )
    #     plt.savefig(os.path.join(output_path, outfile_name))
    #     plt.close()

    @staticmethod
    def _chi_sq(params, x_data, y_data, sigma):
        """
        Calculate the chi-squared value of the residual

        Parameters
        ----------
        :param params: numpy.array
            Array containing initial estimates for gain and offset
        :param x_data: numpy.array
            Array containing WISE orbit data
        :param y_data: numpy.array
            Array containing zodi data generated using the Kelsall model
        :param sigma: numpy.array
            Array containing the WISE orbit uncertainty values
        :return chi_sq:
            The calculated chi-squared value
        """
        residual = x_data - ((y_data * params[0]) + params[1])
        weighted_residual = residual / (np.mean(sigma) ** 2)
        chi_sq = (
            (np.sum(weighted_residual ** 2) / len(x_data)) if len(x_data) > 0 else 0.0
        )
        return chi_sq

    @staticmethod
    def _chi_sq_gain(params, x_data, y_data, sigma, gain):
        """
        Calculate the chi-squared value of the residual

        Parameters
        ----------
        :param params: numpy.array
            Array containing initial estimates for gain and offset
        :param x_data: numpy.array
            Array containing WISE orbit data
        :param y_data: numpy.array
            Array containing zodi data generated using the Kelsall model
        :param sigma: numpy.array
            Array containing the WISE orbit uncertainty values
        :return chi_sq:
            The calculated chi-squared value
        """
        residual = x_data - ((y_data * gain) + params[0])
        weighted_residual = residual / (np.mean(sigma) ** 2)
        chi_sq = (
            (np.sum(weighted_residual ** 2) / len(x_data)) if len(x_data) > 0 else 0.0
        )
        return chi_sq

    def _adjust_data(self, gain, offset, data):
        """
        Subtract residual from original data

        Parameters
        ----------
        :param gain: float
            Fitted gain value
        :param offset: float
            Fitted offset value
        :param data: numpy.array
            Array containing original WISE data
        :return new_data: numpy.array
            Array containing original data with fitted residual subtracted
        """
        residual = ((data - offset) / gain) - self.zodi_data
        new_data = data - gain * residual
        return new_data

    def _segmented_fit(self, orbit_data, orbit_uncs, gain):
        bins = [(-180, -165), (-165, -150), (-150, -135), (-135, -120), (-120, -105), (-105, -90), (-90, -75),
                (-75, -60), (-60, -45), (-45, -30), (-30, -15), (-15, 0), (0, 15), (15, 30), (30, 45), (45, 60),
                (60, 75), (75, 90), (90, 105), (105, 120), (120, 135), (135, 150), (150, 165), (165, 180)]
        offsets = []
        for bin in bins:
            start_theta_deg, end_theta_deg = bin
            start_theta = start_theta_deg + 180
            end_theta = end_theta_deg + 180
            segment_mask = (self.theta >= start_theta) & (self.theta < end_theta)
            segment_data = orbit_data[segment_mask]
            segment_uncs = orbit_uncs[segment_mask]
            segment_zodi = self.zodi_data[segment_mask]
            segment_offset = self._fit_offset(segment_data, segment_zodi, segment_uncs, gain) if len(
                segment_data) > 0 else 0.0
            offsets.append(segment_offset)
        return offsets

    def _fit_offset(self, orbit_data, zodi_data, orbit_uncs, gain):
        init_offset = 0.0
        popt = minimize(
            self._chi_sq_gain,
            np.array([init_offset]),
            args=(orbit_data, zodi_data, orbit_uncs, gain),
            method="Nelder-Mead",
        ).x
        offset = popt[0]
        return offset

    def _fit_to_zodi(self, orbit_data, zodi_data, orbit_uncs):
        """
        Perform chi-squared minimization fit.

        Parameters
        ----------
        :param orbit_data: numpy.array
            Array containing uncalibrated WISE data
        :param zodi_data: numpy.array
            Array containing zodiacal light data generated using the Kelsall model
        :param orbit_uncs: numpy.array
            Array containing uncalibrated WISE uncertainty data
        :return gain, offset: (float, float)
            Fitted gain and offset values
        """
        init_gain = 1.0
        init_offset = 0.0
        popt = minimize(
            self._chi_sq,
            np.array([init_gain, init_offset]),
            args=(orbit_data, zodi_data, orbit_uncs),
            method="Nelder-Mead",
        ).x
        gain, offset = popt
        return gain, offset


class Coadder:
    """
    Class managing the creation of a calibrated full-sky coadd map.

    Parameters
    ----------
    :param band: int
        WISE band number (one of [1,2,3,4])
    :param moon_stripe_file: str
        Filename (including full path) of moon stripe mask
    :param fsm_map_file: str
        Name of output full-sky map file
    :param orbit_file_path: str
        Path to WISE orbit csv files
    :param zodi_file_path: str
        Path to zodi orbit files
    :param output_path: str, optional
        Path to write output files (default is current working directory)

    Methods
    -------
    add_calibrated_orbits :
        Add WISE orbits to the final coadd map after calibration using spline fitting
    load_splines :
        Load gain spline and offset spline from '*.pkl' files saved in a file
    run_iterative_fit :
        Run a modified form of the iterative calibration used in WMAP. For details, see method docstring.
    """

    def __init__(
            self,
            band,
            moon_stripe_file,
            fsm_map_file,
            orbit_file_path,
            zodi_file_path,
            output_path=os.getcwd(),
    ):
        self.band = band
        self.fsm_map_file = fsm_map_file
        self.unc_fsm_map_file = "{}_{}.{}".format(
            fsm_map_file.rpartition(".")[0], "unc", fsm_map_file.rpartition(".")[-1]
        )
        self.output_path = output_path
        setattr(Orbit, "orbit_file_path", orbit_file_path)
        setattr(Orbit, "zodi_file_path", zodi_file_path)

        self.iter = 0
        self.num_orbits = len(os.listdir(orbit_file_path))

        self.moon_stripe_mask = HealpixMap(moon_stripe_file)
        self.moon_stripe_mask.read_data()
        self.moon_stripe_inds = np.arange(len(self.moon_stripe_mask.mapdata))[
            self.moon_stripe_mask.mapdata.astype(bool)
        ]

        self.full_mask = np.zeros_like(self.moon_stripe_mask.mapdata.astype(bool))
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

        self.month_timestamps = OrderedDict(
            [
                ("Jan", 55197),
                ("Feb", 55228),
                ("Mar", 55256),
                ("Apr", 55287),
                ("May", 55317),
                ("Jun", 55348),
                ("Jul", 55378),
                ("Aug", 55378.1),#55409),
            ]
        )
        self.mask_orbit_timestamps = []
        # self.mask_orbit_timestamps = [(55217, 55225), (55229, 55236), (55247, 55254), (55258, 55262), (55276, 55284),
        #                               (55288, 55294), (55306, 55313), (55316, 55325), (55335, 55343), (55346, 55354),
        #                               (55364, 55372), (55376, 55384), (55394, 55400), (55405, 56000)]
        self.mask_rules_stripes = ["phi_pos", "phi_neg", "phi_pos", "phi_neg", "phi_pos", "phi_neg", "phi_pos",
                                   "phi_neg", "phi_pos", "phi_neg", "phi_pos", "phi_pos", "phi_neg", "phi_pos"]

        self.all_orbits = []

        self.gain_spline = None
        self.offset_spline = None

    def add_calibrated_orbits(self, plot=False):
        """Add WISE orbits to the final coadd map after calibration using spline fitting"""
        self._set_output_filenames(label="calibrated")

        # Reset numerator and denominator for keeping track of values using uncertainty propagation
        self.numerator_masked = np.zeros(self.npix)
        self.denominator_masked = np.zeros_like(self.numerator_masked)

        for orbit in self.all_orbits:
            print(f"Adding orbit {orbit.orbit_num}")
            orbit.reset_outliers()  # Include pixels in the galactic plane that were removed for fitting
            orbit.apply_mask()
            if len(orbit._orbit_data_clean_masked) == 0:
                print(f"Empty array for orbit {orbit.orbit_num}")
                continue
            orbit.apply_spline_fit(self.gain_spline, self.offset_spline)
            # self._add_orbit(orbit)
            if orbit.r_squared > 0.9:
                self._add_orbit(orbit)
            else:
                print("fit rsquared < 0.9; excluded")
                continue
            if plot:  # and orbit.orbit_num % 15 == 0.0:
                orbit.plot_fit(label="postadjust")

        self._clean_data()
        self._compile_map()
        self._normalize()
        self._save_maps()

    def load_orbits(self, month="all"):
        mapping_region = 0
        for i in range(self.num_orbits):
            if mapping_region == 2:  # or i % 2 != 0:
                continue
            print(f"Loading orbit {i}")
            # Initialize Orbit object and load data
            orbit = Orbit(i, self.band, self.full_mask, self.nside)
            orbit.load_orbit_data()
            rule = None

            # bad_orbit_list = [5221, 5223, 5227, 5229, 5231, 5235, 5237, 5241, 5243, 5245, 5247, 5249, 5251, 5255, 5257, 5259, 5261, 5263,
            #  5267, 5269, 5271, 5273, 5275, 5277, 5279, 5281, 5283, 5287, 5289, 5291, 5295, 5297, 5301, 5303, 5305, 5307,
            #  5309, 5311, 5315, 5317, 5319, 5321, 5323, 5327, 5329, 5331, 5333, 5335, 5337, 5339, 5341, 5343, 5347, 5349,
            #  5351, 5353, 5355, 5361, 5363, 5365, 5367, 5369, 5371, 5375, 5379, 5381, 5383, 5387, 5389, 5391, 5393, 5395,
            #  5397, 5399, 5401, 5705, 5707, 5711, 5713, 5715, 5719, 5721, 5725, 5727, 5729, 5731, 5733, 5735, 5739, 5741,
            #  5743, 5745, 5747, 5751, 5753, 5755, 5757, 5759, 5761, 5763, 5765, 5767, 5771, 5773, 5775, 5779, 5781, 5785,
            #  5787, 5789, 5791, 5793, 5795, 5799, 5801, 5803, 5805, 5807, 5811, 5813, 5815, 5817, 5819, 5821, 5823, 5825,
            #  5827, 5831, 5833, 5835, 5839, 5841, 5843, 5845, 5847, 5849, 5851, 5853, 5855, 5859, 5861, 5863, 5865, 5867,
            #  5871, 5873, 5875, 5877, 5879, 5881, 5883, 5885, 6037, 6039, 6043, 6045, 6047, 6051, 6057, 6059, 6061, 6063,
            #  6065, 6067, 6069, 6071, 6075, 6077, 6079, 6083, 6085, 6087, 6089, 6091, 6093, 6095, 6097, 6099, 6103, 6105,
            #  6107, 6111, 6113, 6117, 6119, 6121, 6123, 6125, 6127, 6131, 6135, 6137, 6139, 6143, 6145, 6147, 6149, 6151,
            #  6153, 6155, 6157]
            # if orbit.orbit_num in bad_orbit_list:
            #     print("bad orbit: skipping")
            #     continue

            # Check if all orbits should be fitted, or only a subset by month
            include, reason = self._filter_timestamps(month, orbit.mean_mjd_obs)
            if not include:
                print(f"Skipping orbit {i}")
                if reason == "oob" and mapping_region:
                    mapping_region = 2
                continue
            else:
                if reason and reason.startswith("mask"):
                    rule = reason.split(":")[1]
            mapping_region = 1
            orbit.rule = rule
            orbit.load_zodi_orbit_data()
            orbit.apply_mask()
            self.all_orbits.append(orbit)
        return

    def load_splines(self, gain_spline_file, offset_spline_file):
        """Load gain spline and offset spline from '*.pkl' files saved in a file"""
        with open(gain_spline_file, "rb") as g1:
            self.gain_spline = pickle.load(g1)

        with open(offset_spline_file, "rb") as g2:
            self.offset_spline = pickle.load(g2)

    def run_iterative_fit(self, iterations, plot=False):
        """
        The aim of the fitting procedure is to calibrate the WISE data to the zodiacal light modelled using the Kelsall
        model.
        Each WISE orbit is fitted to the corresponding zodiacal light template. The zodiacal light is then subtracted
        from the calibrated orbit data to give the residual of the fit, which serves as a proxy for the galactic signal.
        After all orbits undergo this procedure, a full-sky coadd map is made using the residuals from each orbit. This
        map is equivalent to the zodi-subtracted full-sky WISE map.
        On the next iteration, this zodi-subtracted (i.e. galactic signal) map is subtracted from the data in advance of
        the fit to the zodiacal light template, in order to facilitate a better fit. In addition, any pixels for which
        the residual was an outlier (z-score > 1) are removed from the subsequent fit. These pixels are outliers
        because they are likely to contain strong signal from galactic emission, and are therefore detrimental to a
        good fit to the zodiacal light template.
        This procedure is iterated for the specified number of iterations. The fitted gain and offset values converge
        as the iterations continue.

        Parameters
        ----------
        :param iterations: int
            Number of iterations to use for the fitting algorithm
        :param month: str, optional
            Specify a month between 'Jan' and 'Aug' for partial map creation. Default is 'all'.
        :param plot: bool, optional
            Plot the fit for every fifteenth orbit if True. Default is False.
        """
        for it in range(iterations):
            self._set_output_filenames(label=f"iter{it}")

            # Reset numerator and denominator for keeping track of values using uncertainty propagation
            self.numerator_masked = np.zeros(self.npix)
            self.denominator_masked = np.zeros_like(self.numerator_masked)

            # For each iteration, iterate over all orbits
            for i, orbit in enumerate(self.all_orbits):
                print(f"Iteration {it}; Fitting orbit {orbit.orbit_num}")

                # Perform calibration fit
                orbit.fit()
                orbit.apply_fit()

                # orbit.save_orbit_map(label=it)

                # Add calibrated orbit to full-sky coadd
                self._add_orbit_iter(orbit)
                # if plot and i % 15 == 0.0:
                # orbit.plot_fit(self.output_path, iteration=it)

            # Save gains, offsets and timestamps for current iteration
            self._save_fit_params_to_file(it)

            # Finish making full-sky coadd map
            self._normalize()
            self._save_maps()

            # Set coadd map as the galactic signal template for the next iteration
            setattr(Orbit, "coadd_map", self.fsm_masked.mapdata)
            self.iter += 1

    def _add_orbit(self, orbit):
        """
        As each orbit is added to the final Healpix map, the distribution of values mapping to each Healpix pixel is
        recorded so that outliers can be removed before compiling the full map.

        Parameters
        ----------
        :param orbit: Orbit object
            Orbit to add
        """
        if (
                len(orbit.cal_uncs_clean_masked[orbit.cal_uncs_clean_masked != 0.0]) > 0
                and orbit.gain != 0.0
        ):
            orbit_data = orbit.zs_data_clean_masked
            orbit_uncs = orbit.cal_uncs_clean_masked
            orbit_pixels = orbit.pixel_inds_clean_masked
            for p, px in enumerate(orbit_pixels):
                self.all_data[px].append(orbit_data[p])
                self.all_uncs[px].append(orbit_uncs[p])

    def _add_orbit_iter(self, orbit):
        """
        Add zodi-subtracted values of a given orbit to the full-sky coadd map.

        Parameters
        ----------
        :param orbit: Orbit object
            Orbit to add
        """

        if (
                len(orbit._orbit_uncs_clean_masked[orbit._orbit_uncs_clean_masked != 0.0])
                > 0
                and orbit.gain != 0.0
        ):
            self.numerator_masked[orbit.pixel_inds_clean_masked] += np.divide(
                orbit.zs_data_clean_masked,
                np.square(orbit.cal_uncs_clean_masked),
                where=orbit.cal_uncs_clean_masked != 0.0,
                out=np.zeros_like(orbit.cal_uncs_clean_masked),
            )
            self.denominator_masked[orbit.pixel_inds_clean_masked] += np.divide(
                1,
                np.square(orbit.cal_uncs_clean_masked),
                where=orbit.cal_uncs_clean_masked != 0.0,
                out=np.zeros_like(orbit.cal_uncs_clean_masked),
            )

        return

    def _clean_data(self):
        """For each Healpix pixel, remove any values that have a z-score > 1"""
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

    def _compile_map(self):
        """
        Add cleaned values for each pixel into the Healpix map orbit by orbit.
        As each image is added, uncertainties are propagated as follows (x = data, s = uncertainty (sigma)):

        x_bar = ∑(x/s^2) / ∑(1/x^2)
        s_bar = 1 / ∑(1/x^2)

        The numerator and denominator for x_bar are recorded separately and a running count is maintained as each new
        orbit is added. This method divides the numerator by the denominator to give x_bar, and inverts the
        denominator to give s_bar.
        """
        for p, px_list in enumerate(self.all_data):
            unc_list = self.all_uncs[p]
            self.numerator_masked[p] += sum(
                np.array([px_list[i] / (unc_list[i] ** 2) for i in range(len(px_list))])
            )
            self.denominator_masked[p] += sum(
                np.array([1 / (unc_list[i] ** 2) for i in range(len(px_list))])
            )

    def _normalize(self):
        """
        Following _compile_map(), here the numerator is divided by the denominator to give the final value for each
        Healpix pixel
        """
        self.fsm_masked.mapdata = np.divide(
            self.numerator_masked,
            self.denominator_masked,
            where=self.denominator_masked != 0.0,
            out=np.zeros_like(self.denominator_masked),
        )
        self.unc_fsm_masked.mapdata = np.divide(
            1,
            self.denominator_masked,
            where=self.denominator_masked != 0.0,
            out=np.zeros_like(self.denominator_masked),
        )

    def _save_fit_params_to_file(self, it):
        """Save all fitted gains, fitted offsets and pixel timestamps to file for a given iteration"""
        print("Saving data for iteration {}".format(it))
        all_gains = np.array([orb.gain for orb in self.all_orbits])
        all_offsets = np.array([orb.offset for orb in self.all_orbits])
        all_mjd_vals = np.array([orb.orbit_mjd_obs for orb in self.all_orbits])
        all_segmented_offsets = np.array([orb.segmented_offsets for orb in self.all_orbits])
        with open(
                os.path.join(self.output_path, "fitvals_iter_{}.pkl".format(it)), "wb"
        ) as f:
            pickle.dump(
                [all_gains, all_offsets, all_mjd_vals, all_segmented_offsets],
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return

    def load_fitvals(self, it):
        """Load iteration fit values from pickle file"""
        with open("fitvals_iter_{}.pkl".format(it), "rb") as fitval_file:
            all_gains, all_offsets, all_mjd_vals, all_segmented_offsets = pickle.load(fitval_file)
        for i, orb in enumerate(self.all_orbits):
            orb.gain = all_gains[i]
            orb.offset = all_offsets[i]
            orb.orbit_mjd_obs = all_mjd_vals[i]
            orb.segmented_offsets = all_segmented_offsets[i]
        return

    def _save_maps(self):
        """Save full-sky maps (intensity and uncertainty) to file"""
        self.fsm_masked.save_map()
        self.unc_fsm_masked.save_map()

    def _set_output_filenames(self, label):
        """Initialize WISEMap objects for the full-sky maps (intensity and uncertainty)"""
        self.fsm_masked = WISEMap(
            self.fsm_map_file.replace(".fits", f"_{label}.fits"), self.band
        )
        self.unc_fsm_masked = WISEMap(
            self.unc_fsm_map_file.replace(".fits", f"_{label}.fits"), self.band
        )

    def _filter_timestamps(self, month_list, mjd_obs):
        """
        Return True if orbit timestamp is within desired time range ['Jan',...,'Aug'] or 'all'.
        Otherwise return False
        """
        if month_list == "all":
            month_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]

        if isinstance(month_list, str):
            month_list = [month_list]

        include = False
        reason = None
        for month in month_list:
            if include:
                return include, reason
            if month not in self.month_timestamps:
                print(
                    "Unrecognized time period. Please specify one of ['all', 'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', "
                    "'Aug']. Proceeding with 'all'."
                )
                include = True
            else:
                months = list(self.month_timestamps.keys())
                if months.index(month) == len(months) - 1:
                    if mjd_obs >= self.month_timestamps[month]:
                        include = True
                    else:
                        include = False
                        reason = "oob"  # out of bounds
                # elif months.index(month) == 0:
                #     if mjd_obs < self.month_timestamps[months[months.index(month) + 1]]:
                #         include = True
                #     else:
                #         include = False
                #         reason = "oob"  # out of bounds
                else:
                    current_month_num = months.index(month)
                    if (
                            self.month_timestamps[month]
                            <= mjd_obs
                            < self.month_timestamps[months[current_month_num + 1]]
                    ):
                        include = True
                    else:
                        include = False
                        reason = "oob"  # out of bounds
            if include:
                for c, chunk in enumerate(self.mask_orbit_timestamps):
                    chunk_start, chunk_end = chunk
                    if (chunk_start <= mjd_obs) & (mjd_obs < chunk_end):
                        include = True
                        reason = "mask:{}".format(self.mask_rules_stripes[c])  # moon stripe
        return include, reason
