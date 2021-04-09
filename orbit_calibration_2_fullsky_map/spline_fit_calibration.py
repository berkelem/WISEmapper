"""
:author: Matthew Berkeley
:date: Jul 2020

Module containing spline fitting class to fit a smooth curve through the fitted gains and offsets for all orbits

Main classes
------------
SplineFitter : Class for fitting a spline to the fitted gains and offsets for all orbits
"""

import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import stats
import os
from collections import OrderedDict


class SplineFitter:
    """
    Class for fitting a spline to the fitted gains and offsets for all orbits

    Parameters
    ----------
    :param iter_num: int
        Iteration number to use for the optimal gain/offset fits
    :param path_to_fitvals: str
        If other than the current working directory, provide the path to the "fitvals*pkl" files made by the Coadder
        class

    Methods
    -------
    fit_spline :
        Fit a spline curve through the fitted gains and offsets from selected iteration
    """

    def __init__(self, iter_num, path_to_fitvals=os.getcwd()):
        self.iter_num = iter_num
        self.output_path = path_to_fitvals
        self.fitvals_file = os.path.join(self.output_path, "fitvals_iter_{}.pkl".format(iter_num))
        self.gain_spline_file = os.path.join(self.output_path, "gain_spline.pkl")
        self.offset_spline_file = os.path.join(self.output_path, "offset_spline.pkl")
        self.spl_gain = None
        self.spl_offset = None

    def fit_spline(self, plot=True):
        """
        Fit a spline curve through the fitted gains and offsets from selected iteration. Gains and offsets are fitted
        independently. Aberrant fits associated with orbits that are corrupted by moon stripes are removed (by manual
        inspection) before fitting the spline.

        :param plot: bool, optional
            Plot the spline fit curves. Default is True.
        """
        all_gains, all_offsets, all_mjd_vals = self._load_fitvals()

        # apr_mask = [x[0] > 55287 for x in all_mjd_vals]
        # all_gains = all_gains[apr_mask]
        # all_offsets = all_offsets[apr_mask]
        # all_mjd_vals = all_mjd_vals[apr_mask]

        # all_gains = all_gains[::2]  # Even orbits
        # all_offsets = all_offsets[::2]
        # all_mjd_vals = all_mjd_vals[::2]

        # off galaxy orbits
        # all_gains = np.r_[all_gains[1:1295:2], all_gains[1296:5000:2], all_gains[5001::2]]
        # all_offsets = np.r_[all_offsets[1:1295:2], all_offsets[1296:5000:2], all_offsets[5001::2]]
        # all_mjd_vals = np.r_[all_mjd_vals[1:1295:2], all_mjd_vals[1296:5000:2], all_mjd_vals[5001::2]]
        #
        times_gain_masked, times_offset_masked, gains_masked, offsets_masked = self._clean_data(all_gains, all_offsets,
                                                                                                all_mjd_vals)

        # Off galaxy orbits
        # stripe_gains = ((55217 < times_gain_masked) & (times_gain_masked < 55225)) | (
        #                        (55246 < times_gain_masked) & (times_gain_masked < 55286)) | (
        #                        (55304 < times_gain_masked) & (times_gain_masked < 55315)) | (
        #                        (55335 < times_gain_masked) & (times_gain_masked < 55343)) | (
        #                        (55362 < times_gain_masked) & (times_gain_masked < 55387)) | (
        #                        (55405 < times_gain_masked) & (times_gain_masked < 55414))
        #
        # gains_masked[-15:] = np.mean(gains_masked[~stripe_gains][-15:])
        # stripe_gains[-15:] = False
        #
        # stripe_offsets = ((55217 < times_offset_masked) & (times_offset_masked < 55225)) | (
        #                        (55246 < times_offset_masked) & (times_offset_masked < 55286)) | (
        #                        (55304 < times_offset_masked) & (times_offset_masked < 55315)) | (
        #                        (55335 < times_offset_masked) & (times_offset_masked < 55343)) | (
        #                        (55362 < times_offset_masked) & (times_offset_masked < 55387)) | (
        #                        (55405 < times_offset_masked) & (times_offset_masked < 55414))
        #
        # self.spl_gain = UnivariateSpline(times_gain_masked[~stripe_gains], gains_masked[~stripe_gains], s=1000, k=5)
        # self.spl_offset = UnivariateSpline(times_offset_masked[~stripe_offsets], offsets_masked[~stripe_offsets],
        #                                    s=70000, k=3)

        # on galaxy orbits
        # all_gains = np.r_[all_gains[0:1295:2], all_gains[1295:5000:2], all_gains[5000::2]]
        # all_offsets = np.r_[all_offsets[0:1295:2], all_offsets[1295:5000:2], all_offsets[5000::2]]
        # all_mjd_vals = np.r_[all_mjd_vals[0:1295:2], all_mjd_vals[1295:5000:2], all_mjd_vals[5000::2]]
        #
        # times_gain_masked, times_offset_masked, gains_masked, offsets_masked = self._clean_data(all_gains, all_offsets,
        #                                                                                         all_mjd_vals)

        # stripe_gains = ((55200 < times_gain_masked) & (times_gain_masked < 55207.5)) | (
        #         (55217 < times_gain_masked) & (times_gain_masked < 55221)) | (
        #                        (55227 < times_gain_masked) & (times_gain_masked < 55236)) | (
        #                        (55246 < times_gain_masked) & (times_gain_masked < 55296)) | (
        #                        (55316 < times_gain_masked) & (times_gain_masked < 55325)) | (
        #                        (55345 < times_gain_masked) & (times_gain_masked < 55357)) | (
        #                        (55370 < times_gain_masked) & (times_gain_masked < 55386)) | (
        #                        (55393 < times_gain_masked) & (times_gain_masked < 55414))

        # gains_masked[:15] = np.mean(gains_masked[~stripe_gains][:15])
        # stripe_gains[:15] = False
        # gains_masked[-15:] = np.mean(gains_masked[~stripe_gains][-15:])
        # stripe_gains[-15:] = False

        # stripe_offsets = ((55200 < times_offset_masked) & (times_offset_masked < 55207.5)) | (
        #         (55217 < times_offset_masked) & (times_offset_masked < 55221)) | (
        #                        (55227 < times_offset_masked) & (times_offset_masked < 55236)) | (
        #                        (55246 < times_offset_masked) & (times_offset_masked < 55296)) | (
        #                        (55316 < times_offset_masked) & (times_offset_masked < 55325)) | (
        #                        (55345 < times_offset_masked) & (times_offset_masked < 55357)) | (
        #                        (55370 < times_offset_masked) & (times_offset_masked < 55386)) | (
        #                        (55393 < times_offset_masked) & (times_offset_masked < 55414))

        # offsets_masked[:15] = np.mean(offsets_masked[~stripe_offsets][:15])
        # stripe_offsets[:15] = False
        # offsets_masked[-15:] = np.mean(offsets_masked[~stripe_offsets][-15:])
        # stripe_offsets[-15:] = False

        # self.spl_gain = UnivariateSpline(times_gain_masked[~stripe_gains], gains_masked[~stripe_gains], s=1, k=5)
        # self.spl_offset = UnivariateSpline(times_offset_masked[~stripe_offsets], offsets_masked[~stripe_offsets],
        #                                    s=100000, k=3)

        # Even mask
        # Mask all gain and offset values that have aberrant values due to moon stripe regions causing a bad fit
        # stripe_gains = ((55200 < times_gain_masked) & (times_gain_masked < 55207.5)) | (
        #             (55218 < times_gain_masked) & (times_gain_masked < 55220)) | (
        #                            (55227 < times_gain_masked) & (times_gain_masked < 55236)) | (
        #                            (55247 < times_gain_masked) & (times_gain_masked < 55255)) | (
        #                            (55276 < times_gain_masked) & (times_gain_masked < 55284)) | (
        #                            (55306 < times_gain_masked) & (times_gain_masked < 55313.5)) | (
        #                            (55335 < times_gain_masked) & (times_gain_masked < 55343)) | (
        #                            (55362 < times_gain_masked) & (times_gain_masked < 55372)) | (
        #                            (55376 < times_gain_masked) & (times_gain_masked < 55384)) | (
        #                            (55393 < times_gain_masked) & (times_gain_masked < 55402)) | (
        #                            (55407 < times_gain_masked) & (times_gain_masked < 55414))
        # gains_masked[0:5] = np.mean(gains_masked[~stripe_gains][0:5])
        # stripe_gains[0:5] = False
        #
        # stripe_offsets = ((55200 < times_offset_masked) & (times_offset_masked < 55208)) | (
        #             (55218 < times_offset_masked) & (times_offset_masked < 55220)) | (
        #                              (55228 < times_offset_masked) & (times_offset_masked < 55236)) | (
        #                              (55247 < times_offset_masked) & (times_offset_masked < 55255)) | (
        #                              (55276 < times_offset_masked) & (times_offset_masked < 55284)) | (
        #                              (55305 < times_offset_masked) & (times_offset_masked < 55314)) | (
        #                              (55335 < times_offset_masked) & (times_offset_masked < 55343)) | (
        #                              (55362 < times_offset_masked) & (times_offset_masked < 55372)) | (
        #                              (55376 < times_offset_masked) & (times_offset_masked < 55382)) | (
        #                              (55393 < times_offset_masked) & (times_offset_masked < 55402)) | (
        #                              (55407 < times_offset_masked) & (times_offset_masked < 55414))
        #
        # offsets_masked[0:5] = np.mean(offsets_masked[~stripe_offsets][0:5])
        # stripe_offsets[0:5] = False



        # stripe_gains = ((55200 < times_gain_masked) & (times_gain_masked < 55205)) | (
        #         (55218 < times_gain_masked) & (times_gain_masked < 55224)) | (
        #                        (55230 < times_gain_masked) & (times_gain_masked < 55236)) | (
        #                        (55247 < times_gain_masked) & (times_gain_masked < 55255)) | (
        #                        (55259 < times_gain_masked) & (times_gain_masked < 55264)) | (
        #                        (55276 < times_gain_masked) & (times_gain_masked < 55284)) | (
        #                        (55287 < times_gain_masked) & (times_gain_masked < 55295)) | (
        #                        (55306 < times_gain_masked) & (times_gain_masked < 55313)) | (
        #                        (55318 < times_gain_masked) & (times_gain_masked < 55324)) | (
        #                        (55335 < times_gain_masked) & (times_gain_masked < 55343)) | (
        #                        (55348 < times_gain_masked) & (times_gain_masked < 55354)) | (
        #                        (55364 < times_gain_masked) & (times_gain_masked < 55370)) | (
        #                        (55378 < times_gain_masked) & (times_gain_masked < 55384)) | (
        #                        (55393 < times_gain_masked) & (times_gain_masked < 55402)) | (
        #                        (55408 < times_gain_masked) & (times_gain_masked < 55414))
        #
        # stripe_offsets = ((55200 < times_offset_masked) & (times_offset_masked < 55208)) | (
        #         (55218 < times_offset_masked) & (times_offset_masked < 55224)) | (
        #                          (55230 < times_offset_masked) & (times_offset_masked < 55236)) | (
        #                          (55247 < times_offset_masked) & (times_offset_masked < 55255)) | (
        #                          (55259 < times_offset_masked) & (times_offset_masked < 55266)) | (
        #                          (55276 < times_offset_masked) & (times_offset_masked < 55284)) | (
        #                          (55287 < times_offset_masked) & (times_offset_masked < 55295)) | (
        #                          (55306 < times_offset_masked) & (times_offset_masked < 55313)) | (
        #                          (55318 < times_offset_masked) & (times_offset_masked < 55324)) | (
        #                          (55335 < times_offset_masked) & (times_offset_masked < 55343)) | (
        #                          (55348 < times_offset_masked) & (times_offset_masked < 55354)) | (
        #                          (55364 < times_offset_masked) & (times_offset_masked < 55370)) | (
        #                          (55378 < times_offset_masked) & (times_offset_masked < 55384)) | (
        #                          (55393 < times_offset_masked) & (times_offset_masked < 55402)) | (
        #                          (55407 < times_offset_masked) & (times_offset_masked < 55414))

        stripe_gains = ~times_gain_masked.astype(bool)
        stripe_offsets = ~times_offset_masked.astype(bool)

        self.spl_gain = UnivariateSpline(times_gain_masked[~stripe_gains], gains_masked[~stripe_gains], s=100, k=3)
        self.spl_offset = UnivariateSpline(times_offset_masked[~stripe_offsets], offsets_masked[~stripe_offsets],
                                           s=10000, k=3)

        self._save_spline()

        if plot:
            self._plot_spline(times_gain_masked, stripe_gains, gains_masked,
                              times_offset_masked, stripe_offsets, offsets_masked)

        return

    def _load_fitvals(self):
        """Load iteration fit values from pickle file"""
        with open(self.fitvals_file, "rb") as fitval_file:
            all_gains, all_offsets, all_mjd_vals = pickle.load(fitval_file)
        return all_gains, all_offsets, all_mjd_vals

    def _plot_all_fitvals(self):
        """Helper method for plotting the gains and offsets before fitting a spline"""
        all_gains, all_offsets, all_mjd_vals = self._load_fitvals()
        median_mjd_vals = np.array([np.median(arr) for arr in all_mjd_vals])

        selected_data = (55197 <= median_mjd_vals) & (median_mjd_vals < 55228)

        plt.plot(median_mjd_vals[selected_data], all_gains[selected_data], "r.")
        plt.xlabel("Median MJD value")
        plt.ylabel("Fitted gain")
        plt.ylim((65,105))
        plt.savefig(os.path.join(self.output_path, "all_gains_iter_{}.png".format(self.iter_num)))
        plt.close()

        plt.plot(median_mjd_vals[selected_data], all_offsets[selected_data], "r.")
        plt.xlabel("Median MJD value")
        plt.ylabel("Fitted offset")
        plt.ylim((-300, 150))
        plt.savefig(os.path.join(self.output_path, "all_offsets_iter_{}.png".format(self.iter_num)))
        plt.close()

    def _plot_fit_evolution(self, orbit_num):
        """
        Plot the evolution of gain/offset for a given orbit over n iterations

        :param orbit_num: int
            Orbit number to inspect
        """
        gains = []
        offsets = []
        for it in range(self.iter_num):
            self.fitvals_file = os.path.join(self.output_path, "fitvals_iter_{}.pkl".format(it))
            all_gains, all_offsets, all_mjd_vals = self._load_fitvals()
            gains.append(all_gains)
            offsets.append(all_offsets)

        orbit_gains = [gains[i][orbit_num] for i in range(len(gains))]
        orbit_offsets = [offsets[i][orbit_num] for i in range(len(offsets))]

        plt.plot(range(self.iter_num), orbit_gains, "r.")
        plt.xlabel("Iteration")
        plt.ylabel("Fitted gain")
        plt.savefig(os.path.join(self.output_path, "orbit_{}_gains.png".format(orbit_num)))
        plt.close()

        plt.plot(range(self.iter_num), orbit_offsets, "r.")
        plt.xlabel("Iteration")
        plt.ylabel("Fitted gain")
        plt.savefig(os.path.join(self.output_path, "orbit_{}_offsets.png".format(orbit_num)))
        plt.close()

    @staticmethod
    def _clean_data(all_gains, all_offsets, all_mjd_vals):
        """
        Helper method for removing outliers in gain and offset values.
        As all fitted gain/offset values are very close, a very tight constraint is placed on the z-score (0.1)

        Parameters
        ----------
        :param all_gains: numpy.array
            Gain values for all orbits
        :param all_offsets: numpy.array
            Offset values for all orbits
        :param all_mjd_vals: numpy.array
            Median timestamps for all orbits
        :return times_gain_masked, times_offset_masked, gains_masked, offsets_masked: all numpy.arrays
            Cleaned arrays for timestamps associated with gains, timestamps associated with offsets, gains, and offsets.
        """

        median_mjd_vals = np.array([np.median(arr) for arr in all_mjd_vals])
        return median_mjd_vals, median_mjd_vals, all_gains, all_offsets

        # z_gains = np.abs(stats.zscore(all_gains))
        # mask_gains = z_gains > 1
        #
        # z_offsets = np.abs(stats.zscore(all_offsets))
        # mask_offsets = z_offsets > 1
        #
        # times_gain_masked = median_mjd_vals[~mask_gains]
        # gains_masked = all_gains[~mask_gains]
        #
        # times_offset_masked = median_mjd_vals[~mask_offsets]
        # offsets_masked = all_offsets[~mask_offsets]
        #
        # return times_gain_masked, times_offset_masked, gains_masked, offsets_masked

    def _save_spline(self):
        """Save splines to pickle files, to be read in later by the Coadder object"""

        with open(self.gain_spline_file, "wb") as f:
            pickle.dump(self.spl_gain, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.offset_spline_file, "wb") as g:
            pickle.dump(self.spl_offset, g, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def _plot_spline(self, times_gain_masked, stripe_gains, gains_masked,
                     times_offset_masked, stripe_offsets, offsets_masked):
        """
        Plot all gains/offsets along with the fitted spline curve.
        Original data is red, masked values are blue, spline is green.

        Parameters
        ----------
        :param times_gain_masked: numpy.array
            X-axis values for gain plot
        :param stripe_gains: numpy.array
            Boolean mask identifying aberrant gain values caused by a moon stripe
        :param gains_masked: numpy.array
            Original gain fit values
        :param times_offset_masked: numpy.array
            X-axis values for offset plot
        :param stripe_offsets: numpy.array
            Boolean mask identifying aberrant offset values caused by a moon stripe
        :param offsets_masked: numpy.array
            Original offset fit values
        """

        str_month_dict = OrderedDict([(55197, "Jan"), (55228, "Feb"), (55256, "Mar"), (55287, "Apr"),
                                             (55317, "May"), (55348, "Jun"), (55378, "Jul"), (55409, "Aug")])

        min_time = min(times_gain_masked)
        max_time = max(times_gain_masked)
        month_start_times = list(str_month_dict.keys())

        start_month_ind = month_start_times.index(min(month_start_times, key=lambda x: abs(x - min_time)))
        start_month_ind = start_month_ind if month_start_times[start_month_ind] < min_time else start_month_ind-1

        end_month_ind = month_start_times.index(min(month_start_times, key=lambda x: abs(x - max_time)))
        end_month_ind = end_month_ind if month_start_times[end_month_ind] > max_time else end_month_ind+1

        x_ticks = month_start_times[start_month_ind:end_month_ind+1]

        fig, ax = plt.subplots()
        ax.plot(times_gain_masked[stripe_gains], gains_masked[stripe_gains], 'ko', alpha=0.2, ms=3)
        ax.plot(times_gain_masked[~stripe_gains], gains_masked[~stripe_gains], 'ro', ms=3)
        ax.plot(times_gain_masked, self.spl_gain(times_gain_masked), 'g', lw=2)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str_month_dict[x] for x in x_ticks], rotation=45)
        plt.subplots_adjust(bottom=0.2)
        plt.xlabel("Orbit median timestamp")
        plt.ylabel("Fitted Gain")
        plt.ylim((65, 105))
        plt.savefig(os.path.join(self.output_path, "spline_gains.png"))
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(times_offset_masked[stripe_offsets], offsets_masked[stripe_offsets], 'ko', alpha=0.2, ms=3)
        ax.plot(times_offset_masked[~stripe_offsets], offsets_masked[~stripe_offsets], 'ro', ms=3)
        ax.plot(times_offset_masked, self.spl_offset(times_offset_masked), 'g', lw=2)
        ax.set_xticks(np.array(x_ticks))
        ax.set_xticklabels(np.array([str_month_dict[x] for x in x_ticks]), rotation=45)
        plt.subplots_adjust(bottom=0.2)
        plt.xlabel("Orbit median timestamp")
        plt.ylabel("Fitted Offset")
        plt.ylim((-300, 150))
        plt.savefig(os.path.join(self.output_path, "spline_offsets.png"))
        plt.close()

        return
