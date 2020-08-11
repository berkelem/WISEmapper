import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import stats
import os


class SplineFitter:

    def __init__(self, iter_num, path_to_fitvals=os.getcwd()):
        self.iter_num = iter_num
        self.output_path = path_to_fitvals
        self.fitvals_file = os.path.join(path_to_fitvals, "fitvals_iter_{}.pkl".format(iter_num))
        self.spl_gain = None
        self.spl_offset = None

    def load_fitvals(self):
        with open(self.fitvals_file, "rb") as fitval_file:
            all_gains, all_offsets, all_mjd_vals = pickle.load(fitval_file)
        return all_gains, all_offsets, all_mjd_vals

    def plot_all_fitvals(self):
        all_gains, all_offsets, all_mjd_vals = self.load_fitvals()
        median_mjd_vals = np.array([np.median(arr) for arr in all_mjd_vals])

        plt.plot(median_mjd_vals, all_gains, "r.")
        plt.xlabel("Median MJD value")
        plt.ylabel("Fitted gain")
        plt.savefig(os.path.join(self.output_path, "all_gains_iter_{}.png".format(self.iter_num)))
        plt.close()

        plt.plot(median_mjd_vals, all_offsets, "r.")
        plt.xlabel("Median MJD value")
        plt.ylabel("Fitted offset")
        plt.savefig(os.path.join(self.output_path, "all_offsets_iter_{}.png".format(self.iter_num)))
        plt.close()

    def plot_fit_evolution(self, orbit_num):
        gains = []
        offsets = []
        for it in range(25):
            all_gains, all_offsets, all_mjd_vals = self.load_fitvals()
            gains.append(all_gains)
            offsets.append(all_offsets)

        orbit_gains = [gains[i][orbit_num] for i in range(len(gains))]
        orbit_offsets = [offsets[i][orbit_num] for i in range(len(offsets))]

        plt.plot(range(25), orbit_gains, "r.")
        plt.xlabel("Iteration")
        plt.ylabel("Fitted gain")
        plt.savefig(os.path.join(self.output_path, "orbit_{}_gains.png".format(orbit_num)))
        plt.close()

        plt.plot(range(25), orbit_offsets, "r.")
        plt.xlabel("Iteration")
        plt.ylabel("Fitted gain")
        plt.savefig(os.path.join(self.output_path, "orbit_{}_offsets.png".format(orbit_num)))
        plt.close()

    def clean_data(self, all_gains, all_offsets, all_mjd_vals):
        median_mjd_vals = np.array([np.median(arr) for arr in all_mjd_vals])

        z_gains = np.abs(stats.zscore(all_gains))
        mask_gains = z_gains > 0.1

        z_offsets = np.abs(stats.zscore(all_offsets))
        mask_offsets = z_offsets > 0.1

        times_gain_masked = median_mjd_vals[~mask_gains]
        gains_masked = all_gains[~mask_gains]

        times_offset_masked = median_mjd_vals[~mask_offsets]
        offsets_masked = all_offsets[~mask_offsets]

        return times_gain_masked, times_offset_masked, gains_masked, offsets_masked


    def fit_spline(self, plot=True):
        all_gains, all_offsets, all_mjd_vals = self.load_fitvals()


        times_gain_masked, times_offset_masked, gains_masked, offsets_masked = self.clean_data(all_gains, all_offsets,
                                                                                               all_mjd_vals)

        stripe_gains = ((55200 < times_gain_masked) & (times_gain_masked < 55205)) | (
                    (55218 < times_gain_masked) & (times_gain_masked < 55224)) | (
                                   (55230 < times_gain_masked) & (times_gain_masked < 55236)) | (
                                   (55247 < times_gain_masked) & (times_gain_masked < 55255)) | (
                                   (55259 < times_gain_masked) & (times_gain_masked < 55264)) | (
                                   (55276 < times_gain_masked) & (times_gain_masked < 55284)) | (
                                   (55287 < times_gain_masked) & (times_gain_masked < 55295)) | (
                                   (55306 < times_gain_masked) & (times_gain_masked < 55313)) | (
                                   (55318 < times_gain_masked) & (times_gain_masked < 55324)) | (
                                   (55335 < times_gain_masked) & (times_gain_masked < 55343)) | (
                                   (55348 < times_gain_masked) & (times_gain_masked < 55354)) | (
                                   (55364 < times_gain_masked) & (times_gain_masked < 55370)) | (
                                   (55378 < times_gain_masked) & (times_gain_masked < 55384)) | (
                                   (55393 < times_gain_masked) & (times_gain_masked < 55402)) | (
                                   (55408 < times_gain_masked) & (times_gain_masked < 55414))

        stripe_offsets = ((55200 < times_offset_masked) & (times_offset_masked < 55208)) | (
                 (55218 < times_offset_masked) & (times_offset_masked < 55224)) | (
                               (55230 < times_offset_masked) & (times_offset_masked < 55236)) | (
                               (55247 < times_offset_masked) & (times_offset_masked < 55255)) | (
                               (55259 < times_offset_masked) & (times_offset_masked < 55266)) | (
                               (55276 < times_offset_masked) & (times_offset_masked < 55284)) | (
                               (55287 < times_offset_masked) & (times_offset_masked < 55295)) | (
                               (55306 < times_offset_masked) & (times_offset_masked < 55313)) | (
                               (55318 < times_offset_masked) & (times_offset_masked < 55324)) | (
                               (55335 < times_offset_masked) & (times_offset_masked < 55343)) | (
                               (55348 < times_offset_masked) & (times_offset_masked < 55354)) | (
                               (55364 < times_offset_masked) & (times_offset_masked < 55370)) | (
                               (55378 < times_offset_masked) & (times_offset_masked < 55384)) | (
                               (55393 < times_offset_masked) & (times_offset_masked < 55402)) | (
                               (55407 < times_offset_masked) & (times_offset_masked < 55414))

        self.spl_gain = UnivariateSpline(times_gain_masked[~stripe_gains], gains_masked[~stripe_gains], s=5000, k=5)
        self.spl_offset = UnivariateSpline(times_offset_masked[~stripe_offsets], offsets_masked[~stripe_offsets], s=500000, k=5)

        self.save_spline()

        if plot:
            self.plot_spline(times_gain_masked, stripe_gains, gains_masked,
                             times_offset_masked, stripe_offsets, offsets_masked)

        return

    def save_spline(self):

        with open(os.path.join(self.output_path, "gain_spline.pkl"), "wb") as f:
            pickle.dump(self.spl_gain, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.output_path, "offset_spline.pkl"), "wb") as g:
            pickle.dump(self.spl_offset, g, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def plot_spline(self, times_gain_masked, stripe_gains, gains_masked,
                    times_offset_masked, stripe_offsets, offsets_masked):

        str_month_list = ['1 Jan 10', '1 Feb 10', '1 Mar 10', '1 Apr 10', '1 May 10', '1 Jun 10', '1 Jul 10',
                          '1 Aug 10']

        fig, ax = plt.subplots()
        ax.plot(times_gain_masked[stripe_gains], gains_masked[stripe_gains], 'bo', ms=5)
        ax.plot(times_gain_masked[~stripe_gains], gains_masked[~stripe_gains], 'ro', ms=5)
        ax.plot(times_gain_masked, self.spl_gain(times_gain_masked), 'g', lw=3)
        ax.set_xticks([55197, 55228, 55256, 55287, 55317, 55348, 55378, 55409])
        ax.set_xticklabels(str_month_list, rotation=45)
        plt.subplots_adjust(bottom=0.2)
        plt.xlabel("Orbit median timestamp")
        plt.ylabel("Fitted Gain")
        plt.savefig(os.path.join(self.output_path, "spline_gains.png"))
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(times_offset_masked[stripe_offsets], offsets_masked[stripe_offsets], 'bo', ms=5)
        ax.plot(times_offset_masked[~stripe_offsets], offsets_masked[~stripe_offsets], 'ro', ms=5)
        ax.plot(times_offset_masked, self.spl_offset(times_offset_masked), 'g', lw=3)
        ax.set_xticks([55197, 55228, 55256, 55287, 55317, 55348, 55378, 55409])
        ax.set_xticklabels(str_month_list, rotation=45)
        plt.subplots_adjust(bottom=0.2)
        plt.xlabel("Orbit median timestamp")
        plt.ylabel("Fitted Offset")
        plt.savefig(os.path.join(self.output_path, "spline_offsets.png"))
        plt.close()

        return
