import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import stats


def load_fitvals(iter_num):
    with open("fitvals_iter_{}.pkl".format(iter_num), "rb") as fitval_file:
        all_gains, all_offsets, all_mjd_vals = pickle.load(fitval_file)
    return all_gains, all_offsets, all_mjd_vals

def plot_all_fitvals(iter_num):
    all_gains, all_offsets, all_mjd_vals = load_fitvals(iter_num)
    median_mjd_vals = np.array([np.median(arr) for arr in all_mjd_vals])

    plt.plot(median_mjd_vals, all_gains, "r.")
    plt.xlabel("Median MJD value")
    plt.ylabel("Fitted gain")
    plt.savefig("all_gains_iter_{}.png".format(iter_num))
    plt.close()

    plt.plot(median_mjd_vals, all_offsets, "r.")
    plt.xlabel("Median MJD value")
    plt.ylabel("Fitted offset")
    plt.savefig("all_offsets_iter_{}.png".format(iter_num))
    plt.close()

def plot_fit_evolution(orbit_num):
    gains = []
    offsets = []
    mjd_vals = []
    for it in range(25):
        all_gains, all_offsets, all_mjd_vals = load_fitvals(it)
        gains.append(all_gains)
        offsets.append(all_offsets)

    orbit_gains = [gains[i][orbit_num] for i in range(len(gains))]
    orbit_offsets = [offsets[i][orbit_num] for i in range(len(offsets))]

    plt.plot(range(25), orbit_gains, "r.")
    plt.xlabel("Iteration")
    plt.ylabel("Fitted gain")
    plt.savefig("orbit_{}_gains.png".format(orbit_num))
    plt.close()

    plt.plot(range(25), orbit_offsets, "r.")
    plt.xlabel("Iteration")
    plt.ylabel("Fitted gain")
    plt.savefig("orbit_{}_offsets.png".format(orbit_num))
    plt.close()

def fit_spline(iter_num):
    all_gains, all_offsets, all_mjd_vals = load_fitvals(iter_num)
    median_mjd_vals = np.array([np.median(arr) for arr in all_mjd_vals])

    z_gains = np.abs(stats.zscore(all_gains))
    mask_gains = z_gains > 0.1

    z_offsets = np.abs(stats.zscore(all_offsets))
    mask_offsets = z_offsets > 0.1

    times_gain_masked = median_mjd_vals[~mask_gains]
    gains_masked = all_gains[~mask_gains]

    times_offset_masked = median_mjd_vals[~mask_offsets]
    offsets_masked = all_offsets[~mask_offsets]

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

    spl_gain = UnivariateSpline(times_gain_masked[~stripe_gains], gains_masked[~stripe_gains], s=5000, k=5)
    spl_offset = UnivariateSpline(times_offset_masked[~stripe_offsets], offsets_masked[~stripe_offsets], s=500000, k=5)

    # with open("gain_spline.pkl", "wb") as f:
    #     pickle.dump(spl_gain, f, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open("offset_spline.pkl", "wb") as g:
    #     pickle.dump(spl_offset, g, protocol=pickle.HIGHEST_PROTOCOL)

    str_month_list = ['1 Jan 10', '1 Feb 10', '1 Mar 10', '1 Apr 10', '1 May 10', '1 Jun 10', '1 Jul 10', '1 Aug 10']


    fig, ax = plt.subplots()
    ax.plot(times_gain_masked[stripe_gains], gains_masked[stripe_gains], 'bo', ms=5)
    ax.plot(times_gain_masked[~stripe_gains], gains_masked[~stripe_gains], 'ro', ms=5)
    ax.plot(times_gain_masked, spl_gain(times_gain_masked), 'g', lw=3)
    ax.set_xticks([55197, 55228, 55256, 55287, 55317, 55348, 55378, 55409])
    ax.set_xticklabels(str_month_list, rotation=45)
    plt.subplots_adjust(bottom=0.2)
    plt.xlabel("Orbit median timestamp")
    plt.ylabel("Fitted Gain")
    plt.savefig("spline_gains.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(times_offset_masked[stripe_offsets], offsets_masked[stripe_offsets], 'bo', ms=5)
    ax.plot(times_offset_masked[~stripe_offsets], offsets_masked[~stripe_offsets], 'ro', ms=5)
    ax.plot(times_offset_masked, spl_offset(times_offset_masked), 'g', lw=3)
    ax.set_xticks([55197, 55228, 55256, 55287, 55317, 55348, 55378, 55409])
    ax.set_xticklabels(str_month_list, rotation=45)
    plt.subplots_adjust(bottom=0.2)
    plt.xlabel("Orbit median timestamp")
    plt.ylabel("Fitted Offset")
    plt.savefig("spline_offsets.png")
    plt.close()

    return


if __name__ == "__main__":
    # plot_all_fitvals(24)
    # plot_fit_evolution(100)
    fit_spline(24)