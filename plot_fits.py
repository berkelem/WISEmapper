import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_fits():
    with open("gain_fits.pkl", "rb") as gain_file:
        gains = pickle.load(gain_file)
    with open("offset_fits.pkl", "rb") as offset_file:
        offsets = pickle.load(offset_file)

    return gains, offsets

def plot_fit_values_over_iterations(gains, offsets):
    for i in range(len(gains[0])):
        gain_vals = [gains[j][i] for j in range(len(gains))]
        offset_vals = [offsets[j][i] for j in range(len(offsets))]
        iterations = len(gain_vals)

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Gain', color=color)
        ax1.plot(range(iterations), gain_vals, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Offset', color=color)  # we already handled the x-label with ax1
        ax2.plot(range(iterations), offset_vals, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.suptitle(f"Orbit {i}")
        plt.savefig(f"plots/orbit_{i}.png")
        plt.close()

def plot_optimal_fit_values(gains, offsets):
    opt_gains = gains[-1].flatten()
    opt_offsets = offsets[-1].flatten()
    n_orbits = len(opt_gains)

    bad_fits = [ind for ind, val in enumerate(opt_offsets) if val == 0.0]
    good_gains = np.array([opt_gains[i] for i in range(len(opt_gains)) if i not in bad_fits])
    good_offsets = np.array([opt_offsets[i] for i in range(len(opt_offsets)) if i not in bad_fits])
    good_inds = np.array([i for i in range(len(opt_offsets)) if i not in bad_fits])

    interp_gains = np.interp(bad_fits, good_inds, good_gains)
    interp_offsets = np.interp(bad_fits, good_inds, good_offsets)

    opt_gains = np.array(opt_gains)
    opt_gains[bad_fits] = interp_gains

    opt_offsets = np.array(opt_offsets)
    opt_offsets[bad_fits] = interp_offsets

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Orbit')
    ax1.set_ylabel('Gain', color=color)
    ax1.plot(range(n_orbits), opt_gains, 'r.')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Offset', color=color)  # we already handled the x-label with ax1
    ax2.plot(range(n_orbits), opt_offsets, 'b.')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.suptitle(f"Optimal fit values")
    plt.savefig(f"optimal_fits.png")
    plt.close()
    return opt_gains, opt_offsets

def interpolate_over_pixlimit(npix_limit, gains, offsets):
    opt_gains = gains[-1].flatten()
    opt_offsets = offsets[-1].flatten()

    with open("num_px_for_gain_cal.pkl", "rb") as g1:
        npix_gain = np.array(pickle.load(g1))
    with open("num_px_for_offset_cal.pkl", "rb") as g2:
        npix_offset = np.array(pickle.load(g2))

    orbit_num = np.arange(len(opt_gains))

    bad_fit_mask = opt_offsets != 0.0
    px_num_mask_gain = npix_gain > npix_limit
    px_num_mask_offset = npix_offset > npix_limit
    mask_gain = px_num_mask_gain & bad_fit_mask
    mask_offset = px_num_mask_offset & bad_fit_mask

    good_orbits_gain = orbit_num[mask_gain]
    good_gains = opt_gains[mask_gain]

    good_orbits_offset = orbit_num[mask_offset]
    good_offsets = opt_offsets[mask_offset]

    plt.plot(good_orbits_gain, good_gains, 'r.')
    plt.xlabel("Orbit number")
    plt.ylabel("Good Gains")
    plt.savefig(f"interp_plots/good_gains_{npix_limit}.png")
    plt.close()

    plt.plot(good_orbits_offset, good_offsets, 'r.')
    plt.xlabel("Orbit number")
    plt.ylabel("Good Offsets")
    plt.savefig(f"interp_plots/good_offsets_{npix_limit}.png")
    plt.close()

    f_gains = np.interp(orbit_num[~mask_gain], good_orbits_gain, good_gains)
    f_offsets = np.interp(orbit_num[~mask_offset], good_orbits_offset, good_offsets)

    opt_gains[~mask_gain] = f_gains
    opt_offsets[~mask_offset] = f_offsets

    plt.plot(orbit_num, opt_gains, 'r.')
    plt.xlabel("Orbit number")
    plt.ylabel("Gain")
    plt.savefig(f"interp_plots/interp_gains_{npix_limit}_pixlimit.png")
    plt.close()

    plt.plot(orbit_num, opt_offsets, 'b.')
    plt.xlabel("Orbit number")
    plt.ylabel("Offset")
    plt.savefig(f"interp_plots/interp_offsets_{npix_limit}_pixlimit.png")
    plt.close()

    return opt_gains, opt_offsets

def smooth_params(gains, offsets, pixlimit):

    window_size = 251
    smooth_gains = median_filter(gains, window_size)
    smooth_offsets = median_filter(offsets, window_size)

    plt.plot(gains, 'r.', smooth_gains, 'b')
    plt.xlabel("Orbit number")
    plt.ylabel("Gain")
    plt.savefig(f"smoothed_plots/smoothed_opt_gains_window_{window_size}_pixlimit_{pixlimit}.png")
    plt.close()

    plt.plot(offsets, 'r.', smooth_offsets, 'b')
    plt.xlabel("Orbit number")
    plt.ylabel("Offset")
    plt.savefig(f"smoothed_plots/smoothed_opt_offsets_window_{window_size}_pixlimit_{pixlimit}.png")
    plt.close()
    return smooth_gains, smooth_offsets

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

if __name__ == "__main__":
    gains, offsets = load_fits()
    # plot_optimal_fit_values(gains, offsets)
    # plot_fit_values_over_iterations(gains, offsets)
    # for px_limit in range(300, 1500, 100):
    px_limit = 400
    interp_gains, interp_offsets = interpolate_over_pixlimit(px_limit, gains, offsets)
    smooth_gains, smooth_offsets = smooth_params(interp_gains, interp_offsets, px_limit)
    with open("smoothed_interp_params.pkl", "wb") as params_file:
        pickle.dump(np.array([smooth_gains, smooth_offsets]), params_file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(np.array([smooth_gains, smooth_offsets]), params_file, pickle.HIGHEST_PROTOCOL)