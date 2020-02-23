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

if __name__ == "__main__":
    gains, offsets = load_fits()
    plot_optimal_fit_values(gains, offsets)
    # plot_fit_values_over_iterations(gains, offsets)