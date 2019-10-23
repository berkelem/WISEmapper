from fullskymapping import MapCombiner
import os
import healpy as hp
import matplotlib.pyplot as plt


def combine_days(band, days):
    mc = MapCombiner(band)
    mc.add_days(0, days, smooth_window=11)
    # mc.normalize()
    mc.save_file()
    return mc.fsm.mapdata



def plot_map(data, days, band):
    scale_dict = {1: 1, 2: 1, 3: 35, 4: 70}
    hp.mollview(data, unit='MJy/sr', max=scale_dict[band])
    plt.savefig(f'days{days}_band{band}.png')
    plt.close()

if __name__ == "__main__":
    for band in range(3,4):#1, 5):
        print(f"Creating band {band} map")
        max_day = -1
        day_available = True
        while day_available:
            max_day += 1
            day_available = os.path.exists(f"/home/users/mberkeley/wisemapper/data/output_maps/w{band}/fsm_w{band}_day_{max_day}.fits")
        print(f"Combining {max_day} days")
        mapdata = combine_days(band, max_day)
        plot_map(mapdata, max_day, band)

    # import pickle
    # import numpy as np
    # import numpy.ma as ma
    # popt_file = "popt_w3.pkl"
    # with open(popt_file, "rb") as f:
    #     gain, offset = pickle.load(f)
    #
    # bad_fit_days = [26, 27, 28, 30, 46, 47, 48, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
    #                      67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 85, 86, 87, 88, 95, 98, 99, 100,
    #                      102, 108, 112, 113, 114, 115, 116, 117, 125, 126, 127, 128, 129, 130, 131, 132, 136, 146,
    #                      147, 150, 158, 176, 177]
    # bad_fit_days2 = [0, 23, 24, 25, 36, 37, 38, 39, 41, 42, 49, 84, 89, 90, 91, 92, 93, 94, 96, 97, 101, 103, 104, 105,
    #                  106, 118, 120, 121, 122, 123, 124, 133, 136, 138, 139, 140, 141, 142, 143, 144, 145, 151, 180,
    #                  194, 199, 200, 201, 202, 203, 204, 205]
    #
    # all_bad_days = bad_fit_days + bad_fit_days2
    #
    # mask = np.zeros_like(gain)
    # mask[bad_fit_days] = 1
    # mask[bad_fit_days2] = 1
    # days = np.arange(len(gain))
    # days_masked = ma.array(days, mask=mask)
    # masked_gain = ma.array(gain, mask=mask)
    # masked_offset = ma.array(offset, mask=mask)
    #
    # interp_gain = np.interp(all_bad_days, days_masked.compressed(), masked_gain.compressed())
    # gain[all_bad_days] = interp_gain
    #
    # interp_offset = np.interp(all_bad_days, days_masked.compressed(), masked_offset.compressed())
    # offset[all_bad_days] = interp_offset
    #
    # from calibration import SmoothFit
    # smooth_gain = SmoothFit.mean_filter(gain, 25)
    # smooth_offset = SmoothFit.mean_filter(offset, 25)
    #
    # plt.plot(days_masked, masked_gain, 'r.', days, smooth_gain, 'b')
    # plt.xlabel("Days")
    # plt.ylabel("Gain")
    # plt.savefig("masked_gain.png")
    # plt.close()
    #
    # plt.plot(days, masked_offset, 'r.', days, smooth_offset, 'b')
    # plt.xlabel("Days")
    # plt.ylabel("Offset")
    # plt.savefig("masked_offset.png")
    # plt.close()
    #
    #
    # print(days)