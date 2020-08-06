from fullskymapping import MapCombiner
import os
import healpy as hp
import matplotlib.pyplot as plt

class MapCombiner:

    def __init__(self, band, nside=256):
        self.band = band
        self.path = f"/home/users/mberkeley/wisemapper/data/output_maps/w{band}/"
        self.mapname = f'fsm_w{band}_all_interp.fits'
        self.uncname = self.mapname.replace('.fits', '_unc.fits')
        self.fsm = WISEMap(self.mapname, nside)
        self.unc_fsm = WISEMap(self.uncname, nside)
        self.numerator = np.zeros_like(self.fsm.mapdata)
        self.denominator = np.zeros_like(self.fsm.mapdata)


    def add_days(self, day_start, day_end, smooth_window=11):

        smooth_fitter = SmoothFit(self.band, "/home/users/mberkeley/wisemapper/scripts")
        gain, offset = smooth_fitter.load_fit_vals()

        bad_fit_days = [26, 27, 28, 30, 46, 47, 48, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                        67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 85, 86, 87, 88, 95, 98, 99, 100,
                        102, 108, 112, 113, 114, 115, 116, 117, 125, 126, 127, 128, 129, 130, 131, 132, 136, 146,
                        147, 150, 158, 176, 177]
        bad_fit_days2 = [0, 23, 24, 25, 36, 37, 38, 39, 41, 42, 49, 84, 89, 90, 91, 92, 93, 94, 96, 97, 101, 103, 104,
                         105,
                         106, 118, 120, 121, 122, 123, 124, 133, 136, 138, 139, 140, 141, 142, 143, 144, 145, 151, 180,
                         194, 199, 200, 201, 202, 203, 204, 205]

        all_bad_days = bad_fit_days + bad_fit_days2
        import numpy.ma as ma
        mask = np.zeros_like(gain)
        mask[bad_fit_days] = 1
        mask[bad_fit_days2] = 1
        days = np.arange(len(gain))
        days_masked = ma.array(days, mask=mask)
        masked_gain = ma.array(gain, mask=mask)
        masked_offset = ma.array(offset, mask=mask)

        interp_gain = np.interp(all_bad_days, days_masked.compressed(), masked_gain.compressed())
        gain[all_bad_days] = interp_gain

        interp_offset = np.interp(all_bad_days, days_masked.compressed(), masked_offset.compressed())
        offset[all_bad_days] = interp_offset

        smooth_gain = smooth_fitter.mean_filter(gain, smooth_window)
        smooth_offset = smooth_fitter.mean_filter(offset, smooth_window)
        smooth_fitter.plot_smoothfit_vals(gain, smooth_gain, offset, smooth_offset)

        map_datas = []
        map_uncs = []
        day_maps = []
        for i in range(day_start, day_end):
            filename = f"{self.path}fsm_w{self.band}_day_{i}.fits"
            if not os.path.exists(filename):
                print(f'Skipping file {filename} as it does not exist')
                continue
            day_map = WISEMap(filename, self.band)
            day_maps.append(day_map)
            day_map.read_data()

            orig_gain = gain[i-day_start]
            orig_offset = offset[i-day_start]
            adj_gain = smooth_gain[i-day_start]
            adj_offset = smooth_offset[i-day_start]
            data = day_maps[i-day_start].mapdata

            adj_data = smooth_fitter.adjust_calibration(data, orig_gain, adj_gain, orig_offset, adj_offset)
            map_datas.append(adj_data)

            day_uncmap = WISEMap(filename.replace("day", "unc_day"), self.band)
            day_uncmap.read_data()
            day_uncmap.mapdata = np.sqrt(np.abs(day_uncmap.mapdata)) * np.abs(adj_gain / orig_gain)
            # unc data originally saved as variance, so have to take sqrt. Also adjust for smooth fit by applying
            # gain adjustment factor.
            map_uncs.append(day_uncmap.mapdata)

            # self.numerator += np.divide(adj_data, day_uncmap.mapdata**2, out=np.zeros_like(adj_data),
            #                             where=day_uncmap.mapdata != 0.0)
            # self.denominator += np.divide(np.ones_like(adj_data), day_uncmap.mapdata**2,
            #                               out=np.zeros_like(adj_data), where=day_uncmap.mapdata != 0.0)
        px_vals = np.array(map_datas).T
        unc_vals = np.array(map_uncs).T
        for p, px in enumerate(px_vals):
            if len(px[px > 0.0]) > 2:
                mask = self.mask_outliers(px)
                good_vals = px[px > 0.0][~mask]
                good_unc_vals = unc_vals[p][px > 0.0][~mask]
            else:
                good_vals = px[px > 0.0]
                good_unc_vals = unc_vals[p][px > 0.0]

            if len(good_unc_vals) > 0:
                numerator = np.sum(good_vals / good_unc_vals ** 2)
                denominator = np.sum(1 / good_unc_vals ** 2)
                self.fsm.mapdata[p] = numerator / denominator
                self.unc_fsm.mapdata[p] = 1 / np.sqrt(denominator)
        return

    def normalize(self):
        self.fsm.mapdata = np.divide(self.numerator, self.denominator, out=np.zeros_like(self.numerator),
                                     where=self.numerator != 0.0)
        self.unc_fsm.mapdata = np.sqrt(np.divide(np.ones_like(self.numerator), self.denominator,
                                                 out=np.zeros_like(self.numerator), where=self.denominator != 0.0))

    def save_file(self):
        self.fsm.save_map()
        self.unc_fsm.save_map()

    @staticmethod
    def mask_outliers(data, threshold=1):
        z = np.abs(stats.zscore(data[data > 0.0]))
        mask = z > threshold
        return mask


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