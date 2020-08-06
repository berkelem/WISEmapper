from wise_images_2_orbit_coadd.file_handler import ZodiMap, HealpixMap
import numpy as np
from scipy.optimize import minimize
import pickle
import time


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
        self.pole_region = HealpixMap("/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/mask_map_89.fits")
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


def var_sum(offsets, data):
    if not len(offsets.shape) == 2:
        offsets = np.array(offsets, ndmin=2).T
    return np.ma.sum(np.ma.var(data + offsets, axis=0))


def minimize_var(init_offsets, data):
    # print(var_sum(init_offsets, data))
    optimized_values = []
    offsets = init_offsets
    n = 0
    while n < 50:
        # local_data = data[n:n+20, :]
        # local_offsets = offsets[n:n+20, :]
        tic = time.clock()
        popt = minimize(var_sum, offsets, args=(data), method='Nelder-Mead').x
        toc = time.clock()
        print(f"Fitting iteration {n/5} took {toc - tic} seconds")
        popt = np.array(popt, ndmin=2).T
        offsets = popt
        optimized_values.append(popt)
        n += 5

    # popt = minimize(var_sum, init_offsets, args=(data), method='Nelder-Mead').x
    optimized_offsets = np.array(popt, ndmin=2).T
    # print(var_sum(optimized_offsets, data))
    return optimized_offsets

def calibrate_gain(init_gains, data, uncs, calibration_data, offsets):
    gains = np.zeros(calibration_data.shape[0])
    for i, orbit in enumerate(data):
        # print(f"Fitting orbit {i} gain")
        init_gain = init_gains[i]
        cal_orbit = np.ma.array(calibration_data[i], mask=orbit.mask)
        offset = offsets[i]
        sigma = uncs[i]
        popt = minimize(chi_sq_gain, init_gain, args=(data, cal_orbit, sigma, offset), method='Nelder-Mead').x
        gains[i] = popt
    return np.array(gains, ndmin=2).T

def chi_sq_gain(param, x_data, y_data, sigma, offset):
    residual = (y_data - (param * x_data + offset))
    weighted_residual = residual / sigma
    chi_sq = (np.ma.sum(weighted_residual ** 2) / len(x_data)) if len(x_data) > 0 else 0.0
    return chi_sq

def calibrate(input_data, input_uncs):
    zc = ZodiCalibrator(3)
    cal_map = zc.kelsall_map.mapdata

    nonzero_mask = input_data == 0.0
    moon_mask = np.tile(zc.moon_stripe_mask.mapdata.astype(bool), (input_data.shape[0], 1))
    pole_mask = ~zc.pole_region_mask

    gain_mask = nonzero_mask | moon_mask
    offset_mask = nonzero_mask | moon_mask | pole_mask


    gains = np.ones((input_data.shape[0], 1))
    offsets = np.zeros_like(gains)

    inds = np.arange(len(cal_map))
    compressed_gain_map = np.any(~gain_mask, axis=0)
    compressed_offset_map = np.any(~offset_mask, axis=0)
    gain_inds = inds[compressed_gain_map]
    offset_inds = inds[compressed_offset_map]

    masked_data_for_offset_cal = np.ma.array(input_data, mask=offset_mask)[:, offset_inds]

    n = 0
    offset_fits = []
    gain_fits = []
    while n < 5:
        print(f"Calibration iteration {n}")
        print("Minimizing offset")
        offsets = minimize_var(offsets, masked_data_for_offset_cal)
        print("Adjusting data based on offset fit")
        calibrated_data = gains * input_data + offsets
        calibrated_uncs = np.abs(gains)*input_uncs
        masked_data_for_gain_cal = np.ma.array(calibrated_data, mask=gain_mask)[:, gain_inds]
        masked_unc_for_gain_cal = np.ma.array(calibrated_uncs, mask=gain_mask)[:, gain_inds]
        print("Fitting gains on each orbit")
        gains = calibrate_gain(gains, masked_data_for_gain_cal, masked_unc_for_gain_cal, cal_map[gain_inds], offsets)
        print("Adjusting data based on gain fit")
        calibrated_data = gains * input_data + offsets
        masked_data_for_offset_cal = np.ma.array(calibrated_data, mask=offset_mask)[:, offset_inds]
        print(f"Recording fit parameters for iteration {n}\n")
        offset_fits.append(offsets)
        gain_fits.append(gains)
        n += 1

    with open("fitted_gains.pkl", "wb") as f:
        pickle.dump(gain_fits, f, pickle.HIGHEST_PROTOCOL)

    with open("fitted_offsets.pkl", "wb") as f:
        pickle.dump(offset_fits, f, pickle.HIGHEST_PROTOCOL)

    calibrated_input_data = gains * input_data + offsets
    # print(f"Iter {n}; Fitted offsets: ", offsets)
    # print(f"Iter {n}; Fitted gains: ", gains)
    # print("Calibrated data", calibrated_input_data, "\n")

    return calibrated_input_data


def load_data():
    with open("/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/all_orbits_data.pkl", "rb") as f:
        all_orbits_data = pickle.load(f)
    with open("/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/all_orbits_unc.pkl", "rb") as g:
        all_orbits_unc = pickle.load(g)
    return all_orbits_data, all_orbits_unc


def main():

    all_orbits_data, all_orbits_unc = load_data()
    calibrated_data = calibrate(all_orbits_data, all_orbits_unc)
    with open("calibrated_data.pkl", "wb") as f:
        pickle.dump(calibrated_data, f, pickle.HIGHEST_PROTOCOL)




    # test_matrix = np.array([[2, 3, 6, 7, 0, 5],
    #                         [3, 2, 5, 7, 3, 6],
    #                         [3, 0, 0, 3, 6, 7],
    #                         [9, 6, 7, 3, 5, 0]])
    #
    # calibration_matrix = np.array([[1002, 1003, 1007, 1008, 1003, 1006],
    #                                [1003, 1005, 1007, 1008, 1004, 1007],
    #                                [1002, 1000, 1008, 1009, 1007, 1007],
    #                                [1006, 1006, 1007, 1003, 1005, 1000]])

    # calibrate(test_matrix, calibration_matrix)




if __name__ == "__main__":
    main()