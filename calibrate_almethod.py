from file_handler import ZodiMap, HealpixMap, WISEMap
import numpy as np
from scipy.optimize import minimize
import pickle
import os
from scipy import stats
from fullskymapping import FullSkyMap
import matplotlib.pyplot as plt


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
        self.pole_region = HealpixMap("/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/mask_map_80.fits")
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

def chi_sq_gain(param, x_data, y_data, sigma, offset):
    residual = (y_data - (param * x_data + offset))
    weighted_residual = residual / sigma
    chi_sq = (np.ma.sum(weighted_residual ** 2) / len(x_data)) if len(x_data) > 0 else 0.0
    return chi_sq

def calibrate_gain(data, uncs, calibration_data, offsets):
    init_gains = np.ones((data.shape[0], 1))
    gains = np.zeros(data.shape[0])
    num_px_for_gain_cal = []
    for i, orbit in enumerate(data):
        # print(f"Fitting orbit {i} gain")
        cal_orbit = np.ma.array(calibration_data, mask=orbit.mask).compressed()
        init_gain = init_gains[i]
        offset = offsets[i]
        sigma = uncs[i].compressed()
        orbit_data = data[i].compressed()
        num_px_for_gain_cal.append(len(orbit_data))
        popt = minimize(chi_sq_gain, init_gain, args=(orbit_data, cal_orbit, sigma, offset), method='Nelder-Mead').x
        gains[i] = popt
        # print("gain", popt)
    num_px_filename = "num_px_for_gain_cal.pkl"
    if not os.path.exists(num_px_filename):
        with open(num_px_filename, "wb") as num_px_file:
            pickle.dump(num_px_for_gain_cal, num_px_file, pickle.HIGHEST_PROTOCOL)
    return np.array(gains, ndmin=2).T

def calibrate(input_data, input_uncs):
    print("Initializing calibrator")
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

    with open("gain_inds.pkl", "wb") as gain_inds_file:
        pickle.dump(gain_inds, gain_inds_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open("offset_inds.pkl", "wb") as offset_inds_file:
        pickle.dump(offset_inds, offset_inds_file, protocol=pickle.HIGHEST_PROTOCOL)

    masked_data_for_gain_cal = np.ma.array(input_data, mask=gain_mask)[:, gain_inds]
    masked_uncs_for_gain_cal = np.ma.array(input_uncs, mask=gain_mask)[:, gain_inds]

    masked_data_for_offset_cal = np.ma.array(input_data, mask=offset_mask)[:, offset_inds]
    masked_uncs_for_offset_cal = np.ma.array(input_data, mask=offset_mask)[:, offset_inds]

    n = 0
    offset_fits = []
    gain_fits = []
    while n < 1:
        print(f"Calibration iteration {n}")
        print("Fitting gains on each orbit")
        gains = calibrate_gain(masked_data_for_gain_cal, masked_uncs_for_gain_cal, cal_map[gain_inds], offsets)

        print("Adjusting data based on gain fit")
        calibrated_data_for_offset_cal = gains * masked_data_for_offset_cal  # no offset
        calibrated_uncs_for_offset_cal = np.abs(gains) * masked_uncs_for_offset_cal

        print("Minimizing offset")
        offsets = minimize_var(calibrated_data_for_offset_cal, calibrated_uncs_for_offset_cal).T

        print(f"Recording fit parameters for iteration {n}\n")
        offset_fits.append(offsets)
        gain_fits.append(gains)
        n += 1
    # calibrated_data_for_offset_cal = None
    # calibrated_uncs_for_offset_cal = None
    # calibrated_data_for_gain_cal = gains * masked_data_for_gain_cal + offsets
    # calibrated_uncs_for_gain_cal = np.abs(gains) * masked_data_for_gain_cal
    return masked_data_for_gain_cal, masked_uncs_for_gain_cal, gain_fits, offset_fits

def minimize_var(data, uncs):

    A = ~data.mask.T
    print("Shape of matrix A", A.shape)
    data = data.filled(fill_value=0.0)
    num_px_filename = "num_px_for_offset_cal.pkl"
    if not os.path.exists(num_px_filename):
        num_px_for_offset_cal = np.count_nonzero(data, axis=1)
        with open(num_px_filename, "wb") as num_px_file:
            pickle.dump(num_px_for_offset_cal, num_px_file, pickle.HIGHEST_PROTOCOL)
    unc_sq = np.square(uncs.filled(fill_value=0.0))
    numerator = np.sum(np.divide(data, unc_sq, where=unc_sq != 0.0, out=np.zeros_like(data, dtype=float)), axis=0)
    denominator = np.sum(np.divide(1, unc_sq, where=unc_sq != 0.0, out=np.zeros_like(unc_sq, dtype=float)), axis=0)
    mapdata = np.divide(numerator, denominator, where=denominator != 0.0, out=np.zeros_like(denominator, dtype=float))
    D = np.array(mapdata, ndmin=2)
    B_num = D.dot(A)
    B_denom = np.array(np.sum(np.square(A), axis=0), ndmin=2)
    B = np.divide(-B_num, B_denom, where=B_denom != 0.0, out=np.zeros_like(B_denom, dtype=float))
    return B

def load_data_generator():
    for i in range(3161):
        print(f"Loading data for orbit {i}")
        input_map = WISEMap(
            f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/uncalibrated/fullskymap_band3_fullorbit_{i}_uncalibrated.fits",
            3)
        input_uncmap = WISEMap(
            f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/uncalibrated/fullskymap_band3_unc_fullorbit_{i}_uncalibrated.fits",
            3)
        input_map.read_data()
        input_uncmap.read_data()
        yield input_map.mapdata, input_uncmap.mapdata

def load_data():
    with open("/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/all_orbits_data.pkl", "rb") as f:
        all_orbits_data = pickle.load(f)
    with open("/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/all_orbits_unc.pkl", "rb") as g:
        all_orbits_unc = pickle.load(g)
    return all_orbits_data, all_orbits_unc

def create_final_map(all_data, all_uncs):
    with open("gain_fits.pkl", "rb") as gain_file:
        fitted_gains = pickle.load(gain_file)
    with open("offset_fits.pkl", "rb") as offset_file:
        fitted_offsets = pickle.load(offset_file)
    opt_gains = fitted_gains[-1]
    opt_offsets = fitted_offsets[-1]
    calibrated_data = opt_gains * all_data + opt_offsets
    calibrated_uncs = np.abs(opt_gains) * all_uncs
    unc_sq = np.square(calibrated_uncs)
    numerator = np.sum(np.divide(calibrated_data, unc_sq, where=unc_sq != 0.0, out=np.zeros_like(unc_sq, dtype=float)), axis=0)
    denominator = np.sum(np.divide(1, unc_sq, where=unc_sq != 0.0, out=np.zeros_like(unc_sq, dtype=float)), axis=0)
    mapdata = np.divide(numerator, denominator, where=denominator != 0.0, out=np.zeros_like(denominator, dtype=float))

    with open("mapdata.pkl", "wb") as mapdata_file:
        pickle.dump(mapdata, mapdata_file, protocol=pickle.HIGHEST_PROTOCOL)

def apply_calibration(data, gain, offset):
    """Apply adjustment across entire map and restore original zeros afterwards"""
    zeros = data == 0.0
    adj_data = gain*data + offset
    adj_data[zeros] = 0.0
    return adj_data

def mask_outliers(data, threshold=1):
    z = np.abs(stats.zscore(data[data != 0.0]))
    mask = z > threshold
    return mask

def combine_orbits_from_pkl(all_data, all_uncs, gains, offsets, start, end):
    fsm = FullSkyMap(
        f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/fullskymap_band3_{start}_{end}.fits", 256)
    unc_fsm = FullSkyMap(
        f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/fullskymap_unc_band3_{start}_{end}.fits",
        256)

    for i in range(start, end):
        all_data[i] = all_data[i] * gains[i] + offsets[i]
        all_uncs[i] = all_uncs[i] * abs(gains[i])

    unc_sq = np.square(all_uncs)
    numerator = np.sum(np.divide(all_data, unc_sq, where=unc_sq != 0.0, out=np.zeros_like(unc_sq, dtype=float)),
                       axis=0)
    denominator = np.sum(np.divide(1, unc_sq, where=unc_sq != 0.0, out=np.zeros_like(unc_sq, dtype=float)), axis=0)
    fsm.mapdata = np.divide(numerator, denominator, where=denominator != 0.0, out=np.zeros_like(denominator, dtype=float))
    fsm.save_map()

    unc_fsm.mapdata = np.divide(1, np.sqrt(denominator), where=denominator != 0.0, out=np.zeros_like(denominator, dtype=float))
    unc_fsm.save_map()


def combine_maps_by_pixel(all_data, all_uncs, gains, offsets, start, end, inds):
    import time
    tic = time.time()
    fsm = FullSkyMap(
        f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/fullskymap_band3_{start}_{end}.fits", 256)
    unc_fsm = FullSkyMap(
        f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/fullskymap_unc_band3_{start}_{end}.fits",
        256)
    numerator = np.zeros(len(all_data[0]), dtype=float)
    denominator = np.zeros(len(all_data[0]), dtype=float)
    for p in range(len(all_data[0])):
        data_val = gains * all_data[:,p] + offsets
        unc_val = np.abs(gains) * all_uncs[:,p]
        numerator[p] = np.sum(np.divide(data_val, (unc_val ** 2), where=unc_val != 0.0, out=np.zeros_like(unc_val)))
        denominator[p] = np.sum(np.divide(1, (unc_val ** 2), where=unc_val != 0.0, out=np.zeros_like(unc_val)))
        if p % 10000 == 0.0:
            print("{} pixels mapped in {} seconds".format(p, time.time() - tic))
    px_values = np.divide(numerator, denominator, where=denominator != 0.0, out=np.zeros_like(denominator, dtype=float))
    fsm.mapdata[inds] = px_values
    fsm.save_map()

    unc_values = np.divide(1, np.sqrt(denominator), where=denominator != 0.0,
                                out=np.zeros_like(denominator, dtype=float))
    unc_fsm.mapdata[inds] = unc_values
    unc_fsm.save_map()


def combine_orbits(start, end, gains, offsets):
    fsm = FullSkyMap(f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/fullskymap_band3_{start}_{end}.fits", 256)
    unc_fsm = FullSkyMap(f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/fullskymap_unc_band3_{start}_{end}.fits", 256)
    map_datas = []
    map_uncs = []
    for i in range(start, end):
        filename = f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/uncalibrated/fullskymap_band3_fullorbit_{i}_uncalibrated.fits"
        if not os.path.exists(filename):
            print(f'Skipping file {os.path.basename(filename)} as it does not exist')
            continue
        else:
            print(f"Adding file {os.path.basename(filename)}")
        orbit_map = WISEMap(filename, 3)
        orbit_map.read_data()
        print("gain", gains[i], "offset", offsets[i])
        map_datas.append(apply_calibration(orbit_map.mapdata, gains[i], offsets[i]))

        orbit_uncmap = WISEMap(f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/uncalibrated/fullskymap_band3_unc_fullorbit_{i}_uncalibrated.fits", 3)
        orbit_uncmap.read_data()
        map_uncs.append((orbit_uncmap.mapdata * np.abs(gains[i])))

    px_vals = np.array(map_datas).T
    unc_vals = np.array(map_uncs).T
    for p, px in enumerate(px_vals):
        if np.count_nonzero(px) > 2:
            mask = mask_outliers(px)
            good_vals = px[px != 0.0][~mask]
            good_unc_vals = unc_vals[p][px != 0.0][~mask]
        else:
            good_vals = px[px != 0.0]
            good_unc_vals = unc_vals[p][px != 0.0]

        # if len(good_unc_vals) > 1:
        #     pass

        # if len(good_unc_vals) > 0:
        numerator = np.sum(good_vals / (good_unc_vals ** 2))
        denominator = np.sum(1 / (good_unc_vals ** 2))
        fsm.mapdata[p] = np.divide(numerator, denominator, where=denominator != 0.0, out=np.zeros_like(denominator))
        unc_fsm.mapdata[p] = np.divide(np.ones_like(denominator), np.sqrt(denominator), where=denominator != 0.0, out=np.zeros_like(denominator))
    fsm.save_map()
    unc_fsm.save_map()
    return

def interp_fitted_vals(gains, offsets):
    opt_gains = gains[-1].flatten()
    opt_offsets = offsets[-1].flatten()

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
    return opt_gains, opt_offsets


def run_calibration():
    # npix = 2000
    # norbits = 3161
    # all_orbits_data = np.random.rand(norbits, npix)
    # all_orbits_unc = np.random.rand(norbits, npix)
    # make_zero = all_orbits_data < 0.5
    # all_orbits_data[make_zero] = 0.0
    # all_orbits_unc[make_zero] = 0.0
    print("Loading data")
    all_orbits_data, all_orbits_unc = load_data()
    print("Data successfully loaded")

    calibrated_data_gain, calibrated_uncs_gain, gain_fits, offset_fits = calibrate(all_orbits_data, all_orbits_unc)
    # with open("calibrated_data_offset.pkl", "wb") as f:
    #     pickle.dump(calibrated_data_offset, f, pickle.HIGHEST_PROTOCOL)
    # with open("calibrated_uncs_offset.pkl", "wb") as g:
    #     pickle.dump(calibrated_uncs_offset, g, pickle.HIGHEST_PROTOCOL)
    with open("calibrated_data_gain.pkl", "wb") as f:
        pickle.dump(calibrated_data_gain, f, pickle.HIGHEST_PROTOCOL)
    with open("calibrated_uncs_gain.pkl", "wb") as g:
        pickle.dump(calibrated_uncs_gain, g, pickle.HIGHEST_PROTOCOL)
    # with open("gain_fits.pkl", "wb") as p:
    #     pickle.dump(gain_fits, p, pickle.HIGHEST_PROTOCOL)
    # with open("offset_fits.pkl", "wb") as q:
    #     pickle.dump(offset_fits, q, pickle.HIGHEST_PROTOCOL)

def run_create_map():
    print("Loading data")
    all_orbits_data, all_orbits_unc = load_data()
    print("Data successfully loaded")
    # create_final_map(all_orbits_data, all_orbits_unc)
    with open("gain_fits.pkl", "rb") as gain_file:
        fitted_gains = pickle.load(gain_file)
    with open("offset_fits.pkl", "rb") as offset_file:
        fitted_offsets = pickle.load(offset_file)
    opt_gains, opt_offsets = interp_fitted_vals(fitted_gains, fitted_offsets)

    combine_orbits_from_pkl(all_orbits_data, all_orbits_unc, opt_gains, opt_offsets, 0, 3161)

def run_create_partial_map():

    # with open("/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/calibrated_data_offset_15_iterations.pkl", "rb") as f:
    #     partial_data = pickle.load(f)
    # with open("/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/calibrated_uncs_offset_15_iterations.pkl", "rb") as g:
    #     partial_uncs = pickle.load(g)
    with open("30_iterations/gain_fits.pkl", "rb") as gain_file:
        fitted_gains = pickle.load(gain_file)
    with open("30_iterations/offset_fits.pkl", "rb") as offset_file:
        fitted_offsets = pickle.load(offset_file)
    opt_gains = fitted_gains[14].flatten()
    opt_offsets = fitted_offsets[14].flatten()

    zc = ZodiCalibrator(3)
    moon_mask = zc.moon_stripe_mask.mapdata.astype(bool)
    pole_mask = ~zc.pole_region_mask

    fsm = FullSkyMap(
        f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/fullskymap_band3_offset_region.fits", 256)
    unc_fsm = FullSkyMap(
        f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/fullskymap_unc_band3_offset_region.fits",
        256)

    numerator = np.zeros(len(moon_mask), dtype=float)
    denominator = np.zeros(len(moon_mask), dtype=float)
    for i in range(3161):
        filename = f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/uncalibrated/fullskymap_band3_fullorbit_{i}_uncalibrated.fits"
        unc_filename = f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/uncalibrated/fullskymap_band3_unc_fullorbit_{i}_uncalibrated.fits"
        if not os.path.exists(filename):
            print(f'Skipping file {os.path.basename(filename)} as it does not exist')
            continue
        else:
            print(f"Adding file {os.path.basename(filename)}")


        orbit_map = WISEMap(filename, 3)
        orbit_map.read_data()
        orbit_uncmap = WISEMap(unc_filename, 3)
        orbit_uncmap.read_data()

        nonzero_mask = orbit_uncmap.mapdata == 0.0
        offset_mask = nonzero_mask | moon_mask | pole_mask

        inds = np.arange(len(orbit_map.mapdata))
        compressed_offset_map = orbit_map.mapdata[~offset_mask]
        compressed_offset_uncmap = orbit_uncmap.mapdata[~offset_mask]
        offset_inds = inds[~offset_mask]

        cal_data = compressed_offset_map * opt_gains[i] + opt_offsets[i]
        cal_uncs = compressed_offset_uncmap * abs(opt_gains[i])

        numerator[offset_inds] += np.divide(cal_data, np.square(cal_uncs), where=cal_uncs != 0.0, out=np.zeros_like(cal_uncs))
        denominator[offset_inds] += np.divide(1, np.square(cal_uncs), where=cal_uncs != 0.0, out=np.zeros_like(cal_uncs))

    fsm.mapdata = np.divide(numerator, denominator, where=denominator != 0.0, out=np.zeros_like(denominator))
    unc_fsm.mapdata = np.divide(np.ones_like(denominator), np.sqrt(denominator), where=denominator != 0.0,
                                   out=np.zeros_like(denominator))


    fsm.save_map()
    unc_fsm.save_map()
    # with open("offset_inds.pkl", "rb") as offset_inds_file:
    #     offset_inds = pickle.load(offset_inds_file)
    # with open("gain_inds.pkl", "rb") as gain_inds_file:
    #     gain_inds = pickle.load(gain_inds_file)

    # combine_maps_by_pixel(partial_data, partial_uncs, opt_gains, opt_offsets, 0, 3161, offset_inds)
    # combine_orbits_from_pkl(partial_data, partial_uncs, opt_gains, opt_offsets, 0, 3161)

def run_create_map_smoothed():
    # print("Loading data")
    # all_orbits_data, all_orbits_unc = load_data()
    # print("Data successfully loaded")
    with open("smoothed_interp_params.pkl", "rb") as params_file:
        gains, offsets = pickle.load(params_file)

    # combine_orbits_from_pkl(all_orbits_data, all_orbits_unc, gains, offsets, 0, 3161)
    combine_orbits(2746, 2747, gains, offsets)
    return

def main():
    # run_calibration()
    # run_create_map()
    run_create_partial_map()
    # run_create_map_smoothed()






if __name__ == "__main__":
    main()