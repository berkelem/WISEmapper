from file_handler import ZodiMap, HealpixMap
import numpy as np
from scipy.optimize import minimize
import pickle

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
    for i, orbit in enumerate(data):
        # print(f"Fitting orbit {i} gain")
        cal_orbit = np.ma.array(calibration_data, mask=orbit.mask).compressed()
        init_gain = init_gains[i]
        offset = offsets[i]
        sigma = uncs[i].compressed()
        orbit_data = data[i].compressed()
        popt = minimize(chi_sq_gain, init_gain, args=(orbit_data, cal_orbit, sigma, offset), method='Nelder-Mead').x
        gains[i] = popt
        # print("gain", popt)
    return np.array(gains, ndmin=2).T

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

    masked_data_for_gain_cal = np.ma.array(input_data, mask=gain_mask)[:, gain_inds]
    masked_uncs_for_gain_cal = np.ma.array(input_uncs, mask=gain_mask)[:, gain_inds]

    masked_data_for_offset_cal = np.ma.array(input_data, mask=offset_mask)[:, offset_inds]
    masked_uncs_for_offset_cal = np.ma.array(input_data, mask=offset_mask)[:, offset_inds]

    n = 0
    offset_fits = []
    gain_fits = []
    while n < 5:
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
    return masked_data_for_gain_cal, masked_data_for_gain_cal, gain_fits, offset_fits

def minimize_var(data, uncs):

    A = ~data.mask.T
    data = data.filled(fill_value=0.0)
    unc_sq = np.square(uncs.filled(fill_value=0.0))
    numerator = np.sum(np.divide(data, unc_sq, where=unc_sq != 0.0, out=np.zeros_like(data, dtype=float)), axis=0)
    denominator = np.sum(np.divide(1, unc_sq, where=unc_sq != 0.0, out=np.zeros_like(unc_sq, dtype=float)), axis=0)
    mapdata = np.divide(numerator, denominator, where=denominator != 0.0, out=np.zeros_like(denominator, dtype=float))
    D = np.array(mapdata, ndmin=2)
    B_num = D.dot(A)
    B_denom = np.array(np.sum(np.square(A), axis=0), ndmin=2)
    B = np.divide(-B_num, B_denom, where=B_denom != 0.0, out=np.zeros_like(B_denom, dtype=float))
    return B

def load_data():
    with open("/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/all_orbits_data.pkl", "rb") as f:
        all_orbits_data = pickle.load(f)
    with open("/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/all_orbits_unc.pkl", "rb") as g:
        all_orbits_unc = pickle.load(g)
    return all_orbits_data, all_orbits_unc

def main():
    npix = 786432
    norbits = 150
    all_orbits_data = np.random.rand(norbits, npix)
    all_orbits_unc = np.random.rand(norbits, npix)
    make_zero = all_orbits_data < 0.5
    all_orbits_data[make_zero] = 0.0
    all_orbits_unc[make_zero] = 0.0
    # all_orbits_data, all_orbits_unc = load_data()
    calibrated_data, calibrated_uncs, gain_fits, offset_fits = calibrate(all_orbits_data, all_orbits_unc)
    with open("calibrated_data.pkl", "wb") as f:
        pickle.dump(calibrated_data, f, pickle.HIGHEST_PROTOCOL)
    with open("calibrated_uncs.pkl", "wb") as g:
        pickle.dump(calibrated_uncs, g, pickle.HIGHEST_PROTOCOL)
    with open("gain_fits.pkl", "wb") as p:
        pickle.dump(gain_fits, p, pickle.HIGHEST_PROTOCOL)
    with open("offset_fits.pkl", "wb") as q:
        pickle.dump(offset_fits, q, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()