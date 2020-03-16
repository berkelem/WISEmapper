from file_handler import ZodiMap, HealpixMap, WISEMap
import numpy as np
import os
from scipy.optimize import minimize
import pickle
from fullskymapping import FullSkyMap
from collections import defaultdict


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
        self.pole_region = HealpixMap("/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/mask_map_70.fits")
        self.pole_region.read_data()
        self.pole_region_mask = self.pole_region.mapdata.astype(bool)
        self.nside = None


class RegionSelector:

    def __init__(self, zodi_calibrator):
        self.zc = zodi_calibrator
        self.pole_mask = ~self.zc.pole_region_mask
        self.moon_mask = self.zc.moon_stripe_mask.mapdata.astype(bool)

    def get_pole_map(self):
        pole_full_mask = self.pole_mask | self.moon_mask
        self.pole_region_size = len(self.pole_mask[~pole_full_mask])
        return pole_full_mask


class PoleCalibrator:

    def __init__(self):
        pass



def nonzero_z_score(arr, uncs):
    z_score = np.zeros_like(arr)
    inds = np.arange(len(arr), dtype=int)
    nonzero_mask = uncs != 0.0
    nonzero_inds = inds[nonzero_mask]
    nonzero_arr = arr[nonzero_mask]
    nonzero_uncs = uncs[nonzero_mask]
    if len(nonzero_uncs > 0):
        mean = np.sum(nonzero_arr / np.square(nonzero_uncs)) / np.sum(1 / np.square(nonzero_uncs))
        std = np.std(nonzero_arr)
        nonzero_z_score = ((nonzero_arr - mean) / std) if std != 0.0 else 0.0
        z_score[nonzero_inds] = nonzero_z_score
    return np.abs(z_score)

def chi_sq_gain(param, x_data, y_data, sigma):
    residual = (y_data - (param * x_data))
    weighted_residual = residual / (np.mean(sigma)**2)
    chi_sq = (np.ma.sum(weighted_residual ** 2) / len(x_data)) if len(x_data) > 0 else 0.0
    return chi_sq


def run_calibration():
    n_orbits = 6323

    zc = ZodiCalibrator(3)
    rs = RegionSelector(zc)
    pole_full_mask = rs.get_pole_map()

    A = np.zeros((rs.pole_region_size, n_orbits), dtype=int)

    all_data_pole = np.zeros_like(A, dtype=float)
    all_uncs_pole = np.zeros_like(A, dtype=float)

    gains = np.ones(n_orbits, dtype=float)
    offsets = np.zeros_like(gains)

    gain_data_list_clean = []
    gain_uncs_list_clean = []
    gain_inds_list_clean = []
    gain_zodi_list_clean = []

    gain_data_list_unclean = []
    gain_uncs_list_unclean = []
    gain_inds_list_unclean = []
    gain_zodi_list_unclean = []

    print("Loading data")
    for i in range(0, n_orbits):
        # Load orbit data
        filename = f"/home/users/mberkeley/wisemapper/data/output_maps/uncalibrated/fsm_w3_orbit_{i}_uncalibrated.fits"
        if not os.path.exists(filename):
            print(f'Skipping file {os.path.basename(filename)} as it does not exist')
            continue
        else:
            print(f"Reading file {os.path.basename(filename)}")
        orbit_map = WISEMap(filename, 3)
        orbit_map.read_data()

        orbit_uncmap = WISEMap(filename.replace("orbit", "unc_orbit"), 3)
        orbit_uncmap.read_data()

        # Select polar data for offset calibration

        offset_data_region = orbit_map.mapdata[~pole_full_mask]
        offset_uncs_region = orbit_uncmap.mapdata[~pole_full_mask]
        all_data_pole[:,i] = offset_data_region
        all_uncs_pole[:,i] = offset_uncs_region
        # Make pointing matrix for pole region
        A[:,i] = offset_data_region.astype(bool)

        # Select orbit data for gain calibration
        nonzero_mask = orbit_uncmap.mapdata == 0.0
        orbit_mask = nonzero_mask | moon_mask

        gain_data_region = orbit_map.mapdata[~orbit_mask]
        gain_uncs_region = orbit_uncmap.mapdata[~orbit_mask]
        gain_inds_region = np.arange(orbit_map.npix)[~orbit_mask]
        zodi_map_orbit = zc.kelsall_map.mapdata[~orbit_mask]

        gain_data_list_unclean.append(gain_data_region)
        gain_uncs_list_unclean.append(gain_uncs_region)
        gain_inds_list_unclean.append(gain_inds_region)
        gain_zodi_list_unclean.append(zodi_map_orbit)

        ratio_data = gain_data_region / zodi_map_orbit
        ratio_uncs = gain_uncs_region / np.abs(zodi_map_orbit)
        z_gain = nonzero_z_score(ratio_data, ratio_uncs)
        outlier_inds = np.arange(len(ratio_data))[z_gain > 3]
        clean_data = np.delete(gain_data_region, outlier_inds)
        clean_uncs = np.delete(gain_uncs_region, outlier_inds)
        clean_inds = np.delete(gain_inds_region, outlier_inds)
        clean_zodi_orbit = np.delete(zodi_map_orbit, outlier_inds)


        gain_data_list_clean.append(clean_data)
        gain_uncs_list_clean.append(clean_uncs)
        gain_inds_list_clean.append(clean_inds)
        gain_zodi_list_clean.append(clean_zodi_orbit)

    # max_len = max([len(g) for g in gain_inds_list])
    # gain_data_arr = np.zeros((n_orbits, max_len), dtype=float)
    # gain_uncs_arr = np.zeros_like(gain_data_arr)
    # gain_inds_arr = np.ones_like(gain_data_arr, dtype=int) * -1
    # for orbit_ind in range(len(n_orbits)):
    #     gain_data_arr[orbit_ind, :len(gain_data_region[orbit_ind])] = gain_data_region[orbit_ind]
    #     gain_uncs_arr[orbit_ind, :len(gain_uncs_region[orbit_ind])] = gain_uncs_region[orbit_ind]
    #     gain_inds_arr[orbit_ind, :len(gain_inds_region[orbit_ind])] = gain_inds_region[orbit_ind]
    # gain_data_region = gain_uncs_region = gain_inds_region = None
    #


    calib_iter = 0
    while calib_iter < 15:
        print(f"Calibrating {calib_iter}th iteration")

        # Calibrate gain
        for orbit_ind in range(n_orbits):
            # print(f"Calibrating gain of {orbit_ind}th orbit")
            gain_data = gain_data_list_clean[orbit_ind]
            gain_uncs = gain_uncs_list_clean[orbit_ind]
            zodi_data = gain_zodi_list_clean[orbit_ind]

            # init_params = [gains[orbit_ind], offsets[orbit_ind]]
            if len(gain_uncs) > 0:
                opt_gain = minimize(chi_sq_gain, gains[orbit_ind],
                                    args=(gain_data + offsets[orbit_ind], zodi_data, gain_uncs),
                                    method='Nelder-Mead').x
                gains[orbit_ind] = opt_gain
            else:
                print(f"Orbit {orbit_ind} is empty")
                continue
            # l1, l2 = plt.plot(np.arange(len(gain_data)), (gain_data)*opt_gain, 'b.', np.arange(len(gain_data)), zodi_data, 'g.')
            # plt.xlabel("Pixel")
            # plt.ylabel("MJy/sr")
            # plt.legend((l1,l2), ("cal_data", "zodi_map"))
            # plt.savefig(f"calibration_orbit_{orbit_ind}_iter_{calib_iter}.png")
            # plt.close()

        all_data_applied_gains = (all_data_pole + offsets) * gains
        all_uncs_applied_gains = all_uncs_pole * np.abs(gains)

        z = np.zeros_like(all_data_applied_gains)
        for p in range(all_data_applied_gains.shape[1]):
            px_data = all_data_applied_gains[:, p]
            px_uncs = all_uncs_applied_gains[:, p]
            z[:, p] = nonzero_z_score(px_data, px_uncs)

        all_data_applied_gains = np.where(z > 3, 0.0, all_data_applied_gains)
        all_uncs_applied_gains = np.where(z > 3, 0.0, all_uncs_applied_gains)

        numerator = np.sum(np.divide(all_data_applied_gains, np.square(all_uncs_applied_gains), where=all_uncs_applied_gains != 0.0, out=np.zeros_like(all_uncs_applied_gains, dtype=float)), axis=1)
        denominator = np.sum(np.divide(1, np.square(all_uncs_applied_gains), where=all_uncs_applied_gains != 0.0, out=np.zeros_like(all_uncs_applied_gains, dtype=float)), axis=1)

        D = np.array(np.divide(numerator, denominator, where=denominator != 0.0, out=np.zeros_like(denominator)), ndmin=2)

        B_num = D.dot(A)
        B_denom = np.array(np.sum(np.square(A), axis=0), ndmin=2)
        offsets = np.divide(-B_num, B_denom, where=B_denom != 0.0, out=np.zeros_like(B_denom, dtype=float)).flatten()
        # offsets += secondary_offsets

        calib_iter += 1

    with open("gains.pkl", "wb") as gain_file:
        pickle.dump(gains, gain_file, pickle.HIGHEST_PROTOCOL)

    with open("offsets.pkl", "wb") as offset_file:
        pickle.dump(offsets, offset_file, pickle.HIGHEST_PROTOCOL)

    fsm = FullSkyMap(
        f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/fullskymap_band3_{n_orbits}.fits", 256)
    unc_fsm = FullSkyMap(
        f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/fullskymap_unc_band3_{n_orbits}.fits",
        256)

    fsm_num = np.zeros(fsm.npix, dtype=float)
    fsm_denom = np.zeros_like(fsm_num)
    pixel_values = defaultdict(list)
    pixel_uncs = defaultdict(list)
    for o_ix in range(n_orbits):
        raw_data = gain_data_list_unclean[o_ix]
        raw_uncs = gain_uncs_list_unclean[o_ix]
        orbit_reg_inds = gain_inds_list_unclean[o_ix]
        zodi_data_fullorbit = gain_zodi_list_unclean[o_ix]

        gain = gains[o_ix]
        offset = offsets[o_ix]

        cal_data = (raw_data + offset) * gain #+ offset
        cal_uncs = raw_uncs * abs(gain)

        # l1, l2 = plt.plot(np.arange(len(cal_data)), cal_data, 'b.', np.arange(len(cal_data)),
        #                   zodi_data_fullorbit, 'g.')
        # plt.xlabel("Pixel")
        # plt.ylabel("MJy/sr")
        # plt.legend((l1, l2), ("cal_data", "zodi_map"))
        # plt.savefig(f"calibration_orbit_{orbit_ind}_with_secondary_offset.png")
        # plt.close()

        for j, k in enumerate(orbit_reg_inds):
            pixel_values[k].append(cal_data[j])
            pixel_uncs[k].append(cal_uncs[j])

    for px in pixel_values:
        all_px_data = np.array(pixel_values[px])
        all_px_uncs = np.array(pixel_uncs[px])
        if len(all_px_uncs) > 0:
            z_score = nonzero_z_score(all_px_data, all_px_uncs)

            outlier_inds = np.arange(len(all_px_data))[z_score > 3]
            clean_px_data = np.delete(all_px_data, outlier_inds)
            clean_px_uncs = np.delete(all_px_uncs, outlier_inds)


            fsm_num[px] += np.sum(np.divide(clean_px_data, np.square(clean_px_uncs), where=clean_px_uncs != 0.0, out=np.zeros_like(clean_px_uncs)))
            fsm_denom[px] += np.sum(np.divide(1, np.square(clean_px_uncs), where=clean_px_uncs != 0.0, out=np.zeros_like(clean_px_uncs)))

    fsm.mapdata = np.divide(fsm_num, fsm_denom, where=fsm_denom != 0.0, out=np.zeros_like(fsm_denom))
    unc_fsm.mapdata = np.divide(np.ones_like(fsm_denom), np.sqrt(fsm_denom), where=fsm_denom != 0.0,
                                   out=np.zeros_like(fsm_denom))
    fsm.save_map()
    unc_fsm.save_map()