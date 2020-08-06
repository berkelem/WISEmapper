from calibration import ZodiCalibrator
from file_handler import WISEMap
from fullskymapping import FullSkyMap
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from scipy import stats


def converge_cal(input_map, input_uncmap, zc, id):

    input_map.read_data()
    input_uncmap.read_data()
    input_map_data = input_map.mapdata
    input_uncmap_data = input_uncmap.mapdata
    popt, npix = zc.calibrate(input_map_data, input_uncmap_data, id)
    return popt, npix

def plot_optimized_params(gains, offsets, n_calpix):
    plt.plot(gains, 'r.')
    plt.xlabel("Orbit number")
    plt.ylabel("Gain")
    plt.savefig("optimized_gains.png")
    plt.close()

    plt.plot(offsets, 'r.')
    plt.xlabel("Orbit number")
    plt.ylabel("Offset")
    plt.savefig("optimized_offsets.png")
    plt.close()

    plt.plot(n_calpix, 'r.')
    plt.xlabel("Orbit number")
    plt.ylabel("Num Cal Px")
    plt.savefig("num_cal_px.png")
    plt.close()

    return

def load_all_orbits():
    import healpy as hp
    all_orbits_data = np.zeros((3161, hp.nside2npix(256)), dtype=float)
    all_orbits_unc = np.zeros_like(all_orbits_data)
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
        all_orbits_data[i] = input_map.mapdata
        all_orbits_unc[i] = input_uncmap.mapdata

    with open("/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/all_orbits_data.pkl", "wb") as f:
        pickle.dump(all_orbits_data, f, pickle.HIGHEST_PROTOCOL)
    with open("/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/all_orbits_unc.pkl", "wb") as f:
        pickle.dump(all_orbits_unc, f, pickle.HIGHEST_PROTOCOL)
    print(len(all_orbits_data[all_orbits_data != 0.0]))
    print(len(all_orbits_unc[all_orbits_unc != 0.0]))

def load_data():
    with open("/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/all_orbits_data.pkl", "rb") as f:
        all_orbits_data = pickle.load(f)
    # with open("/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/all_orbits_unc.pkl", "rb") as g:
    #     all_orbits_unc = pickle.load(g)
    return all_orbits_data, None#all_orbits_unc

def main():
    # load_all_orbits()
    all_orbits_data, all_orbits_unc = load_data()
    zc = ZodiCalibrator(3)
    zc.calibrate(all_orbits_data, all_orbits_unc)



def old_main():
    band = 3
    zc = ZodiCalibrator(band)
    gains = []
    offsets = []
    n_calpix = []
    for i in range(3161):#6323):
        print(f"Getting calibration parameters for orbit {i}")
        input_map = WISEMap(
            f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/uncalibrated/fullskymap_band3_fullorbit_{i}_uncalibrated.fits", 3)
        input_uncmap = WISEMap(
            f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/uncalibrated/fullskymap_band3_unc_fullorbit_{i}_uncalibrated.fits", 3)
        popt, npix = converge_cal(input_map, input_uncmap, zc, i)
        gains.append(popt[0])
        offsets.append(popt[1])
        n_calpix.append(npix)

    plot_optimized_params(gains, offsets, n_calpix)

    with open("/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/optimized_cal_params.pkl", "wb") as f:
        pickle.dump(np.array([np.array(gains), np.array(offsets)]), f, pickle.HIGHEST_PROTOCOL)

    with open("/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/num_calibration_pix.pkl", "wb") as f:
        pickle.dump(np.array(n_calpix), f, pickle.HIGHEST_PROTOCOL)

def smooth_params(gains, offsets):
    # with open("optimized_cal_params.pkl", "rb") as f:
    #     gains, offsets = pickle.load(f)

    window_size = 251
    smooth_gains = median_filter(gains, window_size)
    smooth_offsets = median_filter(offsets, window_size)

    # smooth_gains2 = median_filter(smooth_gains, 451)
    # smooth_offsets2 = median_filter(smooth_offsets, 451)

    plt.plot(gains, 'r.', smooth_gains, 'b')#, smooth_gains2, 'g')
    plt.xlabel("Orbit number")
    plt.ylabel("Gain")
    plt.savefig(f"smoothed_opt_gains_window_{window_size}.png")
    plt.close()

    plt.plot(offsets, 'r.', smooth_offsets, 'b')#, smooth_offsets2, 'g')
    plt.xlabel("Orbit number")
    plt.ylabel("Offset")
    plt.savefig(f"smoothed_opt_offsets_window_{window_size}.png")
    plt.close()
    return smooth_gains, smooth_offsets

def adjust_params(npix_limit):
    with open("optimized_cal_params.pkl", "rb") as f:
        gains, offsets = pickle.load(f)

    gains = np.append(gains, gains[0])
    offsets = np.append(offsets, offsets[0])

    with open("num_calibration_pix.pkl", "rb") as g:
        npix = pickle.load(g)

    npix = np.append(npix, npix[0])

    orbit_num = np.arange(len(gains))

    mask = npix > npix_limit
    good_orbits = orbit_num[mask]
    good_gains = gains[mask]
    good_offsets = offsets[mask]
    # print("good orbits", npix_limit)
    # print(good_orbits, "\n")
    # even_inds = good_orbits % 2 == 0

    plt.plot(good_orbits, good_gains, 'r.')
    plt.xlabel("Orbit number")
    plt.ylabel("Good Gains")
    plt.savefig(f"good_gains_{npix_limit}.png")
    plt.close()

    # plt.plot(good_orbits[even_inds], good_gains[even_inds], 'r.')
    # plt.xlabel("Orbit number")
    # plt.ylabel("Good Gains")
    # plt.savefig(f"good_gains_{npix_limit}_even.png")
    # plt.close()
    #
    # plt.plot(good_orbits[~even_inds], good_gains[~even_inds], 'r.')
    # plt.xlabel("Orbit number")
    # plt.ylabel("Good Gains")
    # plt.savefig(f"good_gains_{npix_limit}_odd.png")
    # plt.close()

    plt.plot(good_orbits, good_offsets, 'r.')
    plt.xlabel("Orbit number")
    plt.ylabel("Good Offsets")
    plt.savefig(f"good_offsets_{npix_limit}.png")
    plt.close()

    # plt.plot(good_orbits[even_inds], good_offsets[even_inds], 'r.')
    # plt.xlabel("Orbit number")
    # plt.ylabel("Good Offsets")
    # plt.savefig(f"good_offsets_{npix_limit}_even.png")
    # plt.close()
    #
    # plt.plot(good_orbits[~even_inds], good_offsets[~even_inds], 'r.')
    # plt.xlabel("Orbit number")
    # plt.ylabel("Good Offsets")
    # plt.savefig(f"good_offsets_{npix_limit}_odd.png")
    # plt.close()

    from scipy import interpolate

    f_gains = interpolate.interp1d(good_orbits, good_gains)
    f_offsets = interpolate.interp1d(good_orbits, good_offsets)

    new_orbits = orbit_num[~mask]
    new_gains = f_gains(new_orbits)
    new_offsets = f_offsets(new_orbits)

    gains[~mask] = new_gains
    offsets[~mask] = new_offsets

    # with open("adjusted_opt_cal_params.pkl", "wb") as f:
    #     pickle.dump(np.array([np.array(gains), np.array(offsets)]), f, pickle.HIGHEST_PROTOCOL)

    plt.plot(gains, 'r.')
    plt.xlabel("Orbit number")
    plt.ylabel("Gain")
    plt.savefig(f"adjusted_gains_{npix_limit}.png")
    plt.close()

    plt.plot(offsets, 'r.')
    plt.xlabel("Orbit number")
    plt.ylabel("Offset")
    plt.savefig(f"adjusted_offsets_{npix_limit}.png")
    plt.close()

    return gains, offsets


def combine_orbits(start, end, gains, offsets):
    fsm = WISEMap(f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/fullskymap_band3_{start}_{end}.fits", 3)
    unc_fsm = WISEMap(f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/fullskymap_unc_band3_{start}_{end}.fits", 3)
    # fsm = WISEMap(f"/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/orbit_batches/fullskymap_band3_{start}_{end}.fits", 3)
    # unc_fsm = WISEMap(f"/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/orbit_batches/fullskymap_unc_band3_{start}_{end}.fits", 3)
    map_datas = []
    map_uncs = []
    for i in range(start, end):
        filename = f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/uncalibrated/fullskymap_band3_fullorbit_{i}_uncalibrated.fits"
        # filename = f"/home/users/mberkeley/wisemapper/data/output_maps/fsm_attempt6/w3/uncalibrated/fsm_w3_orbit_{i}_uncalibrated.fits"
        # filename = f"/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/orbit_maps/fsm_w3_orbit_{i}_uncalibrated.fits"
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
        # orbit_uncmap = WISEMap(f"/home/users/mberkeley/wisemapper/data/output_maps/fsm_attempt6/w3/uncalibrated/fsm_w3_unc_orbit_{i}_uncalibrated.fits", 3)
        # orbit_uncmap = WISEMap(f"/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/orbit_maps/fsm_w3_unc_orbit_{i}_uncalibrated.fits", 3)
        orbit_uncmap.read_data()
        map_uncs.append((orbit_uncmap.mapdata/np.abs(gains[i])))

    px_vals = np.array(map_datas).T
    unc_vals = np.array(map_uncs).T
    for p, px in enumerate(px_vals):
        if len(px[px > 0.0]) > 2:
            mask = mask_outliers(px)
            good_vals = px[px > 0.0][~mask]
            good_unc_vals = unc_vals[p][px > 0.0][~mask]
        else:
            good_vals = px[px > 0.0]
            good_unc_vals = unc_vals[p][px > 0.0]

        if len(good_unc_vals) > 1:
            pass

        if len(good_unc_vals) > 0:
            numerator = np.sum(good_vals / good_unc_vals ** 2)
            denominator = np.sum(1 / good_unc_vals ** 2)
            fsm.mapdata[p] = np.divide(numerator, denominator, where=denominator!=0.0, out=np.zeros_like(denominator))
            unc_fsm.mapdata[p] = np.divide(np.ones_like(denominator), np.sqrt(denominator), where=denominator!=0.0, out=np.zeros_like(denominator))
    fsm.save_map()
    unc_fsm.save_map()
    return

def make_index_map():
    fsm = WISEMap("/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/index_map.fits", 3)
    for i in range(len(fsm.mapdata)):
        fsm.mapdata[i] = i
    fsm.save_map()
    return

def mask_outliers(data, threshold=1):
    z = np.abs(stats.zscore(data[data > 0.0]))
    mask = z > threshold
    return mask


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

def apply_calibration(data, gain, offset):
    """Apply adjustment across entire map and restore original zeros afterwards"""
    zeros = data == 0.0
    adj_data = (data - offset)/gain
    adj_data[zeros] = 0.0
    return adj_data

def pair_orbits():
    for i in range(3161):
        filename1 = f"/home/users/mberkeley/wisemapper/data/output_maps/fsm_attempt6/w3/uncalibrated/fsm_w3_orbit_{2*i}_uncalibrated.fits"
        filename2 = f"/home/users/mberkeley/wisemapper/data/output_maps/fsm_attempt6/w3/uncalibrated/fsm_w3_orbit_{2*i+1}_uncalibrated.fits"
        orbit_map1 = WISEMap(filename1, 3)
        orbit_map1.read_data()
        orbit_map2 = WISEMap(filename2, 3)
        orbit_map2.read_data()

        filename1_unc = f"/home/users/mberkeley/wisemapper/data/output_maps/fsm_attempt6/w3/uncalibrated/fsm_w3_unc_orbit_{2 * i}_uncalibrated.fits"
        filename2_unc = f"/home/users/mberkeley/wisemapper/data/output_maps/fsm_attempt6/w3/uncalibrated/fsm_w3_unc_orbit_{2 * i + 1}_uncalibrated.fits"
        orbit_map1_unc = WISEMap(filename1_unc, 3)
        orbit_map1_unc.read_data()
        orbit_map2_unc = WISEMap(filename2_unc, 3)
        orbit_map2_unc.read_data()

        fsm = WISEMap(
            f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/uncalibrated/fullskymap_band3_fullorbit_{i}_uncalibrated.fits",
            256)
        combined_map = orbit_map1.mapdata + orbit_map2.mapdata
        combined_count = orbit_map1.mapdata.astype(bool).astype(int) + orbit_map2.mapdata.astype(bool).astype(int)
        combined_map = np.divide(combined_map, combined_count, where=combined_count != 0.0,
                                 out=np.zeros_like(combined_map))
        fsm.mapdata = combined_map
        fsm.save_map()

        fsm_unc = WISEMap(
            f"/home/users/mberkeley/wisemapper/data/output_maps/pole_fitting/w3/uncalibrated/fullskymap_band3_unc_fullorbit_{i}_uncalibrated.fits",
            256)
        combined_map_unc = orbit_map1_unc.mapdata + orbit_map2_unc.mapdata
        combined_count_unc = orbit_map1_unc.mapdata.astype(bool).astype(int) + orbit_map2_unc.mapdata.astype(
            bool).astype(int)
        combined_map_unc = np.divide(combined_map_unc, combined_count_unc, where=combined_count_unc != 0.0,
                                     out=np.zeros_like(combined_map_unc))
        fsm_unc.mapdata = combined_map_unc
        fsm_unc.save_map()

        if any(orbit_map1.mapdata.astype(bool) & orbit_map2.mapdata.astype(bool)):

            num_overlap = np.sum((orbit_map1.mapdata.astype(bool) & orbit_map2.mapdata.astype(bool)).astype(int))
            print(f"{num_overlap} doubled pixel(s) in {2*i} and {2*i+1}")






if __name__ == "__main__":
    main()
    # make_index_map()
    # pair_orbits()
    # for npix_limit in [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]:
    # gains, offsets = adjust_params(900)#npix_limit)
    # gains, offsets = smooth_params(gains, offsets)
    # combine_orbits(0, 3161, gains, offsets)
