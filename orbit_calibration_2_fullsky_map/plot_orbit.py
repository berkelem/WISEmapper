from orbit_calibration_2_fullsky_map.coadd_orbits import Orbit, Coadder
from orbit_calibration_2_fullsky_map.spline_fit_calibration import SplineFitter
from wise_images_2_orbit_coadd.file_handler import WISEMap
import os
import pickle

def load_splines(gain_spline_file, offset_spline_file):
    """Load gain spline and offset spline from '*.pkl' files saved in a file"""
    with open(gain_spline_file, "rb") as g1:
        gain_spline = pickle.load(g1)

    with open(offset_spline_file, "rb") as g2:
        offset_spline = pickle.load(g2)
    return gain_spline, offset_spline

def save_orbit_map(orbit, label):
    orbit_map = WISEMap("orbit_{}_{}.fits".format(orbit.orbit_num, label), orbit.band)
    orbit_map.mapdata[orbit.pixel_inds_clean_masked] = orbit.zs_data_clean_masked
    orbit_map.save_map()
    return

if __name__ == "__main__":

    # Declare paths to important input files/directories
    moon_stripe_file = "/home/users/mberkeley/wisemapper/data/masks/moon_stripe_mask_G_thick3.fits"  # Input file, already exists
    orbit_file_path = "/home/users/mberkeley/wisemapper/data/output_maps/w3_days_mooncut/csv_files/"  # Input files, path already exists
    zodi_file_path = "/home/users/jguerraa/AME/cal_files/W3_days_mooncut/"  # Input files, path already exists

    # Declare outputs
    output_path = os.getcwd()  # Specify where to store output files from calibration
    # Specify desired name (and path) of output full-sky map
    fsm_map_file = os.path.join(output_path, "fullskymap_band3_masked.fits")
    iterations = 15

    sf = SplineFitter(iter_num=iterations - 1, path_to_fitvals=output_path)
    gain_spline, offset_spline = load_splines(sf.gain_spline_file, sf.offset_spline_file)

    for orb_num in range(212):
        # Initialize Coadder object for managing calibration
        coadd_map = Coadder(3, moon_stripe_file, fsm_map_file, orbit_file_path, zodi_file_path, output_path)
        orbit = Orbit(orb_num, 3, None, 256)
        orbit.load_orbit_data()
        orbit.load_zodi_orbit_data()
        orbit.apply_mask()


        # Fit a spline through the converged fit values for gains and offsets
        orbit.apply_spline_fit(gain_spline, offset_spline)
        orbit.plot_fit(output_path)
        orbit.plot_diff(output_path)
        save_orbit_map(orbit, "subtraction")

        rot_zs_data, rot_pix_inds, theta_rot, phi_rot = orbit.rotate_data("G", "E", orbit.zs_data_clean_masked, orbit.pixel_inds_clean_masked, orbit._nside)
        rot_cal_data, rot_pix_inds, theta_rot, phi_rot = orbit.rotate_data("G", "E", orbit._cal_data_clean_masked,
                                                                       orbit.pixel_inds_clean_masked, orbit._nside)
        rot_zodi_data, rot_pix_inds, theta_rot, phi_rot = orbit.rotate_data("G", "E", orbit._zodi_data_clean_masked,
                                                                       orbit.pixel_inds_clean_masked, orbit._nside)
        orbit.zs_data_clean_masked = rot_zs_data
        orbit._cal_data_clean_masked = rot_cal_data
        orbit._zodi_data_clean_masked = rot_zodi_data
        orbit.pixel_inds_clean_masked = rot_pix_inds
        orbit.plot_fit(output_path, label="_ecl")

        orbit.zs_data_clean_masked = orbit.zs_data_clean_masked.astype(bool)
        save_orbit_map(orbit, "region")


