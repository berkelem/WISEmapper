from orbit_calibration_2_fullsky_map.coadd_orbits import Coadder
from orbit_calibration_2_fullsky_map.spline_fit_calibration import SplineFitter

if __name__ == "__main__":

    # Declare paths to important files/directories
    moon_stripe_file = "/home/users/mberkeley/wisemapper/data/masks/stripe_mask_G.fits"
    gain_pickle_file = "/home/users/mberkeley/wisemapper/data/output_maps/w3/gain_spline.pkl"
    offset_pickle_file = "/home/users/mberkeley/wisemapper/data/output_maps/w3/offset_spline.pkl"
    fsm_map_file = "/home/users/mberkeley/wisemapper/data/output_maps/w3/fullskymap_band3_masked.fits"
    orbit_file_path = "/home/users/mberkeley/wisemapper/data/output_maps/w3/csv_files/"
    zodi_file_path = "/home/users/jguerraa/AME/cal_files/W3/"

    # Initialize Coadder object for managing calibration
    coadd_map = Coadder(3, moon_stripe_file, gain_pickle_file, offset_pickle_file, fsm_map_file, orbit_file_path,
                        zodi_file_path)
    coadd_map.run_iterative_fit(iterations=25, month='all')

    # Fit a spline through the converged fit values for gains and offsets
    sf = SplineFitter(iter_num=24)
    sf.fit_spline()

    # Load the spline back into the Coadder and do a final calibration
    coadd_map.load_splines()
    coadd_map.add_calibrated_orbits()