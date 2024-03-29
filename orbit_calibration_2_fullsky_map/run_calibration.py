from orbit_calibration_2_fullsky_map.coadd_orbits import Coadder
from orbit_calibration_2_fullsky_map.spline_fit_calibration import SplineFitter
import os

if __name__ == "__main__":

    # Declare paths to important input files/directories
    moon_stripe_file = "/home/users/mberkeley/wisemapper/data/masks/moon_stripe_mask_G_thick3.fits"#south_mask.fits"#moon_stripe_mask_G_thick3.fits"  # Input file, already exists
    orbit_file_path = "/home/users/mberkeley/wisemapper/data/output_maps/w3/csv_files/"#w2/csv_files/"#w4/csv_files_orbits/"#w3_days/csv_files/"  # Input files, path already exists
    zodi_file_path = "/home/users/jguerraa/AME/cal_files/W3_new/"#W2/"#W3_new/"#_days_new/"#W3_days_mooncut/"  # Input files, path already exists

    # Declare outputs
    output_path = os.getcwd()  # Specify where to store output files from calibration
    # Specify desired name (and path) of output full-sky map
    fsm_map_file = os.path.join(output_path, "fullskymap_band3_masked.fits")
    iterations = 1

    # Initialize Coadder object for managing calibration
    coadd_map = Coadder(3, moon_stripe_file, fsm_map_file, orbit_file_path, zodi_file_path, output_path)
    coadd_map.load_orbits(month=["Jan"])#"all")#["Feb", "Mar", "Apr", "May", "Jun", "Jul"])
    # coadd_map.load_fitvals(0)
    coadd_map.run_iterative_fit(iterations=iterations)

    # Fit a spline through the converged fit values for gains and offsets
    # sf = SplineFitter(iter_num=iterations-1, path_to_fitvals=output_path)
    # sf.fit_spline()

    # Load the spline back into the Coadder and do a final calibration
    # coadd_map.load_splines(sf.gain_spline_file, sf.offset_spline_file)
    coadd_map.add_calibrated_orbits(plot=True)
