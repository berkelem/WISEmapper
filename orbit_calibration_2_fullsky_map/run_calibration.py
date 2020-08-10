from orbit_calibration_2_fullsky_map.coadd_orbits import Coadder

if __name__ == "__main__":
    moon_stripe_file = "/home/users/mberkeley/wisemapper/data/masks/stripe_mask_G.fits"
    gain_pickle_file = "/home/users/mberkeley/wisemapper/data/output_maps/w3/gain_spline.pkl"
    offset_pickle_file = "/home/users/mberkeley/wisemapper/data/output_maps/w3/offset_spline.pkl"
    fsm_map_file = "/home/users/mberkeley/wisemapper/data/output_maps/w3/fullskymap_band3_masked.fits"
    orbit_file_path = "/home/users/mberkeley/wisemapper/data/output_maps/w3/csv_files/"
    zodi_file_path = "/home/users/jguerraa/AME/cal_files/W3/"
    coadd_map = Coadder(3, moon_stripe_file, gain_pickle_file, offset_pickle_file, fsm_map_file, orbit_file_path,
                        zodi_file_path)
    coadd_map.run_iterative_fit(num_orbits=6323, iterations=25)
