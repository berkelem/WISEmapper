from orbit_calibration_2_fullsky_map.coadd_orbits import Orbit


if __name__ == "__main__":
    orbit_file_path = "/home/users/mberkeley/wisemapper/data/output_maps/w3_days_mooncut/csv_files/"  # Input files, path already exists
    zodi_file_path = "/home/users/jguerraa/AME/cal_files/W3_days_mooncut/"  # Input files, path already exists
    setattr(Orbit, "orbit_file_path", orbit_file_path)
    setattr(Orbit, "zodi_file_path", zodi_file_path)
    orbit = Orbit(141, 3, None, 256)
    orbit.load_orbit_data()
    orbit.load_zodi_orbit_data()
    orbit.fit()
    orbit.plot_fit(".")