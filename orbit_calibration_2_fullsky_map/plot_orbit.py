from orbit_calibration_2_fullsky_map.coadd_orbits import Orbit, Coadder
from orbit_calibration_2_fullsky_map.spline_fit_calibration import SplineFitter
import os



if __name__ == "__main__":
    output_path = os.getcwd()
    orbit_file_path = "/home/users/mberkeley/wisemapper/data/output_maps/w3_days_mooncut/csv_files/"  # Input files, path already exists
    zodi_file_path = "/home/users/jguerraa/AME/cal_files/W3_days_mooncut/"  # Input files, path already exists
    coadd_map = Coadder(3, None, None, orbit_file_path, zodi_file_path, ".")

    setattr(Orbit, "orbit_file_path", orbit_file_path)
    setattr(Orbit, "zodi_file_path", zodi_file_path)
    orbit = Orbit(141, 3, None, 256)
    orbit.load_orbit_data()
    orbit.load_zodi_orbit_data()
    sf = SplineFitter(iter_num=24, path_to_fitvals=output_path)
    coadd_map.load_splines(sf.gain_spline_file, sf.offset_spline_file)
    orbit.plot_fit(output_path)