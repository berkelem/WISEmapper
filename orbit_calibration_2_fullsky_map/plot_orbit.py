from orbit_calibration_2_fullsky_map.coadd_orbits import Orbit


if __name__ == "__main__":
    orbit = Orbit(141, 3, None, 256)
    orbit.load_orbit_data()
    orbit.load_zodi_orbit_data()
    orbit.fit()
    orbit.plot_fit(".")