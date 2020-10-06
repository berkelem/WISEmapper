from orbit_calibration_2_fullsky_map.coadd_orbits import Orbit
from wise_images_2_orbit_coadd.file_handler import HealpixMap
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

if __name__ == "__main__":
    moon_stripe_file = "/home/users/mberkeley/wisemapper/data/masks/stripe_mask_G.fits"  # Input file, already exists
    orbit_file_path = "/home/users/mberkeley/wisemapper/data/output_maps/w3/csv_files/"  # Input files, path already exists
    zodi_file_path = "/home/users/jguerraa/AME/cal_files/W3/"  # Input files, path already exists

    output_path = os.getcwd()

    n_orbits = 30

    moon_stripe_mask = HealpixMap(moon_stripe_file)
    moon_stripe_mask.read_data()
    moon_stripe_inds = np.arange(len(moon_stripe_mask.mapdata))[moon_stripe_mask.mapdata.astype(bool)]

    full_mask = moon_stripe_mask.mapdata.astype(bool)

    odd_peaks = []
    even_peaks = []
    for i in range(n_orbits):
        orbit = Orbit(i, 3, full_mask, 256)
        orbit.load_orbit_data()
        orbit.load_zodi_orbit_data()

        peak_zodi = np.max(orbit._zodi_data)
        if i % 2 == 0:
            even_peaks.append(peak_zodi)
        else:
            odd_peaks.append(peak_zodi)


        plt.plot(range(0, n_orbits, 2), even_peaks, 'r.', ms=5)
        plt.plot(range(1, n_orbits, 2), odd_peaks, 'b.', ms=5)
        plt.xlabel("Orbit number")
        plt.yaxis("Peak zodi value (MJy/sr)")
        plt.savefig(os.path.join(output_path, "peak_zodi.png"))
        plt.close()