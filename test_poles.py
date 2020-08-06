from fullskymapping import FullSkyMap
import numpy as np
from healpy.rotator import Rotator
import healpy as hp

def select_pole_region(deg):
    # pole_test_map = WISEMap("pole_test.fits", 3)
    input_map = WISEMap(
            "/home/users/mberkeley/wisemapper/data/output_maps/fsm_attempt6/w3/fullskymap_band3.fits", 3)
    input_map.read_data()

    inds = np.arange(input_map.npix)

    mask = np.zeros_like(inds)
    mask_map = WISEMap(f"mask_map_{int(deg)}.fits", 3)
    mask_map.mapdata = mask

    ra_arr, dec_arr = input_map.ind2wcs(inds)

    mask_map.mapdata = (dec_arr < -deg) | (dec_arr > deg)

    mask_map.rotate_map("E", "G")

    mask_map.save_map()

    # pole_test_map.mapdata[mask_map.mapdata.astype(bool)] = input_map.mapdata[mask_map.mapdata.astype(bool)]
    #
    # pole_test_map.save_map()
    return

def main():
    select_pole_region(80.0)



if __name__ == "__main__":
    main()