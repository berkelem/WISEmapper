from wise_images_2_orbit_coadd.file_handler import WISEMap


def subtract_zodi():
    mapfilename = "fullskymap_band3_6323.fits"
    zodifilename = "kelsall_model_wise_scan_lam_12_v3.fits"

    wise_map = WISEMap(mapfilename, 3)
    zodi_map = WISEMap(zodifilename, 3)

    wise_map.read_data()
    zodi_map.read_data()

    output_map = WISEMap("band_3_zodi_subtracted.fits", 3)
    output_map.mapdata = wise_map.mapdata - zodi_map.mapdata
    output_map.save_map()


if __name__ == "__main__":
    subtract_zodi()