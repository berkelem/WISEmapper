from wise_images_2_orbit_coadd.file_handler import HealpixMap

def moon_stripe_thickness():
    fsm = HealpixMap(
        "/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/fullsky_ongalaxyspline/fullskymap_band3_masked_calibrated.fits"
    )
    fsm.read_data()

    moon_mask = HealpixMap(
        "/Users/Laptop-23950/PycharmProjects/WISEmapper/wise_images_2_orbit_coadd/moon_stripe_mask_G_thick5.fits"
    )
    moon_mask.read_data()

    new_map = HealpixMap("stripe_thickness_5.fits")

    map_data = fsm.mapdata.copy()
    map_data[moon_mask.mapdata.astype(bool)] = 0.0

    new_map.mapdata = map_data
    new_map.save_map()

def mask_southern_latitudes():
    fsm = HealpixMap("south_mask.fits")
    fsm.set_resolution(256)
    fsm.rotate_map(old_coord="G", new_coord="G")
    mask = fsm.theta > 0
    fsm.mapdata[mask] = 1
    fsm.save_map()

if __name__ == "__main__":
    mask_southern_latitudes()o
