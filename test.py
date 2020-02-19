from fullskymapping import FullSkyMap

def main():

    filename = "/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/orbit_maps/fsm_w3_orbit_100_uncalibrated.fits"
    filemap = FullSkyMap(filename, 256)
    filemap.read_data()
    mapdata = filemap.mapdata

    unc_filename = filename.replace("_orbit_", "_unc_orbit_")
    unc_filemap = FullSkyMap(unc_filename, 256)
    unc_filemap.read_data()
    uncdata = unc_filemap.mapdata

if __name__ == "__main__":
    main()