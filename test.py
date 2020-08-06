from fullskymapping import FullSkyMap
import matplotlib.pyplot as plt
import numpy as np

def main():

    # filename = "/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/orbit_maps/fsm_w3_orbit_100_uncalibrated.fits"
    filename1 = "/Users/Laptop-23950/projects/wisemapping/data/output_maps/day_analysis/pole_fitting/fullskymap_band3_200_202.fits"
    filemap1 = WISEMap(filename1, 3)
    filemap1.read_data()
    mapdata1 = filemap1.mapdata

    filename2 = "/Users/Laptop-23950/projects/wisemapping/data/output_maps/day_analysis/pole_fitting/fullskymap_band3_5480_5482.fits"
    filemap2 = WISEMap(filename2, 3)
    filemap2.read_data()
    mapdata2 = filemap2.mapdata

    filename3 = "/Users/Laptop-23950/projects/wisemapping/data/kelsall_maps/kelsall_model_wise_scan_lam_12_v3.fits"
    filemap3 = WISEMap(filename3, 3)
    filemap3.read_data()
    mapdata3 = filemap3.mapdata

    overlap_pixels = mapdata1.astype(bool) & mapdata2.astype(bool)

    overlap_data1 = mapdata1[overlap_pixels]
    overlap_data2 = mapdata2[overlap_pixels]
    overlap_data3 = mapdata3[overlap_pixels]

    plt.plot(overlap_data1, overlap_data2, 'r.', np.arange(110), np.ones(110)*np.arange(110), 'b--')
    plt.xlabel("Orbit 101")
    plt.ylabel("Orbit 2740")
    plt.savefig("repeated_orbit_200_5480.png")
    plt.close()

    l1, = plt.plot(np.arange(len(overlap_data1)), overlap_data1, 'r.', ms=0.7, alpha=1)
    l2, = plt.plot(np.arange(len(overlap_data2)), overlap_data2, 'b.', ms=0.7, alpha=1)
    l3, = plt.plot(np.arange(len(overlap_data3)), overlap_data3, 'g.', ms=0.7, alpha=1)
    plt.legend((l1, l2, l3), ("Orbit 200-202", "Orbit 5480-5482", "Kelsall map"), markerscale=10)
    plt.xlabel("Pixel id")
    plt.ylabel("MJy/sr")
    plt.savefig("repeated_orbit_comparison_200_5480.png")
    plt.close()


if __name__ == "__main__":
    main()