from file_handler import HealpixMap
from fullskymapping import FullSkyMap

file1 = HealpixMap("/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/may18_smooth525_iterations50/fullskymap_band3_iter_49.fits")
file2 = HealpixMap("/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/fullskymap_band3_iter_42.fits")
file1.read_data()
file2.read_data()

diff_map = WISEMap("/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/diff_map.fits", 3)
diff_map.mapdata = file2.mapdata - file1.mapdata
diff_map.save_map()