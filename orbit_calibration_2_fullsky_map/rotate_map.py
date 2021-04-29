import sys
from wise_images_2_orbit_coadd.file_handler import WISEMap

filename = sys.argv[1]
wisemap = WISEMap(filename, 3)
wisemap.read_data()
wisemap.filename = filename.replace(".fits", "_E.fits")
wisemap.rotate_map("G", "E")
wisemap.save_map("E")