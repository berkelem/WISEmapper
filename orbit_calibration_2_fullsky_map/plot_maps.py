from wise_images_2_orbit_coadd.file_handler import WISEMap
import sys

filename = sys.argv[1]
title = sys.argv[2]
fsm = WISEMap(filename, 3)
fsm.read_data()
fsm.plot_mollweide(title, min=0, max=2, rot=[-90, -30])

