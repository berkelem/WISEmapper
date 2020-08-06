import pickle
from wise_images_2_orbit_coadd.file_handler import WISEMap

def uncalibrate_file(orbit_id, gain, offset):
    mapfile = WISEMap(f"/home/users/mberkeley/wisemapper/data/output_maps/w3/fsm_w3_orbit_{orbit_id}.fits", 3)
    mapfile.read_data()
    mapfile.mapdata[mapfile.mapdata != 0.0] = (mapfile.mapdata[mapfile.mapdata != 0.0] - offset)/gain
    mapfile.filename = mapfile.filename.replace(".fits", "_uncalibrated.fits")
    mapfile.write_data()

    uncmapfile = WISEMap(f"/home/users/mberkeley/wisemapper/data/output_maps/w3/fsm_w3_unc_orbit_{orbit_id}.fits", 3)
    uncmapfile.read_data()
    uncmapfile.mapdata[uncmapfile.mapdata != 0.0] = (uncmapfile.mapdata[mapfile.mapdata != 0.0]) / abs(gain)
    uncmapfile.filename = uncmapfile.filename.replace(".fits", "_uncalibrated.fits")
    uncmapfile.write_data()

    return


def main():
    with open("/home/users/mberkeley/wisemapper/scripts/popt_w3.pkl", "rb") as cal_params:
        gains, offsets = pickle.load(cal_params)

    for i in range(100, 6323):
        print(f"Uncalibrating file {i} of 6323")
        uncalibrate_file(i, gains[i], offsets[i])



if __name__ == "__main__":
    main()