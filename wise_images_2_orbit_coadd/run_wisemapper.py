"""
:author: Matthew Berkeley
:date: Jun 2019

Main script for creating coadds of WISE data. Each coadd represents an individual WISE scan.
Each WISE scan covers approximately half a full orbit.

To run:

python run_wisemapper.py <band> <metadata_file>

where
<band> can be any of [1,2,3,4]
<metadata_file> is the full path to the table containing WISE metadata for all orbits in the band
"""

from wise_images_2_orbit_coadd.fullskymapping import MapMaker
from wise_images_2_orbit_coadd.data_management import FileBatcher
from wise_images_2_orbit_coadd.process_manager import RunLinear, RunRankZero, RunDistributed
import sys
import os


def main(band, filename, output_path):

    # Create batches of raw WISE images corresponding to individual WISE scans/half-orbits.
    batch = FileBatcher(filename)
    RunLinear(batch.group_days)
    n_orbits = len(batch.groups)
    print("{} orbits".format(n_orbits))

    # Create a generator to iterate over the file batches
    filelist_gen = RunLinear(batch.filelist_generator).retvalue

    # Create a single coadd map of each batch of files
    n = 0
    orbit_num = None
    while n < n_orbits:
        # If coadd map for this orbit already exists, skip
        if os.path.exists(os.path.join(output_path, f"fsm_w{band}_orbit_{n}.fits")):
            RunRankZero(print, data=f"Already mapped orbit {n + 1} of {n_orbits}")
            filelist, mjd_list, orbit_num = next(filelist_gen)
            n += 1
            continue


        RunRankZero(print, data=f"Mapping week {n + 1} of {n_orbits}")
        filelist = []
        mjd_list = []

        # Generate next batch of files
        while orbit_num != n:
            filelist, mjd_list, orbit_num = next(filelist_gen)

        # Create coadd map of all files in batch
        mapmaker = MapMaker(band, n, output_path)
        process_map = RunDistributed(mapmaker.add_image, list(zip(filelist, mjd_list)), iterate=True,
                                     gather_items=[mapmaker.numerator_cumul, mapmaker.denominator_cumul,
                                                   mapmaker.time_numerator_cumul, mapmaker.time_denominator_cumul])
        process_map.run()
        alldata = process_map.retvalue
        RunRankZero(mapmaker.unpack_multiproc_data, data=alldata)
        RunRankZero(mapmaker.normalize)
        RunRankZero(mapmaker.save_map)

        n += 1
    print("Finished code")


if __name__ == "__main__":
    band = int(sys.argv[1])

    filename = str(sys.argv[2])
    # The appropriate metadata files are on Clusty at
    # f"/home/users/mberkeley/wisemapper/data/filelists/filtered_df_band{band}.csv"

    output_path = str(sys.argv[3])
    # The current output path is
    # f"/home/users/mberkeley/wisemapper/data/test_output/w{band}/"

    main(band, filename, output_path)
