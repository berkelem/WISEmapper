from fullskymapping import MapMaker, FileBatcher, MapCombiner
from process_manager import RunProcess, RunParallel, RunLinear, RunRankZero, RunDistributed
import matplotlib.pyplot as plt
import healpy as hp
import sys
import os

def main():

    band = int(sys.argv[1])
    filename = f"/home/users/mberkeley/wisemapper/data/filelists/filtered_df_band{band}.csv"
    # filename = f"/Users/Laptop-23950/projects/wisemapping/data/filelists/filtered_df_band{band}.csv"

    # Send filelists to ranks
    batch = FileBatcher(filename)
    RunLinear(batch.group_days)
    n_days = len(batch.groups)

    filelist_gen = RunLinear(batch.filelist_generator).retvalue

    n = 0
    while n < n_days:
        if os.path.exists(f"/home/users/mberkeley/wisemapper/data/output_maps/w{band}/fsm_w{band}_day_{n}.fits"):
            RunRankZero(print, data=f"Already mapped day {n + 1} of {n_days}")
            filelist = next(filelist_gen)
            n += 1
            continue
        RunRankZero(print, data=f"Mapping day {n+1} of {n_days}")
        filelist = next(filelist_gen)
        mapmaker = MapMaker(band, n)
        process_map = RunDistributed(mapmaker.add_image, filelist, iterate=True,
                   gather_items=[mapmaker.numerator_cumul, mapmaker.denominator_cumul])
        process_map.run()
        alldata = process_map.retvalue
        process_map.run_rank_zero(mapmaker.unpack_multiproc_data, data=alldata)
        process_map.run_rank_zero(mapmaker.normalize)
        process_map.run_rank_zero(mapmaker.calibrate)
        process_map.run_rank_zero(mapmaker.save_map)
        n += 1
    print("Finished code")

if __name__ == "__main__":
    main()
    # band = 4
    # days = 212
    # mc = MapCombiner(band)
    # mc.add_days(0, days)
    # mc.normalize()
    # mc.save_file()
    # scale_dict = {1: 1, 2: 1, 3: 35, 4: 70}
    # hp.mollview(mc.fsm.mapdata, unit='MJy/sr', max=scale_dict[band])
    # plt.savefig(f'days{days}_band{band}.png')
    # plt.close()

    # To do:
    # - Calibrate fit to ceiling
    # - Configure logging
    # - Clean up process manager
    # - pep8
    # - filepaths
    # - Github
    # - run map combiner script


