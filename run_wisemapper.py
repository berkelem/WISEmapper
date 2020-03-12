from fullskymapping import MapMaker, FileBatcher
from process_manager import RunLinear, RunRankZero, RunDistributed

import sys
import os


def main():

    band = int(sys.argv[1])
    filename = f"/home/users/mberkeley/wisemapper/data/filelists/filtered_df_band{band}.csv"
    # filename = f"/Users/Laptop-23950/projects/wisemapping/data/filelists/filtered_df_band{band}.csv"

    # Send filelists to ranks
    batch = FileBatcher(filename)
    RunLinear(batch.group_orbits)
    # RunLinear(batch.group_days)
    n_orbits = len(batch.groups)
    # n_days = len(batch.groups)
    print("{} orbits".format(n_orbits))
    # print("{} days".format(n_days))

    filelist_gen = RunLinear(batch.filelist_generator).retvalue

    # try:
    #     popt_vals = load_pickle('popt_vals.pkl')
    # except IOError:
    #     popt_vals = np.array([])
    n = 0
    while n < n_days:#n_orbits:
        # if os.path.exists(f"/home/users/mberkeley/wisemapper/data/output_maps/w{band}/fsm_w{band}_orbit_{n}.fits"):
        #     RunRankZero(print, data=f"Already mapped orbit {n + 1} of {n_orbits}")
        if os.path.exists(f"/home/users/mberkeley/wisemapper/data/output_maps/w{band}/fsm_w{band}_day_{n}.fits"):
            RunRankZero(print, data=f"Already mapped day {n + 1} of {n_days}")
            filelist = next(filelist_gen)
            n += 1
            continue

        # RunRankZero(print, data=f"Mapping orbit {n+1} of {n_orbits}")
        RunRankZero(print, data=f"Mapping day {n + 1} of {n_days}")
        filelist = next(filelist_gen)
        mapmaker = MapMaker(band, n)
        process_map = RunDistributed(mapmaker.add_image, filelist, iterate=True,
                   gather_items=[mapmaker.numerator_cumul, mapmaker.denominator_cumul])
        process_map.run()
        alldata = process_map.retvalue
        process_map.run_rank_zero(mapmaker.unpack_multiproc_data, data=alldata)
        try:
            process_map.run_rank_zero(mapmaker.normalize)
            # process_map.run_rank_zero(mapmaker.calibrate)
            process_map.run_rank_zero(mapmaker.save_map)
        except ValueError:
            continue
        # process_map.run_rank_zero(np.append, data=(popt_vals, mapmaker.calibrator.popt))
        # process_map.run_rank_zero(save_pickle, data=('popt_vals.pkl', popt_vals))
        n += 1
    print("Finished code")

if __name__ == "__main__":
    main()


    # To do:
    # - Calibrate fit to ceiling
    # - Configure logging
    # - Clean up process manager
    # - pep8
    # - filepaths
    # - Github
    # - run map combiner script


