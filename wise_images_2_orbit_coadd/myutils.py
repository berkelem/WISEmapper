import sys


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar

    @params:
        iteration   - Required  : current iteration (Int) \n
        total       - Required  : total iterations (Int) \n
        prefix      - Optional  : prefix string (Str) \n
        suffix      - Optional  : suffix string (Str) \n
        decimals    - Optional  : positive number of decimals in percent complete (Int) \n
        bar_length  - Optional  : character length of bar (Int) \n
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '#' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def diff_maps(file1, file2):
    from wise_images_2_orbit_coadd.file_handler import HealpixMap
    import numpy as np

    map1 = HealpixMap(file1)
    map1.read_data()

    map2 = HealpixMap(file2)
    map2.read_data()

    diffmap = HealpixMap("diff_map.fits")
    mask = (map1.mapdata == 0.0) | (map2.mapdata == 0.0)
    diffmap.mapdata = np.zeros_like(map1.mapdata)
    diffmap.mapdata[~mask] = map1.mapdata[~mask] - map2.mapdata[~mask]
    diffmap.save_map()

if __name__ == "__main__":
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    diff_maps(file1, file2)