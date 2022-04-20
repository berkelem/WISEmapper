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

def smooth_map(file1, smoothing_fwhm):
    from file_handler import HealpixMap
    import healpy as hp
    mapfile = HealpixMap(file1)
    mapfile.read_data()
    mapfile.set_resolution(256)
    i = 0
    while i < 1:
        print("i", i)
        mapfile.mapdata = hp.sphtfunc.smoothing(mapfile.mapdata, fwhm=smoothing_fwhm)
        i += 1
    mapfile.filename = "smooth_map.fits"
    mapfile.save_map()


def downgrade_map(filename):
    from file_handler import HealpixMap
    mapfile = HealpixMap(filename)
    mapfile.read_data()
    mapfile.set_resolution(8)
    mapfile.filename = "downgraded_map.fits"
    mapfile.save_map()

def upgrade_map(filename):
    from file_handler import HealpixMap
    import healpy as hp
    mapfile = HealpixMap(filename)
    mapfile.read_data()
    mapfile.set_resolution(256)
    i = 0
    while i < 1:
        print("i", i)
        mapfile.mapdata = hp.sphtfunc.smoothing(mapfile.mapdata, fwhm=0.1)
        i += 1
    mapfile.filename = "upgraded_map.fits"
    mapfile.save_map()

if __name__ == "__main__":
    file1 = sys.argv[1]
    fwhm = float(sys.argv[2])
    # file2 = sys.argv[2]
    # diff_maps(file1, file2)
    # upgrade_map(file1)
    smooth_map(file1, fwhm)