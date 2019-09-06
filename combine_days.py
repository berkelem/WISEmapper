from fullskymapping import MapCombiner
import os
import healpy as hp
import matplotlib.pyplot as plt


def combine_days(band, days):
    mc = MapCombiner(band)
    mc.add_days(0, days)
    mc.normalize()
    mc.save_file()
    return mc.fsm.mapdata



def plot_map(data, days, band):
    scale_dict = {1: 1, 2: 1, 3: 35, 4: 70}
    hp.mollview(data, unit='MJy/sr', max=scale_dict[band])
    plt.savefig(f'days{days}_band{band}.png')
    plt.close()

if __name__ == "__main__":
    for band in range(1, 5):
        print(f"Creating band {band} map")
        max_day = -1
        day_available = True
        while day_available:
            max_day += 1
            day_available = os.path.exists(f"/home/users/mberkeley/wisemapper/data/output_maps/w{band}/fsm_w{band}_day_{max_day}.fits")
        print(f"Combining {max_day} days")
        mapdata = combine_days(band, max_day)
        plot_map(mapdata, max_day, band)
