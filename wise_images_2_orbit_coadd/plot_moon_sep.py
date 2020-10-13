from wise_images_2_orbit_coadd.wise_file_selection import MetaDataReader
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filename = "~/PAH_Project/all_band3.tbl"
    meta_reader = MetaDataReader()
    meta_reader.load_metadata(filename)

    moon_sep_by_scan = meta_reader.dataframe[["moon_sep", "scan_id"]]

    scan_groups = moon_sep_by_scan.groupby(moon_sep_by_scan["scan_id"])

    min_seps = []
    max_seps = []
    for name, group in scan_groups:
        minval = min(group["moon_sep"])
        maxval = max(group["moon_sep"])
        min_seps.append(minval)
        max_seps.append(maxval)

    plt.plot(range(0, len(min_seps), 2), min_seps[::2], 'r.', ms=0.5, label="Even")
    plt.plot(range(1, len(min_seps), 2), min_seps[1::2], 'b.', ms=0.5, label="Odd")
    plt.xlabel("Scan number")
    plt.ylabel("Moon Sep minimum (Deg)")
    plt.savefig("moon_sep_min.png")
    plt.close()

    plt.plot(range(0, len(max_seps), 2), max_seps[::2], 'r.', ms=0.5, label="Even")
    plt.plot(range(1, len(max_seps), 2), max_seps[1::2], 'b.', ms=0.5, label="Odd")
    plt.xlabel("Scan number")
    plt.ylabel("Moon Sep maximum (Deg)")
    plt.savefig("moon_sep_max.png")
    plt.close()

    # print(min_seps)
    # print(max_seps)