from wise_images_2_orbit_coadd.wise_file_selection import MetaDataReader

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


    print(min_seps)
    print(max_seps)