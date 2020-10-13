from wise_images_2_orbit_coadd.wise_file_selection import MetaDataReader

if __name__ == "__main__":
    filename = "~/PAH_Project/all_band3.tbl"
    meta_reader = MetaDataReader()
    meta_reader.load_metadata(filename)

    moon_sep_by_scan = meta_reader.dataframe[["moon_sep", "scan_id"]]
    print(moon_sep_by_scan)