import pandas as pd
import numpy as np

class MetaDataReader:

    def __init__(self):
        self.dataframe = None
        self.filtered_df = None

    def load_metadata(self, filename):

        self.dataframe = pd.read_csv(filename,
                                     sep='\s+',
                                     names=["band", "crval1", "crval2", "ra1", "dec1", "ra2", "dec2", "ra3", "dec3",
                                            "ra4", "dec4", "magzp", "magzpunc", "modeint", "scan_id", "scangrp",
                                            "frame_num", "date_obs_date", "date_obs_time", "mjd_obs", "dtanneal",
                                            "utanneal_date", "utanneal_time", "exptime", "qa_status", "qual_frame",
                                            "debgain", "febgain", "moon_sep", "saa_sep", "qual_scan"],
                                     dtype={"band": np.int32, "crval1": np.float64, "crval2": np.float64,
                                            "ra1": np.float64, "dec1": np.float64, "ra2": np.float64,
                                            "dec2": np.float64, "ra3": np.float64, "dec3": np.float64,
                                            "ra4": np.float64, "dec4": np.float64, "magzp": np.float64,
                                            "magzpunc": np.float64, "modeint": np.float64, "scan_id": np.object_,
                                            "scangrp": np.object_, "frame_num": np.object_, "date_obs_date": np.object_,
                                            "date_obs_time": np.object_, "mjd_obs": np.float64, "dtanneal": np.float64,
                                            "utanneal_date": np.object_, "utanneal_time": np.object_,
                                            "exptime": np.float64, "qa_status": np.object_, "qual_frame": np.int32,
                                            "debgain": np.float64, "febgain": np.float64, "moon_sep": np.float64,
                                            "saa_sep": np.float64, "qual_scan": np.int32},
                                     skiprows=4
                                     )
        return

    def filter_files(self):

        selection = ((self.dataframe['qual_frame'] == 10)
                     & (self.dataframe['qual_scan'] > 0)
                     & (self.dataframe['saa_sep'] > 0)
                     & (self.dataframe['dtanneal'] > 2000)
                     & (self.dataframe['moon_sep'] > 90)
                     )

        self.filtered_df = self.dataframe[selection].copy()
        return

    @staticmethod
    def get_basepath(row):
        if int(row['scangrp'][0]) < 3 or str(row['scangrp']) == '3a':
            wisefolder = 'wise1'
        elif str(row['scangrp']) == '5c':
            wisefolder = 'wise4'
        elif (int(row['scangrp'][0]) < 7 or str(row['scangrp']) == '3b' or str(row['scangrp']) == '8b'
              or (str(row['scangrp']) == '8a' and int(row['scan_id'][:5]) < 1999)):
            wisefolder = 'wise2'
        elif (int(row['scangrp'][0]) == 7 or int(row['scangrp'][0]) == 9 or
              (int(row['scangrp'][0]) == 8 and int(row['scan_id'][:5]) > 1999)):
            wisefolder = 'wise3'
        else:
            raise IOError(f"File {row['filename']} not assigned a wisefolder")

        return f"/mnt/wise/{wisefolder}/{row['scangrp']}/{row['scan_id']}/{row['frame_num']}/"


class FileSelector(MetaDataReader):

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        super().load_metadata(self.filename)
        super().filter_files()

    def add_basepath(self):
        self.filtered_df['basepath'] = self.filtered_df.apply(self.get_basepath, axis=1)
        return

    def add_filename(self):
        self.filtered_df['filename'] = self.filtered_df['scan_id'] + self.filtered_df['frame_num'] + '-w' \
                                       + self.filtered_df['band'].map(str) + '-int-1b.fits'
        return

    def combine_basepath_and_filename(self):
        self.filtered_df['full_filepath'] = self.filtered_df['basepath'] + self.filtered_df['filename']
        self.filtered_df.drop(['filename', 'basepath'], axis=1, inplace=True)
        return

    def write_file(self, output_file):
        np.savetxt(output_file, self.filtered_df['full_filepath'].values, fmt='%s')
        return


if __name__ == '__main__':
    for band in [2,3,4]:
        # band = 2
        filename = f"all_band{band}.tbl"
        fileselector = FileSelector(filename)
        print(fileselector.filtered_df)
        fileselector.add_filename()
        fileselector.add_basepath()
        fileselector.combine_basepath_and_filename()
        print(fileselector.filtered_df)
        fileselector.filtered_df.to_csv(f"filtered_df_band{band}.csv", index=False)
