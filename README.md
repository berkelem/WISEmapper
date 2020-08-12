# WISEmapper

Create full sky maps using all of the WISE data in each of the four mid-infrared bands.

The pipeline from raw WISE images to full-sky calibrated Healpix maps involves the following major steps:

[1. Filter WISE data](#1-filter-wise-data):
Select WISE images that meet certain quality criteria.

[2. Create scan coadds](#2-create-scan-coadds):
Create coadds of individual WISE images that correspond to complete WISE scans (approximately one half-orbit)

[3. Calibration](#3-calibration):
Calibrate the orbit coadds using a zodiacal light template generated for the same timestamps using the Kelsall model.


##1. Filter WISE data

The code for this step is found in `wise_file_selection.py`

First of all, the metadata for all WISE images was downloaded. The WISE data was filtered according to the following criteria, in line with the recommendations in the [WISE Explanatory Supplement](http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec1_4d.html#singlexp_img):

- `qual_frame` == 10
- `qual_scan` > 0
- `saa_sep` > 0
- `dtanneal` > 2000
- `moon_sep` > 24

The filtered metadata was stored in a CSV file named `filtered_df_band#.csv` where `#` is replaced by each of `[1,2,3,4]`.

##2. Create scan coadds

`run_wisemapper.py` is the master script controlling how individual scan coadds are created from the raw WISE data.

This script is called as follows

`python run_wisemapper.py <band> <metadata_filename> <output_path>`

where 

`<band>` is one of `[1,2,3,4]` and correponds to the four WISE bands (3.4, 4.6, 12, 22) microns,

`<metadata_filename>` is the path to the previously created `filtered_df_band#.csv` file (see [previous section](#1-filter-wise-data)), and 

`<output_path>` is the path to the directory where the output files should be written.

######Note: The script was written to be run in an MPI environment, so the `python` command should be preceded by `mpirun` in a sbatch script file.

### Steps

1. Firstly, the WISE images are grouped into scans. A single WISE image has the following name structure:
`<scan_id><frame_num>-w<band>-int-1b.fits`. Each scan contains an average of about 201 frames, with a minimum of 7 and a maximum of 273. There are 6323 scans in total. A single scan typically covers roughly half a full orbit. Nevertheless, throughout the code "orbit" and "scan" are used interchangeably, unless specified otherwise.

2. The scan batches are distributed among the available processing units. The MPI interface is managed in the `process_manager.py` module.



##3. Calibration