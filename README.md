# WISEmapper

Create full sky maps using all of the WISE data in each of the four mid-infrared bands.

The pipeline from raw WISE images to full-sky calibrated Healpix maps involves the following major steps:

1. Create coadds of individual WISE images that correspond to complete WISE scans (approximately one half-orbit)
2. Calibrate the orbit coadds using a zodiacal light template generated for the same timestamps using the Kelsall model.


Part 1

`wise_file_selection.py`

First of all, the metadata for all WISE images was downloaded. The WISE data was filtered according to the following criteria, in line with the recommendations in the WISE Explanatory Supplement:

- `qual_frame` == 10
- `qual_scan` > 0
- `saa_sep` > 0
- `dtanneal` > 2000
- `moon_sep` > 24

The filtered metadata was stored in a CSV file named `filtered_df_band#.csv` where `#` is replaced by each of `[1,2,3,4]`

`run_wisemapper.py`

This is the master script controlling how individual scan coadds are created from the raw WISE data.
