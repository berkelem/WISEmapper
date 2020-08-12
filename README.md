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

1. Firstly, the WISE images are grouped into scans, using the `FileBatcher` object in the `data_management.py` module. A single WISE image has the following name structure:
`<scan_id><frame_num>-w<band>-int-1b.fits`. Each scan contains an average of about 201 frames, with a minimum of 7 and a maximum of 273. There are 6323 scans in total. A single scan typically covers roughly half a full orbit. Nevertheless, throughout the code "orbit" and "scan" are used interchangeably, unless specified otherwise.

2. The scan batches are distributed among the available processing units. The MPI interface is managed in the `process_manager.py` module.

3. In `fullskymapping.py` the `MapMaker` class manages the creation of coadds for the WISE scan. This involves the following steps:
    1. The WISE images are read in using the `WISEDataLoader` class in the `data_management.py` module. Each WISE intensity image has corresponding mask and uncertainty files. The mask is applied to the intensity image and extended to include all NaN values and negative values.
    2. Using the WCS information in the FITS file header, the WISE image is mapped to a Healpix grid. In order to propagate uncertainties, the final values (intensity and uncertainty) of a single Healpix pixel are calculated as follows:
        
        ![equation](https://latex.codecogs.com/svg.latex?%5Cbegin%7Bequation%7D%20%5Cbar%7Bx%7D%20%3D%20%5Cfrac%7B%5Csum_i%5Cleft%28%5Cfrac%7Bx_i%7D%7B%5Csigma_i%5E2%7D%5Cright%29%7D%7B%5Csum_j%5Cleft%28%5Cfrac%7B1%7D%7B%5Csigma_j%5E2%7D%5Cright%29%7D%20%5Cend%7Bequation%7D)

        ![euqation](https://latex.codecogs.com/svg.latex?%5Cbegin%7Bequation%7D%20%5Cbar%7B%5Csigma%7D%20%3D%20%5Cfrac%7B1%7D%7B%5Csum_j%5Cleft%28%5Cfrac%7B1%7D%7B%5Csigma_j%5E2%7D%5Cright%29%7D%20%5Cend%7Bequation%7D)

        where *i* and *j* denote every WISE image pixel that maps to the Healpix pixel in question. To facilitate this propagation over many pixels and many WISE images, the numerator and denominator of the above equations are accumulated independently before combining them at the end.
    
    3. The timestamp for the final Healpix pixel is also weighted according to the contribution of each WISE image to that pixel, as follows:
    
        ![equation](https://latex.codecogs.com/svg.latex?%5Cbegin%7Bequation%7D%20%5Cbar%7B%5Ctau%7D%20%3D%20%5Cfrac%7B%5Csum_i%5Cleft%28%5Cfrac%7BN_i%20t_i%7D%7B%5Cbar%7B%5Csigma%7D_i%5E2%7D%5Cright%29%7D%7B%5Csum_j%5Cleft%28%5Cfrac%7BN_j%7D%7B%5Cbar%7B%5Csigma%7D_j%5E2%7D%5Cright%29%7D%20%5Cend%7Bequation%7D)
    
        Here *i* indexes the WISE images that contribute to the Healpix pixel, and *N* is the number of WISE pixels contributed by WISE image *i* to the Healpix pixel.
        
    4. The numerator and denominator values for all of the processors are pooled to a single processor. The cumulative numerator is divided by the cumulative denominator to give the final Healpix pixel value.

    5. The final maps (intensity and uncertainty) are saved to file. Two types of file are created:
    - Full-sky Healpix maps: These FITS files are sparse and only contain non-zero pixels in the WISE scan region. There is one for intensity and one for uncertainty. The filenames are of the form `fsm_w<band>_orbit_<orbit_num>.fits` and `fsm_w<band>_unc_orbit_<orbit_num>.fits`.
    - Table of pixel values: This CSV file contains the following columns, for non-zero pixels only: 
    (Healpix pixel index, pixel intensity, pixel uncertainty, pixel timestamp). The filename is of the form `band_w<band>_orbit_<orbit_num>_pixel_timestamps.csv`.


##3. Calibration

