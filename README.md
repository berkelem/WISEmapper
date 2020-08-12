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

###### Note: The script was written to be run in an MPI environment, so the `python` command should be preceded by `mpirun` in a sbatch script file.

### Scan Coadd Steps

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

`run_calibration.py` is the master script controlling the calibration of the scan coadds and the creation of the final full-sky zodi-subtracted map.

The script requires the following paths to be declared:

- `moon_stripe_file`: The path to the file containing the moon stripe mask.
- `gain_pickle_file`: The filename in which to store the spline for the gain values. It should end in ".pkl" as it is a binary pickle file.
- `offset_pickle_file`: The filename in which to store the spline for the offset values. As before it should end in ".pkl".
- `fsm_map_file`: The desired name of the final output file.
- `orbit_file_path`: The path to the orbit coadd CSV files (see [previous section](#2-create-scan-coadds))
- `zodi_file_path`: The path to the zodiacal light maps corresponding to the individual orbits, generated using the Kelsall model.

### Calibration Steps

1. A `Coadder` object is initialized from the module `coadd_orbits.py`. This manages the coadding pipeline for all `Orbit` objects from the same module.
2. The `Orbit` class reads the CSV file for a given orbit and its corresponding zodiacal light template. The moon stripe regions are masked in the data. This class also provides methods for fitting the coadd data to the zodi template.
3. The iterative fitting procedure proceeds as follows:
    1. For each orbit, an initial fit for gain and offset is performed in the `Orbit` class.
    2. The non-negative residual of the fit is taken as a proxy for galactic signal. This positive-definite zodi-subtracted signal is added to a full-sky Healpix map in the same manner as described in [Step 3 of the previous section](#scan-coadd-steps)
    3. After processing all orbits, the full-sky map containing all the residuals is normalized and passed back to the `Orbit` object as a proxy for galactic signal in the next iteration
    4. For the next iteration, each orbit is processed as before, but the galactic signal is subtracted from the WISE data in advance of the fitting procedure. This facilitates a better fit with the zodiacal light template. In addition, any pixels in which the residual was an outlier (z-score > 1) were removed before the subsequent fit, as these pixels are likely to contain a strong galactic component.
    5. The steps i-iv are repeated for as many iterations as specified, and the fit values for gain and offset converge as the iterations continue.
4. After the specified number of iterations, the converged values for the gain and offset are loaded into a `SplineFitter` object in the `spline_fit_calibration.py` module.
5. Outlier values are removed, and then by manual inspection other aberrant values are removed. In particular, the moon stripe regions are associated with bad fit parameters for the gain and offset and need to be removed.
6. A Univariate spline is fit through the remaining points. Two such splines are made - one for gain and one for offset.
7. The splines are saved and passed back to the previously-created `Coadder` class.
8. The `Coadder` object loops through all of the orbits one more time, this time telling the `Orbit` class to draw fit parameters from the splines.
9. The resulting zodi-subtracted residuals (no longer forced to be positive-definite) are used to create the final calibrated full-sky map. This follows the same process as Step 3.ii except the data is cleaned before being added, as follows:
    1. For every Healpix pixel, the contribution from all orbits mapping to that Healpix pixel are stored in a list, rather than accumulated directly.
    2. The distribution of values for each Healpix pixel is considered, and any outlier values (z-score > 1) are removed.
    3. The remaining values go through the accumulation process for numerator and denominator, and subsequent normalization, described [previously](#scan-coadd-steps).
10. The final map is saved to file.
    

    