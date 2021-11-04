"""
:author: Matthew Berkeley
:date: Jun 2020

Module for creating coadds on a Healpix grid.

Main classes
------------
Mapmaker : Class managing how batches of WISE images are co-added to form a single WISE scan
"""
import numpy as np
import healpy as hp
import pandas as pd
from wise_images_2_orbit_coadd.file_handler import WISEMap
from wise_images_2_orbit_coadd.data_management import WISEDataLoader
from functools import reduce
import os


class BaseMapper:
    """
    Base class shared by other map making classes
    """

    def __init__(self, band, label, path):
        self.band = band
        self.label = label
        self.path = path
        self.orbit_fits_name = f'fsm_w{self.band}_orbit_{self.label}.fits'
        self.orbit_unc_fits_name = self.orbit_fits_name.replace('orbit', 'unc_orbit')
        self.orbit_csv_name = f"band_w{self.band}_orbit_{self.label}_pixel_timestamps.csv"


class MapMaker(BaseMapper):
    """
    Class managing how batches of WISE images are co-added to form a single WISE scan (approximately one half-orbit)

    Parameters
    ----------
    :param band: int/str
        One of [1,2,3,4]
    :param label: int/str
        Orbit label
    :param out_path: str
        Directory in which to write output files (default is current directory)
    :param nside: int
        Healpix NSIDE parameter

    Methods
    -------
    add_image :
        Add individual WISE image to orbit coadd.
    normalize :
        Divide running count of numerator by running count of denominator. This enables propagation of uncertainties.
        See method docstring for details
    save_map :
        Save orbit placed on a full sky map to a Healpix FITS file and a tabular csv file containing all orbit data,
        including pixel timestamps
    unpack_multiproc_data :
        Retrieve all data into current object from other processors when running on a distributed system
    """

    def __init__(self, band, label, out_path, nside=256):
        super().__init__(band, label, out_path)
        self.fsm = WISEMap(self.orbit_fits_name, nside)
        self.unc_fsm = WISEMap(self.orbit_unc_fits_name, nside)
        self.numerator_cumul = np.zeros_like(self.fsm.mapdata)
        self.denominator_cumul = np.zeros_like(self.fsm.mapdata)
        self.time_numerator_cumul = np.zeros_like(self.fsm.mapdata)
        self.time_denominator_cumul = np.zeros_like(self.fsm.mapdata)

    def add_image(self, args):
        """
        Add individual WISE image to orbit coadd.

        Parameters
        ----------
        :param args: tuple
            Tuple (filelist, mjd_list) where filelist is the path to each WISE file and mjd_list is the mjd_obs
            timestamp associated with each file
        """
        filename, mjd_obs = args
        self._load_image(filename)
        self._place_image(mjd_obs)
        return

    def normalize(self):
        """
        As each image is added, uncertainties are propagated as follows (x = data, s = uncertainty (sigma)):

        x_bar = ∑(x/s^2) / ∑(1/x^2)
        s_bar = 1 / ∑(1/x^2)

        The numerator and denominator for x_bar are recorded separately and a running count is maintained as each new
        WISE image is added. This method divides the numerator by the denominator to give x_bar, and inverts the
        denominator to give s_bar.
        A similar process determines the weighted timestamp per Healpix pixel.
        """
        self.fsm.mapdata = np.divide(self.numerator_cumul, self.denominator_cumul, out=np.zeros_like(self.fsm.mapdata),
                                     where=self.denominator_cumul > 0)
        self.unc_fsm.mapdata = np.sqrt(np.divide(np.ones_like(self.denominator_cumul), self.denominator_cumul,
                                                 out=np.zeros_like(self.unc_fsm.mapdata),
                                                 where=self.denominator_cumul > 0))
        self.fsm.timedata = np.divide(self.time_numerator_cumul, self.time_denominator_cumul,
                                      out=np.zeros_like(self.fsm.mapdata),
                                      where=self.time_denominator_cumul > 0)
        return

    def save_map(self):
        """
        Save orbit placed on a full sky map to a Healpix FITS file and a tabular csv file containing all orbit data,
        including pixel timestamps
        """
        self._save_fits()
        self._save_csv()
        return

    def unpack_multiproc_data(self, alldata):
        """
        Unpack all data from other processors into attributes of current object

        Parameters
        ----------
        :param alldata: tuple
            Tuple containing (numerators, denominators, time_numerators, time_denominators) from each processor.
        """
        numerators, denominators, time_numerators, time_denominators = zip(*alldata)
        self.numerator_cumul = reduce(np.add, numerators)
        self.denominator_cumul = reduce(np.add, denominators)
        self.time_numerator_cumul = reduce(np.add, time_numerators)
        self.time_denominator_cumul = reduce(np.add, time_denominators)
        return

    @staticmethod
    def _calc_hp_pixel(data, unc, time):
        """
        As each image is added, uncertainties are propagated as follows (x = data, s = uncertainty (sigma)):

        x_bar = ∑(x/s^2) / ∑(1/x^2)
        s_bar = 1 / ∑(1/x^2)

        The numerator and denominator for x_bar are recorded separately and a running count is maintained as each new
        WISE image is added. The timestamp per pixel is recorded and tracked in a similar manner.

        Parameters
        ----------
        :param data: numpy.array
            intensity data
        :param unc: numpy.array
            uncertainty data
        :param time: numpy.array
            mjd_obs data
        :return: tuple of numpy.arrays
            numerator, denominator, time_numerator, time_denominator
        """
        numerator = np.sum(data / unc ** 2)
        denominator = np.sum(1 / unc ** 2)
        N = len(data)
        time_numerator = N * time * denominator
        time_denominator = N * denominator
        return numerator, denominator, time_numerator, time_denominator

    def _fill_map(self, inds, ints, uncs, mjd_obs):
        """
        This method maps the WISE pixel data to the corresponding Healpix pixels.

        As each image is added, uncertainties are propagated as described in _calc_hp_pixel().
        The cumulative values for numerator and denominator are tracked in this method.

        Parameters
        ----------
        :param inds: numpy.array
            Indices of Healpix pixels to which the WISE image maps
        :param ints: numpy.array
            intensity values of WISE image
        :param uncs: numpy.array
            Uncertainty values of WISE image
        :param mjd_obs: numpy.array
            mjd_obs timestamp values corresponding to current WISE image (all pixels have the same timestamp) in an
            array the same size as ints.
        """
        data_grouped, uncs_grouped = self._groupby(inds, ints, uncs)
        numerator, denominator, t_numerator, t_denominator = zip(
            *np.array([self._calc_hp_pixel(data_grouped[i], uncs_grouped[i], mjd_obs)
                       if len(data_grouped[i]) > 0 else (0, 0, 0, 0)
                       for i in range(len(data_grouped))
                       ]))
        self.numerator_cumul[:len(numerator)] += numerator
        self.denominator_cumul[:len(denominator)] += denominator
        self.time_numerator_cumul[:len(t_numerator)] += t_numerator
        self.time_denominator_cumul[:len(t_denominator)] += t_denominator

        return

    @staticmethod
    def _groupby(inds, data, uncs):
        """
        There are typically N WISE image pixels mapping to a single Healpix pixel (N is generally large as the
        Healpix resolution is coarser than WISE). This method groups the pixels in the input WISE image into groups
        associated with individual Healpix pixels.

        Parameters
        ----------
        :param inds: numpy.array
            Indices of Healpix pixels to which the WISE image maps
        :param data: numpy.array
            Intensity values of WISE image
        :param uncs: numpy.array
            Uncertainty values of WISE image
        :return: numpy.array of numpy.arrays
            data_out, uncs_out - each numpy.array contains values associated with a single Healpix pixel
        """
        # Get argsort indices, to be used to sort arrays in the next steps
        sidx = inds.argsort()
        data_sorted = data[sidx]
        inds_sorted = inds[sidx]
        uncs_sorted = uncs[sidx]

        # Get the group limit indices (start, stop of groups)
        cut_idx = np.flatnonzero(np.r_[True, inds_sorted[1:] != inds_sorted[:-1], True])

        # Create cut indices for all unique IDs in b
        n = inds_sorted[-1]+2
        cut_idxe = np.full(n, cut_idx[-1], dtype=int)

        insert_idx = inds_sorted[cut_idx[:-1]]
        cut_idxe[insert_idx] = cut_idx[:-1]
        cut_idxe = np.minimum.accumulate(cut_idxe[::-1])[::-1]

        # Split input array with those start, stop ones
        data_out = np.array([data_sorted[i:j] for i, j in zip(cut_idxe[:-1], cut_idxe[1:])])
        uncs_out = np.array([uncs_sorted[i:j] for i, j in zip(cut_idxe[:-1], cut_idxe[1:])])

        return data_out, uncs_out

    def _load_image(self, filename):
        """
        Load data from WISE image

        Parameters
        ----------
        :param filename: str
            Name of WISE image file, including full path
        """
        # Function to get image and coordinate data from File and map to the Healpix grid
        self.data_loader = WISEDataLoader(filename)
        self.data_loader.load_data()
        self.data_loader.load_coords()
        return

    def _place_image(self, mjd_obs):
        """
        Place WISE image on Healpix grid

        Parameters
        ----------
        :param mjd_obs: numpy.array
            Array of the same size as the input image, with every pixel equal to the same timestamp
        """
        int_data = self.data_loader.int_data.compressed()
        unc_data = self.data_loader.unc_data.compressed()
        coords = self.data_loader.wcs_coords
        ra, dec = coords.T
        hp_inds = self.fsm.wcs2ind(ra, dec)
        self._fill_map(hp_inds, int_data, unc_data, mjd_obs)
        del self.data_loader

        return

    def _save_csv(self):
        """
        Save orbit data to csv file. The columns are
        (numeric_index, Healpix_index, pixel_intensity_value, pixel_uncertainty_value, pixel_mjd_obs)
        """
        pixel_inds = np.arange(len(self.fsm.mapdata), dtype=int)
        nonzero_inds = self.unc_fsm.mapdata != 0.0

        data_to_save = {"hp_pixel_index": pixel_inds[nonzero_inds],
                        "pixel_value": self.fsm.mapdata[nonzero_inds],
                        "pixel_unc": self.unc_fsm.mapdata[nonzero_inds],
                        "pixel_mjd_obs": self.fsm.timedata[nonzero_inds]}

        dataframe = pd.DataFrame(data=data_to_save)
        dataframe.to_csv(os.path.join(self.path, self.orbit_csv_name))

    def _save_fits(self):
        """Save orbit data in a full sky Healpix map with most pixels zero-valued."""
        hp.fitsfunc.write_map(self.path + self.orbit_fits_name, self.fsm.mapdata,
                              coord='G', overwrite=True)
        hp.fitsfunc.write_map(self.path + self.orbit_unc_fits_name, self.unc_fsm.mapdata,
                              coord='G', overwrite=True)
