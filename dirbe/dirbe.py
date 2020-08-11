"""
Helper classes for mapping DIRBE data.
:author: Matthew Berkeley
"""

from astropy.io import fits
import sys
import numpy as np
import healpy as hp

class DIRBEFile:
    """
    Base class for all DIRBE file types.
    """

    skymap_info = None

    def __init__(self, filename, crd_sys, nside):
        self.filename = filename
        self.crd_sys = crd_sys
        self.nside = nside

    def inspect_file(self):
        "Print out all the header and data extensions."
        hdu = fits.open(self.filename)
        hdu_info = hdu.info()
        print("HDU info:\n", hdu_info)
        for i in range(len(hdu)):
            print("hdu{}".format(i))
            print(hdu[i].header)
            print(hdu[i].data)
        return

    @staticmethod
    def interpolate_gaps(data):
        "Fill in any gaps left after projecting from the quadcube to Healpix."
        nside = hp.npix2nside(len(data))
        zeros = data == 0.0
        px_nos = np.arange(hp.nside2npix(nside))

        while any(zeros):
            nbs = hp.pixelfunc.get_all_neighbours(nside, px_nos[zeros])
            data_nbs = data[nbs]
            data_nbs_masked = np.ma.masked_where(data_nbs == 0.0, data_nbs)
            interp_vals = np.mean(data_nbs_masked, axis=0, keepdims=True)
            data[zeros] = interp_vals.squeeze()
            zeros = data == 0.0

        return data

    def load_coords(self):
        "Load the appropriate coordinates from the SKYMAP_INFO file."
        dt = np.dtype('int,float,float,float,float,float,float')
        hdu = fits.open(self.skymap_info)
        coord_data = np.array(hdu[1].data, dtype=dt)
        coord_data.dtype.names = ['pix_num', 'elon', 'elat', 'ra', 'dec', 'glon', 'glat']
        if self.crd_sys == 'E':
            elon = coord_data['elon']
            elat = coord_data['elat']
            return elon, elat
        elif self.crd_sys == 'C':
            ra = coord_data['ra']
            dec = coord_data['dec']
            return ra, dec
        elif self.crd_sys == 'G':
            glon = coord_data['glon']
            glat = coord_data['glat']
            return glon, glat

    def create_map(self):
        """
        Create a Healpix map array after mapping the quadcube data to Healpix format.
        :return skymap: array
        """

        map_data = self.read_file()

        glon, glat = self.load_coords()
        pxs = hp.ang2pix(self.nside, glon, glat, nest=True, lonlat=True)

        skymap = np.bincount(pxs, weights=map_data)
        count = np.bincount(pxs)
        skymap = np.divide(skymap, count, where=count != 0.0, out=np.zeros_like(skymap))
        ind_order = np.arange(hp.nside2npix(self.nside))
        ind_neworder = hp.ring2nest(self.nside, ind_order)
        skymap = skymap[ind_neworder]
        skymap[np.isnan(skymap)] = 0.0
        skymap = self.interpolate_gaps(skymap)

        return skymap

    def downgrade_map(self, nside_out):
        """
        Change the resolution of the map.
        :param nside_out: int
            specified output resolution
        :return:
        """
        skymap_in = hp.fitsfunc.read_map(self.filename)
        data_downgrade = hp.ud_grade(skymap_in, nside_out)
        hp.fitsfunc.write_map(filename.replace('.fits', '_nside{}.fits'.format(nside_out)), data_downgrade, coord=self.crd_sys,
                              overwrite=True)
        return

    def save_map(self, data, output_filename):
        hp.fitsfunc.write_map(output_filename, data, nest=False, coord=self.crd_sys, overwrite=True)


class AAM_File(DIRBEFile):
    """Subclass allowing a read-in of AAM file data."""

    def __init__(self, filename, crd_sys="G", nside=128):
        super().__init__(filename, crd_sys, nside)

    def read_file(self):
        hdu = fits.open(self.filename)
        aam_data = np.array(hdu[1].data)
        aam_data.dtype.names = ['Pixel_no', 'PSubPos', 'Time', 'Photomet', 'StdDev', 'WtNumObs', 'SumNRecs']
        return aam_data["Photomet"].T


class ZSMA_File(DIRBEFile):
    """Subclass allowing a read-in of ZSMA file data."""

    def __init__(self, filename, crd_sys="G", nside=128):
        super().__init__(filename, crd_sys, nside)

    def read_file(self):
        filedata = fits.getdata(self.filename)
        dt = np.dtype('int,float,int,int,float')
        zsma_data = np.array(filedata, dtype=dt)
        zsma_data.dtype.names = ['pix_num', 'resid', 'num_obs', 'num_wks', 'std']
        return zsma_data["resid"]


if __name__ == "__main__":

    """Usage: python dirbe.py <DIRBE_filename> DIRBE_SKYMAP_INFO.fits"""

    filename = sys.argv[1]
    skymap_info = sys.argv[2]  # 'DIRBE_SKYMAP_INFO.fits'
    setattr(DIRBEFile, "skymap_info", skymap_info)

    if "ZSMA" in filename:
        file_obj = ZSMA_File(filename)
    elif "AAM" in filename:
        file_obj = AAM_File(filename)
    else:
        raise SystemExit("Filename should contain either 'AAM' or 'ZSMA'.")

    map_data = file_obj.create_map()
    file_obj.save_map(map_data, filename.replace(".fits", "_healpix.fits"))
