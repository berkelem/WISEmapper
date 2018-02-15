# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:05:33 2016

@author: Matthew Berkeley
"""

import numpy as np
import numpy.ma as ma
import os
import sys
import math
import time

from astropy.io import fits
from astropy import wcs
import cPickle
import healpy as hp

try:
    from mpi4py import MPI
    mpi_available = True
except ImportError:
    mpi_available = False


class File:


    def __init__(self, file_name, read_header=True):
        """Create a new file instance.
        
        Attributes
        ----------
        
        .name : str
            File name \n
        .header : dict
            File header \n
        .maskname : str
            Mask file name \n
        .mask : ndarray
            Mask data \n
        .uncname : str
            Uncertainty file name \n
        .data : ndarray
            File data \n
        .unc : ndarray
            Uncertainty data
        """
        
        self.name = file_name
        self.basename = self.name.split("/")[-1]
        self.maskname = self.name.replace('int', 'msk')
        self.uncname = self.name.replace('int', 'unc')
        if read_header:
            self.header = fits.open(self.name)[0].header
            self.header.rename_keyword('RADECSYS', 'RADESYS')
        #self.maskname = None
        self.mask = None
        #self.uncname = None
        self.data = None
        self.unc = None


    def wcs2px(self):
        """Read file header and return a 1-dim array of wcs pixel coordinates.
        """
        x_dim = self.header['NAXIS1']
        y_dim = self.header['NAXIS2']
        coord_array = [(x, y) for y in xrange(y_dim) for x in xrange(x_dim)]
        wcs_file = wcs.WCS(self.header).wcs_pix2world(coord_array, 0,
                                                      ra_dec_order=True)
        return wcs_file

    def calibrate(self, bandcheck):
        """Calibrate data using the magnitude zero point in the header.
        
        The conversion factor for each WISE frequency band is different, 
        so a band check is included to avoid accidental error.
        The conversion factors can be found in the WISE supplement at the 
        following link:
        http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/figures/sec4_4ht9.gif
        """
        if self.header['BUNIT'].strip() != 'DN':
            raise TypeError("Wrong BUNIT %s" % self.header['BUNIT'])
        band = self.header['BAND']
        if int(band) != int(bandcheck):
            raise ValueError("Wrong band!")
        try:
            self.mask = fits.getdata(self.maskname).astype(bool).flatten()
        except IOError:
            self.mask = fits.getdata(self.maskname+'.gz').astype(bool).flatten()
            
        try:
            dn_unc = fits.getdata(self.uncname).astype(float).flatten()
        except IOError:
            dn_unc = fits.getdata(self.uncname+'.gz').astype(float).flatten()
        
        
        dn_data = fits.getdata(self.name).astype(float).flatten()
        neg_mask = dn_data < 0.0
        nan_mask = np.isnan(dn_data)
        nan_mask_unc = np.isnan(dn_unc)
        self.mask += neg_mask + nan_mask + nan_mask_unc
        self.data = ma.array(dn_data, mask=self.mask)        
        
        
        self.unc = ma.array(dn_unc, mask=self.mask)
        
        magzp = self.header['MAGZP']
        

        dn_to_jy = {
            1:306.682,#1.9350E-06,
            2:170.663,#2.7048E-06,
            3:29.0448,#1.8326e-06,
            4:8.2839#5.2269E-05,
            }
        
        if band not in dn_to_jy:
            raise ValueError("WISE only has 4 bands!  This file has band=%s" % band)
            
        #beam_area = {
        #    1:  6.08   * 5.60 * np.pi / (8*np.log(2)) / 206265.**2,
        #    2:  6.84   * 6.12 * np.pi / (8*np.log(2)) / 206265.**2,
        #    3:  7.36   * 6.08 * np.pi / (8*np.log(2)) / 206265.**2,
        #    4:  11.99  * 11.65* np.pi / (8*np.log(2)) / 206265.**2,
        #    }
        
        self.data *= dn_to_jy[band] *10**(-0.4*magzp) #* 1e-6 / beam_area[band]
        self.unc *= dn_to_jy[band] *10**(-0.4*magzp) #* 1e-6 / beam_area[band]
        
        return


class Mosaic:


    def __init__(self, filelist, deg_ppx, ra_span, dec_span,
                 lowerleft_ra, lowerleft_dec, path, region,
                 band, parallel, adjust):
        """Create instance of a mosaic.
        
        Attributes
        ----------
        
        .filelist : list of strings
            List of files to be mosaicked \n
        .ra_span : float
            Span of mosaic in right ascension \n
        .dec_span : float
            Span of mosaic in declination \n
        .deg_ppx : float
            Pixel resolution \n
        .x_ax : int
            Number of pixels along x-axis \n
        .y_ax : int
            Number of pixels along y-axis \n
        .pixels : int
            Total number of pixels in mosaic \n
        .llra : float
            Right ascension value in lower left corner of mosaic \n
        .lldec : float
            Declination value in lower left corner of mosaic \n
        .path : str
            Path to files to be mosaicked \n
        .region : str
            Region of the sky, denoted as RA_DEC \n
        .band : int (1,2,3,4)
            WISE frequency band \n
        .intensity : ndarray of floats
            Mosaic data \n
        .unc : ndarray of floats
            Mosaic uncertainty \n
        .count : ndarray of ints
            Count of nhits for mosaic pixels \n
        .tile_positions : ndarray of strings
            Array of temporary '*pos*' files for files in mosaic \n
        .image_values : ndarray of strings
            Array of temporary '*val*' files for files in mosaic \n
        .uncertainties : ndarray of strings
            Array of temporary '*unc*' files for files in mosaic \n
        .center_coords : ndarray of tuples
            Array of coordinates of the central pixels of each file \n
        .possible_overlaps : dict
            Dictionary of indices to keep track of the neighboring files 
            of each file \n
        .parallel : bool
            If True, the code will run in parallel \n
        .comm :
            If parallel == True, this calls MPI COMM_WORLD \n
        .size : int
            If parallel == True, this stores the number of available cores \n
        .rank : int
            If parallel == True, this stores the rank of each processor \n
        """
        self.filelist = filelist
        self.total_filelist = np.copy(self.filelist)
        #self.msklist = [s.replace('int', 'msk') for s in filelist]
        #self.unclist = [s.replace('int', 'unc') for s in filelist]
        self.ra_span = ra_span
        self.dec_span = dec_span
        self.deg_ppx = deg_ppx
        self.x_ax = int(np.ceil(ra_span/deg_ppx))
        self.y_ax = int(np.ceil(dec_span/deg_ppx))
        self.pixels = self.x_ax*self.y_ax
        self.llra = lowerleft_ra
        self.lldec = lowerleft_dec
        self.path = path
        self.region = region
        self.band = band
        self.intensity = np.zeros(self.pixels, dtype=float)
        self.unc = np.zeros(self.pixels, dtype=float)
        self.count = np.zeros(self.pixels, dtype=int)
        #self.tile_positions = []
        self.image_values = []
        self.uncertainties = []
        self.center_coords = []
        self.possible_overlaps = {}
        self.adjust = adjust
        if mpi_available:
            self.parallel = parallel
            if self.parallel == True:
                self.comm = MPI.COMM_WORLD
                self.size = self.comm.Get_size()
                self.rank = self.comm.Get_rank()
        else:
            self.parallel = False

    def position_tile(self,File):
        """Place file on mosaic grid.
        
        Three temporary files are created: \n
        1) A boolean stamp of the mosaic pixels covered by the input file \n
        2) A full mosaic-sized grid with just the file data values included \n
        3) A full mosaic-sized grid with just the file uncertainty values
        included
        """
        #File.image_pos = np.zeros(self.pixels, dtype=bool)
        File.image_val = np.zeros(self.pixels, dtype=float)
        File.image_unc = np.zeros(self.pixels, dtype=float)
        File.calibrate(self.band)
        wcs_file = File.wcs2px()[~File.data.mask]
        File.data = File.data.compressed()
        File.unc = np.zeros_like(File.data)#File.unc.compressed()
        for i in xrange(len(wcs_file)):
            ra,dec = wcs_file[i]
            if (self.llra <= ra < (self.llra + self.ra_span) and
                    self.lldec <= dec < (self.lldec + self.dec_span)):
                file_ra_bin = int((ra-self.llra)/self.deg_ppx)
                file_dec_bin = int((dec-self.lldec)/self.deg_ppx)
                mosaic_pixel = file_dec_bin*self.x_ax + file_ra_bin
                #File.image_pos[mosaic_pixel] = True
                File.image_val[mosaic_pixel] = File.data[i]
                File.image_unc[mosaic_pixel] = File.unc[i]
        return File.image_val, File.image_unc

    def pickle(self):
        """Save the positioned files as binary Pickle files."""
        if self.parallel:
            self.index_sublist = range(len(self.filelist))[self.rank::self.size]
            self.filelist = [self.total_filelist[i] for i in self.index_sublist]#[operator.itemgetter(*self.index_sublist)(self.filelist)]
            #if self.rank == 0:
            #    file_sublist = [
            #        self.filelist[i::self.size] for i in xrange(self.size)
            #        ]
            #else:
            #    file_sublist = None
            #file_sublist = self.comm.scatter(file_sublist, root=0)
            #self.filelist = file_sublist
        else:
            self.index_sublist = range(len(self.filelist))
            self.filelist = self.total_filelist
        if not self.parallel or (self.parallel and self.rank == 0):
            pickle_start = time.clock()
            print 'Placing files on mosaic grid ...'
        for i in xrange(len(self.filelist)):
            #print 'file', self.filelist[i]
            f = File(self.filelist[i])
            #if not os.path.isfile(f.maskname) or not os.path.isfile(f.uncname):
            #    self.image_values.append(None)
            #    self.uncertainties.append(None)
            #    self.center_coords.append(None)
            #    continue
            f.crvals = (f.header['CRVAL1'],f.header['CRVAL2'])
            #pos_db_name = (self.path + f.name[len(self.path):len(self.path)+13]
            #                         + 'pos-1b.pkl')
            val_db_name = self.path + (f.basename.replace('int', 'val')).replace('fits', 'pkl')
            unc_db_name = self.path + (f.basename.replace('int', 'unc')).replace('fits', 'pkl')
            if not os.path.isfile(val_db_name):
                image_val, image_unc = self.position_tile(f)
                #pos_db = open(pos_db_name, 'wb')
                #cPickle.dump(image_pos, pos_db, cPickle.HIGHEST_PROTOCOL)
                #pos_db.close()
                val_db = open(val_db_name, 'wb')
                cPickle.dump(image_val, val_db, cPickle.HIGHEST_PROTOCOL)
                val_db.close()
                unc_db = open(unc_db_name, 'wb')
                cPickle.dump(image_unc, unc_db, cPickle.HIGHEST_PROTOCOL)
                unc_db.close()
            #self.tile_positions.append(pos_db_name)
            self.image_values.append(val_db_name)
            self.uncertainties.append(unc_db_name)
            self.center_coords.append(f.crvals)
            if not self.parallel or (self.parallel and self.rank == 0):
                print_progress(i+1, len(self.filelist))
        if not self.parallel or (self.parallel and self.rank == 0):
            pickle_end = time.clock()
            print 'Files positioned.'
            print 'Time: ' + str(pickle_end - pickle_start)

    def calc_offsets(self):
        """Adjust each input file in order to smooth the final mosaic.
        
        Perform a least-squares minimization calculation on the differences 
        between the overlapping regions of adjacent files.
        """
        Vlist = []
        differences = []
        del_diffs = []
        n_vals = []
        total_nn = self.find_neighbors()
        if self.parallel:
            self.comm.barrier()
            if self.rank == 0:
                print ('Approx ' 
                    + str(total_nn*self.size) 
                    + ' possible overlaps to consider.')
        else:
            print str(total_nn) + ' possible overlaps.'
        overlap_generator = self.generate_overlaps()
        if not self.parallel or (self.parallel and self.rank == 0):
            print 'Determining overlap offsets...'
            overlap_start = time.clock()
        for index in xrange(total_nn):
            overlap_region,n,m = next(overlap_generator)
            if not self.parallel or (self.parallel and self.rank == 0):
                if total_nn > 1:
                    print_progress(index, total_nn-1)
            if not any(overlap_region):
                continue
            with open(self.image_values[n], 'rb') as tile1:
                image_vals1 = cPickle.load(tile1)
            with open(self.image_values[m], 'rb') as tile2:
                image_vals2 = cPickle.load(tile2)
            
            with open(self.uncertainties[n], 'rb') as tile1unc:
                unc_vals1 = cPickle.load(tile1unc)
            with open(self.uncertainties[m], 'rb') as tile2unc:
                unc_vals2 = cPickle.load(tile2unc)
            
            sigma1 = unc_vals1[overlap_region]
            #for p in xrange(len(sigma1)):
            #    if np.isnan(sigma1[p]):
            #        print 'sigma1 NaN', p
            sigma2 = unc_vals2[overlap_region]
            #for p in xrange(len(sigma2)):
            #    if np.isnan(sigma2[p]):
            #        print 'sigma2 NaN', p
            sigma12 = [np.sqrt(sigma1[i]**2 + sigma2[i]**2) for i in xrange(len(unc_vals1[overlap_region]))]
            #for l in xrange(len(sigma12)):
            #    if np.isnan(sigma12[l]):
            #        print 'sigma12 NaN', l
            n_val = np.sqrt(1./np.sum([1/sigma12[i]**2 for i in xrange(len(sigma12))]))
            
            # In the following lines, the background value is obtained by 
            # creating a histogram of pixel values. This prevents the 
            # background value being accidentally skewed by bright stars.
            
            val1,bins1 = np.histogram(image_vals1[overlap_region], bins=500)
            val2,bins2 = np.histogram(image_vals2[overlap_region], bins=500)
            std_err1 = round(np.std(image_vals1[overlap_region],
                                    dtype=np.float64) /
                                    len(image_vals1[overlap_region]),2)
            std_err2 = round(np.std(image_vals2[overlap_region],
                                    dtype=np.float64) /
                                    len(image_vals2[overlap_region]),2)
            #n_val = np.var(image_vals1[overlap_region].flatten()) + np.var(image_vals2[overlap_region].flatten()) - np.cov([image_vals1[overlap_region], image_vals2[overlap_region]], ddof=0)
            #cov = np.cov([image_vals1[overlap_region], image_vals2[overlap_region]], ddof=0)
            #if cov[0,0] == 0.0:
            #    continue
            #try:
            #    assert(cov[0,1] == cov[1,0])
            #    n_val = cov[0,0] + cov[1,1] - 2*cov[0,1]
            #except AssertionError:
            #    print cov
            #    raise(ValueError('Covariance matrix not symmetric...'))
            max_index1 = np.where(val1 == max(val1))[0][0]
            background1 = bins1[max_index1]
            max_index2 = np.where(val2 == max(val2))[0][0]
            background2 = bins2[max_index2]
            diff = background1 - background2
            del_diff = np.sqrt(std_err1**2 + std_err2**2)
            
            V = np.zeros(len(self.total_filelist))
            V[n] = V[n] + 1.
            V[m] = V[m] -1.
            Vlist.append(V)
            differences.append(diff)
            del_diffs.append(del_diff**2)
            #if np.isnan(n_val):
            #    print 'NaN found: ', self.rank, n,m
            n_vals.append(n_val)
        if not self.parallel or (self.parallel and self.rank == 0):
            overlap_end = time.clock()
            print 'Time: ' + str(overlap_end - overlap_start)
        if self.parallel:
            self.comm.barrier()
            if len(Vlist) == 0:
                Vlist = None
                differences = None
            elements_to_send = (Vlist,differences, del_diffs, self.index_sublist, n_vals)
            elements_received = self.comm.gather(elements_to_send, root=0)
            if self.rank == 0:
                print 'Gathering all data and calculating least squares offset solution...'
                all_Vlist = []
                all_differences = []
                all_deldiffs = []
                all_nvals = []
                index_order = []
                for array in elements_received:
                    # Unpack received data package into lists
                    if array[0] is not None:
                        all_Vlist = all_Vlist + array[0]
                        all_differences = all_differences + array[1]
                        all_deldiffs = all_deldiffs + array[2]
                        index_order = index_order + array[3]
                        all_nvals = all_nvals + array[4]
                #inv_order = [x for y,x in sorted(zip(index_order, range(len(index_order))))]
                #all_differences = [all_differences[i] for i in inv_order]
                #all_deldiffs = [all_deldiffs[i] for i in inv_order]
                #all_Vlist = [all_Vlist[i] for i in inv_order]
                #print 'reordered', len(all_Vlist), len(all_deldiffs), len(all_nvals)
                #all_Vlist = np.ndarray(all_Vlist).flatten()
                #print 'len(all_Vlist)', len(all_Vlist)
                all_Vlist = filter(lambda x: x is not None, all_Vlist)

                
                # The following lines contain the least squares minimization
                # calculation. Formula adapted from Tegmark (1996).
                A = np.vstack(all_Vlist)
                all_ninvvals = [1./all_nvals[i] for i in xrange(len(all_nvals))]
                #N = np.diag(all_nvals)#np.identity(A.shape[0])
                Ninv = np.diag(all_ninvvals)#np.linalg.inv(N)
                #print 'N', N
                NinvA = np.dot(Ninv, A)
                
                epsilon = np.identity(A.shape[1])*0.000001
                W = np.linalg.inv(
                        np.dot(A.T, NinvA)
                        + epsilon
                        ).dot(
                              np.dot(A.T,Ninv)
                              )
                offsets = np.dot(W,np.array(all_differences))
                del_offsets = np.dot(W,np.array(all_deldiffs))
                #offsets = [offsets[i] for i in inv_order]
                #del_offsets = [del_offsets[i] for i in inv_order]
            else:
                offsets = del_offsets = None#np.empty_like(len(Vlist),
            #                                          dtype=float)
            #    print self.rank, 'len(offsets)', len(offsets)
            offset_data = [offsets,del_offsets]
            self.comm.barrier()
            offsets, del_offsets = self.comm.bcast(offset_data, root=0)
        elif len(Vlist) == 0:
            offsets = del_offsets = np.zeros(len(self.total_filelist), dtype=float)
        else:
            # The following lines contain the least squares minimization
            # calculation. Formula adapted from Tegmark (1996).
            print 'Calculating least squares offset solution...'
            A = np.vstack(Vlist)
            N = np.identity(A.shape[0])
            epsilon = np.identity(A.shape[1])*0.000001
            W = np.linalg.inv(
                    np.dot(A.T, np.dot(np.linalg.inv(N),A))
                    + epsilon
                    ).dot(
                          np.dot(A.T,np.linalg.inv(N))
                          )
            offsets = np.dot(W,np.array(differences))
            del_offsets = np.dot(W,np.array(del_diffs))
        return offsets,del_offsets

    def find_neighbors(self):
        """Index files that are adjacent to each other.
        
        Using the center pixel coordinate, find all other files that are within
        1 degree of the current file, and hence may have a shared overlapping 
        region.
        """
        if not self.parallel or (self.parallel and self.rank == 0):
            print 'Filtering neighboring files...'
        length = len(self.total_filelist)
        total_nn = 0
        for i in self.index_sublist:
            #if self.center_coords[i] is None:
            #    continue
            neighbours = []
            ra1,dec1 = self.center_coords[i]
            for j in xrange(i+1,length):
                #print j, length
                #print len(self.center_coords)
                #print self.center_coords[j]
                #if self.center_coords[j] is None:
                #    print 'None!'
                #    continue
                ra2,dec2 = self.center_coords[j]
                angular_dist =  math.acos(
                                math.cos(math.radians(90-dec1))*
                                math.cos(math.radians(90-dec2))+
                                math.sin(math.radians(90-dec1))*
                                math.sin(math.radians(90-dec2))*
                                math.cos(math.radians(ra1-ra2))
                                )
                if angular_dist < 1.:
                    neighbours.append(j)
            self.possible_overlaps[str(i)] = neighbours
            total_nn += len(neighbours)
        return total_nn

    def generate_overlaps(self):
        """Cycle through the adjacent files to generate the set of overlapping 
        pixels for each pair of neighboring files.
        """
        index = 0
        while True:
            ind1 = self.index_sublist[index]
            if self.image_values[ind1] is None:
                index += 1
                continue
            file1 = open(self.image_values[ind1], 'rb')
            file1_data = cPickle.load(file1).astype(bool)
            neighbours = self.possible_overlaps[str(ind1)]
            for ind2 in neighbours:
                if self.image_values[ind2] is None:
                    continue
                file2 = open(self.image_values[ind2], 'rb')
                file2_data = cPickle.load(file2).astype(bool)
                overlap = file1_data & file2_data
                file2.close()
                yield overlap,ind1,ind2
            file1.close()
            index += 1

    def atlas_offset(self, tile_pos_data, tile_val_data):
        offset_reg = self.intensity.astype(bool) & tile_pos_data
        med1 = np.median(self.intensity[offset_reg])
        med2 = np.median(tile_val_data[offset_reg])
        offset = med1 - med2
        return offset
        

    def build_mosaic(self):
        """Apply calculated offsets to smooth the final mosaic.
        
        All the temporary data files are added together along with their 
        offsets and divided by the overall nhits to create the final mosaic.
        """
        if self.adjust:
            offsets, del_offsets = self.calc_offsets()
            print 'offsets', offsets
        else:
            offsets = del_offsets = np.zeros(len(self.total_filelist))
        if not self.parallel or (self.parallel and self.rank == 0):
            print 'Compiling final mosaic...'
        #print 'rank', self.rank, len(index_sublist)
        #print 'rank', self.rank, index_sublist[0], index_sublist[len(index_sublist)-1]
        mins = np.empty(self.intensity.shape)*(np.nan)
        maxs = np.empty(self.intensity.shape)*(np.nan)
        print mins.shape
        for i in self.index_sublist:
            if self.image_values[i] is None:
                continue
            offset = offsets[i]
            d_offset = del_offsets[i]
            tile_val = open(self.image_values[i], 'rb')
            tile_val_data = cPickle.load(tile_val)
            #tile_pos = open(self.tile_positions[i], 'rb')
            tile_pos_data = (tile_val_data.astype(bool)).astype(int)#cPickle.load(tile_pos)
            tile_unc = open(self.uncertainties[i],'rb')
            tile_unc_data = cPickle.load(tile_unc)
            #atlas_offset = self.atlas_offset(tile_pos_data, tile_val_data)
            print len(mins[tile_pos_data.astype(bool)])
            print len(tile_val_data[tile_pos_data.astype(bool)])
            mins[tile_pos_data.astype(bool)] = np.nanmin(np.array([mins[tile_pos_data.astype(bool)], tile_val_data[tile_pos_data.astype(bool)]]), axis=0)
            maxs[tile_pos_data.astype(bool)] = np.nanmax(np.array([maxs[tile_pos_data.astype(bool)], tile_val_data[tile_pos_data.astype(bool)]]), axis=0)
            self.intensity += tile_val_data - (offset*tile_pos_data)# + atlas_offset
            self.unc = np.sqrt(self.unc**2 + tile_unc_data**2 
                                           + (d_offset*tile_pos_data)**2)
            self.count += tile_pos_data
            #tile_pos.close()
            tile_val.close()
            tile_unc.close()
        if self.parallel:
            if self.rank != 0:
                self.comm.Isend([self.intensity, MPI.FLOAT],
                                dest=0, tag=77)
                self.comm.Isend([self.unc, MPI.FLOAT],
                                dest=0, tag=78)
            elif self.rank == 0:
                for proc in xrange(1,self.size):
                    part_intensity = np.empty(len(self.intensity),
                                              dtype=float)
                    part_uncertainty = np.empty(len(self.intensity),
                                                dtype=float)
                    self.comm.Recv([part_intensity, MPI.FLOAT],
                                   source=proc, tag=77)
                    self.comm.Recv([part_uncertainty, MPI.FLOAT],
                                   source=proc, tag=78)
                    # Blocking receive is necessary to avoid a SegFault.
                    self.intensity += part_intensity
                    #print 'nonzero_pixels after proc', proc, len(self.intensity[np.where(self.intensity!=0)])
                    self.unc = np.sqrt(self.unc**2 + part_uncertainty**2)

            if self.rank != 0:
                self.comm.Isend([self.count, MPI.INT], dest=0, tag=135)
                full_mosaic, full_unc = None, None
            elif self.rank == 0:
                for proc in xrange(1,self.size):
                    part_count = np.empty(len(self.count), dtype=int)
                    self.comm.Recv([part_count, MPI.INT],
                                   source=proc, tag=135)
                    self.count += part_count

                full_mosaic, full_unc, full_count = (self.intensity,
                                                     self.unc, self.count)
                multihits = full_count > 1
                full_mosaic[multihits] = (full_mosaic[multihits] /
                                          full_count[multihits])
                full_unc[multihits] = (full_unc[multihits] /
                                       full_count[multihits])
                full_mosaic = full_mosaic.reshape(self.y_ax,self.x_ax)
                full_unc = full_unc.reshape(self.y_ax,self.x_ax)
            self.comm.barrier()    # Necessary to avoid SegFault.
        else:
            print 'mins', mins[int(len(mins)/2.):int(len(mins)/2.)+1000]
            print 'maxs', maxs[int(len(mins)/2.):int(len(mins)/2.)+1000]
            avgs = np.nanmean(np.array([mins, maxs]), axis=0)
            print 'avgs', avgs[int(len(mins)/2.):int(len(mins)/2.)+1000]
            nans = np.isnan(avgs)
            avgs[nans] = 0.0
            print 'avgs', avgs[int(len(mins)/2.):int(len(mins)/2.)+1000]
            self.intensity -= maxs
            self.count[~nans] -= 1
            
            multihits = self.count > 1
            self.intensity[multihits] = (self.intensity[multihits] /
                                         self.count[multihits])
            self.unc[multihits] = (self.unc[multihits] /
                                   self.count[multihits])
            full_mosaic = self.intensity.reshape(self.y_ax,self.x_ax)
            full_unc = self.unc.reshape(self.y_ax,self.x_ax)
        if not self.parallel or (self.parallel and self.rank == 0):
            print 'Full mosaic created.'
        return full_mosaic, full_unc
        
        
        
class FullSkyMap():
    
    def __init__(self, NSIDE, file_list, parallel, band):
        self.nside = NSIDE
        self.npix = hp.nside2npix(self.nside)
        self.skymap = np.zeros(self.npix)
        self.count = np.zeros_like(self.skymap, dtype=int)
        self.uncmap = np.zeros_like(self.skymap, dtype=float)
        self.contd_map = np.zeros_like(self.skymap)
        self.contd_count = np.zeros_like(self.skymap, dtype=int)
        self.contd_uncmap = np.zeros_like(self.skymap, dtype=float)
        self.parallel = parallel
        self.band = int(band)
        if mpi_available and self.parallel:
            self.comm = MPI.COMM_WORLD
            self.size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
            if self.rank == 0:
                file_sublist = [
                    file_list[i::self.size] for i in xrange(self.size)
                    ]
            else:
                file_sublist = None
            file_sublist = self.comm.scatter(file_sublist, root=0)
            self.filelist = file_sublist
            self.index_sublist = range(len(file_list))[self.rank::self.size]
        else:
            self.parallel = False
            self.filelist = file_list
            self.index_sublist = range(len(self.filelist))
        
    def ind2wcs(self, index):
        theta,phi=hp.pixelfunc.pix2ang(self.nside,index)
        return np.degrees(phi),np.degrees(np.pi/2.-theta)

    def wcs2ind(self, ra_arr, dec_arr, nest=False):
        theta = [np.radians(-float(dec)+90.) for dec in dec_arr]
        phi = [np.radians(float(ra)) for ra in ra_arr]
        return hp.pixelfunc.ang2pix(self.nside, theta, phi, nest=nest)
        
    def mapfile(self, file_name, badfile=False):
        if badfile:
            orig_name = file_name
            badfile_name = '/mnt/wise/wise4/badfiles/'+str(file_name.split("/")[-1])
            f = File(badfile_name)
            f.maskname = orig_name.replace('int', 'msk')
        else:
            f = File(file_name)
        f.calibrate(self.band)
        wcs_file = f.wcs2px()[~f.data.mask]
        f.data = f.data.compressed()
        f.unc = f.unc.compressed()
        if any(np.isnan(f.unc)):
            print 'Problem detected with {}'.format(file_name.replace('int', 'unc'))
        
        sig_sq = np.array([(f.unc[i]**2) for i in xrange(len(f.unc))])
        
        ra, dec = wcs_file.T
        hp_inds = self.wcs2ind(ra,dec)
        #sig_sq_hp = np.bincount(hp_inds, weights=sig_sq)        
        #data_hp = np.bincount(hp_inds, weights=f.data)
        var_inv = np.array([1/val for val in sig_sq])
        data_adj = f.data / (f.unc**2)
        data_hp = np.bincount(hp_inds, weights=(data_adj))
        sig_sq_inv = np.bincount(hp_inds, weights=var_inv)
        posvals = data_hp != 0.0

        data_hp[posvals] = data_hp[posvals] / sig_sq_inv[posvals]
        sig_hp = np.zeros_like(data_hp)
        sig_hp[posvals] = np.array([1./np.sqrt(val) for val in sig_sq_inv[posvals]])
        self.skymap[:len(data_hp)] += data_hp
        self.uncmap[:len(sig_hp)] += sig_hp
        self.count[:len(data_hp)] += posvals.astype(int)
        return 
        
    def build_map(self, map_name):
        skip_file = 0
        use_file = 0
        for i in xrange(len(self.filelist)):
            if self.parallel:
                if self.rank == 0:
                    print_progress(i+1,len(self.filelist),used=use_file,skipped=skip_file)
            else:
                print_progress(i+1,len(self.filelist),used=use_file,skipped=skip_file)
            file_name = str(self.filelist[i])
            try:
                self.mapfile(file_name)
                use_file += 1
            except IOError:
                try:
                    self.mapfile(file_name, badfile=True)
                    use_file += 1
                except IOError:
                    #    if not os.path.isfile(file_name) or not os.path.isfile(file_name.replace('int', 'msk')):
                    print file_name + ' not found.'
                    skip_file += 1
                    continue
            except TypeError:
                print('Problem with '+str(file_name))
            #if os.path.isfile(file_name):
            #    print 'File Ok', file_name
            #else:
            #    raise ValueError('File name problem: ' + file_name)
            
            #if self.parallel:
            #    if self.index_sublist[i] > save_num*self.size:
            #        print 'Saving progress: ', save_num, self.rank
            #        self.save_map()
            #        save_num += 1
            #    print 'build_map Ready ', save_num, self.rank
            #    self.comm.barrier()
            #else:
            #    if self.index_sublist[i] > save_num*100:
            #        print 'Saving progress: ', save_num
            #        self.save_map()
            #        save_num += 1
        use_file, skip_file = self.save_map(map_name, use_file, skip_file)
        if not self.parallel or (self.parallel and self.rank == 0):
            print 'Skipped ' + str(skip_file) + ' files.'
            print 'Used ' + str(use_file) + ' files.'
            print 'Map complete.'
        return

    def save_map(self, map_name, use_file, skip_file):                
        if self.parallel:
            if self.rank != 0:
                self.comm.Isend([self.skymap, MPI.FLOAT], dest=0, tag=77)
                self.comm.Isend([self.uncmap, MPI.FLOAT], dest=0, tag=84)
                self.comm.Isend([self.count, MPI.INT], dest=0, tag=135)
                self.comm.isend(use_file, dest=0, tag=104)
                self.comm.isend(skip_file, dest=0, tag=105)
            elif self.rank == 0:
                for proc in xrange(1,self.size):
                    part_map = np.empty(len(self.skymap), dtype=float)
                    part_count = np.empty(len(self.count), dtype=int)
                    part_unc = np.empty(len(self.uncmap), dtype=float)
                    #part_used = 0
                    #part_skip = 0
                    self.comm.Recv([part_map, MPI.FLOAT], source=proc, tag=77)
                    # Blocking receive is necessary to avoid a SegFault.
                    self.skymap += part_map
                    self.comm.Recv([part_unc, MPI.FLOAT], source=proc, tag=84)
                    self.uncmap += part_unc
                    self.comm.Recv([part_count, MPI.INT], source=proc, tag=135)
                    self.count += part_count
                    part_used = self.comm.recv(source=proc, tag=104)
                    part_skip = self.comm.recv(source=proc, tag=105)
                    use_file += int(part_used)
                    skip_file += int(part_skip)
                
                self.skymap += self.contd_map
                self.uncmap += self.contd_uncmap
                self.count += self.contd_count
                self.partmap = np.copy(self.skymap)
                self.partunc = np.copy(self.uncmap)
                self.partcount = np.copy(self.count)
                multihits = self.count > 1
                self.skymap[multihits] = (self.skymap[multihits] /
                                          self.count[multihits])
                self.uncmap = np.array([np.sqrt(val) for val in self.uncmap])
                hp.fitsfunc.write_map(map_name, self.skymap)
                hp.fitsfunc.write_map(map_name.replace('.fits', '_unc.fits'), self.uncmap)
            self.comm.barrier()
        else:
            self.skymap += self.contd_map
            self.count += self.contd_count
            multihits = self.count > 1
            self.partmap = np.copy(self.skymap)
            self.partunc = np.copy(self.uncmap)
            self.partcount = np.copy(self.count)
            self.skymap[multihits] = (self.skymap[multihits] /
                                      self.count.astype(float)[multihits])
            hp.fitsfunc.write_map(map_name, self.skymap)
            hp.fitsfunc.write_map(map_name.replace('.fits', '_unc.fits'), self.uncmap)    
        return use_file, skip_file
        
        
            
        
def print_progress(iteration, total, used=0, skipped=0, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar

    @params:
        iteration   - Required  : current iteration (Int) \n
        total       - Required  : total iterations (Int) \n
        prefix      - Optional  : prefix string (Str) \n
        suffix      - Optional  : suffix string (Str) \n
        decimals    - Optional  : positive number of decimals in percent complete (Int) \n
        bar_length  - Optional  : character length of bar (Int) \n
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    #used_files = 'Used: '+str(used) + ' '
    #skipped_files = 'Skipped: ' +str(skipped) + ' '

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix))#, used_files, skipped_files)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
'''    
if __name__ == '__main__':
    print 'Second file'
    files_list = sys.argv[1]
    with open(files_list,'rb') as f:
        split_lines = f.read().splitlines()
        file_list = [x for x in split_lines if 'int' in x]
    testmap = FullSkyMap(256, file_list[2:3], False, 1)
    with open('part_count_test.pkl', 'rb') as partcount:
        testmap.contd_count = cPickle.load(partcount)
    with open('part_map_test.pkl', 'rb') as partmap:
        testmap.contd_map = cPickle.load(partmap)
    testmap.build_map('testmap.fits')
    with open('part_map_test2.pkl', 'wb') as partmap:
        cPickle.dump(testmap.partmap, partmap, cPickle.HIGHEST_PROTOCOL)
    with open('part_count_test2.pkl', 'wb') as partcount:
        cPickle.dump(testmap.count, partcount, cPickle.HIGHEST_PROTOCOL)
        
'''
'''
    test_file = File(file_list[0])
    test_file.calibrate(1)
    img = test_file.data.reshape((1016,1016))
    test_wcs = test_file.wcs2px()
    print type(test_wcs)
    ra, dec = test_wcs.T
    ra_file = ra.reshape((1016,1016))
    dec_file = dec.reshape((1016,1016))
    healpxs = testmap.wcs2ind(ra, dec)
    wcs_coords = np.array(testmap.ind2wcs(healpxs))
    print wcs_coords
    ra_new, dec_new = wcs_coords
    ra_file_new = ra_new.reshape((1016,1016))
    dec_file_new = dec_new.reshape((1016,1016))
'''
'''    
    test_img = np.zeros((100,100), dtype=float)
    for i in xrange(100):
        for j in xrange(100):
            test_img[i,j] = np.mean(img[10*i:min(10*(i+1),1016),10*j:min(10*(j+1),1016)])
    cdelt1 = test_file.header['WCDELT1']
    cdelt2 = test_file.header['WCDELT2']
    w_hdr = wcs.WCS(naxis = 2)
    w_hdr.wcs.crpix = [50.5, 50.5]
    w_hdr.wcs.cdelt = np.array([cdelt1, cdelt2])
    w_hdr.wcs.crval = [test_file.header['CRVAL1'], test_file.header['CRVAL2']]
    w_hdr.wcs.ctype = [test_file.header['CTYPE1'], test_file.header['CTYPE2']]
    #test_file.header['CRPIX1'] = 50.5  
    #test_file.header['CRPIX2'] = 50.5    
    #test_file.header['WCDELT1'] = cdelt1*1016.#/100.
    #test_file.header['WCDELT2'] = cdelt2*1016.#/100.
'''
'''
    hdu1 = fits.PrimaryHDU(header = test_file.header)
    hdu1.data = img
    hdu1.writeto('test_img.fits', clobber=True)
    hdu2 = fits.PrimaryHDU()
    hdu2.data = ra_file
    hdu2.writeto('test_ra.fits', clobber=True)
    hdu3 = fits.PrimaryHDU()
    hdu3.data = dec_file
    hdu3.writeto('test_dec.fits', clobber=True)
    hdu4 = fits.PrimaryHDU()
    hdu4.data = ra_file_new
    hdu4.writeto('test_ra_new.fits', clobber=True)
    hdu5 = fits.PrimaryHDU()
    hdu5.data = dec_file_new
    hdu5.writeto('test_dec_new.fits', clobber=True)
'''    
    #print test_file.header['WCDELT1']
    #ra_arr = np.arange(5)*5# + np.ones(5)*0.1
    #dec_arr = np.arange(5)*5# + np.ones(5)*0.1
    #print zip(ra_arr, dec_arr)
    #pix_arr = testmap.wcs2ind(ra_arr, dec_arr, nest=True)
    #print pix_arr
    #print testmap.ind2wcs(1157802)
