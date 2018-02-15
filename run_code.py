# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 00:14:01 2016

@author: Matthew Berkeley
"""

import pah_mapping
import os
#import inpaint
import numpy as np
from astropy import wcs
from astropy.io import fits
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.misc
import cPickle

def calculate_offsets(NSIDE, file_list, parallel, band):
    power = np.log2(NSIDE)
    powers = [2**(i+1) for i in range(power)]
    for nside in powers:
        pixmap = pah_mapping.FullSkyMap(nside, file_list, parallel, band)

def create_mosaic(
        file_list, deg_ppx, ra_span, dec_span, 
        lowerleft_ra, lowerleft_dec, path, 
        region, band, parallel=False, adjust=False
        ):
    coadd = pah_mapping.Mosaic(file_list, deg_ppx, ra_span, dec_span, 
                               lowerleft_ra, lowerleft_dec, path, 
                               region, band, parallel, adjust)
    coadd.pickle()
    if parallel:
        #index_sublist = coadd.filelist#range(len(coadd.filelist))[coadd.rank::coadd.size]
        elements = (coadd.image_values, coadd.uncertainties, 
                    coadd.center_coords, coadd.index_sublist)
        data = coadd.comm.allgather(elements)
        #coadd.all_tile_positions = []
        coadd.all_image_values = []
        coadd.all_uncertainties = []
        coadd.all_center_coords = []
        index_order = []
        for array in data:
            #coadd.all_tile_positions = coadd.all_tile_positions + array[0]
            coadd.all_image_values = coadd.all_image_values + array[0]
            coadd.all_uncertainties = coadd.all_uncertainties + array[1]
            coadd.all_center_coords = coadd.all_center_coords + array[2]
            index_order = index_order + array[3]
    
        reorder = [i[0] for i in sorted(enumerate(index_order), 
                   key=lambda x:x[1])]
        #coadd.tile_positions = [coadd.all_tile_positions[i] for i in reorder]
        coadd.image_values = [coadd.all_image_values[i] for i in reorder]
        coadd.uncertainties = [coadd.all_uncertainties[i] for i in reorder]
        coadd.center_coords = [coadd.all_center_coords[i] for i in reorder]
        full_mosaic, full_unc = coadd.build_mosaic()
    else:
        full_mosaic, full_unc = coadd.build_mosaic(range(len(coadd.filelist)))
    if parallel and coadd.rank != 0:
        return None, None
    
    
    #plt.imshow(full_mosaic)
    #plt.savefig(path + region+'_'+str(band)+'_2files.png')
    #hdu2 = fits.PrimaryHDU(header = header)
    #hdu2.data = full_unc
    #hdu2.writeto(path + region+'_'+str(band)+'_unc_2files.fits', clobber = True)        
    print 'Mosaic created'
    return full_mosaic, full_unc

'''
parallel = True
adjust = True
region = '84_-5_offset'
band = int(sys.argv[1])
path = 'mosaic/'#'tiles/'+region+'/w' + str(band) + '/'#'../testdir2/'#'./84_-5/w1/'
files = 'datafiles_0907b_band'+str(band)+'.txt'
with open(files,'rb') as f:
    split_lines = f.read().splitlines()
    file_list = [x for x in split_lines if 'int' in x]
    #msk_list = [x for x in split_lines if 'msk' in x]
    #unc_list = [x for x in split_lines if 'unc' in x]
#file_list = []
#msk_list = []
#for item in os.listdir(path):
#    if 'int' in item and '.fits' in item:
#        file_list.append(path + item)
#    elif 'msk' in item and '.fits' in item:
#        msk_list.append(path + item)
#unc_list = msk_list

deg_ppx = 0.005
ra_span = 4.
dec_span = 4.
lowerleft_ra = 82.
lowerleft_dec = -7.
'''

'''
inds = [11, 12, 14, 15, 16, 28, 29, 30, 31, 45, 46, 47, 48, 67, 68, 69, 70, 71, 72, 93, 94, 95, 96, 97, 98, 99, 122, 123, 124, 125, 126, 127, 128, 153, 154, 155, 156, 157, 158, 159, 185, 186, 187, 188, 189, 190, 191, 218, 219, 220, 221, 222, 223, 224, 234, 235, 236, 237, 238, 239, 240, 241, 267, 268, 269, 270, 271, 272, 298, 299, 300, 301, 302, 303, 328, 329, 330, 331, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400]

#for i in xrange(60,len(inds)):
#    print '\n' + str((i)) + ' files'
#    file_inds = inds[0:i]
file_lists = [[file_list[j-1] for j in inds], [msk_list[j-1] for j in inds], [unc_list[j-1] for j in inds]]
'''

'''
#file_lists = [file_list, msk_list, unc_list]
full_mosaic, full_unc = create_mosaic(file_list, deg_ppx, ra_span, dec_span, 
                                      lowerleft_ra, lowerleft_dec, path, 
                                      region, band, parallel, adjust)
#scipy.misc.toimage(full_mosaic, cmin=0, cmax=8*10**-5).save('./'+region+'/anim_files2/offsetfiles.png')
if full_mosaic is not None:
    print 'Saving to file.'
    w_hdr = wcs.WCS(naxis = 2)
    w_hdr.wcs.crpix = [-lowerleft_ra/deg_ppx, -lowerleft_dec/deg_ppx]
    w_hdr.wcs.cdelt = np.array([deg_ppx,deg_ppx])
    w_hdr.wcs.crval = [0.,0.]
    w_hdr.wcs.ctype = ["RA---CAR","DEC--CAR"]
    header = w_hdr.to_header()
    header['BAND'] = band
    hdu1 = fits.PrimaryHDU(header = header)
    hdu1.data = full_mosaic
    hdu1.writeto(path + region+'_'+str(band)+'.fits', clobber=True)
#else:
#    print 'No action taken.'
'''

#sys.argv order: datafile_list.txt BAND start end TRUE/FALSE
if __name__ == '__main__':
    parallel = True
    NSIDE = 1024
    files_list = sys.argv[1]
    band = sys.argv[2]
    start_ind = int(sys.argv[3])
    end_ind = int(sys.argv[4])
    continued = sys.argv[5] == 'True'
    with open(files_list,'rb') as f:
        split_lines = f.read().splitlines()
        file_list = [x for x in split_lines]# if 'int' in x]
    fsm = pah_mapping.FullSkyMap(NSIDE, file_list[start_ind:end_ind], parallel, band)
    map_name = 'sky_map_w'+str(band)+'_'+str(end_ind)+'.fits'
    if continued and (not parallel or fsm.comm.rank == 0):
        with open('part_count_w'+str(band)+'.pkl', 'rb') as partcount:
            fsm.contd_count = cPickle.load(partcount)
        with open('part_map_w'+str(band)+'.pkl', 'rb') as partmap:
            fsm.contd_map = cPickle.load(partmap)
        with open('part_unc_w'+str(band)+'.pkl', 'rb') as partunc:
            fsm.contd_uncmap = cPickle.load(partunc)
    fsm.build_map(map_name)
    if (not parallel or fsm.comm.rank == 0):
        with open('part_map_w'+str(band)+'_'+str(int(end_ind/1000))+'.pkl', 'wb') as partmap:
            cPickle.dump(fsm.partmap, partmap, cPickle.HIGHEST_PROTOCOL)
        with open('part_count_w'+str(band)+'_'+str(int(end_ind/1000))+'.pkl', 'wb') as partcount:
            cPickle.dump(fsm.count, partcount, cPickle.HIGHEST_PROTOCOL)
        with open('part_unc_w'+str(band)+'_'+str(int(end_ind/1000))+'.pkl', 'wb') as partunc:
            cPickle.dump(fsm.partunc, partunc, cPickle.HIGHEST_PROTOCOL)
