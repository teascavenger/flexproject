#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load a large dataset using numpy.memmap - array mapped on disk. Reconstruct it. 
"""
#%% Imports

from flexdata import io
from flexdata import array
from flexdata import display
from flextomo import project

import numpy

#%% Read data

path = '/ufs/ciacc/flexbox/al_test/90KV_no_filt/'

dark = io.read_tiffs(path, 'di')
flat = io.data.read_tiffs(path, 'io')    
proj = io.data.read_tiffs(path, 'scan_', memmap = '/ufs/ciacc/flexbox/swap/swap.prj')

meta = io.read_meta(path, 'flexray')   
 
#%% Prepro:
    
# Now, since the data is on the harddisk, we shouldn't lose the pointer to it!    
# Be careful which operations to apply. Implicit are OK.
proj -= dark
proj /= (flat.mean(0) - dark)

numpy.log(proj, out = proj)
proj *= -1

proj = array.raw2astra(proj)    

display.display_slice(proj)

#%% Recon

vol = numpy.zeros([50, 2000, 2000], dtype = 'float32')

project.FDK(proj, vol, meta['geometry'])

display.display_slice(vol)

#%% SIRT

vol = numpy.ones([50, 2000, 2000], dtype = 'float32')

options = {'block_number':10, 'index':'sequential'}
project.SIRT(proj, vol, meta['geometry'], iterations = 5)

display.display_slice(vol, title = 'SIRT')