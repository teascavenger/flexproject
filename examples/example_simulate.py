#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test forward / backward projection of a 2D phantom.
"""
#%%
from flexdata import io
from flexdata import display
from flextomo import project
from flextomo import phantom

import numpy

#%% Check project default settings:
project.settings

#%% Create volume and forward project:
    
# Initialize images:    
vol = numpy.zeros([1, 512, 512], dtype = 'float32')
proj = numpy.zeros([1, 361, 512], dtype = 'float32')

# Define a simple projection geometry:
geometry = io.init_geometry(src2obj = 100, det2obj = 100, det_pixel = 0.01, theta_range = [0, 360], geom_type = 'simple')

# Create phantom and project into proj:
vol = phantom.spheroid(vol.shape, geometry, 0.5, 0.7, 0.25, [0, 0.2, 0.05])
display.display_slice(vol, title = 'Phantom')

# Forward project:
project.forwardproject(proj, vol, geometry)
display.display_slice(proj)

#%% Reconstruct

vol_rec = numpy.zeros_like(vol)

project.FDK(proj, vol_rec, geometry)
display.display_slice(vol_rec)

#%% EM
vol_rec = numpy.zeros_like(vol)

options = {'bounds':[0, 10], 'l2_update':True, 'block_number':5, 'mode':'random'}
project.EM(proj, vol_rec, geometry, iterations = 10, options = options)
display.display_slice(vol_rec)

#%% SIRT
vol = numpy.zeros([1, 512, 512], dtype = 'float32')

options = {'bounds':[0, 10], 'l2_update':True, 'block_number':5, 'mode':'random'}
project.SIRT(proj, vol, geometry, iterations = 10, options = options)

display.display_slice(vol, title = 'SIRT')
