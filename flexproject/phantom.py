#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 2017

@author: kostenko

Genereation of CT phantoms from geometrical primitives. Reads geometry data
to compute dimensions correctly.

"""
import numpy

def _coords_(shape, geometry, offset = [0.,0.,0.]):
    """
    Coordinate space in mm.
    """
    voxel = geometry['img_pixel']
    
    xx = (numpy.arange(0, shape[0]) - shape[0] / 2) * voxel - offset[0] 
    yy = (numpy.arange(0, shape[1]) - shape[1] / 2) * voxel - offset[1]
    zz = (numpy.arange(0, shape[2]) - shape[2] / 2) * voxel - offset[2]
    
    return xx, yy, zz

def sphere(shape, geometry, r, offset = [0., 0., 0.]):
    """
    Make sphere. Radius is in units (geometry['unit'])
    """
    return spheroid(shape, geometry, r, r, r, offset)
    
def spheroid(shape, geometry, r1, r2, r3, offset = [0., 0., 0.]):
    """
    Make a spheroid. 
    """
    # Get the coordinates in mm:
    xx,yy,zz = _coords_(shape, geometry, offset)
    
    # Volume init: 
    return  numpy.array((((xx[:, None, None]/r1)**2 + (yy[None, :, None]/r2)**2 + (zz[None, None, :]/r3)**2) < 1), dtype = 'float32') 
    
def cuboid(shape, geometry, a, b, c, offset = [0., 0., 0.]):
    """
    Make a cuboid. Dimensions are in units (geometry['unit'])
    """
    # Get the coordinates in mm:
    xx,yy,zz = _coords_(shape, geometry, offset)
     
    return  numpy.array((abs(xx[:, None, None]) < a / 2) * (abs(yy[None, :, None]) < b / 2) * (abs(zz[None, None, :]) < c / 2), dtype = 'float32')  
    
       
def cylinder(shape, geometry, r, h, offset = [0., 0., 0.]):
    """
    Make a cylinder with a specified radius and height.
    """
    
    volume = numpy.zeros(shape, dtype = 'float32')
    
    # Get the coordinates in mm:
    xx,yy,zz = _coords_(shape, geometry, offset)
     
    volume = numpy.array(((zz[None, None, :])**2 + (yy[None, :, None])**2) < r ** 2, dtype = 'float32')  
    
    return (numpy.abs(xx) < h / 2)[:, None, None] * volume
        
def checkers(shape, geometry, frequency, offset = [0., 0., 0.]):
    """
    Make a 3D checkers board.
    """
    
    volume = numpy.zeros(shape, dtype = 'float32')
    
    # Get the coordinates in mm:
    xx,yy,zz = _coords_(shape, geometry, offset)
    
    volume_ = numpy.zeros(shape, dtype='bool')
    
    step = shape[1] // frequency
    
    for ii in range(0, frequency):
        sl = slice(ii*step, int((ii + 0.5) * step))
        volume_[sl, :, :] = ~volume_[sl, :, :]
    
    for ii in range(0, frequency):
        sl = slice(ii*step, int((ii + 0.5) * step))
        volume_[:, sl, :] = ~volume_[:, sl, :]

    for ii in range(0, frequency):
        sl = slice(ii*step, int((ii + 0.5) * step))
        volume_[:, :, sl] = ~volume_[:, :, sl]
 
    volume *= volume_
    
    return volume