#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 2018
@author: kostenko

This module some wrappers around ASTRA to make lives of people slightly less horrible.
A lone traveller seeking for some algorithms can find bits and pieces of SIRT, FISTA and more.
We will do our best to be memmap compatible and make sure that large data will not make your PC sink into dispair.
"""

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Imports >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import numpy
import astra
import sys
import random
import scipy 
from tqdm import tqdm       # progress bar
from flexdata import io     # geometry to astra conversions
from flexdata import display# show images
import traceback
        
# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class project():
    '''
    This class contain reconstruction algorithms. It will also remember some 
    settings like the number of iterations etc. 
    '''
    
    # Default settings (many of them are only used with iterative algorthms):
    block_number = 10
    mode = 'sequential'      # This field can be 'random', 'sequential' or 'equidistant'
    poisson_weight = False
    
    bounds = None
    preview = False
    
    norm_update = False
    norm = 0
    
    def init_volume(projections, geometry = None):
        """
        Initialize a standard volume array.
        """          
        # Use geometry to compute additional offset if needed:
        if geometry:
            sample = geometry['proj_sample']
    
            offset = int(abs(geometry['vol_tra'][2]) / geometry['img_pixel'] / sample[2])
    
        else:
            offset = 0
            
            sample = [1, 1, 1]
    
        shape = projections[::sample[0], ::sample[1], ::sample[2]].shape
        return numpy.zeros([shape[0], shape[2]+offset, shape[2]+offset], dtype = 'float32')
    
    def FDK(self, projections, volume, geometry):
        """
        Feldkamp, Davis and Kress cone beam reconstruction.
        Args:
            projections : input numpy.array (dtype = float32) with the following dimensions: [vrt, rot, hrz]
            volume      : output numpy.array (dtype = float32) with the following dimensions: [vrt, mag, hrz]
            geometry    : geometry description - one of threee types: 'simple', 'static_offsets', 'linear_offsets'
        """
        self.backproject(projections, volume, geometry, 'FDK_CUDA')
    
    def backproject(self, projections, volume, geometry, algorithm = 'BP3D_CUDA'):
        """
        Backproject using standard ASTRA functionality. If data array is memmap, backprojection is done in blocks to save RAM.
        Args:
            projections : input numpy.array (dtype = float32) with the following dimensions: [vrt, rot, hrz]
            volume      : output numpy.array (dtype = float32) with the following dimensions: [vrt, mag, hrz]
            geometry    : geometry description - one of threee types: 'simple', 'static_offsets', 'linear_offsets'
            algorithm   : ASTRA algorithm type ['BP3D_CUDA', 'FDK_CUDA' etc.]
        """
        # Check if projections should be subsampled:
        sam = geometry['proj_sample']
        if sum(sam) > 3:
            projections = projections[sam[0], sam[1], sam[2]]
        
        # If algorithm is FDK we use single-block projection unless data is a memmap
        if (self.block_number == 1) | ((algorithm == 'FDK_CUDA') & (not isinstance(projections, numpy.memmap))):
            
            projections = self._contiguous_check_(projections) 
            
            # Initialize ASTRA geometries:
            vol_geom = io.astra_vol_geom(geometry, volume.shape)
            proj_geom = io.astra_proj_geom(geometry, projections.shape)    
            
            # Progress bar:
            pbar = tqdm(total=1)
            self._backproject_block_add_(projections, volume, proj_geom, vol_geom, algorithm)
            pbar.update(1)
            pbar.close()
            
        else:   
            # Here is the multi-block version:
            
            # Initialize ASTRA volume geometry:
            vol_geom = io.astra_vol_geom(geometry, volume.shape)
            
            # Progress bar:
            pbar = tqdm(unit = 'block', total=self.block_number)
            
            # Loop over blocks:
            for ii in range(self.block_number):
                
                # Extract a block:
                index = self._block_index_(ii, self.block_number, projections.shape[1], self.mode)
                if index is []: break
                
                proj_geom = io.astra_proj_geom(geometry, projections.shape, index)    
                block = projections[:, index,:]
                block = self._contiguous_check_(block)
                
                # Backproject:    
                self._backproject_block_add_(block, volume, proj_geom, vol_geom, algorithm)  
                
                pbar.update(1)
                
            pbar.close()
            
            # ASTRA is not aware of the number of blocks:    
            volume /= self.block_number
                           
    def forwardproject(self, projections, volume, geometry):
        """
        Forwardproject using standard ASTRA functionality. If data array is memmap, projection is done in blocks to save RAM.
        Args:
            projections : output numpy.array (dtype = float32) with the following dimensions: [vrt, rot, hrz]
            volume      : input numpy.array (dtype = float32) with the following dimensions: [vrt, mag, hrz]
            geometry    : geometry description - one of threee types: 'simple', 'static_offsets', 'linear_offsets'
        """
        
        # Check if projections should be subsampled:
        sam = geometry['vol_sample']
        if sum(sam) > 3:
            volume = volume[sam[0], sam[1], sam[2]]
        
        # Non-memmap case is a single block:
        volume = self._contiguous_check_(volume) 
            
        # Forward project will always use blocks:     
        if self.block_number == 1:
            
            # Initialize ASTRA geometries:
            vol_geom = io.astra_vol_geom(geometry, volume.shape)
            proj_geom = io.astra_proj_geom(geometry, projections.shape)    
            
            # Progress bar:
            pbar = tqdm(total=1)
            self._forwardproject_block_add_(projections, volume, proj_geom, vol_geom)
            pbar.update(1)
            pbar.close()
            
        else:   
            # Multi-block:
            
            # Initialize ASTRA geometries:
            vol_geom = io.astra_vol_geom(geometry, volume.shape)
            
            # Progress bar:
            pbar = tqdm(unit = 'block', total=self.block_number)
            
            # Loop over blocks:
            for ii in range(self.block_number):
                
                index = self._block_index_(ii, self.block_number, projections.shape[1], self.mode)
                if index is []: break
            
                # Extract a block:
                proj_geom = io.astra_proj_geom(geometry, projections.shape, index)    
                block = projections[:, index,:]
                block = self._contiguous_check_(block)
                
                # Backproject:    
                self._forwardproject_block_add_(block, volume, proj_geom, vol_geom)  
                
                projections[:, index,:] = block
                
                pbar.update(1)
                
            pbar.close()
   
    def SIRT(self, projections, volume, geometry, iterations):
        """
        Simultaneous Iterative Reconstruction Technique.
        """     
        # Sampling:
        samp = geometry['proj_sample']
        anisotropy = geometry['vol_sample']
        
        shp = numpy.array(projections.shape)
        shp //= samp
    
        # TODO: Test this!!!
        prj_weight = 1 / (shp[1] * numpy.prod(anisotropy) * max(volume.shape)) 
                        
        # Initialize L2:
        self.norm = []   
    
        print('Feeling SIRTy...')
        
        for ii in tqdm(range(iterations)):
        
            # Update volume:
            if sum(samp) > 3:
                proj = projections[::samp[0], ::samp[1], ::samp[2]]
                self.L2_step(proj, prj_weight, volume, geometry)
                
            else:
                self.L2_step(projections, prj_weight, volume, geometry)
                
            # Preview
            if self.preview:
                display.display_slice(volume, dim = 1)
                
        if self.norm_update:   
             display.plot(self.norm, semilogy = True, title = 'Resudual L2')   
       
        def EM(self, projections, volume, geometry, iterations, options = {'preview':False, 'bounds':None, 'block_number':1, 'mode':'sequential', 'l2_update': True}):
            """
            Expectation Maximization
            """ 
            # Make sure that the volume is positive:
            if volume.max() <= 0: 
                volume *= 0
                volume += 1
            elif volume.min() < 0: volume[volume < 0] = 0
        
            projections[projections < 0] = 0
        
            # Initialize L2:
            self.norm= []
                    
            print('Em Emm Emmmm...')
            
            for ii in tqdm(range(iterations)):
                 
                # Update volume:
                self.EM_step(projections, 1, volume, geometry)
                            
                # Preview
                if self.preview:
                    display.display_slice(volume, dim = 1)
                    
            if self.norm_update:   
                 display.plot(self.norm, semilogy = True, title = 'Resudual norm')      
        
    def FISTA(self, projections, volume, geometry, iterations):
        '''
        FISTA reconstruction. Right now there is no TV minimization substep here!
        '''
        # Sampling:
        samp = geometry['proj_sample']
        anisotropy = geometry['vol_sample']
        
        shp = numpy.array(projections.shape)
        shp //= samp
        
        prj_weight = 1 / (shp[1] * numpy.prod(anisotropy) * max(volume.shape)) 
                        
        # Initialize L2:
        self.norm = []   
        t = 1
        
        volume_t = volume.copy()
        volume_old = volume.copy()
    
        print('FISTING in progress...')
            
        for ii in tqdm(range(iterations)):
        
            # Update volume:
            if sum(samp) > 3:
                proj = projections[::samp[0], ::samp[1], ::samp[2]]
                self.FISTA_step(proj, prj_weight, volume, volume_old, volume_t, t, geometry)
            
            else:
                self.FISTA_step(projections, prj_weight, volume, volume_old, volume_t, t, geometry)
            
            # Preview
            if self.preview:
                display.display_slice(volume, dim = 1)
                
        if self.norm_update:   
             display.plot(self.norm, semilogy = True, title = 'Resudual norm')   
             
    def MULTI_SIRT(self, projections, volume, geometries, iterations):
        """
        A multi-dataset version of SIRT. Here prjections and geometries are lists.
        """ 
        
        # Make sure array is contiguous (if not memmap):
        # if not isinstance(projections, numpy.memmap):
        #    projections = numpy.ascontiguousarray(projections)        
        
        # Initialize L2:
        self.norm = []
    
        print('Doing SIRT`y things...')

        for ii in tqdm(range(iterations)):
            
            self.norm = 0
            for ii, proj in enumerate(projections):
                
                # This weight is half of the normal weight to make sure convergence is ok:
                prj_weight = 1 / (proj.shape[1] * max(volume.shape)) 
        
                # Update volume:
                self.L2_step(proj, prj_weight, volume, geometries[ii])
                                        
            # Preview
            if self.preview:
                display.display_slice(volume, dim = 1)
                
        if self.norm_update:   
             display.plot(self.norm, semilogy = True, title = 'Resudual norm')        
             
    def MULTI_PWLS(self, projections, volume, geometries, iterations = 10, student = False, pwls = True, weight_power = 1): 
        '''
        Penalized Weighted Least Squares based on multiple inputs.
        '''
        #error log:
        self.norm = []
    
        fac = volume.shape[2] * geometries[0]['img_pixel'] * numpy.sqrt(2)
    
        print('PWLS-ing in progress...')
                        
        # Iterations:
        for ii in tqdm(range(iterations)):
        
            # Error:
            L_mean = 0
            
            #Blocks:
            for jj in range(self.block_number):        
                
                # Volume update:
                vol_tmp = numpy.zeros_like(volume)
                bwp_w = numpy.zeros_like(volume)
                
                for kk, projs in enumerate(projections):
                    
                    index = self._block_index_(jj, self.block_number, projs.shape[1], 'random')
                    
                    proj = numpy.ascontiguousarray(projs[:,index,:])
                    geom = geometries[kk]
    
                    proj_geom = io.astra_proj_geom(geom, projs.shape, index = index) 
                    vol_geom = io.astra_vol_geom(geom, volume.shape) 
                
                    prj_tmp = numpy.zeros_like(proj)
                    
                    # Compute weights:
                    if pwls & ~ student:
                        fwp_w = numpy.exp(-proj * weight_power)
                        
                    else:
                        fwp_w = numpy.ones_like(proj)
                                            
                    #fwp_w = scipy.ndimage.morphology.grey_erosion(fwp_w, size=(3,1,3))
                    
                    self._backproject_block_add_(fwp_w, bwp_w, proj_geom, vol_geom, 'BP3D_CUDA')
                    self._forwardproject_block_add_(prj_tmp, volume, proj_geom, vol_geom)
                    
                    prj_tmp = (proj - prj_tmp) * fwp_w / fac

                    if student:
                        prj_tmp = studentst(prj_tmp, 5)
                        
                    self._backproject_block_add_(prj_tmp, vol_tmp, proj_geom, vol_geom, 'BP3D_CUDA')
                    
                    # Mean L for projection
                    L_mean += (prj_tmp**2).mean() 
                    
                eps = bwp_w.max() / 100    
                bwp_w[bwp_w < eps] = eps
                    
                volume += vol_tmp / bwp_w
                volume[volume < 0] = 0
    
                #print((volume<0).sum())
                    
            self.norm.append(L_mean / self.block_number / len(projections))
            
        display.plot(numpy.array(self.norm), semilogy=True)     
    
    def L2_step(self, projections, prj_weight, volume, geometry):
        """
        A single L2 minimization step. Supports blocking and subsets.
        """
            
        # Initialize ASTRA geometries:
        vol_geom = io.astra_vol_geom(geometry, volume.shape)      
        
        for ii in range(self.block_number):
            
            # Create index slice to address projections:
            index = self._block_index_(ii, self.block_number, projections.shape[1], self.mode)
            if index is []: break
    
            # Extract a block:
            proj_geom = io.astra_proj_geom(geometry, projections.shape, index = index)    
            
            # The block will contain the discrepancy eventually (that's why we need a copy):
            if (self.mode == 'sequential') & (self.block_number == 1):
                block = projections.copy()
                
            else:
                block = (projections[:, index, :]).copy()
                block = self._contiguous_check_(block)
                    
            # Forwardproject:
            self._forwardproject_block_(block, volume, proj_geom, vol_geom, '-')   
                        
            # Take into account Poisson:
            if self.poisson_weight:
                
                # Some formula representing the effect of photon starvation...
                block *= numpy.exp(-projections[:, index, :])
                
            block *= prj_weight * self.block_number
            
            # Apply ramp to reduce boundary effects:
            #block = array.ramp(block, 0, 5, mode = 'linear')
            #block = array.ramp(block, 2, 5, mode = 'linear')
                    
            # L2 norm (use the last block to update):
            if self.norm_update:
                self.norm.append(numpy.sqrt((block ** 2).mean()))
            else:
                self.norm = []
              
            # Project
            self._backproject_block_add_(block, volume, proj_geom, vol_geom, 'BP3D_CUDA')    
        
        # Apply bounds
        if self.bounds:
            numpy.clip(volume, a_min = self.bounds[0], a_max = self.bounds[1], out = volume)   
        
    def FISTA_step(self, projections, prj_weight, vol, vol_old, vol_t, t, geometry, options):
        """
        A single FISTA step. Supports blocking and subsets.
        """
        # Initialize ASTRA geometries:
        vol_geom = io.astra_vol_geom(geometry, vol.shape)      
        
        vol_old[:] = vol.copy()  
        
        t_old = t 
        t = (1 + numpy.sqrt(1 + 4 * t**2))/2
    
        vol[:] = vol_t.copy()
        
        for ii in range(self.block_number):
            
            # Create index slice to address projections:
            index = self._block_index_(ii, self.block_number, projections.shape[1], self.mode)
            if index is []: break
    
            # Extract a block:
            proj_geom = io.astra_proj_geom(geometry, projections.shape, index = index)    
            
            # Copy data to a block or simply pass a pointer to data itself if block is one.
            if (self.mode == 'sequential') & (self.block_number == 1):
                block = projections.copy()
                
            else:
                block = (projections[:, index, :]).copy()
                block = numpy.ascontiguousarray(block)
                    
            # Forwardproject:
            self._forwardproject_block_add_(block, vol_t, proj_geom, vol_geom, negative = True)   
                        
            # Take into account Poisson:
            if options.get('poisson_weight'):
                # Some formula representing the effect of photon starvation...
                block *= numpy.exp(-projections[:, index, :])
                
            block *= prj_weight * self.block_number
            
            # Apply ramp to reduce boundary effects:
            #block = block = flexData.ramp(block, 2, 5, mode = 'linear')
            #block = block = flexData.ramp(block, 0, 5, mode = 'linear')
                    
            # L2 norm (use the last block to update):
            if self.norm_update:
                self.norm.append(numpy.sqrt((block ** 2).mean()))
                
            else:
                self.norm = []
              
            # Project
            self._backproject_block_add_(block, vol, proj_geom, vol_geom, 'BP3D_CUDA')   
            
            vol_t[:] = vol + ((t_old - 1) / t) * (vol - vol_old)
                    
        # Apply bounds
        if self.bounds is not None:
            numpy.clip(vol, a_min = self.bounds[0], a_max = self.bounds[1], out = vol)  
    
    def EM_step(self, projections, prj_weight, volume, geometry, options):
        """
        A single Expecrtation Maximization step. Supports blocking and subsets.
        """              
        # Initialize ASTRA geometries:
        vol_geom = io.astra_vol_geom(geometry, volume.shape)      
        
        for ii in range(self.block_number):
            
            # Create index slice to address projections:
            index = self._block_index_(ii, self.block_number, projections.shape[1], self.mode)
            if index is []: break
    
            # Extract a block:
            proj_geom = io.astra_proj_geom(geometry, projections.shape, index = index)    
            
            # Copy data to a block or simply pass a pointer to data itself if block is one.
            if (self.mode == 'sequential') & (self.block_number == 1):
                block = projections
                
            else:
                block = (projections[:, index, :]).copy()
            
            # Reserve memory for a forward projection (keep it separate):
            synth = self._contiguous_check_(numpy.zeros_like(block))
            
            # Forwardproject:
            self._forwardproject_block_add_(synth, volume, proj_geom, vol_geom)   
      
            # Compute residual:        
            synth[synth < synth.max() / 100] = numpy.inf  
            synth = (block / synth)
                        
            # L2 norm (use the last block to update):
            if self.norm_update:
                self.norm.append(synth[synth > 0].std())
                
            else:
                self.norm = [] 
              
            # Project
            self._backproject_block_mult_(synth * prj_weight * self.block_number, volume, proj_geom, vol_geom, 'BP3D_CUDA')    
        
        # Apply bounds
        if self.bounds:
            numpy.clip(volume, a_min = self.bounds[0], a_max = self.bounds[1], out = volume)               
              
    def _backproject_block_add_(projections, volume, proj_geom, vol_geom, algorithm = 'BP3D_CUDA', negative = False):
        """
        Additive backprojection of a single block. 
        Use negative = True if you want subtraction instead of addition.
        """           
        try:
            if negative:
                projections *= -1
            
            sin_id = astra.data3d.link('-sino', proj_geom, projections)        
            vol_id = astra.data3d.link('-vol', vol_geom, volume)    
            
            projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
            
            # We are using accumulate version to avoid creating additional copies of data.
            if algorithm == 'BP3D_CUDA':
                astra.experimental.accumulate_BP(projector_id, vol_id, sin_id)                
            elif algorithm == 'FDK_CUDA':
                astra.experimental.accumulate_FDK(projector_id, vol_id, sin_id)               
            else:
                raise ValueError('Unknown ASTRA algorithm type.')
                             
        except:
            # The idea here is that we try to delete data3d objects even if ASTRA crashed
            try:
                if negative:
                    projections *= -1
            
                astra.algorithm.delete(projector_id)
                astra.data3d.delete(sin_id)
                astra.data3d.delete(vol_id)
                
            finally:
                info = sys.exc_info()
                traceback.print_exception(*info)        
        
        if negative:
            projections *= -1
        
        astra.algorithm.delete(projector_id)
        astra.data3d.delete(sin_id)
        astra.data3d.delete(vol_id)  
            
    def _backproject_block_mult_(projections, volume, proj_geom, vol_geom, algorithm = 'BP3D_CUDA', operation = '+'):
        """
        Multiplicative backprojection of a single block. 
        """           
        try:
            # Need to create a copy of the volume:
            volume_ = numpy.zeros_like(volume)
            
            sin_id = astra.data3d.link('-sino', proj_geom, projections)        
            vol_id = astra.data3d.link('-vol', vol_geom, volume_)    
            
            projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
            
            # We are using accumulate version to avoid creating additional copies of data.
            if algorithm == 'BP3D_CUDA':
                astra.experimental.accumulate_BP(projector_id, vol_id, sin_id)                
            elif algorithm == 'FDK_CUDA':
                astra.experimental.accumulate_FDK(projector_id, vol_id, sin_id)               
            else:
                raise ValueError('Unknown ASTRA algorithm type.')
                             
        except:
            # The idea here is that we try to delete data3d objects even if ASTRA crashed
            try:
               
                astra.algorithm.delete(projector_id)
                astra.data3d.delete(sin_id)
                astra.data3d.delete(vol_id)
                
            finally:
                info = sys.exc_info()
                traceback.print_exception(*info)        

        volume *= volume_
        
        astra.algorithm.delete(projector_id)
        astra.data3d.delete(sin_id)
        astra.data3d.delete(vol_id)      
                
    def _forwardproject_block_add_(projections, volume, proj_geom, vol_geom, negative = False):
        """
        Additive forwardprojection of a single block. 
        Use negative = True if you want subtraction instead of addition.
        """           
        
        try:
            # We are goint to negate the projections block and not the whole volume:
            if negative:
                projections *= -1
                    
            sin_id = astra.data3d.link('-sino', proj_geom, projections)        
            vol_id = astra.data3d.link('-vol', vol_geom, volume)    
            
            projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
            
            # Project!
            astra.experimental.accumulate_FP(projector_id, vol_id, sin_id)
            
            # Negate second time:
            if negative:
                projections *= -1
                 
        except:
            # Always try to delete data3d:
            try:
                astra.algorithm.delete(projector_id)
                astra.data3d.delete(sin_id)
                astra.data3d.delete(vol_id)   
            finally:
                info = sys.exc_info()
                traceback.print_exception(*info)
            
        astra.algorithm.delete(projector_id)
        astra.data3d.delete(sin_id)
        astra.data3d.delete(vol_id)   
        
    def _contiguous_check_(data):
        '''
        Check if data is contiguous, if not - convert. This makes ASTRA happy.
        Careful, it may copy the data and overflow RAM.
        '''
        if not data.flags['C_CONTIGUOUS']:
            data = numpy.ascontiguousarray(data)
        
        # Check if data type is correct:
        if data.dtype != 'float32':
            data = data.astype('float32')
            
        # Sometimes data shape is weird. Check.    
        if min(data.shape) == 0:
            raise Exception('Strange data shape:' + str(data.shape))
        
        return data  

    def _block_index_(ii, block_number, length, mode = 'sequential'):
        """
        Create a slice for a projection block
        """   
        
        # Length of the block and the global index:
        block_length = int(numpy.round(length / block_number))
        index = numpy.arange(length)
    
        # Different indexing modes:    
        if (mode == 'sequential')|(mode is None):
            # Index = 0, 1, 2, 4
            pass
            
        elif mode == 'random':   
            # Index = 2, 3, 0, 1 for instance...        
            random.shuffle(index)    
             
        elif mode == 'equidistant':   
            
            # Index = 0, 2, 1, 3   
            index = numpy.mod(numpy.arange(length) * block_length, length)
            
        else:
            raise ValueError('Indexer type not recognized! Use: sequential/random/equidistant')
        
        first = ii * block_length
        last = min((length + 1, (ii + 1) * block_length))
        
        return index[first:last]        

    
def studentst(res, deg = 1, scl = None):
    
    # nD to 1D:
    shape = res.shape
    res = res.ravel()
    
    # Optimize scale:
    if scl is None:    
        fun = lambda x: misfit(res[::70], x, deg)
        scl = scipy.optimize.fmin(fun, x0 = [1,], disp = 0)[0]
        #scl = numpy.percentile(numpy.abs(res), 90)
        #print(scl)
        #print('Scale in Student`s-T is:', scl)
        
    # Evaluate:    
    grad = numpy.reshape(st(res, scl, deg), shape)
    
    return grad

def misfit(res, scl, deg):
    
    c = -numpy.size(res) * (scipy.special.gammaln((deg + 1) / 2) - 
            scipy.special.gammaln(deg / 2) - .5 * numpy.log(numpy.pi*scl*deg))
    
    return c + .5 * (deg + 1) * sum(numpy.log(1 + numpy.conj(res) * res / (scl * deg)))
    
def st(res, scl, deg):   
    
    grad = numpy.float32(scl * (deg + 1) * res / (scl * deg + numpy.conj(res) * res))
    
    return grad
          

           
