# coding: utf-8
from tqdm import tqdm
from time import perf_counter
import cupy as cp
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
from cupyx.scipy.signal import convolve2d as conv2
#from cupy.fft import fftn
from time import perf_counter
import numpy as np

import hdf5plugin
import h5py

from IPython.display import clear_output

from scipy import ndimage as ndi
from scipy import signal
from scipy.fftpack import fft2
from scipy.signal import find_peaks
from scipy.special import jv
import scipy.io

from skimage import img_as_float, restoration
from skimage import restoration
from skimage.feature import peak_local_max
from skimage.measure import profile_line
from skimage.registration import phase_cross_correlation


import torch
from torchvision import transforms

from scipy.interpolate import griddata
from xml import dom
from scipy.io import loadmat

import random

import pandas as pd


def RL_deconvblind(img,PSF,iterations,verbose=False,TV=False):
    #Richardson Lucy (RL) algorithm for deconvoluting a measured image with a known point-spread-function image to return underlying object image
    if verbose:
        print('Calculating deconvolution...')
    #float32 type for diffraction pattern (img) and probe, point spread function (PSF)
    img = img.astype(cp.float32)
    PSF = PSF.astype(cp.float32)
    
    #find minimum value in img excluding <=0 pixels
    a = np.nanmin(np.where(img<=0, np.nan, img))
    #replace <=0 values in image
    img = cp.where(img <= 0, a, img)
    
    #RL deconvolution iterations
    init_img = img
    PSF_hat = self.flip180(PSF)
    for i in range(iterations):
        if verbose:
            print('Iteration: {}'.format(i+1))
        est_conv = conv2(init_img,PSF,'same', boundary='symm')
        relative_blur = (img / est_conv)
        error_est = conv2(relative_blur,PSF_hat, 'same',boundary='symm')
        if TV: #with total variation regularization
            alpha=0.001 #regularization term weight
            tv_factor=cp.asarray(total_variation_term(init_img,alpha))
            init_img=cp.nan_to_num(init_img*error_est*tv_factor)
        else: #without regularization
            init_img = init_img* error_est
    return init_img #recovered, deconvoluted, underlying object image
def flip180(arr):
    #inverts 2D array, used to invert probe array for Richardson Lucy deconvoltuion algorithm
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

def roi(image):
    #define a region of interest overwhich to perform deconvolution
    #requires prompted raw input from the user: (startX, deltaX), (startY, deltaY)
    result=0
    not_satisfied=True
    while not_satisfied:
        plt.figure()
        plt.imshow(image)#,norm=colors.LogNorm())
        plt.show()
        
        x,hx=(int(n) for n in input("Select region of interest (x) ").split())
        y,hy=(int(n) for n in input("Select region of interest (y) ").split())

        image_cropped = image[x:x+hx,y:y+hy]
    
        plt.imshow(image_cropped)
        plt.show()
        s=input("Satisfied? ")
        if s=='y':
            not_satisfied=False
            result=image_cropped
    return result #return roi (cropped) image
    


def plotter(images,labels,cmap='jet',log=False):
    # display n plots side by side
    n=len(images)
    fig, axes = plt.subplots(1, n, figsize=(8, 3))#, sharex=True, sharey=True)
    ax = axes.ravel()
    for i in range(0,n):
        if log:
            ax[i].imshow(images[i],norm=colors.LogNorm(),cmap=cmap)#clim=(1,1000),cmap=cmap)
        else:
            ax[i].imshow(images[i])
        #ax[i].axis('off')
        ax[i].set_title(labels[i])
    plt.show()

def RL_deconvblind(img,PSF,iterations,verbose=False,TV=False):
    #Richardson Lucy (RL) algorithm for deconvoluting a measured image with a known point-spread-function image to return underlying object image
    if verbose:
        print('Calculating deconvolution...')
    #float32 type for diffraction pattern (img) and probe, point spread function (PSF)
    img = img.astype(cp.float32)
    PSF = PSF.astype(cp.float32)
    
    #find minimum value in img excluding <=0 pixels
    a = np.nanmin(np.where(img<=0, np.nan, img))
    #replace <=0 values in image
    img = cp.where(img <= 0, a, img)
    
    #RL deconvolution iterations
    init_img = img
    PSF_hat = flip180(PSF)
    for i in range(iterations):
        if verbose:
            print('Iteration: {}'.format(i+1))
        est_conv = conv2(init_img,PSF,'same', boundary='symm')
        relative_blur = (img / est_conv)
        error_est = conv2(relative_blur,PSF_hat, 'same',boundary='symm')
        if TV:
            alpha=0.001 #regularization term weight
            tv_factor=cp.asarray(total_variation_term(init_img,alpha))
            init_img=cp.nan_to_num(init_img*error_est*tv_factor)

            
            # print('-----------------------------------------')
        else:
            init_img = init_img* error_est
    return init_img #recovered, deconvoluted, underlying object image
