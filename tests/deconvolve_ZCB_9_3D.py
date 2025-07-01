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

from tqdm import tqdm

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

def load_h5py_dp(file):
    #file format for diffraction pattern data is hdf5 (hierarchical data format)
    with h5py.File(file,"r") as f:    
        # Print all root level object names (aka keys) of hdf5 file
        # these can be group or dataset names 
        print("Keys: %s" % f.keys())
        # get object names/keys for import information
        dp_group_key = 'dp' #diffraction patterns (key)
        sdd=f['detector_distance'][()][0] #sample to detector distance
        wavelength=f['lambda'][()][0] #X-ray wavelength
        # preferred methods to get dataset values:
        ds_obj = f[dp_group_key]      # returns as a h5py dataset object
        ds_arr = f[dp_group_key][()]  # returns as a numpy array
        f.close()
    return ds_arr,sdd,wavelength #return tuple consisting of np.array of dps, sdd, wavelength
        

from scipy.interpolate import griddata
from xml import dom

#TO USE MATLAB WRAPPER, WILL HAVE TO SET THIS UP ON "USER" REFINER
#import matlab.engine
#eng=matlab.engine()



#plt.clim is a great way to scale colors in mages plt.clim(1,1000) for log scale images


class Deconvolve:
    def __init__(self,filetype,dp,probe):
        if filetype=='mat':
            self.dp=self.load_data_mat(dp)
        elif filetype=='h5':
            self.dp=self.load_data_h5(dp)
        else:
            print('Invalid filetype (need h5 or mat)')
        self.probe=probe

    def __repr__(self):
        return f'Deconvolve(DP: {self.dp!r}, Probe: {self.probe!r})'

    #def __iter__(self):
    #def __next__(self):

    def load_data_mat(self,filename):
        #load in an mat (MATLAB) diffraction patterns file
        return scipy.io.loadmat(filename)
    
    def load_data_h5(self,filename):
        #file format for diffraction pattern data is hdf5 (hierarchical data format) 
        with h5py.File(filename,"r") as f:    
            # Print all root level object names (aka keys) of hdf5 file
            # these can be group or dataset names 
            print("Keys: %s" % f.keys())
            # get object names/keys for import information
            dp_group_key = 'dp' #diffraction patterns (key)
            sdd=f['detector_distance'][()][0] #sample to detector distance
            wavelength=f['lambda'][()][0] #X-ray wavelength
            # preferred methods to get dataset values:
            ds_obj = f[dp_group_key]      # returns as a h5py dataset object
            ds_arr = f[dp_group_key][()]  # returns as a numpy array
            f.close()
        return ds_arr,sdd,wavelength #return tuple consisting of np.array of dps, sdd, wavelength`
        
        
    def RL_deconvblind(self,img,PSF,iterations,verbose=False,TV=False):
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

        
    def flip180(self,arr):
        #inverts 2D array, used to invert probe array for Richardson Lucy deconvoltuion algorithm
        new_arr = arr.reshape(arr.size)
        new_arr = new_arr[::-1]
        new_arr = new_arr.reshape(arr.shape)
        return new_arr
        
        
    def roi(self,image):
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
    
    def FT_image(self,image):
        #calculate fourier transform of image with necessary shift to center result
        return np.abs(np.fft.fftshift(np.fft.fft2(image)))**2
    
    
    def plotter(self,images,labels,cmap='jet',log=False):
        # display n plots side by side
        n=len(images)
        fig, axes = plt.subplots(1, n, figsize=(8, 3))#, sharex=True, sharey=True)
        ax = axes.ravel()
        for i in range(0,n):
            if log:
                ax[i].imshow(images[i],norm=colors.LogNorm(),clim=(1,1000),cmap=cmap)
            else:
                ax[i].imshow(images[i],cmap=cmap)
            #ax[i].axis('off')
            ax[i].set_title(labels[i])
        plt.show()
        
        
        
        
        
    def run(self,device=1):
        with cp.cuda.Device(device):
            ##load in diffraction patterns (dps) data, including sample-to-detector distance (sdd) and xray wavelength (wavelength)
            dps = self.dp
            probe = self.probe
            
            probe=roi(probe) #FOR JUST CENTER FZP PATTERN 
                        #BEAM: x,y: 118 21

            #matrix that will be used for output in appropriate format for matlab reciprocal space scripts
            result_mtx=np.zeros((len(dps[0]),len(dps[0][0]),len(dps))) #x, y, phis
            phis = dps['phi'].squeeze()
            
            #delete bad frames
            #phis = np.delete(phis,[112,113])       
            
            #crop probe
            probe_gray=(probe*255/np.max(probe)).astype(np.uint8)
            bgr = cv2.cvtColor(probe_gray, cv2.COLOR_GRAY2BGR)
            img = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(256,256))
            gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            _,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
            edges = cv2.dilate(cv2.Canny(thresh,0,255),None)
            cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
            mask = np.zeros((256,256), np.uint8)
            masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
            mask_2=cv2.resize(mask,probe.shape)
            probe_masked=cv2.bitwise_and(probe,probe,mask=mask_2)
            plt.imshow(probe_masked,norm=colors.LogNorm())
            plt.clim(1,1000)
            plt.show()
            probe=probe_masked

            #convolution counter
            count=0
            for dp in tqdm(dps):
                #deconvolute dp and PSF
                iterations=50
                #convert to cupy array
                probe=cp.asarray(probe) 
                dp=cp.asarray(dp)
                
                #initialize GPU timer
                start_time=perf_counter()            
                
                #result = RL_deconvblind(dp, psf, iterations,TV=True)
                result = self.RL_deconvblind(dp, probe, iterations,TV=False)

                #acquire probe, dp, and result from GPUs and put on CPU
                result_cpu=result.get()
                dp_cpu=dp.get()
                probe_cpu=probe.get()
            
                #calculate time of deconvolution on GPU
                cp.cuda.Device(device).synchronize()
                stop_time = perf_counter( )
                time=str(round(stop_time-start_time,4))
                #print("Computation time on GPU: "+time+" seconds")
                
                #print first result
                if count==1:
                    plotter([probe_cpu,dp_cpu,result_cpu],['psf','dp','recovered'],log=True)

                #load result matrix with deconvoluted image
                for i in range(0,len(dps[count])):
                    for j in range(0,len(dps[count][i])):
                        result_mtx[j][i][count] = result_cpu[i][j] #loaded with transpose so need to flip again i <-> j
                
                count+=1
                
            #SHOW PROGRESSION OF DECONVOLUTION WITH MORE DP FRAMES
            more=False
            if more:
                iterations=50
                num_frames=len(dps)
                start=len(dps)-1
                probe=cp.asarray(probe)
                
                for i in range(start,len(dps[:num_frames])):
                    
                    dp=cp.asarray(avg_frames(dps[0:i],i))
                    result = self.RL_deconvblind(dp, probe, iterations)
                    result_cpu=result.get()
                    clear_output(wait=True)
                    plt.imshow(result_cpu,norm=colors.LogNorm())
                    plt.show()
                    #plt.pause(0.2)
                dp_cpu=dp.get()
                probe_cpu=probe.get()
                plotter([probe_cpu,dp_cpu,result_cpu],['psf','dp','recovered'],log=True)                      

            final = {'img':result_mtx,'phi':phis}
            
            #save *mat file for matlab
            #scipy.io.savemat("output.mat",final)
            
            return final
        
        
        
        
    
    
    
    
    
    
    

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
            #alpha=0.001
            tv_factor=cp.asarray(total_variation_term(init_img,alpha))
            # print(init_img,error_est,tv_factor)
            # plt.pause(2)
            init_img=cp.nan_to_num(init_img*error_est*tv_factor)
            # clear_output(wait=True)
            # plt.imshow(tv_factor.get(),norm=colors.LogNorm())
            # plt.show()
            # print(init_img,error_est,tv_factor)
            # plt.pause(2)

            
            # print('-----------------------------------------')
        else:
            init_img = init_img* error_est
    return init_img #recovered, deconvoluted, underlying object image


#def load_h5py_dp(file):
    #now named load_data_h5

def interpolate(dp,cmap='jet'):
    size=dp.shape
    grid_x,grid_y=np.meshgrid(np.arange(0,size[0],1),np.arange(0,size[1],1))
    grid_x=grid_x.T
    grid_y=grid_y.T
    points=np.asarray([(x,y) for x in range(0,size[0],1) for y in range(0,size[1],1)])
    d=dp
    
    #create mask of detector image
    mask=list(zip(np.where(d<=0)[0],np.where(d<=0)[1]))
    mask=np.asarray([list(m) for m in mask])

    plt.plot(mask[:,1],mask[:,0],color='k',alpha=0.5)

    plt.imshow(d,norm=colors.LogNorm(),cmap=cmap);plt.clim(1,1000);plt.plot(mask[:,1],mask[:,0],color='k',alpha=0.5); plt.show()
    plt.show()
    
    
    #filter out mask points
    out=points
    for m in mask:
        row = np.where( (out == m).all(axis=1))
        out=np.delete(out, row, axis=0)
    
    
    #interpolate
    #d=mat['dt'].T[0]
    values=d[out[:,0], out[:,1]]
    method='linear'
    grid_z2 = griddata(out, values, (grid_x, grid_y), method=method)
    plt.imshow(grid_z2, norm=colors.LogNorm(),cmap=cmap)
    plt.clim(1,1000)
    plt.title(method+' (griddata method)')
    plt.show()
            


#def load_h5py_dp(file):
    #now named load_data_h5
        
    
def flip180(arr):
    #inverts 2D array, used to invert probe array for Richardson Lucy deconvoltuion algorithm
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

def normal_gray(array):
    #normalize image to gray scale
    array = array/cp.amax(array)*255
    array = cp.where(array < 0,  0, array)
    array = cp.where(array > 255, 255, array)
    array = array.astype(cp.int16)
    return array



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

def FT_image(image):
    #calculate fourier transform of image with necessary shift to center result
    return np.abs(np.fft.fftshift(np.fft.fft2(image)))**2

def plotter_gray(images,labels,log=False):
    # display n plots side by side
    n=len(images)
    fig, axes = plt.subplots(1, n, figsize=(8, 3))#, sharex=True, sharey=True)
    ax = axes.ravel()
    for i in range(0,n):
        if log:
            ax[i].imshow(images[i], cmap=plt.cm.gray,norm=colors.LogNorm())
        else:
            ax[i].imshow(images[i], cmap=plt.cm.gray)
        #ax[i].axis('off')
        ax[i].set_title(labels[i])
    plt.show()
    
def plotter(images,labels,cmap='jet',log=False):
    # display n plots side by side
    n=len(images)
    fig, axes = plt.subplots(1, n, figsize=(8, 3))#, sharex=True, sharey=True)
    ax = axes.ravel()
    for i in range(0,n):
        if log:
            ax[i].imshow(images[i],norm=colors.LogNorm())
            ax[i].imshow(images[i],norm=colors.LogNorm(),clim=(1,1000),cmap=cmap)
            ax[i].imshow(images[i],norm=colors.LogNorm(),clim=(1,1000))
        else:
            ax[i].imshow(images[i])
        #ax[i].axis('off')
        ax[i].set_title(labels[i])
    plt.show()
    

def gamma_manip(image, gamma):
    #increase contrast for gray scale image
    result = np.uint8(cv2.pow(image / 255., gamma) * 255.)
    return result 

    
def peak_finder(dp_image_file):
    image=cv2.imread(dp_image_file)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #im = img_as_float(image)

    # image_max is the dilation of im with a 30*30 structuring element
    # It is used within peak_local_max function
    image_max = ndi.maximum_filter(image, size=30, mode='constant')
    # Comparison between image_max and im to find the coordinates of local maxima
    coordinates = peak_local_max(image, min_distance=10, threshold_abs=.01)
    plotter([image,image_max],['Original','Maximum filter'],log=True)
    # plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    # plt.show()
    # print("Peaks (coordinates)")
    # print(coordinates)
    # print('Peaks (q, theta)')
    peaks=peak_converter(coordinates)
    return coordinates#peaks


def peak_converter(peaks):
    center=200 #same for x and y
    q_hk=[]
    theta_hk=[]
    for peak in peaks:
        q_hk.append(np.sqrt((peak[0]-center)**2+(peak[1]-center)**2))
        theta_hk.append(np.arctan((center-peak[0])/(center-peak[1]))*180/np.pi)
        #print(q_hk,theta_hk)
    return q_hk,theta_hk

def sum_frames(dp_list,total_frames,live_plot=False):
    #sum diffraction patterns from hdf5_file
    total=np.zeros(dp_list[0].shape)
    for dp in dp_list[0:total_frames]:
        #plot result of summation for each frame
        if live_plot:
            clear_output(wait=True)
            plt.imshow(total,norm=colors.LogNorm())
            plt.show()
        total+=dp
    return total #returns total sum of dp frames

def avg_frames(dp_list,total_frames,live_plot=False):
    #diffraction patterns from hdf5_file
    total=np.zeros(dp_list[0].shape)
    for dp in tqdm(dp_list[0:total_frames]):
        #plot result of summation for each frame
        if live_plot:
            clear_output(wait=True)
            plt.imshow(total,norm=colors.LogNorm())
            plt.show()
        total+=dp
    return total/total_frames  #returns total sum of dp frames/number of summed frames

def live_plot_images(images):
    #live plot
    #plt.figure()
    for x in images[:]:
        try:
            plt.imshow(x,norm=colors.LogNorm())
            plt.clim(1,1000)
            clear_output(wait=True)
            plt.pause(.3)
            plt.clf()
        except KeyboardInterrupt:
            break
    plt.show()


def hor_line_profile(image, line):
    #horizontal line profile of image
    start = (line, 0) #Start of the profile line row=line, col=0
    end = (line, image.shape[1] - 1) #End of the profile line row=100, col=last
    profile = profile_line(image, start, end)
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Image (log)')
    ax[0].imshow(image,norm=colors.LogNorm(),clim=(1,1000))
    ax[0].plot([start[1], end[1]], [start[0], end[0]], 'r')
    ax[1].set_title('Profile (log)')
    ax[1].set_yscale('log')
    ax[1].plot(profile)
    peaks=find_peaks(profile,prominence=1,width=1.2,distance=18)
    xs=np.linspace(0,image.shape[0],image.shape[0])
    ax[0].scatter([xs[p] for p in peaks[0]],[line for p in peaks[0]],color='r',alpha=0.2)
    ax[1].scatter([xs[p] for p in peaks[0]],[profile[p] for p in peaks[0]],color='r')


def vert_line_profile(image, line):
    #vertical line profile of image
    start = (0,line) #Start of the profile line row=line, col=0
    end = (image.shape[0] - 1,line) #End of the profile line row=100, col=last
    profile = profile_line(image, start, end)
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Image (log)')
    ax[0].imshow(image,norm=colors.LogNorm(),clim=(1,1000))
    ax[0].plot([start[1], end[1]], [start[0], end[0]], 'r')
    ax[1].set_title('Profile (log)')
    ax[1].set_yscale('log')
    ax[1].plot(profile)
    peaks=find_peaks(profile,prominence=1,width=1.2,distance=18)
    xs=np.linspace(0,image.shape[0],image.shape[0])
    ax[0].scatter([line for p in peaks[0]],[xs[p] for p in peaks[0]],color='r',alpha=0.2)
    ax[1].scatter([xs[p] for p in peaks[0]],[profile[p] for p in peaks[0]],color='r')


def center(gray_image):
    # https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
    # convert the grayscale image to binary image
    ret,thresh = cv2.threshold(gray_image,127,255,0)
    # calculate moments of binary image
    M = cv2.moments(thresh)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    #plot calculated center and image
    plt.imshow(gray_image,norm=colors.LogNorm())
    plt.scatter(cX,cY,color='r')
    return cX,cY

def radial_average(image,cx,cy):
    #https://stackoverflow.com/questions/48842320/what-is-the-best-way-to-calculate-radial-average-of-the-image-with-python
    # create array of radii
    x,y = np.meshgrid(np.arange(image.shape[1]),np.arange(image.shape[0]))
    R = np.sqrt((x- cx)**2+(y-cy)**2)
    # calculate the mean
    dr=0.5 #half a pixel
    f = lambda r : image[(R >= r-dr) & (R < r+dr)].mean()
    r  = np.linspace(0,np.max(image.shape),num=np.max(image.shape))
    mean = np.vectorize(f)(r)
    # plot it
    fig,ax=plt.subplots()
    ax.plot(r,mean,norm=colors.LogNorm())
    plt.xlabel('pixels from center')
    plt.show()
    return r,mean

def plot_q_radial_avg(image):
    #find center of image
    cx,cy=center(image)
    #calculate radial average of full image
    pixels,intensity=radial_average(image,cx,cy)
    #convert to q
    #example using realistic sdd, wavelength, pixel-size
    sdd=2 #2m distance to detector from sample
    wavelength=1.240*10**(-10) # wavelength of xray
    pixel_size=200*10**(-6) #pixel linear dimension
    thetas=[np.arctan(p*pixel_size/sdd)/2 for p in pixels]
    q=[4*np.pi*np.sin(th)/wavelength/(1*10**9) for th in thetas] # q in inverse nm
    plt.loglog(q,intensity)
    plt.xlabel('q (nm$^{-1}$)')
    plt.show()

def normalize_log(image):
    #normalized based on log-normal distribution
    #https://en.wikipedia.org/wiki/Log-normal_distribution
    mean=np.mean(image)
    std=np.std(image)
    meanlog=np.log(mean**2/(np.sqrt(mean**2+std**2)))
    stdlog=np.sqrt(np.log(1+(std**2/mean**2)))
    return (image - meanlog) / stdlog


def normalize_T(array):
    #normalize based on mean and standard deviation
    #noramlization using pytorch tensor transforms           
    # plt.hist(array.ravel(), density=True)
    # plt.show()
    
    array_T=torch.from_numpy(array)
    mean, std = array_T.mean(),array_T.std()
    # define custom transform
    # here we are using our calculated
    # mean & std
    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # get normalized image
    img_normalized = transform_norm(array)
    # convert normalized image to numpy
    # array
    img_np = np.array(img_normalized)
    # transpose from shape of (1,,) to shape of (,,1)
    img_np = img_np.transpose(1, 2, 0)
    # plt.imshow(img_np,norm=colors.LogNorm())
    # plt.show()
    return(img_np)



def correlate(im1,im2):
    #calcuate 2D correlation of two images
    cx,cy=center(im1)
    im1=im1[cx-50:cx+50,cy-50:cy+50]
    im2=im2[cx-50:cx+50,cy-50:cy+50]
    cor_max=signal.correlate2d(im1,im1,boundary='symm',mode='same')
    cor=signal.correlate2d(im1,im2,boundary='symm',mode='same')-cor_max+1
    pcc=phase_cross_correlation(im1,im2)
    return cor.flatten() #returned flattened 1D array of 2D correlation


def mae(y1, y2):
    #mean aboslute error of two arrays, used for comparing flattened correlation functions
    y1, y2 = np.array(y1), np.array(y2)
    return np.mean(np.abs(y1 - y2))



def total_variation_term(array1,alpha):
    a=torch.tensor(array1)
    da=torch.gradient(a) #size=2xnxn
    da_mag=torch.sqrt(da[0]*da[0]+da[1]*da[1]) #nxn
    #da_mag1=torch.hypot(da[0],da[1])
    #print(da_mag1==da_mag)
    dax_da=torch.gradient(da[0])[0]#nxn
    day_da=torch.gradient(da[1])[1]#nxn
    div=(dax_da+day_da)/da_mag#divergence term: nxn
    result=1/(1-alpha*div)
    result_array=np.nan_to_num(result.numpy())
    # plt.imshow(result_array)
    # print(result_array)
    # plt.pause(3)
    return result_array


def run(dps,probe,device=1):
    with cp.cuda.Device(device):
        ##load in diffraction patterns (dps) data, including sample-to-detector distance (sdd) and xray wavelength (wavelength)
        #dps,sdd,wavelength = load_h5py_dp(dp)
        
        #dps=scipy.io.loadmat('/home/beams/B304014/ptychosaxs/chansong/ckim_data585_766.mat')
        phis=scipy.io.loadmat('/home/beams/B304014/ptychosaxs/chansong/ckim_data585_766.mat')['phi'].squeeze()
        phis = np.delete(phis,[112,113])       
        print(len(phis))
        
        # #load in point-spread function (psf) (the probe) data, *npy data file
        # #the reconstructed probe from ptychography data
        # psf_real = np.load(probe,allow_pickle=True)
        
        
        #psf_real=dps['dt'].T[0]

        psf_real=dps[0]
        
        
        #psf_real=normalize_T(psf_real).squeeze()
        # # to create an image file to save
        # psf_real[psf_real == 0.] = np.nan
        # logimage = np.log(psf_real+0.02)
        # cv2.imwrite('images/test.png',logimage)
        # plt.imshow(logimage)
        # plt.show()
        
        # #plt.imshow(np.abs(psf_real)) #magnitude
        # #plt.imshow(np.angle(psf_real)) #phase
        
        # #intensity (magnitude**2) of the reconstructed probe (real space)
        # psf_real_mag_2=np.abs(psf_real)**2
        
        # #fourier transform of reconstructed probe (reciprocal space)
        # psf=np.abs(np.fft.fftshift(np.fft.fft2(psf_real)))**2 # amplitude i.,e., magnitude squared
        # #psf=np.abs(np.fft.fftshift(np.fft.fft2(psf_real))) #sqrt i.e. magnitude
        
        psf=psf_real
        
        #cv2.imwrite(directory+'psf.png',psf)
        #235 42 235 42
        #psf=roi(psf) #FOR JUST CENTER FZP PATTERN 
                    #BEAM: x,y: 118 21
        #psf=psf[118:139,118:139]
        #cv2.imwrite(directory+'psf.png',psf)
        
        
        
        
        #dps=np.asarray([np.sqrt(dp) for dp in dps])
        
        count=0
        #result_mtx=np.zeros((len(dps['dt']),len(dps['dt'][0]),len(dps['dt'][0][0])))
        result_mtx=np.zeros((len(dps[0]),len(dps[0][0]),len(dps)))
        
        
        result_test=[]
        
        psf=dps[111][230:284,230:284]
        plt.imshow(psf,norm=colors.LogNorm())
        plt.clim(1,1000)
        plt.show()
        
        #crop probe
        probe_gray=(psf*255/np.max(psf)).astype(np.uint8)
        bgr = cv2.cvtColor(probe_gray, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(256,256))
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        _,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
        edges = cv2.dilate(cv2.Canny(thresh,0,255),None)
        cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
        mask = np.zeros((256,256), np.uint8)
        masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
        mask_2=cv2.resize(mask,psf.shape)
        psf_masked=cv2.bitwise_and(psf,psf,mask=mask_2)
        plt.imshow(psf_masked,norm=colors.LogNorm())
        plt.clim(1,1000)
        plt.show()
        psf=psf_masked

        
        #for dp in tqdm(dps['dt'].T):
        for dp in tqdm(dps):
            #write dp to image file and load in the image
            #cv2.imwrite('dp.png',dp)
            
            #psf=dp[230:284,230:284] #roi(235,42,235,42)
            
            #psf=dp[235:277,235:277] 
            
            #psf=normalize_T(psf).squeeze()
            #psf=normalize_log(psf)
            #psf=dp[205:307,205:307]

            # dp=dp[190:320,190:320]
            
            #dp=normalize_T(dp).squeeze()
            #dp=normalize_log(dp)
            
            #plotter([psf,dp],['psf','dp'],log=True)
            
            
            # psf=normalize_log(psf)
            # dp=normalize_log(dp)
            
            
            
            #plotter([psf,dp],['psf','dp'],log=True)
            
            #process or done process ***
            #dp_image = cv2.imread('dp.png')
            #dp_image=process_dp('dp.png')
  
            #convert images to gray scale
            #dp_image_gray=dp_image
            # if processed, comment out this line ***
            #dp_image_gray = cv2.cvtColor(dp_image, cv2.COLOR_BGR2GRAY)
        
            #deconvolute dp and PSF
            iterations=50
            ##convert to cupy array
            #dp_gray=cp.asarray(dp_image_gray)
            
            psf=cp.asarray(psf) 
            dp=cp.asarray(dp)
            
            #initialize GPU timer
            start_time=perf_counter()        
            
                
            #result = restoration.richardson_lucy(dp,psf,iterations)#,filter_epsilon=1)
            #result = restoration.wiener(dp_image_gray,psf,10000)
            #result = restoration.unsupervised_wiener(dp_image_gray,psf)[0]#,iterations)

            
            #result = RL_deconvblind(dp, psf, iterations,TV=True)
            result = RL_deconvblind(dp, psf, iterations,TV=False)
            # TVcorr=correlate(result.get(),resultTV.get())

            # #get cupy arrays from GPU
            # result_norm=normal(result)
            # dp_gray_norm=normal(dp_gray)
            # psf_norm=normal(psf)
            # result_norm_cpu=result_norm.get()
            # dp_gray_norm_cpu=dp_gray_norm.get()
            # psf_norm_cpu=psf_norm.get()
            result_cpu=result.get()
            #dp_gray_cpu=dp_gray.get()
            dp_cpu=dp.get()
            psf_cpu=psf.get()
            # psf_cpu=psf
            # result_cpu=result
            

        
            #calculate time of deconvolution on GPU
            cp.cuda.Device(device).synchronize()
            stop_time = perf_counter( )
            time=str(round(stop_time-start_time,4))
            
            #print("Computation time on GPU: "+time+" seconds")
            
            if count==1:
                plotter([psf_cpu,dp_cpu,result_cpu],['psf','dp','recovered'],log=True)
            #plotter([np.nan_to_num(psf_cpu),np.nan_to_num(dp_cpu),np.nan_to_num(result_cpu)],['psf nan2num','dp nan2num','recovered nan2num'],log=True)
            
            # bkg= np.min(result_cpu[np.nonzero(result_cpu)]) #to get rid of zero 
            bkg=1e-1
            #normalize_T(dp_cpu)
            #plt.imshow((result_cpu - np.mean(result_cpu))/np.ptp(result_cpu),norm=colors.LogNorm())
            #plt.show()            
            
            #plotter([psf_cpu+abs(bkg),dp_cpu+abs(bkg),result_cpu+abs(bkg)],['psf (bkg)','dp (bkg)','recovered (bkg)'],log=True)
            
            
            # #save image
            # save='y'
            # if save == 'y':
            #      #out_filename="output_frame{}_iterations{}.png".format(frame,iterations)
            #      out_filename="tests/images/output_frame{}.png".format(count)
            #      #cv2.imwrite(out_filename, result_cpu)
            #      plt.imshow(result_cpu,norm=colors.LogNorm());
            #      plt.savefig(out_filename)
            #      print("Plots written to "+out_filename)
            # else:
            #      print("Plots not saved")
        
            # for i in range(0,len(dps['dt'])):
            #     for j in range(0,len(dps['dt'][i])):
            #         result_mtx[j][i][count] = result_cpu[i][j] #loaded with transpose so need to flip again i <-> j
            for i in range(0,len(dps[count])):
                 for j in range(0,len(dps[count][i])):
                     result_mtx[j][i][count] = result_cpu[i][j] #loaded with transpose so need to flip again i <-> j
            
            result_test.append(result_cpu)
            
            count+=1
        
        # #filter diffraction pattern images by correlation
        # im_fixed=dps[6] #pick a diffraction image to compare others too for correlation
        # dps_filtered=[dp for dp in tqdm(dps) if mae(correlate(im_fixed,dp),correlate(im_fixed,im_fixed))<22000000]
        # print(len(dps_filtered))
        # dps=dps_filtered
        
        #SHOW PROGRESSION OF DECONVOLUTION WITH MORE DP FRAMES
        more=False
        if more:
            iterations=50
            num_frames=len(dps)
            start=len(dps)-1
            psf=cp.asarray(psf)
            #for i in tqdm(range(start,len(dps[:num_frames]))):
            
            for i in range(start,len(dps[:num_frames])):
                
                dp=cp.asarray(avg_frames(dps[0:i],i))
                result = RL_deconvblind(dp, psf, iterations)
                result_cpu=result.get()
                clear_output(wait=True)
                plt.imshow(result_cpu,norm=colors.LogNorm())
                plt.show()
                #plt.pause(0.2)
            dp_cpu=dp.get()
            psf_cpu=psf.get()
            plotter([psf_cpu,dp_cpu,result_cpu],['psf','dp','recovered'],log=True)
            # cv2.imwrite('dp.png',dp_cpu)
            # np.save('dp.npy',dp_cpu)
            # cv2.imwrite('recovered.png',result_cpu)
            # np.save('recovered.npy',result_cpu)
        return result_cpu
                    
        
        #final = {'img':result_mtx.,'phi':dps['phi'].squeeze()}
        final = {'img':result_mtx,'phi':phis}
        #save *mat file for matlab
        #scipy.io.savemat("output.mat",final)
        return final
    
    
    
    
# test=np.zeros((len(mat['dt']),len(mat['dt'][0]),len(mat['dt'][0][0])))
# for i in range(0,len(mat['dt'])):
#     for j in range(0,len(mat['dt'][i])):
#         for k in range(0,len(mat['dt'][i][j])):
#             test[i][j][k] = mat['dt'][i][j][k]
#             test[i][j][k] = mat['dt'][i][j][k]
