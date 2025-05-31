#%%
import numpy as np
import sys
import os
from pathlib import Path
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
import re
plt.rcParams['image.cmap']='jet'

from skimage.restoration import richardson_lucy

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../'))) 
from src.utils.deconvolutionRL import *
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../NN/ptychosaxsNN/')))
import utils.ptychosaxsNN_utils as ptNN_U
import importlib
importlib.reload(ptNN_U)

#%%
#set paths
basepath1=Path("/scratch/2025_Feb/ptycho/")

scan_num=5065
dps=ptNN_U.load_h5_scan_to_npy(basepath1, scan_num, plot=False,point_data=True)

#%%
ri=666
fig,ax=plt.subplots(1,2)
ax[0].imshow(dps[ri],norm=colors.LogNorm(),cmap='jet')
ax[0].set_title('Diffraction Pattern',fontsize=12)
ax[1].imshow(dps[ri],norm=colors.LogNorm(),cmap='jet',clim=(1,1000))
ax[1].set_title('Rescaled (RL)',fontsize=12)
plt.show()
#%%
probe_scan=5065
probe=loadmat(f'/scratch/2025_Feb/results/ZCB_9_3D_/fly{probe_scan}/roi0_Ndp256/MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm/MLc_L1_p10_g100_Ndp256_mom0.5_pc800_maxPosError500nm_bg0.1_vp4_vi_mm/Niter1000.mat')['probe']
print(probe.shape)

#%%
#########################################################################################################
#probe_tests_new=object_tests_new['probe'].T[0][0].T #take first mode
probe_tests_new=probe.T[0][0].T #take first mode
probe_tests_new_FT=np.abs(np.fft.fftshift(np.fft.fft2(probe_tests_new)))

size=128
psf=probe_tests_new_FT[probe_tests_new_FT.shape[0]//2-size:probe_tests_new_FT.shape[0]//2+size,probe_tests_new_FT.shape[1]//2-size:probe_tests_new_FT.shape[1]//2+size] #zhihua

#crop probe
probe_gray=(psf*255/np.max(psf)).astype(np.uint8)
bgr = cv2.cvtColor(probe_gray, cv2.COLOR_GRAY2BGR)
img = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(256,256))
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#_,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
_,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
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
plt.imshow(psf,norm=colors.LogNorm())
plt.show()

#%%
device=1
count=0
result_test=[]
probes=[]

full_dps=np.sum(dps,axis=0)
print(full_dps.shape)
#%%

#scan informtion
center=(517,575)

#crop diffraction patterns
dpsize=256
dp=full_dps[center[0]-256//2:center[0]+256//2,
    center[1]-256//2:center[1]+256//2]

plt.imshow(dp,norm=colors.LogNorm())
plt.show()

probe_tests_new=probe.T[0][0].T #take first mode
probe_tests_new_FT=np.abs(np.fft.fftshift(np.fft.fft2(probe_tests_new)))


size=128
psf=probe_tests_new_FT[probe_tests_new_FT.shape[0]//2-size:probe_tests_new_FT.shape[0]//2+size,probe_tests_new_FT.shape[1]//2-size:probe_tests_new_FT.shape[1]//2+size] #zhihua


#crop probe
probe_gray=(psf*255/np.max(psf)).astype(np.uint8)
bgr = cv2.cvtColor(probe_gray, cv2.COLOR_GRAY2BGR)
img = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(256,256))
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
_,thresh = cv2.threshold(gray, np.mean(gray)+10, 255, cv2.THRESH_BINARY_INV)
edges = cv2.dilate(cv2.Canny(thresh,0,255),None)
cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
mask = np.zeros((256,256), np.uint8)
masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
mask_2=cv2.resize(mask,psf.shape)
psf_masked=cv2.bitwise_and(psf,psf,mask=mask_2)
psf=psf_masked

SNRs=[]
ITERATIONS=[]
TIMES=[]
figs=[]
PSNRs=[]

def psnr_iterative(prev, curr, max_val=1.0):
    mse = np.mean((curr - prev)**2)
    return float('inf') if mse == 0 else 10 * np.log10((max_val**2) / mse)
mask = np.load('/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy')

def preprocess_dp(dp,mask):
    size=256
    dp_pp=dp
    dp_pp=dp_pp*mask
    #dp_pp=ptNN_U.log10_custom(dp_pp)
    sf=np.max(dp_pp)-np.min(dp_pp)
    bkg=np.min(dp_pp)
    dp_pp=np.asarray((dp_pp-bkg)/(sf))
    return dp_pp

for i in tqdm(range(50,51)):
    #deconvolute dp and PSF
    iterations=i+1
    prev_iteration=i
    
    # #normalize dp and psf
    # dp_norm=dp/np.max(dp)
    # psf_norm=psf/np.max(psf)

   
    print(psf.shape)
    print(dp.shape)
    #preprocess dp
    dp_norm=dp*mask
    # #remove nan and zero
    dp_norm=np.where(dp_norm<=0,np.min(dp_norm[dp_norm>0]),dp_norm)
    dp_norm=np.where(np.isnan(dp_norm),np.min(dp_norm[dp_norm>0]),dp_norm)
    sf=np.max(dp_norm)-np.min(dp_norm)
    bkg=np.min(dp_norm)
    dp_norm=np.asarray((dp_norm-bkg)/(sf))
    
    #normalize psf
    psf_norm=psf/np.max(psf)
    
    psf=cp.asarray(psf_norm) 
    dp=cp.asarray(dp_norm)

    #initialize GPU timer
    start_time=perf_counter()        

    result = RL_deconvblind(dp, psf, iterations,TV=False)
    result_prev = RL_deconvblind(dp, psf, prev_iteration,TV=False)
    result_cpu=result.get()
    result_prev_cpu=result_prev.get()
    dp_cpu=dp.get()
    psf_cpu=psf.get()

    snr=np.mean(result_cpu)/np.std(result_cpu)
    # print("SNR: ",snr)
    SNRs.append(snr)
    psnr=psnr_iterative(result_prev_cpu,result_cpu)
    PSNRs.append(psnr)
    print(f"PSNR: {psnr}")
    #calculate time of deconvolution on GPU
    cp.cuda.Device(device).synchronize()
    stop_time = perf_counter( )
    time=str(round(stop_time-start_time,4))
    
    #remove nan and zero
    result_cpu=np.where(result_cpu<=0,np.min(result_cpu[result_cpu>0]),result_cpu)
    result_cpu=np.where(np.isnan(result_cpu),np.min(result_cpu[result_cpu>0]),result_cpu)
    

    ITERATIONS.append(iterations)
    # print("time: ",time)
    TIMES.append(time)

    fig=plotter([psf_cpu,dp_cpu,result_cpu],['psf','dp','recovered'],log=True)
    plt.savefig(f'TEMP/iterTEST{i}.png')
    plt.close(fig)


#%%

plt.xlabel('Iterations')
plt.ylabel('SNR')
plt.plot(ITERATIONS,SNRs)
plt.show()
# %%
fig,ax=plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(ptNN_U.log10_custom(dp_cpu))
ax[1].imshow(ptNN_U.log10_custom(result_cpu))#,norm=colors.PowerNorm(gamma=.1))
plt.show()
# %%
