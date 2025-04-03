#%%
import numpy as np
import sys
import os
from pathlib import Path
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
import re

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../'))) 
from src.utils.deconvolutionRL import *

#%%
#set paths
basepath1=Path("W:/ptychosaxs/")
basepath2=Path("Y:/")

#chansong
dps=np.load(basepath1 / 'interpolated_chansong.npy')
phis=np.load(basepath2 / '2023_Mar/results/ML_recon/scans_585_to_765/phis.npz')['phi']
scans=np.load(basepath2 / '2023_Mar/results/ML_recon/scans_585_to_765/phis.npz')['scan']
print(phis,scans)

#%%
test=np.load('/home/beams/PTYCHOSAXS/NN/interpolated_chansong.npy')
test2=np.load('/home/beams/PTYCHOSAXS/NN/interpolated_deconvolved.npy')
ri=109

fig,ax=plt.subplots(1,2)
ax[0].imshow(test[ri],norm=colors.LogNorm(),cmap='jet')
ax[0].set_title('Diffraction Pattern',fontsize=12)
ax[1].imshow(test2[ri],norm=colors.LogNorm(),cmap='jet')
ax[1].set_title('Recovered (RL)',fontsize=12)

# fig,ax=plt.subplots(2,2)
# ax[0,0].imshow(test[ri],norm=colors.LogNorm(),cmap='jet')
# ax[0,0].set_title('Diffraction Pattern (scan {:03})'.format(ri),fontsize=12)
# ax[0,1].imshow(test2[ri],norm=colors.LogNorm(),cmap='jet')
# ax[0,1].set_title('Deconvolved (RL) (scan {:03})'.format(ri),fontsize=12)
# ax[1,0].imshow(test[ri][test[ri].shape[0]//2-128:test[ri].shape[0]//2+128,test[ri].shape[1]//2-128:test[ri].shape[1]//2+128],norm=colors.LogNorm(),cmap='jet')
# ax[1,0].set_title('Diffraction Pattern (scan {:03})'.format(ri),fontsize=12)
# ax[1,1].imshow(test2[ri][test2[ri].shape[0]//2-128:test2[ri].shape[0]//2+128,test2[ri].shape[1]//2-128:test2[ri].shape[1]//2+128],norm=colors.LogNorm(),cmap='jet')
# ax[1,1].set_title('Deconvolved (RL) (scan {:03})'.format(ri),fontsize=12)
plt.show()
#%%

# ##set paths
# basepath2=Path("/net/micdata/data2/12IDC/")

# #%%
# #zhihua (comissioning)
# df =pd.read_csv(basepath2 / '2024_Dec/indices_phi_samplename.txt',header=None)
# filtered_df=df[df[df.columns[2]]==' JM01_3D_']
# scans=filtered_df[filtered_df.columns[0]].values
# phis=filtered_df[filtered_df.columns[1]].values
# dps=np.asarray([np.sum(load_h5_scan_to_npy(basepath2 / f'2024_Dec/ptycho/',scan,plot=False),axis=0) for scan in tqdm(scans[10:12])])

##%%
#ri=random.choice(scans)
#print(ri)
#ri=ri-scans[0]
#print(ri)

#plt.figure()
#plt.imshow(dps[ri],norm=colors.LogNorm())#[128:-128,128:-128])
#plt.show()

#probe_scan=ri+scans[0]
#object_tests_new=loadmat(basepath2 / '2023_Mar/results/ML_recon/scan{:03}/roi0_Ndp256_rs1/MLs_L1_p10_g50_Ndp256_pc0_noModelCon_vp5_vi_mm/Niter1000.mat'.format(probe_scan))
######################################################################

##cindy
#dps=np.load(basepath2 / 'ptychosaxs/test_cindy_zheng_dps.npy') 
#phis=pd.read_csv(basepath1 / 'Sample6_tomo6_projs_1537_1789_shifts.txt',header=0)['# Angle'].values
#scans=pd.read_csv(basepath1 / 'Sample6_tomo6_projs_1537_1789_shifts.txt',header=0)[' scanNo'].values
#scans=np.asarray([int(s) for s in scans])
#print(phis.shape,scans.shape)

#ri=random.choice(scans)
#print(ri)
#ri=ri-scans[0]
#print(ri)

#plt.figure()
#plt.imshow(dps[ri],norm=colors.LogNorm())#[128:-128,128:-128])
#plt.show()

#probe_scan=ri+scans[0]

#object_tests_new=loadmat(basepath2 / '2021_Nov/results/ML_recon/tomo_scan3/scan{probe_scan}/roi0_Ndp512/MLc_L1_p10_g10_Ndp512_mom0.5_pc0_noModelCon_bg0.01_vp5_vi_mm_ff/Niter500.mat')

#######################################################################

#zhihua

probe_scan=106
object_tests_new=loadmat(basepath2 / f'2024_Dec/results/JM01_3D_/fly{probe_scan}/roi0_Ndp512/MLc_L1_p10_gInf_Ndp256_mom0.5_pc200_model_scale_rotation_shear_asymmetry_noModelCon_bg0.1_vi_mm/MLs_L1_p10_g200_Ndp378_pc200_model_scale_asymmetry_rotation_shear_maxPosError200nm_noModelCon_bg0.1_vi_mm/Niter1000.mat')



#########################################################################################################
#probe_tests_new=object_tests_new['probe'].T[0][0].T #take first mode
probe_tests_new=object_tests_new['probe'].T[0].T #take first mode
probe_tests_new_FT=np.abs(np.fft.fftshift(np.fft.fft2(probe_tests_new)))

#psf=probe_tests_new_FT[224-64:288+64,220-64:284+64] #cindy
#psf=probe_tests_new_FT[96:160,96:160] #chansong

size=64
psf=probe_tests_new_FT[probe_tests_new_FT.shape[0]//2-size:probe_tests_new_FT.shape[0]//2+size,probe_tests_new_FT.shape[1]//2-size:probe_tests_new_FT.shape[1]//2+size] #zhihua

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
plt.imshow(psf,norm=colors.LogNorm())
plt.show()


result_mtx=np.zeros((len(dps[0]),len(dps[0][0]),len(scans)))

device=1
count=0
result_test=[]
probes=[]
#cindy
total_scans=np.arange(1537,1790,1) #all possible scans for this range
#chansong
total_scans=np.arange(585,766,1) #all possible scans for this range
#zhihua
total_scans=len(dps)


full_dps=dps.copy()

center_x,center_y=full_dps.shape[2]//2+8,full_dps.shape[1]-200
dps=full_dps[:,center_x-256:center_x+256,center_x-256:center_x+256]

#for index,scan in tqdm(enumerate(total_scans)):
for index,scan in enumerate(tqdm(scans[10:12])):
    if scan in scans:
        print(scan,index)
        probe_scan=scan
        
        #chansong
        #object_tests_new=loadmat(basepath2 / '2023_Mar/results/ML_recon/scan{:03}/roi0_Ndp256_rs1/MLs_L1_p10_g50_Ndp256_pc0_noModelCon_vp5_vi_mm/Niter1000.mat'.format(probe_scan))
#        #cindy
#        object_tests_new=loadmat(fbasepath2 / '2021_Nov/results/ML_recon/tomo_scan3/scan{probe_scan}/roi0_Ndp512/MLc_L1_p10_g10_Ndp512_mom0.5_pc0_noModelCon_bg0.01_vp5_vi_mm_ff/Niter500.mat')


        #zhihua
        object_tests_new=loadmat(basepath2 / f'2024_Dec/results/JM01_3D_/fly085/roi0_Ndp512/MLc_L1_p10_gInf_Ndp256_mom0.5_pc200_model_scale_rotation_shear_asymmetry_noModelCon_bg0.1_vi_mm/MLs_L1_p10_g200_Ndp378_pc200_model_scale_asymmetry_rotation_shear_maxPosError200nm_noModelCon_bg0.1_vi_mm/Niter1000.mat')
       
        
        
        #probe_tests_new=object_tests_new['probe'].T[0][0].T #take first mode
        probe_tests_new=object_tests_new['probe'].T[0].T #take first mode
        probe_tests_new_FT=np.abs(np.fft.fftshift(np.fft.fft2(probe_tests_new)))
        
        
        #chansong
        #psf=probe_tests_new_FT[96:160,96:160]
#        #cindy
#        psf=probe_tests_new_FT[224-64:288+64,220-64:284+64]
        #zhihua
        size=64
        psf=probe_tests_new_FT[probe_tests_new_FT.shape[0]//2-size:probe_tests_new_FT.shape[0]//2+size,probe_tests_new_FT.shape[1]//2-size:probe_tests_new_FT.shape[1]//2+size] #zhihua
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
        psf=psf_masked
        
        
        #deconvolute dp and PSF
        iterations=50
        
        psf=cp.asarray(psf) 
        #dp=cp.asarray(dps[scan-scans[0]])
        dp=cp.asarray(dps[index])
        
        #initialize GPU timer
        start_time=perf_counter()        
        
        result = RL_deconvblind(dp, psf, iterations,TV=False)
        result_cpu=result.get()
        dp_cpu=dp.get()
        psf_cpu=psf.get()
        


        #calculate time of deconvolution on GPU
        cp.cuda.Device(device).synchronize()
        stop_time = perf_counter( )
        time=str(round(stop_time-start_time,4))
        
        
        #if count==1:
        plotter([psf_cpu,dp_cpu,result_cpu],['psf','dp','recovered'],log=True)


        bkg=1e-1

        for i in range(0,len(dps[count])):
             for j in range(0,len(dps[count][i])):
                 result_mtx[j][i][count] = result_cpu[i][j] #loaded with transpose so need to flip again i <-> jlen
        
        result_test.append(result_cpu)
        
        probes.append(psf_cpu)
        
        count+=1
    else:
        continue
    
fig,ax=plt.subplots(1,2)
test_i=90
ax[0].imshow(result_mtx.T[test_i],norm=colors.LogNorm(),cmap='jet',clim=(1,1000))
ax[1].imshow(probes[test_i],cmap='jet')
plt.show()
