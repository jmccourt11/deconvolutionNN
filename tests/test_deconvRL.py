#%%
import numpy as np
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../'))) 
from src.utils.deconvolutionRL import *

#%%
#set paths
basepath1=Path("W:/ptychosaxs/")
basepath2=Path("Y:/")

#chansong
dps=np.load(basepath1 / 'interpolated.npy')
phis=np.load(basepath2 / '2023_Mar/results/ML_recon/scans_585_to_765/phis.npz')['phi']
scans=np.load(basepath2 / '2023_Mar/results/ML_recon/scans_585_to_765/phis.npz')['scan']
print(phis,scans)

ri=random.choice(scans)
print(ri)
ri=ri-scans[0]
print(ri)

plt.figure()
plt.imshow(dps[ri],norm=colors.LogNorm())#[128:-128,128:-128])
plt.show()

probe_scan=ri+scans[0]
object_tests_new=loadmat(basepath2 / '2023_Mar/results/ML_recon/scan{:03}/roi0_Ndp256_rs1/MLs_L1_p10_g50_Ndp256_pc0_noModelCon_vp5_vi_mm/Niter1000.mat'.format(probe_scan))
######################################################################

#cindy
dps=np.load(basepath2 / 'ptychosaxs/test_cindy_zheng_dps.npy') 
phis=pd.read_csv(basepath1 / 'Sample6_tomo6_projs_1537_1789_shifts.txt',header=0)['# Angle'].values
scans=pd.read_csv(basepath1 / 'Sample6_tomo6_projs_1537_1789_shifts.txt',header=0)[' scanNo'].values
scans=np.asarray([int(s) for s in scans])
print(phis.shape,scans.shape)

ri=random.choice(scans)
print(ri)
ri=ri-scans[0]
print(ri)

plt.figure()
plt.imshow(dps[ri],norm=colors.LogNorm())#[128:-128,128:-128])
plt.show()

probe_scan=ri+scans[0]

object_tests_new=loadmat(basepath2 / '2021_Nov/results/ML_recon/tomo_scan3/scan{probe_scan}/roi0_Ndp512/MLc_L1_p10_g10_Ndp512_mom0.5_pc0_noModelCon_bg0.01_vp5_vi_mm_ff/Niter500.mat')

#######################################################################

probe_tests_new=object_tests_new['probe'].T[0][0].T #take first mode
probe_tests_new_FT=np.abs(np.fft.fftshift(np.fft.fft2(probe_tests_new)))

#psf=probe_tests_new_FT[224-64:288+64,220-64:284+64] #cindy
psf=probe_tests_new_FT[96:160,96:160] #chansong
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

for index,scan in tqdm(enumerate(total_scans)):
    
    if scan in scans:
        print(scan,index)
        probe_scan=scan
        
        #chansong
        object_tests_new=loadmat(basepath2 / '2023_Mar/results/ML_recon/scan{:03}/roi0_Ndp256_rs1/MLs_L1_p10_g50_Ndp256_pc0_noModelCon_vp5_vi_mm/Niter1000.mat'.format(probe_scan))
#        #cindy
#        object_tests_new=loadmat(fbasepath2 / '2021_Nov/results/ML_recon/tomo_scan3/scan{probe_scan}/roi0_Ndp512/MLc_L1_p10_g10_Ndp512_mom0.5_pc0_noModelCon_bg0.01_vp5_vi_mm_ff/Niter500.mat')

        probe_tests_new=object_tests_new['probe'].T[0][0].T #take first mode
        probe_tests_new_FT=np.abs(np.fft.fftshift(np.fft.fft2(probe_tests_new)))
        
        
        #chansong
        psf=probe_tests_new_FT[96:160,96:160]
#        #cindy
#        psf=probe_tests_new_FT[224-64:288+64,220-64:284+64]
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
        
        
        if count==1:
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