#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.io import loadmat
import numpy as np
from skimage.transform import resize
import sys
import os
import importlib
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors
import random
from tqdm import tqdm
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader
import pdb
from torchmetrics.regression import PearsonCorrCoef
import h5py
#%%
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../NN/ptychosaxsNN/'))) 
import utils.ptychosaxsNN_utils as ptNN_U
import ptychosaxsNN.ptychosaxsNN as ptNN
importlib.reload(ptNN_U)
importlib.reload(ptNN)
#%%

# class ConvAutoencoderSkip(nn.Module):
#     def __init__(self, probe_kernel):
#         super(ConvAutoencoderSkip, self).__init__()

#         # Reduce channel sizes
#         self.enc1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, stride=1, padding=(1,1)),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.pool1 = nn.MaxPool2d(2, 2)

#         self.enc2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, 3, stride=1, padding=(1,1)),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )
#         self.pool2 = nn.MaxPool2d(2, 2)
        
        
#         self.enc3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, 3, stride=1, padding=(1,1)),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
#         self.pool3 = nn.MaxPool2d(2, 2)
        
        
#         # Bottleneck
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  
#             nn.ReLU(),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, 3, stride=1, padding=(1,1)),
#             nn.BatchNorm2d(512),
#             nn.ReLU()
#         )

#         # Decoder with Skip Connections (reduced channels)
#         self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
#         self.dec1 = nn.Sequential(
#             nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, 3, stride=1, padding=(1,1)),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )

#         self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
#         self.dec2 = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, 3, stride=1, padding=(1,1)),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )
    
#         self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
#         self.dec3 = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, stride=1, padding=(1,1)),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )

#         # Modify final layer to encourage sharp peaks
#         self.final_layer = nn.Sequential(
#             nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
#             #nn.ReLU(),  # Keep non-negative
#             nn.Sigmoid()
#         )
#         # # Final reconstruction layer
#         # self.final_layer = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        
#         self.sigmoid = nn.Sigmoid()
#         self.drop = nn.Dropout(0.75)

#         # Convert probe kernel to torch tensor
#         probe_kernel = torch.from_numpy(probe_kernel).float()
#         # Add batch and channel dimensions
#         probe_kernel = probe_kernel.unsqueeze(0).unsqueeze(0)
#         #print("Probe kernel shape:", probe_kernel.shape)  # Added print statement
#         self.register_buffer("probe_kernel", probe_kernel)
        
class ConvAutoencoderSkip(nn.Module):
    def __init__(self, probe_kernel):
        super(ConvAutoencoderSkip, self).__init__()

        # Reduce channel sizes
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Decoder with Skip Connections (reduced channels)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
    

        # Modify final layer to encourage sharp peaks
        self.final_layer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),  # Keep non-negative
            nn.Sigmoid()
        )
        # # Final reconstruction layer
        # self.final_layer = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.75)

        # Convert probe kernel to torch tensor
        probe_kernel = torch.from_numpy(probe_kernel).float()
        # Add batch and channel dimensions
        probe_kernel = probe_kernel.unsqueeze(0).unsqueeze(0)
        #print("Probe kernel shape:", probe_kernel.shape)  # Added print statement
        self.register_buffer("probe_kernel", probe_kernel)

    def conv2d_probe(self, x):
        return F.conv2d(x, self.probe_kernel, padding='same')
    
    def fft_conv2d(self, x):
        # Performing convolution
        
        # Compute FFT of decoded image
        x_fft = torch.fft.ifft2(x)
        
        # 1. Multiply probe and object in real space
        real_space_product = x_fft * self.probe_kernel
        
        # 2. Take FFT of product to get diffraction pattern
        output_fft = torch.fft.fft2(real_space_product)
        
        # 3. Take magnitude squared to get intensity
        output = torch.abs(output_fft)**2
        
        # Normalize output (prevent division by zero)
        #output = output / (torch.max(output) + 1e-8)
        
        # Normalize
        batch_min = output.view(output.size(0), -1).min(dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        batch_max = output.view(output.size(0), -1).max(dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        output = (output - batch_min) / (batch_max - batch_min + 1e-8)
        
        # Verify output size
        assert output.size() == x.size(), f"Output size {output.size()} doesn't match input size {x.size()}"
        
        return output


    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        enc1_pooled = self.drop(self.pool1(enc1_out))

        enc2_out = self.enc2(enc1_pooled)
        enc2_pooled = self.drop(self.pool2(enc2_out))

        # Bottleneck
        bottleneck_out = self.bottleneck(enc2_pooled)

        # Decoder with Skip Connections
        up1_out = self.up1(bottleneck_out)
        dec1_out = self.dec1(torch.cat([up1_out, enc2_out], dim=1))

        up2_out = self.up2(dec1_out)
        dec2_out = self.dec2(torch.cat([up2_out, enc1_out], dim=1)) 

        # Final output
        decoded = self.sigmoid(self.final_layer(dec2_out))
        
        #print("Decoded shape:", decoded.shape)  # Added print statement

        # Use FFT-based convolution instead of spatial convolution
        #probe_convolved_output = self.fft_conv2d(decoded)
        probe_convolved_output = self.conv2d_probe(decoded)
        
        return decoded, probe_convolved_output






#%%
#Zhihua probe
#probe=loadmat("/net/micdata/data2/12IDC/2024_Dec/results/JM02_3D_/fly482/roi2_Ndp1024/MLc_L1_p10_gInf_Ndp256_mom0.5_pc100_noModelCon_bg0.1_vi_mm/MLc_L1_p10_g400_Ndp512_mom0.5_pc400_noModelCon_bg0.1_vp4_vi_mm/Niter1000.mat")['probe'].T[0][0].T
#probe=loadmat("/net/micdata/data2/12IDC/2024_Dec/results/JM02_3D_/fly585/roi0_Ndp512/MLc_L1_p10_g1000_Ndp256_mom0.5_pc200_model_scale_rotation_shear_asymmetry_noModelCon_bg0.1_vi_mm/MLc_L1_p10_g100_Ndp512_mom0.5_pc200_model_scale_asymmetry_rotation_shear_maxPosError200nm_noModelCon_bg0.1_vi_mm/Niter600.mat")['probe'].T[0].T
with h5py.File("/net/micdata/data2/12IDC/2025_Feb/ptychi_recons/S5008/Ndp256_LSQML_c1000_m0.5_p15_cp_mm_opr3_ic_pc_ul2/recon_Niter1000.h5",'r') as f:
    probe=f['probe'][0][0]




print(probe.shape)
plt.imshow(np.abs(probe))
plt.colorbar()
plt.show()
dpsize=256#512
# probe=resize(probe,(dpsize,dpsize),preserve_range=True,anti_aliasing=True)
# plt.imshow(probe)
# plt.colorbar()
# plt.show()
plt.imshow(np.abs(np.asarray(np.fft.fftshift(np.fft.fft2(probe)))),norm=colors.LogNorm())
plt.colorbar()
plt.show()


#%%

# probe=np.asarray(np.fft.fftshift(np.fft.fft2(probe))[256-64:256+64,256-64:256+64])
# plt.imshow(np.abs(np.asarray(np.fft.fftshift(np.fft.fft2(probe))))[256-64:256+64,256-64:256+64],norm=colors.LogNorm())
# plt.colorbar()
# plt.show()


#%%
endsize=128
#%%
# Separate resize for real and imaginary components
probe_real = resize(np.real(probe), (endsize,endsize), preserve_range=True, anti_aliasing=True)
probe_imag = resize(np.imag(probe), (endsize,endsize), preserve_range=True, anti_aliasing=True)
# Recombine into complex array
probe = probe_real + 1j * probe_imag
# Verify the resize maintained complex structure
plt.figure(figsize=(12,4))
plt.subplot(131)
plt.imshow(np.abs(probe))
plt.title('Magnitude')
plt.colorbar()
plt.subplot(132)
plt.imshow(np.angle(probe))
plt.title('Phase')
plt.colorbar()

#%%
all_dps=[]
for scan in np.arange(5101,5105):
    #dps = ptNN_U.load_h5_scan_to_npy(Path(f'/net/micdata/data2/12IDC/2024_Dec/ptycho/'),scan,plot=False)
    dps = ptNN_U.load_h5_scan_to_npy(Path(f'/net/micdata/data2/12IDC/2025_Feb/ptycho/'),scan,plot=False)
    all_dps.append(dps)
temp_dps=np.asarray(all_dps)
#%%
dps=temp_dps.reshape(-1,temp_dps.shape[2],temp_dps.shape[3])
#%%
#center=np.array([dps.shape[1]//2-100,dps.shape[2]//2])
center=np.array([dps.shape[1]//2-4,dps.shape[2]//2+85])
print(f"center: {center}")
ri=random.randint(0,dps.shape[0]-1)
test=dps[ri][center[0]-dpsize//2:center[0]+dpsize//2,center[1]-dpsize//2:center[1]+dpsize//2]
plt.imshow(test,norm=colors.LogNorm())
plt.colorbar()
plt.show()
#%%
test=resize(test,(endsize,endsize),preserve_range=True,anti_aliasing=True)
plt.imshow(test,norm=colors.LogNorm())
plt.colorbar()
plt.show()
#%%
#dps preprocss
conv_DPs=dps
print('shuffling indices')
indices = np.arange(conv_DPs.shape[0])
np.random.shuffle(indices)
print('shuffling diffraction patterns')
conv_DPs_shuff = conv_DPs[indices]
print('log10')
amp_conv = ptNN_U.log10_custom(conv_DPs_shuff)

#%%
print('resizing')
amp_conv_red=np.asarray([resize(d[center[0]-128:center[0]+128,center[1]-128:center[1]+128],(endsize,endsize),preserve_range=True,anti_aliasing=True) for d in tqdm(amp_conv)])
print('normalizing')
amp_conv_red=np.asarray([(a-np.min(a))/(np.max(a)-np.min(a)) for a in tqdm(amp_conv_red)])

#%%
def create_center_mask(shape, center_radius=40):
    """Create a mask that excludes the central beam region"""
    y, x = np.ogrid[:shape[0], :shape[1]]
    center_y, center_x = shape[0]//2, shape[1]//2
    
    # Distance from center for each pixel
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create mask (True for pixels we want to keep)
    mask = dist_from_center > center_radius
    return mask


amp_conv_red_filtered=[]
mask=create_center_mask((endsize,endsize),center_radius=endsize//7)
for dp in amp_conv_red:
    sum = np.sum(dp*mask)
    if sum>10000:
        amp_conv_red_filtered.append(dp)
    else:
        continue
print(f"{len(amp_conv_red_filtered)} total diffraction patterns after filtering")

#%%
plt.imshow(amp_conv_red_filtered[random.randint(0,len(amp_conv_red_filtered)-1)]*mask)
plt.colorbar()
plt.show()
#%%
amp_conv_red=np.asarray(amp_conv_red_filtered)
    
    


#%%
NTEST = amp_conv_red.shape[0]//4
NTRAIN = amp_conv_red.shape[0]-NTEST
NVALID = NTEST//2 # NTRAIN//

print(NTRAIN,NTEST,NVALID)

EPOCHS = 128
NGPUS = torch.cuda.device_count()
BATCH_SIZE = NGPUS*16#*8
LR = NGPUS * 1e-3
print("GPUs:", NGPUS, "Batch size:", BATCH_SIZE, "Learning rate:", LR)

no_probe=True
H,W=amp_conv_red[0].shape[0],amp_conv_red[0].shape[1]
print(H,W)





#%%
tst_start = amp_conv_red.shape[0]-NTEST


X_train = amp_conv_red[:NTRAIN].reshape(-1,H,W)[:,np.newaxis,:,:]
X_test = amp_conv_red[tst_start:].reshape(-1,H,W)[:,np.newaxis,:,:]

ntrain=X_train.shape[0]
ntest=X_test.shape[0]

X_train = shuffle(X_train, random_state=0)

#Training data
print('train to tensor')
X_train_tensor = torch.Tensor(X_train)


#Test data
print('test to tensor')
X_test_tensor = torch.Tensor(X_test)


print(X_train_tensor.shape)


print('setting TensorDataset')
if no_probe:
    train_data = TensorDataset(X_train_tensor)
    test_data = TensorDataset(X_test_tensor)
else:
    train_data = TensorDataset(X_train_tensor)
    test_data = TensorDataset(X_test_tensor)


#N_TRAIN = X_combined_train_tensor.shape[0]
N_TRAIN = X_train_tensor.shape[0]

print('random split')
train_data2, valid_data = torch.utils.data.random_split(train_data,[N_TRAIN-NVALID,NVALID])
print(len(train_data2),len(train_data2[0]),len(valid_data),len(test_data))


train_dataV=train_data2
valid_dataV=valid_data
test_dataV=test_data


#download and load training data
print('train to DataLoader')
trainloader = DataLoader(train_dataV, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
print('valid to DataLoader')
validloader = DataLoader(valid_dataV, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

#same for test
#download and load training data
print('test to DataLoader')
testloader = DataLoader(test_dataV, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


























#%%
# Create a fixed probe kernel (e.g., an edge detection filter)
#probe_kernel = torch.tensor(probe, 
#    dtype=torch.float32
#)  # Shape: (out_channels, in_channels, kernel_height, kernel_width)
probe_kernel=probe
#probe_kernel=np.asarray(np.fft.fftshift(np.fft.fft2(probe)))
# Initialize the model
model = ConvAutoencoderSkip(probe_kernel)
#model=recon_model(probe_kernel)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

#Optimizer details
iterations_per_epoch = np.floor((NTRAIN-NVALID)/BATCH_SIZE)+1 #Final batch will be less than batch size
step_size = 6*iterations_per_epoch #Paper recommends 2-10 (6) number of iterations, step_size is half cycle
print(iterations_per_epoch)
print("LR step size is:", step_size, "which is every %d epochs" %(step_size/iterations_per_epoch))




def pearson_loss(output, target):
    """
    Compute 1 - Pearson correlation coefficient as a loss function.
    Args:
        output: Predicted values (B, C, H, W)
        target: Target values (B, C, H, W)
    Returns:
        loss: 1 - correlation (to minimize)
    """
    # Flatten the spatial dimensions
    output_flat = output.view(output.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # Mean of each image
    output_mean = output_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)
    
    # Centered variables
    output_centered = output_flat - output_mean
    target_centered = target_flat - target_mean
    
    # Correlation
    numerator = (output_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt((output_centered**2).sum(dim=1) * (target_centered**2).sum(dim=1))
    
    # Avoid division by zero
    correlation = numerator / (denominator + 1e-8)
    
    # Average over batch and convert to loss (1 - correlation)
    loss = 1 - correlation.mean()
    
    #Negaive pearson loss
    #loss = 1 + correlation.mean()
    
    return loss
#%%
# Modify the custom loss function to handle the scale difference
def custom_loss(output, target,decoded):
    #total_loss = pearson_loss(output, target)
    
    # Create central beam mask
    offset=(9,-6)
    h, w = output.shape[2:]
    y, x = torch.meshgrid(torch.arange(h, device=output.device), 
                         torch.arange(w, device=output.device))
    center_y, center_x = h // 2, w // 2
    r = torch.sqrt((x - (center_x+offset[0]))**2 + (y - (center_y+offset[1]))**2)
    
    # Create mask that de-emphasizes central beam (radius can be adjusted)
    central_beam_radius = 64  # adjust this based on your beam size
    beam_mask = (r > central_beam_radius).float()
    beam_mask = beam_mask.to(output.device)[None, None, :, :]
    
    # Apply mask to correlation loss
    conv_loss = pearson_loss(output * beam_mask, target * beam_mask)
    
    return conv_loss #total_loss


def custom_loss2(output, target, decoded):
    # Main correlation loss between convolved output and target
    conv_loss = pearson_loss(output, target)
    
    # Encourage sparsity (sharp peaks) in decoded image
    # Using a modified L1 loss that's less harsh on peaks
    peak_loss = torch.mean(torch.where(
        decoded > 0.1,  # For values above threshold
        0.1 * torch.abs(decoded),  # Small penalty for peaks
        torch.abs(decoded)  # Larger penalty for non-peak areas
    ))
    
    # Encourage local maxima to be sharp
    # Calculate local max in 3x3 neighborhoods
    max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    is_local_max = (decoded == max_pool(decoded))
    sharpness_loss = torch.mean(torch.where(
        is_local_max,
        -decoded,  # Encourage higher values at peaks
        torch.zeros_like(decoded)
    ))
    
    # Combine losses
    plc=1
    slc=plc/2
    total_loss = conv_loss + plc*peak_loss + slc*sharpness_loss
    #total_loss = peak_loss + sharpness_loss
    
    return total_loss

def custom_loss3(output, target, decoded):    
    # Add small epsilon to prevent numerical instability
    eps = 1e-6
    
    # # Create central beam mask
    # h, w = output.shape[2:]
    # y, x = torch.meshgrid(torch.arange(h, device=output.device), 
    #                      torch.arange(w, device=output.device))
    # center_y, center_x = h // 2, w // 2
    # r = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # # Create mask that de-emphasizes central beam (radius can be adjusted)
    # central_beam_radius = 32  # adjust this based on your beam size
    # beam_mask = (r > central_beam_radius).float()
    # beam_mask = beam_mask.to(output.device)[None, None, :, :]
    
    # # Apply mask to correlation loss
    # conv_loss = pearson_loss(output * beam_mask, target * beam_mask)
    
    # Main correlation loss between convolved output and target
    conv_loss = pearson_loss(output, target)
    
    # Add small epsilon to prevent numerical instability
    eps = 1e-6
    
    # Center distance map for radial weighting
    h, w = decoded.shape[2:]
    y, x = torch.meshgrid(torch.arange(h, device=decoded.device), 
                         torch.arange(w, device=decoded.device))
    center_y, center_x = h // 2, w // 2
    r = torch.sqrt((x - center_x)**2 + (y - center_y)**2 + eps)
    
    
    
    # 1. Encourage centro-symmetry in decoded image
    flipped = torch.flip(decoded, [-2, -1])
    symmetry_loss = F.mse_loss(decoded, flipped)
    
    # 2. Higher-q peaks should be weaker (with numerical stability)
    radial_weight = torch.exp(-r / (h/4))
    radial_loss = torch.mean(decoded * (1 - radial_weight)[None, None, :, :])
    
    # 3. Peaks should be sharp
    # Use Sobel filters for gradient calculation
    sobel_x = torch.tensor([[-1, 0, 1], 
                           [-2, 0, 2], 
                           [-1, 0, 1]], 
                           device=decoded.device,
                           dtype=decoded.dtype).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor([[-1, -2, -1], 
                           [0, 0, 0], 
                           [1, 2, 1]], 
                           device=decoded.device,
                           dtype=decoded.dtype).view(1, 1, 3, 3)
    
    # Calculate gradients using convolution
    pad = nn.ReplicationPad2d(1)
    decoded_pad = pad(decoded)
    dx = F.conv2d(decoded_pad, sobel_x)
    dy = F.conv2d(decoded_pad, sobel_y)
    
    # Detect peaks (with numerical stability)
    max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    is_peak = (torch.abs(decoded - max_pool(decoded)) < eps)
    
    # Encourage high gradients at peaks (with numerical stability)
    gradient_magnitude = torch.sqrt(dx**2 + dy**2 + eps)
    sharpness_loss = -torch.mean(gradient_magnitude * is_peak.float())
    
    # 4. Background should be close to zero
    background_mask = ~is_peak
    background_loss = torch.mean(torch.abs(decoded * background_mask.float() + eps))
    
    # Clip losses to prevent extreme values
    conv_loss = torch.clamp(conv_loss, -100, 100)
    symmetry_loss = torch.clamp(symmetry_loss, -100, 100)
    radial_loss = torch.clamp(radial_loss, -100, 100)
    sharpness_loss = torch.clamp(sharpness_loss, -100, 100)
    background_loss = torch.clamp(background_loss, -100, 100)
    
    # Combine losses with smaller weights to start
    c=3
    total_loss = (conv_loss + 
                  0.05*c * symmetry_loss +
                  0.02*c * radial_loss +
                  0.05*c * sharpness_loss +
                  0.1*c * background_loss)
    
    return total_loss



# Update optimizer parameters
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)  # Lower learning rate
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LR/10, max_lr=LR, step_size_up=step_size,
                                              cycle_momentum=False, mode='triangular2')
# Print model summary
print(model)




#%%
def train(trainloader, metrics):
    tot_loss = 0.0
    
    for i, ft_images in tqdm(enumerate(trainloader)):
        ft_images = ft_images[0].to(device)
        decoded, probe_convolved = model(ft_images)
        
        optimizer.zero_grad()
         
        loss = custom_loss(probe_convolved, ft_images,decoded)
        
        loss.backward()
        optimizer.step()

        tot_loss += loss.detach().item()
        plot_random=True
        if plot_random:
            # Plot a random sample from the first batch of each epoch
            if i == 0:
                # Select random index from batch
                rand_idx = random.randint(0, ft_images.shape[0]-1)
                
                # Create figure with 4 subplots
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
                
                # Plot input image
                im1 = ax1.imshow(ft_images[rand_idx, 0].cpu().detach().numpy())
                ax1.set_title('Input Image')
                plt.colorbar(im1, ax=ax1)
                
                # Plot decoded image
                im2 = ax2.imshow(decoded[rand_idx, 0].cpu().detach().numpy())
                ax2.set_title('Decoded Image')
                plt.colorbar(im2, ax=ax2)
                
                # Plot probe convolved image
                im3 = ax3.imshow(probe_convolved[rand_idx, 0].cpu().detach().numpy())
                ax3.set_title('Probe Convolved')
                plt.colorbar(im3, ax=ax3)
                
                im4 = ax4.imshow(ft_images[rand_idx, 0].cpu().detach().numpy()-probe_convolved[rand_idx, 0].cpu().detach().numpy())
                ax4.set_title('Difference')
                plt.colorbar(im4, ax=ax4)
                
                plt.tight_layout()
                plt.show()
                    
        scheduler.step()
        metrics['lrs'].append(scheduler.get_last_lr())
        
    metrics['losses'].append([tot_loss/i])

def validate(validloader, metrics):
    tot_val_loss = 0.0
    for j, ft_images in enumerate(validloader):
        ft_images = ft_images[0].to(device)
        decoded, probe_convolved = model(ft_images)
         
        val_loss = custom_loss(probe_convolved, ft_images,decoded)
    
        tot_val_loss += val_loss.detach().item()
    metrics['val_losses'].append([tot_val_loss/j])




#%%
metrics = {'losses':[],'val_losses':[], 'lrs':[], 'best_val_loss' : np.inf}
for epoch in range (EPOCHS):

    #Set model to train mode
    model.train() 
    #Training loop
    train(trainloader,metrics)

    #Switch model to eval mode
    model.eval()

    #Validation loop
    validate(validloader,metrics)
    #validate2(validloader,metrics)  
    print('Epoch: %d | Total  | Train Loss: %.5f | Val Loss: %.5f' %(epoch, metrics['losses'][-1][0], metrics['val_losses'][-1][0]))
    print('Epoch: %d | Ending LR: %.6f ' %(epoch, metrics['lrs'][-1][0])) 





#%%
batches = np.linspace(0,len(metrics['lrs']),len(metrics['lrs'])+1)
epoch_list = batches/iterations_per_epoch

plt.plot(epoch_list[1:],metrics['lrs'], 'C3-')
plt.grid()
plt.ylabel("Learning rate")
plt.xlabel("Epoch")


losses_arr = np.array(metrics['losses'])
val_losses_arr = np.array(metrics['val_losses'])
losses_arr.shape
fig, ax = plt.subplots(1,sharex=True, figsize=(15, 8))
ax.plot(losses_arr[:,0], 'C3o-', label = "Total Train loss")
ax.plot(val_losses_arr[:,0], 'C0o-', label = "Total Val loss")
ax.set(ylabel='Loss')
ax.grid()
ax.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.xlabel("Epochs")
plt.show()


#%%
model.eval()
results = []
results_pc=[]
for i, test in enumerate(testloader):
    tests = test[0].to(device)
    result_d,result_pc = model(tests)
    for j in range(tests.shape[0]):
        results.append(result_d[j].detach().to("cpu").numpy())
        results_pc.append(result_pc[j].detach().to("cpu").numpy())
        
results = np.array(results).squeeze()
results_pc = np.array(results_pc).squeeze()




#%%
h,w = H,W
ntest=results.shape[0]
plt.figure()
n = 5
f,ax=plt.subplots(4,n,figsize=(15, 12))
plt.gcf().text(0.02, 0.8, "Input", fontsize=20)
plt.gcf().text(0.02, 0.6, "Output", fontsize=20)
plt.gcf().text(0.02, 0.4, "PC", fontsize=20)
plt.gcf().text(0.02, 0.2, "Difference", fontsize=20)

for i in range(0,n):
    j=int(round(np.random.rand()*ntest))

    # display FT
    im=ax[0,i].imshow(X_test[j].reshape(h, w))
    plt.colorbar(im, ax=ax[0,i], format='%.2f')
    ax[0,i].get_xaxis().set_visible(False)
    ax[0,i].get_yaxis().set_visible(False)
    
    # display predicted intens
    im=ax[1,i].imshow(results[j].reshape(h, w))
    plt.colorbar(im, ax=ax[1,i], format='%.2f')
    ax[1,i].get_xaxis().set_visible(False)
    ax[1,i].get_yaxis().set_visible(False)

    
    #Probe convolved
    im=ax[2,i].imshow(results_pc[j].reshape(h, w))
    plt.colorbar(im, ax=ax[2,i], format='%.2f')
    ax[2,i].get_xaxis().set_visible(False)
    ax[2,i].get_yaxis().set_visible(False)
    
    #Difference in amplitude
    im=ax[3,i].imshow(X_test[j].reshape(h, w)-results_pc[j].reshape(h, w))
    plt.colorbar(im, ax=ax[3,i], format='%.2f')
    ax[3,i].get_xaxis().set_visible(False)
    ax[3,i].get_yaxis().set_visible(False)
plt.show()








# %%
def azimuthal_average(image, center=None):
    # Get image dimensions
    y, x = np.indices(image.shape)
    
    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    
    # Calculate radius for each pixel
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int32)
    
    # Get sorted radii
    tbin = np.bincount(r.ravel(), weights=image.ravel())
    nr = np.bincount(r.ravel())
    
    # Avoid division by zero
    radialprofile = np.zeros_like(tbin)
    mask = nr > 0
    radialprofile[mask] = tbin[mask] / nr[mask]
    
    return radialprofile

h, w = H, W
ntest = results.shape[0]
n = 5

# Create figure with both 2D images and 1D profiles
f, ax = plt.subplots(5, n, figsize=(20, 20), gridspec_kw={'height_ratios': [1, 1, 1, 1, 1.2]})
plt.gcf().text(0.02, 0.85, "Input", fontsize=20)
plt.gcf().text(0.02, 0.67, "Output", fontsize=20)
plt.gcf().text(0.02, 0.49, "PC", fontsize=20)
plt.gcf().text(0.02, 0.31, "Difference", fontsize=20)
plt.gcf().text(0.02, 0.13, "Radial Profiles", fontsize=20)

for i in range(0, n):
    j = int(round(np.random.rand()*ntest))
    
    # Get images
    input_img = X_test[j].reshape(h, w)
    output_img = results[j].reshape(h, w)
    pc_img = results_pc[j].reshape(h, w)
    diff_img = input_img - pc_img
    
    # Create circle coordinates
    theta = np.linspace(0, 2*np.pi, 100)
    radius = 160
    circle_x = center[0]+9 + radius * np.cos(theta)
    circle_y = center[1]-6 + radius * np.sin(theta)
    
    # 2D plots with circles
    for row in range(4):
        im = ax[row,i].imshow([input_img, output_img, pc_img, diff_img][row])
        ax[row,i].plot(circle_x, circle_y, 'r--', linewidth=1, alpha=0.8)
        ax[row,i].get_xaxis().set_visible(False)
        ax[row,i].get_yaxis().set_visible(False)
    
    
    # 2D plots
    im = ax[0,i].imshow(input_img)
    plt.colorbar(im, ax=ax[0,i], format='%.2f')
    ax[0,i].get_xaxis().set_visible(False)
    ax[0,i].get_yaxis().set_visible(False)
    
    im = ax[1,i].imshow(output_img)
    plt.colorbar(im, ax=ax[1,i], format='%.2f')
    ax[1,i].get_xaxis().set_visible(False)
    ax[1,i].get_yaxis().set_visible(False)
    
    im = ax[2,i].imshow(pc_img)
    plt.colorbar(im, ax=ax[2,i], format='%.2f')
    ax[2,i].get_xaxis().set_visible(False)
    ax[2,i].get_yaxis().set_visible(False)
    
    im = ax[3,i].imshow(diff_img)
    plt.colorbar(im, ax=ax[3,i], format='%.2f')
    ax[3,i].get_xaxis().set_visible(False)
    ax[3,i].get_yaxis().set_visible(False)
    
    # Calculate radial profiles
    center = np.array([w/2, h/2])
    r_input = azimuthal_average(input_img, center)
    r_output = azimuthal_average(output_img, center)
    r_pc = azimuthal_average(pc_img, center)
    r_diff = azimuthal_average(diff_img, center)
    
    # Plot radial profiles
    radii = np.arange(len(r_input))
    #ax[4,i].plot(radii, r_input, 'b-', label='Input')
    ax[4,i].plot(radii, r_output, 'r--', label='Output')
    #ax[4,i].plot(radii, r_pc, 'g:', label='PC')
    #ax[4,i].plot(radii, r_diff, 'k-', label='Difference')
    # Add vertical line at r=128
    ax[4,i].axvline(x=radius, color='r', linestyle='--', alpha=0.8)
    if i == 0:  # Only show legend for first plot
        ax[4,i].legend()
    ax[4,i].set_xlabel('Radius (pixels)')
    ax[4,i].set_ylabel('Intensity')
    ax[4,i].grid(True)

plt.tight_layout()
plt.show()
# %%
