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
#%%
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../NN/ptychosaxsNN/'))) 
import utils.ptychosaxsNN_utils as ptNN_U
import ptychosaxsNN.ptychosaxsNN as ptNN
importlib.reload(ptNN_U)
importlib.reload(ptNN)
# #%%
# # Define the Convolutional Autoencoder with Skip Connections
# class ConvAutoencoderSkip(nn.Module):
#     def __init__(self, probe_kernel):
#         super(ConvAutoencoderSkip, self).__init__()

#         # Encoder
#         self.enc1 = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU()
#         )
#         self.pool1 = nn.MaxPool2d(2, 2)

#         self.enc2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU()
#         )
#         self.pool2 = nn.MaxPool2d(2, 2)
        
        
#         self.enc3 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU()
#         )
#         self.pool3 = nn.MaxPool2d(2, 2)

#         # Bottleneck
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU()
#         )
        
#         # Decoder with Skip Connections
#         self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
#         self.dec1 = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU()
#         )

#         # Decoder with Skip Connections
#         self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
#         self.dec2 = nn.Sequential(
#             nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU()
#         )

#         self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0)
#         self.dec3 = nn.Sequential(
#             nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU()
#         )

#         # Final reconstruction layer
#         self.final_layer = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
#         self.sigmoid = nn.Sigmoid()

#         # Probe Convolution (Fixed Kernel)
#         self.register_buffer("probe_kernel", probe_kernel)

#     def forward(self, x):
#         # Encoder
#         enc1_out = self.enc1(x)
#         enc1_pooled = self.pool1(enc1_out)

#         enc2_out = self.enc2(enc1_pooled)
#         enc2_pooled = self.pool2(enc2_out)

#         enc3_out = self.enc3(enc2_pooled)
#         enc3_pooled = self.pool3(enc3_out)

#         # Bottleneck
#         bottleneck_out = self.bottleneck(enc3_pooled)

#         # Decoder with Skip Connections
#         up1_out = self.up1(bottleneck_out)
#         dec1_out = self.dec1(torch.cat([up1_out, enc3_out], dim=1))

#         up2_out = self.up2(dec1_out)
#         dec2_out = self.dec2(torch.cat([up2_out, enc2_out], dim=1))

#         up3_out = self.up3(dec2_out)
#         dec3_out = self.dec3(torch.cat([up3_out, enc1_out], dim=1))

#         # Final output
#         decoded = self.sigmoid(self.final_layer(dec3_out))

#         # Apply probe convolution
#         kernel_size = self.probe_kernel.shape[0]
#         pad_size = kernel_size // 2  # Adjusted padding calculation
        
#         # Flip the kernel (to match scipy.convolve2d behavior)
#         flipped_kernel = torch.flip(self.probe_kernel, [0, 1])
        
#         # Calculate total padding needed
#         total_pad = kernel_size - 1
#         left_pad = total_pad // 2
#         right_pad = total_pad - left_pad
        
#         # Apply convolution with proper padding to maintain exact size
#         probe_convolved_output = F.conv2d(F.pad(decoded, (left_pad, right_pad, left_pad, right_pad), mode='reflect'),flipped_kernel.unsqueeze(0).unsqueeze(0))
        
        
#         probe_convolved_output = (probe_convolved_output - probe_convolved_output.min()) / (probe_convolved_output.max() - probe_convolved_output.min() + 1e-8)
    
#         # Verify shapes match
#         assert probe_convolved_output.shape == decoded.shape, f"Shape mismatch: {probe_convolved_output.shape} vs {decoded.shape}"
        
#         return decoded, probe_convolved_output


#%%

class ConvAutoencoderSkip(nn.Module):
    def __init__(self, probe_kernel):
        super(ConvAutoencoderSkip, self).__init__()

        # Reduce channel sizes
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # 16->8
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride=1, padding=(1,1)),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # 32->16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=(1,1)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 64->32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 128->64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 256->128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Decoder with Skip Connections (reduced channels)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0)
        self.dec3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=(1,1)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.up4 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=0)
        self.dec4 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride=1, padding=(1,1)),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        # Final reconstruction layer
        self.final_layer = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

        # Probe FFT optimization
        self.register_buffer("probe_kernel", probe_kernel)
        # Get kernel size before padding
        self.kernel_size = probe_kernel.shape[0]
        # Center and pad kernel to match input size
        padded_kernel = F.pad(probe_kernel.unsqueeze(0).unsqueeze(0), 
                            (0, 256 - self.kernel_size, 0, 256 - self.kernel_size))
        # Apply FFT shift before computing FFT
        padded_kernel = torch.fft.fftshift(padded_kernel)
        kernel_fft = torch.fft.rfft2(padded_kernel)
        self.register_buffer("kernel_fft", kernel_fft)

    def fft_conv2d(self, x):
        # Center input before FFT
        x_centered = torch.fft.fftshift(x)
        # Compute FFT of input
        x_fft = torch.fft.rfft2(x_centered)
        
        # Multiply in frequency domain (element-wise)
        output_fft = x_fft * self.kernel_fft
        
        # Inverse FFT and shift back
        output = torch.fft.irfft2(output_fft)
        output = torch.fft.ifftshift(output)
        
        # Normalize
        batch_min = output.view(output.size(0), -1).min(dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        batch_max = output.view(output.size(0), -1).max(dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        output = (output - batch_min) / (batch_max - batch_min + 1e-8)
        
        return output

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        enc1_pooled = self.pool1(enc1_out)

        enc2_out = self.enc2(enc1_pooled)
        enc2_pooled = self.pool2(enc2_out)

        enc3_out = self.enc3(enc2_pooled)
        enc3_pooled = self.pool3(enc3_out)
        
        enc4_out = self.enc4(enc3_pooled)
        enc4_pooled = self.pool4(enc4_out)

        # Bottleneck
        bottleneck_out = self.bottleneck(enc4_pooled)

        # Decoder with Skip Connections
        up1_out = self.up1(bottleneck_out)
        dec1_out = self.dec1(torch.cat([up1_out, enc4_out], dim=1))

        up2_out = self.up2(dec1_out)
        dec2_out = self.dec2(torch.cat([up2_out, enc3_out], dim=1))

        up3_out = self.up3(dec2_out)
        dec3_out = self.dec3(torch.cat([up3_out, enc2_out], dim=1))

        up4_out = self.up4(dec3_out)
        dec4_out = self.dec4(torch.cat([up4_out, enc1_out], dim=1))

        # Final output
        decoded = self.sigmoid(self.final_layer(dec4_out))

        # Use FFT-based convolution instead of spatial convolution
        probe_convolved_output = self.fft_conv2d(decoded)
        
        return decoded, probe_convolved_output



#%%
#Zhihua probe
probe=loadmat("/net/micdata/data2/12IDC/2024_Dec/results/JM02_3D_/fly482/roi2_Ndp1024/MLc_L1_p10_gInf_Ndp256_mom0.5_pc100_noModelCon_bg0.1_vi_mm/MLc_L1_p10_g400_Ndp512_mom0.5_pc400_noModelCon_bg0.1_vp4_vi_mm/Niter1000.mat")['probe'].T[0][0].T
print(probe.shape)
plt.imshow(np.abs(probe))
plt.colorbar()
plt.show()
dpsize=512
# probe=resize(probe,(dpsize,dpsize),preserve_range=True,anti_aliasing=True)
# plt.imshow(probe)
# plt.colorbar()
# plt.show()
probe=np.asarray(np.fft.fftshift(np.fft.fft2(probe)))
plt.imshow(np.abs(probe))
plt.colorbar()
plt.show()
endsize=256
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
for scan in np.arange(578,581):
    dps = ptNN_U.load_h5_scan_to_npy(Path(f'/net/micdata/data2/12IDC/2024_Dec/ptycho/'),scan,plot=False)
    all_dps.append(dps)
temp_dps=np.asarray(all_dps)
#%%
dps=temp_dps.reshape(-1,temp_dps.shape[2],temp_dps.shape[3])
center=np.array([dps.shape[1]//2-100,dps.shape[2]//2])
ri=random.randint(0,dps.shape[0]-1)
test=dps[ri][center[0]-dpsize//2:center[0]+dpsize//2,center[1]-dpsize//2:center[1]+dpsize//2]
plt.imshow(test,norm=colors.LogNorm())
plt.colorbar()
plt.show()
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
print('resizing')
amp_conv_red=np.asarray([resize(d[center[0]-256:center[0]+256,center[1]-256:center[1]+256],(endsize,endsize),preserve_range=True,anti_aliasing=True) for d in tqdm(amp_conv)])
print('normalizing')
amp_conv_red=np.asarray([(a-np.min(a))/(np.max(a)-np.min(a)) for a in tqdm(amp_conv_red)])


#%%
plt.imshow(amp_conv_red[620],norm=colors.LogNorm())
plt.colorbar()
plt.show()

#%%
NTEST = conv_DPs.shape[0]//4
NTRAIN = conv_DPs.shape[0]-NTEST
NVALID = NTEST//2 # NTRAIN//

print(NTRAIN,NTEST,NVALID)

EPOCHS = 256
NGPUS = torch.cuda.device_count()
BATCH_SIZE = NGPUS*16#8
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
probe_kernel = torch.tensor(probe, 
    dtype=torch.float32
)  # Shape: (out_channels, in_channels, kernel_height, kernel_width)

# Initialize the model
model = ConvAutoencoderSkip(probe_kernel)

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

# Modify the custom loss function to handle the scale difference
def custom_loss(output, target):
    # # Spatial domain loss
    # mse_loss = F.mse_loss(output, target)
    # # Frequency domain loss
    # #output_fft = torch.fft.rfft2(output)
    # #target_fft = torch.fft.rfft2(target)
    # #fft_loss = F.mse_loss(torch.abs(output_fft), torch.abs(target_fft))
    
    # # Combine losses
    # total_loss = mse_loss# + 0.1 * fft_loss
    
    total_loss = pearson_loss(output, target)
    return total_loss

# Update optimizer parameters
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)  # Lower learning rate
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
         
        loss = custom_loss(probe_convolved, ft_images)
        
        loss.backward()
        optimizer.step()

        tot_loss += loss.detach().item()
        
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
         
        val_loss = custom_loss(probe_convolved, ft_images)
    
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
#model_new.eval() #imp when have dropout etc
results = []
for i, test in enumerate(testloader):
    tests = test[0].to(device)
    result = model(tests)
    for j in range(tests.shape[0]):
        results.append(result[0][j].detach().to("cpu").numpy())
        
results = np.array(results).squeeze()

#%%
h,w = H,W
ntest=results.shape[0]
plt.figure()
n = 5
f,ax=plt.subplots(3,n,figsize=(15, 12))
plt.gcf().text(0.02, 0.8, "Input", fontsize=20)
plt.gcf().text(0.02, 0.6, "True I", fontsize=20)
plt.gcf().text(0.02, 0.2, "Difference I", fontsize=20)

for i in range(0,n):
    j=int(round(np.random.rand()*ntest))

    # display FT
    im=ax[0,i].imshow(X_test[j].reshape(h, w))#,norm=colors.LogNorm())
    plt.colorbar(im, ax=ax[0,i], format='%.2f')
    ax[0,i].get_xaxis().set_visible(False)
    ax[0,i].get_yaxis().set_visible(False)
    
    # display predicted intens
    im=ax[1,i].imshow(results[j].reshape(h, w))#,norm=colors.LogNorm())
    plt.colorbar(im, ax=ax[1,i], format='%.2f')
    ax[1,i].get_xaxis().set_visible(False)
    ax[1,i].get_yaxis().set_visible(False)

    #Difference in amplitude
    im=ax[2,i].imshow(X_test[j].reshape(h, w)-results[j].reshape(h, w))#,norm=colors.LogNorm())
    plt.colorbar(im, ax=ax[2,i], format='%.2f')
    ax[2,i].get_xaxis().set_visible(False)
    ax[2,i].get_yaxis().set_visible(False)
plt.show()

# %%
