#!/usr/bin/env python
# coding: utf-8

# In[1]:


############################################

import os,time,scipy.io
import rawpy
import glob
import numpy as np
import scipy.misc
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms


class SeeInDarkUNet(nn.Module):
    
    
    #initialize network
    def __init__(self, num_classes=10):
        super(SeeInDarkUNet, self).__init__()
        
        #self.leakyRelu = nn.LeakyReLU(0.2, inplace=True);
        
        # compress block-1
        self.conv1_1 = nn.Conv2d(4, 32, kernel_size = 3, stride = 1, padding = 1);
        self.conv1_2 = nn.Conv2d(32,32,kernel_size = 3, stride = 1, padding = 1);
        self.mxpool1 = nn.MaxPool2d(kernel_size=2);
        
        # compress block-2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1);
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1);
        self.mxpool2 = nn.MaxPool2d(kernel_size=2);

        # compress block-3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1);
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1);
        self.mxpool3 = nn.MaxPool2d(kernel_size=2);

        # compress block-4
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1);
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1);
        self.mxpool4 = nn.MaxPool2d(kernel_size=2);
        
        # expand block- 5
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1);
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1);
        
        
        # expand block- 6
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2);
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size = 3, stride = 1, padding = 1);
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1);

        # expand block- 7
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2);
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size = 3, stride = 1, padding = 1);
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1);

        # expand block- 8
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2);
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size = 3, stride = 1, padding = 1);
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1);

        
        # expand block- 9
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size = 2, stride = 2);
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size = 3, stride = 1, padding = 1);
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1);
        
        #final block-10
        self.finalLayer = nn.Conv2d(32, 12, kernel_size = 1, stride = 1);
        
    
    #forward U-Net
    def forward(self, x):
        
        block1 = self.leakyRelu(self.conv1_1(x))
        block1 = self.leakyRelu(self.conv1_2(block1))
        block1_pool1 = self.mxpool1(block1)
        
        
        block2 = self.leakyRelu(self.conv2_1(block1_pool1));
        block2 = self.leakyRelu(self.conv2_2(block2));
        block2_pool2 = self.mxpool2(block2);
        
        block3 = self.leakyRelu(self.conv3_1(block2_pool2));
        block3 = self.leakyRelu(self.conv3_2(block3));
        block3_pool3 = self.mxpool3(block3);

        block4 = self.leakyRelu(self.conv4_1(block3_pool3));
        block4 = self.leakyRelu(self.conv4_2(block4));
        block4_pool4 = self.mxpool4(block4);
        
        block5 = self.leakyRelu(self.conv5_1(block4_pool4));
        block5 = self.leakyRelu(self.conv5_2(block5));
        
        
        
        block6_up6 = torch.cat([self.up6(block5), block4], 1);
        block6 = self.leakyRelu(self.conv6_1(block6_up6));
        block6 = self.leakyRelu(self.conv6_2(block6));

        block7_up7 = torch.cat([self.up7(block6), block3], 1);
        block7 = self.leakyRelu(self.conv7_1(block7_up7));
        block7 = self.leakyRelu(self.conv7_2(block7));

        block8_up8 = torch.cat([self.up8(block7), block2], 1);
        block8 = self.leakyRelu(self.conv8_1(block8_up8));
        block8 = self.leakyRelu(self.conv8_2(block8));

        block9_up9 = torch.cat([self.up9(block8), block1],1);
        block9 = self.leakyRelu(self.conv9_1(block9_up9));
        block9 = self.leakyRelu(self.conv9_2(block9));
        
        
        out = self.finalLayer(block9);
        output = F.pixel_shuffle(out, 2);
        
        return output;
    
           
    def leakyRelu(self, x):
        return torch.max(0.2*x, x);

    def init_weights(self):
        
        for i in self.modules():
            if isinstance(i, nn.ConvTranspose2d):
                i.weight.data.normal_(0.0, 0.02)
                
            if isinstance(i, nn.Conv2d):
                i.weight.data.normal_(0.0, 0.02)
                if i.bias is not None:
                    i.bias.data.normal_(0.0, 0.02)



#############################################

# Training Model


# define paths
input_dir = './dataset/Sony/short/'
golden_dir = './dataset/Sony/long/'
model_dir = './model/Sony/'
result_dir = './result/Sony/'

# sony raw-info
blackLevel = 512;
bitsPerPixel = 14;
maxPixelVal = (2**bitsPerPixel - 1);
minPixelVal = 0;

# Load Training Input Data
train_input_fns = glob.glob(input_dir + '0*.ARW');
train_input_ids = [];
for i in range(len(train_input_fns)):
    _, train_input_fn = os.path.split(train_input_fns[i]);
    train_input_ids.append(int(train_input_fn[0:5]));
    

# Load Training Golden Data
golden_input_fns = glob.glob(golden_dir + '0*.ARW');
golden_input_ids = [];
for i in range(len(golden_input_fns)):
    _, golden_input_fn = os.path.split(golden_input_fns[i]);
    golden_input_ids.append(int(golden_input_fn[0:5]));

    
save_freq = 100;

DEBUG = 1
if DEBUG == 1:
    save_freq = 10
    train_input_ids = train_input_ids[0:100]
    golden_input_ids = golden_input_ids[0:100]
    

# Preprocess RAW (black-level subtraction + PACK + Amplification)
def preProcessRaw(raw):
    # convert to float
    image = raw.raw_image_visible.astype(np.float32); 
    
    # subtract black-level
    image_BLC = np.maximum((image - blackLevel), minPixelVal)/(maxPixelVal - blackLevel);
    
    image_BLC = np.expand_dims(image_BLC, axis = 2);
    
    # pack raw into 4-channels
    height = image_BLC.shape[0];
    width = image_BLC.shape[1];
    
    image_BLC_PCK = np.concatenate((image_BLC[0:height:2,0:width:2,:], image_BLC[0:height:2,1:width:2,:],
                                    image_BLC[1:height:2,0:width:2,:], image_BLC[1:height:2,1:width:2,:]), axis=2);
    
    return image_BLC_PCK;


# costfunction
def costFunc(output, golden):
    return torch.abs(output - golden).mean()
    #np.((output - golden)**2)

    
# Store data in memory
golden_set = [None]*len(train_input_ids);
input_set = {};
input_set['100'] = [None]*len(train_input_ids);
input_set['250'] = [None]*len(train_input_ids);
input_set['300'] = [None]*len(train_input_ids);

#golden_image = {};
#input_image = {};

g_loss = np.zeros(len(train_input_ids));


# Train the model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
learning_rate = 1e-4;
model = SeeInDarkUNet().to(device);
model.init_weights();
model_params = list(model.parameters())
opt = optim.Adam(model_params, lr = learning_rate);

numEpoch = 400;
for epoch in range(1,numEpoch):
    print(epoch);
    if(epoch > 200):
        for m in opt.param_groups:
            m['lr'] = 1e-5;
        
    for ind in np.random.permutation(len(train_input_ids)):
        training_id = train_input_ids[ind];
        input_files = glob.glob(input_dir + '%05d_00*.ARW' % training_id);
        input_path = input_files[np.random.random_integers(0, (len(input_files) - 1))];
        input_fileName = os.path.basename(input_path);
        
        golden_filepath = glob.glob(golden_dir + '%05d_00*.ARW' % training_id);
        golden_fileName = os.path.basename(golden_filepath[0]);
        
        
        input_exposure = float(input_fileName[9:-5]);
        golden_exposure = float(golden_fileName[9:-5]);
        ratio = min(golden_exposure/input_exposure, 300);
        
        #if 0 :
            #if input_set[str(ratio)[0:3]][ind] is None :
            #imraw = rawpy.imread(input_dir + input_fileName);
            #input_image[str(ratio)[0:3]] = ratio * np.expand_dims(preProcessRaw(imraw), axis = 0);
            
            #golden_raw = rawpy.imread(golden_dir + golden_fileName);
            #golden_process = golden_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16);
            #golden_image = np.expand_dims(np.float32(golden_process/65535.0), axis = 0);
            
            #h = input_image[str(ratio)[0:3]].shape[1]
            #w = input_image[str(ratio)[0:3]].shape[2]
            #ps = 512;
            #xx = np.random.randint(0, w - ps);
            #yy = np.random.randint(0, h - ps);
            
            #input_patch = input_image[str(ratio)[0:3]][ :, yy:yy + ps, xx:xx + ps, : ];
            #golden_patch = golden_image[:, yy*2 : yy*2 + 2*ps , xx*2 : xx*2 + 2*ps, :];
            
        if 1 :
            if input_set[str(ratio)[0:3]][ind] is None :
                imraw = rawpy.imread(input_dir + input_fileName);
                input_set[str(ratio)[0:3]][ind] = ratio * np.expand_dims(preProcessRaw(imraw), axis = 0);
                
                golden_raw = rawpy.imread(golden_dir + golden_fileName);
                golden_process = golden_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16);
                golden_set[ind] = np.expand_dims(np.float32(golden_process/65535.0), axis = 0);
                
                
                h = input_set[str(ratio)[0:3]][ind].shape[1]
                w = input_set[str(ratio)[0:3]][ind].shape[2]
                
                ps = 512;
                
                xx = np.random.randint(0, w - ps);
                yy = np.random.randint(0, h - ps);
                
                input_patch = input_set[str(ratio)[0:3]][ind][ :, yy:yy + ps, xx:xx + ps, : ];
                golden_patch = golden_set[ind][:, yy*2 : yy*2 + 2*ps , xx*2 : xx*2 + 2*ps, :];
        
        
        
        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1);
            golden_patch = np.flip(golden_patch, axis=1);
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2);
            golden_patch = np.flip(golden_patch, axis=2);
            
        input_patch = np.minimum(input_patch,1.0);
        golden_patch = np.maximum(golden_patch, 0.0);
        
        
        input_tensor = torch.from_numpy(input_patch).permute(0,3,1,2).to(device);
        golden_tensor = torch.from_numpy(golden_patch).permute(0,3,1,2).to(device);
        
        # forward U-Net
        model.zero_grad();
        output_tensor = model(input_tensor);
        
        
        loss = costFunc(golden_tensor, output_tensor);
        loss.backward();
        
        opt.step();
        g_loss[ind] = loss.data;
        
        
        if epoch%save_freq==0:
            print(epoch);
            print(g_loss[ind]);
            if not os.path.isdir(result_dir + '%04d'%epoch):
                os.makedirs(result_dir + '%04d'%epoch)
            #output = output_tensor.data.numpy()
            output = output_tensor.permute(0, 2, 3, 1).cpu().data.numpy()
            output = np.minimum(np.maximum(output,0),1)
            
            temp = np.concatenate((golden_patch[0,:,:,:], output[0,:,:,:]),axis=1)
            #scipy.misc.toimage(temp*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%04d/%05d_00_train_%d.jpg'%(epoch,training_id,ratio))
            im = Image.fromarray(np.uint8(temp*255))
            im.save(result_dir + '%04d/%05d_00_train_%d.jpg'%(epoch,training_id,ratio))
            torch.save(model.state_dict(), model_dir+'checkpoint_sony_e%04d.pth'%epoch)
        
        


# In[ ]:




