#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
        
#        print(block9.size());
        
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



# In[ ]:




