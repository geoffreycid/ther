#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:39:14 2019

@author: geoffreycideron
"""
"""
This script contains the DQN model
"""

import torch
import torch.nn as nn

class DQN(nn.Module):
    
    def __init__(self, h, w, frames, n_actions):
        """
        h: height of the screen
        w: width of the screen
        frames: number of frames taken into account for the state
        n_actions: number of actions
        """
        super(DQN, self).__init__()

        def size_after_conv(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
            
        self.net = nn.Sequential(
                nn.Conv2d(in_channels=3*frames, out_channels=24, kernel_size=3),
                nn.ReLU(True),
                nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3),
                nn.ReLU(True),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
                )
        
        output_conv_h = size_after_conv(size_after_conv(size_after_conv(h, kernel_size=3, stride=1)))
        output_conv_w = size_after_conv(size_after_conv(size_after_conv(w, kernel_size=3, stride=1)))
        in_linear = 32 * output_conv_h * output_conv_w
        
        self.head = nn.Linear(in_features=in_linear, out_features=n_actions)
        
    def forward(self, x):
        out = self.net(x)
        return self.head(out.view(out.size(0), -1))

#%%
#class Preprocessing():
#    
#    def __init__(self):
#        self.transformations = T.Compose([
#            # To a pytorch tensor
#            T.ToTensor()
#            ])
#    
#    def batch_transform(self, imgs):
#        if imgs.ndim==3:
#            imgs = imgs.reshape(-1, imgs.shape[0], imgs.shape[1], imgs.shape[2])
#        imgs_preprocessed = []
#       for img in imgs:
#            imgs_preprocessed.append(self.transformations(img))
#        
#        return torch.stack(imgs_preprocessed)

#%%

#prepro = Preprocessing()
#DQNetwork = DQN(8,8,4)


#imgs = np.random.randint(0,10,size=(16,8,8,3)).astype("uint8")
#imgs_prepro = prepro.batch_transform(imgs)
#DQNetwork(imgs_prepro).shape
