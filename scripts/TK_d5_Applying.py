#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:22:24 2023
python this.py --data XXX.tsv --coln 3 --model G_YYY.pth --up 2 --width 14 --height 14 --odd width --shape 2,11 --save d5_XXX_YYY
(PID:v16)
@author: tk
"""

import cv2
e1 = cv2.getTickCount()
import argparse
import sys
import pandas as pd
import numpy as np
import glob
import torch
import pickle
import pdb # pdb.set_trace()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', '-d', help = 'Path of 1_ims.tsv data')
parser.add_argument('--coln', '-c', default=3, type=int, help = 'Start column of m/z')
parser.add_argument('--shape', '-sh', help = 'tensor shape of mass spectrum e.g 100,300')
parser.add_argument('--height', '-he', type=int, help = 'The number of origin picture height pixel')
parser.add_argument('--width', '-wi', type=int, help = 'The number of origin picture width pixel')
parser.add_argument('--up', '-u', type=int, help = 'Upscale (high spatial resolution rate) e.g 2')
parser.add_argument('--modelG', '-mg', help = 'Saved modelG.pkl')
parser.add_argument('--header', '-hd', default=3, help = 'Line number with header')
parser.add_argument('--thresh', '-th', default=0, help = 'Threshold for zero value')
parser.add_argument('--odd', '-o', type=str, default='none',help = 'The number of original pixels is odd => --odd height or width or both')
parser.add_argument('--save', '-s', type=str, help = 'Save basename')
args = parser.parse_args()    


class load_data():
    def __init__(self):
        self.input_data = pd.DataFrame()
        self.new_pixel_size = 0
        self.generated_pixels = []
        self.results = []
        self.comments = []
        self.center_pixels = []
    
    def loading(self, args):
        self.input_data = pd.read_csv(args.data, delimiter='\t', comment='#', header=None)
        
        # Separate header
        with open(args.data, 'r') as f:
            for line in f:
                if len(self.comments) < args.header:
                    self.comments.append(line.strip())
                    if len(self.comments) == args.header-1:
                        break
        
        # Format
        if self.input_data.iloc[:, -1].isnull().any():
            self.input_data = self.input_data.dropna(axis=1, how='any')
        self.input_data = self.input_data.iloc[:, args.coln-1:]
        
        # Calc and stock before padding
        self.origin_std = self.input_data.std(axis=0).values
        self.origin_mean = self.input_data.mean(axis=0).values
        self.origin = self.input_data
        
        """" Standardization """
        self.input_data = (self.input_data-self.input_data.mean(axis=0))/self.input_data.std(axis=0)
        self.input_data = self.input_data.fillna(0)
        
        # Padding by treating index
        np_data = np.array(list(range(len(self.input_data)))).reshape(args.height, args.width)
        np_data = self.padding(np_data)
        self.padded_width = np_data.shape[1]
        self.input_data = self.input_data.iloc[np_data.reshape(np_data.shape[0]*np_data.shape[1])]
        print(">> Padding done. Current pixels => {}".format(len(self.input_data)))
        
        # Non-standardized pattern
        self.origin = self.origin.iloc[np_data.reshape(np_data.shape[0]*np_data.shape[1])].values
       
    def predicting(self, args, model_G):
        """ Targeting the padded data """
        KERNEL_SIZE = 1
        self.shape = tuple(int(x) for x in args.shape.split(','))
        pixel_num = len(self.input_data)        
        count = 0
        uncount = 0
        for i in range(0, pixel_num):
            inx1 = i - self.padded_width*KERNEL_SIZE - KERNEL_SIZE
            inx2 = inx1 + KERNEL_SIZE
            inx3 = inx2 + KERNEL_SIZE
            inx4 = i - KERNEL_SIZE
            inx5 = i
            inx6 = i + KERNEL_SIZE
            inx7 = i + self.padded_width*KERNEL_SIZE - KERNEL_SIZE
            inx8 = inx7 + KERNEL_SIZE
            inx9 = inx8 + KERNEL_SIZE
            
            group_x = [inx1, inx2, inx3, inx4, inx5, inx6, inx7, inx8, inx9]
            
            # Skipping unnecessary regions
            if min(group_x) < 0 or max(group_x) >= pixel_num or not i//self.padded_width - KERNEL_SIZE == inx1//self.padded_width or not i//self.padded_width - KERNEL_SIZE == inx3//self.padded_width or not i//self.padded_width + KERNEL_SIZE == inx7//self.padded_width or not i//self.padded_width + KERNEL_SIZE == inx9//self.padded_width:
                uncount += 1
                continue
            
            # Tensor (1, pixels, m/z) -> (1, 9, shape of m/z)
            target_pixels = self.input_data.iloc[group_x]
            target_pixels = torch.tensor(target_pixels.values).view(1, len(group_x), target_pixels.shape[1])
            target_pixels = target_pixels.view(1, 9, self.shape[0], self.shape[1])
            
            # For CPU         
            target_pixels = target_pixels.to(deviceGC)
            target_pixels = target_pixels.to(torch.float32)
            
            # Predicting
            with torch.no_grad():
                generated_pixels = model_G(target_pixels)
            self.generated_pixels.append(generated_pixels) # index = original pixel order
            self.center_pixels.append(self.origin[inx5, :])
            
            if count%(len(self.input_data)//10) == 0:
                print(">>> Predicted pixels (origin) {}".format(count))
            count += 1
            print(inx5)
        print(">> Toatl predicted pixels {}\n>> Skipped => {}".format(count, uncount))
        del self.origin

    def reconstructing(self, args):        
        self.new_pixel_size = len(self.generated_pixels)*args.up*args.up
        print(">> Once, total {} pixels generated as high spatial resolution data (including padding area, here)".format(self.new_pixel_size))
        zero_pic_pd = pd.DataFrame(np.zeros((self.new_pixel_size, self.shape[0]*self.shape[1]))) / 0 # intensionally making NaN pd data
        print(">> Zero picture generated => {}".format(len(zero_pic_pd)))
        
        """ Targeting generated data """
        count_new_pix = 0
        count_origin_pix = 0
        new_width = args.width*args.up
        for j in range(0, len(zero_pic_pd), args.up):
            if not (j//new_width)%args.up == 0: # skipped in the interpolated lanes
                continue
            try:
                splits = list(torch.chunk(self.generated_pixels[count_origin_pix], chunks=3, dim=1)) # chunking (1, 3, shape[0], shape[1]) to list of (1, 1, shape[0], shape[1])
            except:
                print("*** ERROR@chunk *** => j:{}".format(j))

            iny1 = j + 1
            iny2 = j + args.width*args.up
            iny3 = iny2 + 1
            try:
                zero_pic_pd.iloc[j] = self.center_pixels[count_origin_pix] ###
                zero_pic_pd.iloc[iny1] = splits[0].flatten().numpy()*(self.origin_std) + self.origin_mean
                zero_pic_pd.iloc[iny2] = splits[1].flatten().numpy()*(self.origin_std) + self.origin_mean
                zero_pic_pd.iloc[iny3] = splits[2].flatten().numpy()*(self.origin_std) + self.origin_mean
            except:
                print("*** ERROR **** iny1:{}, iny2:{}, iny3:{}".format(iny1, iny2, iny3))
            count_origin_pix += 1
            count_new_pix = count_new_pix + len(splits)
        if zero_pic_pd.isna().any().any():
            print("*** Still NaN present ***")
        zero_pic_pd[zero_pic_pd < args.thresh] = 0
        print(">> Reconstructd pixels =  {} (origin) + {} (generated)".format(count_origin_pix, count_new_pix))
        
        # Odd treatment (deleted)
        np_generated_index = np.array(range(0,len(zero_pic_pd))).reshape(args.height*args.up, args.width*args.up)
        if args.odd == 'height' or args.odd == 'both':
            print(">>> odd treatment in height")
            np_generated_index = np_generated_index[:-1, :]
        if args.odd == 'width' or args.odd == 'both':
            print(">>> odd treatment in width")
            np_generated_index = np_generated_index[:, :-1]
        zero_pic_pd = zero_pic_pd.iloc[np_generated_index.reshape(np_generated_index.size)]
        print(">>> Final pixels => {}".format(len(zero_pic_pd)))
        
        # Reformatting
        new_column_label = pd.Series([0] * len(zero_pic_pd), name='#Label')
        new_column_spot = pd.Series(range(1, len(zero_pic_pd) + 1), name='Spot')
        zero_pic_pd.insert(0, '#Label', new_column_label)
        zero_pic_pd.insert(1, 'Spot', new_column_spot)
        
        # Saving
        save_file = args.save + '_up' + str(args.up) + '.tsv'
        with open(save_file, 'w') as f:
            for com in self.comments:
                f.write(com + "\n")
            zero_pic_pd.to_csv(f, sep='\t', index=False, header=None)
    
    def padding(self, data): # numpy
        data = np.insert(data, 0, data[0], axis=0)
        data = np.insert(data, -1, data[-1], axis=0)
        data = np.insert(data, 0, data[:,0], axis=1)
        data = np.insert(data, -1, data[:,-1], axis=1)
        return data

# Model (same as TK_d4) #==============================#
class kernel_param():
    def __int__(self):
        self.k0 = []
        self.k00 = []
        self.k1 = []
        self.k2 = []
        self.k3 = []
        self.k4 = []
    
    def set_parm(self, args):
        if args.shape == '2,11':
            self.k0 = [1,3]
            self.k00 = [2,3]
            self.k1 = [2,3]
            self.k2 = [1,3]
            self.k3 = [1,3]
            self.k4 = [1,3]
            self.s0 = 1

def load_pkl(load_file):
    with open(load_file, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def index_loading(path):
    tensors_data_y = load_pkl(glob.glob(path+'/*data_y*.pkl')[0]) # (num, 30000)
    tensors_index_x = load_pkl(glob.glob(path+'/*index_x*.pkl')[0]) # (numx8, 9)
    tensors_index_y =  load_pkl(glob.glob(path+'/*index_y*.pkl')[0]) # (numx8, 4)
    return tensors_data_y, tensors_index_x, tensors_index_y

class Generator(torch.nn.Module):
  def __init__(self, kparam):
      super(Generator, self).__init__()
      # Encoder
      self.down0 = torch.nn.Conv2d(9, 128, kernel_size=kparam.k0, stride=kparam.s0)
      self.down1 = self.__encoder_block(128, 256, ksize = kparam.k0, stride=kparam.s0)
      self.down2 = self.__encoder_block(256, 24, ksize = kparam.k00, stride=kparam.s0)
      # Decoder
      self.up7 = self.__decoder_block(24, 256, use_dropout=False, ksize=kparam.k1, stride=kparam.s0)
      self.up1 = self.__decoder_block(512, 128, use_dropout=False, ksize=kparam.k3, stride=kparam.s0)
      self.up0 = torch.nn.Sequential(
          self.__decoder_block(256, 3, use_norm=False, use_dropout=False, ksize=kparam.k4, stride=kparam.s0),
          )

  def __encoder_block(self, input, output, ksize=[2,1], stride=1, use_norm=True):
      layer = [
          torch.nn.Tanh(),
          torch.nn.Conv2d(input, output, kernel_size=ksize, stride=stride)
          ]
      if use_norm:
          layer.append(torch.nn.BatchNorm2d(output))
      return torch.nn.Sequential(*layer)
  
  def __decoder_block(self, input, output, use_norm=True, use_dropout=False, ksize=4, stride=3):
      layer = [
      torch.nn.Tanh(),
      torch.nn.ConvTranspose2d(input, output, kernel_size=ksize, stride=stride)
      ]
      if use_norm:
          layer.append(torch.nn.BatchNorm2d(output))
          if use_dropout:
              layer.append(torch.nn.Dropout(0.25))
      return torch.nn.Sequential(*layer)

  def forward(self, x):
      x0 = self.down0(x)
      x1 = self.down1(x0)
      x2 = self.down2(x1)
      ### latent space ###
      mean = x2
      lnvar = x2
      std = lnvar.exp().sqrt()
      ZDIM = x2.shape
      epsilon = torch.randn(ZDIM).to(deviceGC)
      x2 = mean + std * epsilon
      ####################
      y2 = self.up7(x2)
      # SkipConnection
      y1 = self.up1(self.concat(x1, y2)) # 
      y0 = self.up0(self.concat(x0, y1)) # 
      return y0
          
  def concat(self, x, y):
      return torch.cat([x, y], dim=1)
#==============================#

""" BODY """
# GPU
deviceGC = "cuda:0" if torch.cuda.is_available() else "cpu"
print("#{}\n".format(sys.argv))
print("> Device: "+str(deviceGC))
print("> The number of original pixels is odd in {}".format(args.odd))

# Model
kparam = kernel_param()
kparam.set_parm(args)
model_G = Generator(kparam).to(deviceGC)
try:
    if deviceGC == 'cpu':
        model_G.load_state_dict(torch.load(args.modelG, map_location=torch.device('cpu')), strict=False)
    else:
        model_G.load_state_dict(torch.load(args.modelG), strict=False)
    model_G.eval() # prediction mode
except Exception as e:
    print(">> Failed import of modelG {}: {}".format(args.modelG, e))
    sys.exit()

# Import data
my_load = load_data()
print("> Loading...")
my_load.loading(args)
print("> Predicting...")
my_load.predicting(args, model_G)
print("> Reconstructing & Saving as tsv ...")
my_load.reconstructing(args)

#==============================##==============================#
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
min = int(time/60)
sec = int(time%60)
print("TIME >", "{0:02d}".format(min)+":"+"{0:02d}".format(sec))




