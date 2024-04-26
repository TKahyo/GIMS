#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:14:21 2023
Evaluating the image quality by PSNR and SSIM
python this.py --data1 d5.tsv --data2 d3_narrowed.tsv --coln 3 --height 28 --width 27 --save Z
*--data1 is generated data, --data2 is measured true data
@author: tk
"""

import cv2
e1 = cv2.getTickCount()
import argparse
import pandas as pd
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import pdb # pdb.set_trace()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data1', '-d1', help = 'Path of data1 e.g. d5.tsv')
parser.add_argument('--data2', '-d2', help = 'Path of data2 e.g. d3.tsv')
parser.add_argument('--coln', '-c', default=3, type=int, help = 'Start column of m/z')
parser.add_argument('--height', '-he', type=int, help = 'The number of origin picture height pixel')
parser.add_argument('--width', '-wi', type=int, help = 'The number of origin picture width pixel')
parser.add_argument('--caption', '-ca', type=int, default=2, help = 'Caption(line of m/z) e.g 2')
parser.add_argument('--excluded', '-ex', type=str, help = 'The number of excluded column')
parser.add_argument('--save', '-s', type=str, help = 'Save basename')
args = parser.parse_args()    

# Importing
pd1 = pd.read_csv(args.data1, delimiter='\t', comment='#', header=None)
pd2 = pd.read_csv(args.data2, delimiter='\t', comment='#', header=None)
pd1 = pd1.iloc[:, args.coln-1:]
pd2 = pd2.iloc[:, args.coln-1:]
if args.excluded:
    excluded = [int(x) for x in args.excluded.split(",")]
    excluded = excluded - args.coln
    pd2 = pd2.drop(pd2.columns[excluded], axis=1)
    print(">> Excluded {}".format(excluded))
    
mz_list = []
cap = 1
with open(args.data1, 'r') as f:
    for line in f:
        if  cap == args.caption:
            line = line.strip()
            mz_list = line.split("\t")
            break
        cap += 1
mz_list = mz_list[args.coln-1:]

# Calc
max1_results = []
max2_results = []
mean1_results = []
mean2_results = []
p25_1_results = []
p25_2_results = []
p75_1_results = []
p75_2_results = []
psnr_results = []
ssim_results =[]
for mz in range(0,len(pd1.columns)):
    img1 = pd1.iloc[:,mz].values.reshape((args.height,args.width))
    img2 = pd2.iloc[:,mz].values.reshape((args.height,args.width))
    max1_results.append(img1.max())
    max2_results.append(img2.max())
    mean1_results.append(img1.mean())
    mean2_results.append(img2.mean())
    p25_1_results.append(np.percentile(img1, 25))
    p25_2_results.append(np.percentile(img2, 25))
    p75_1_results.append(np.percentile(img1, 75))
    p75_2_results.append(np.percentile(img2, 75))
    
    if img2.max() == 0 or img1.max() == 0:
        psnr_results.append('NA')
        ssim_results.append('NA')
    else:
        img1 = img1*255/img1.max()
        img2 = img2*255/img2.max()
        try:
            psnr_results.append(peak_signal_noise_ratio(img1, img2, data_range=255.0))
            ssim_results.append(ssim(img1, img2, data_range=255.0))
        except:
            print("*** ERROR in PSNR and SSIM ***")
            pdb.set_trace()

data_dict = {'m/z': mz_list,
             'Max1': max1_results, 'Mean1': mean1_results, '25%per1': p25_1_results, '75%per1': p75_1_results,
             'Max2': max2_results, 'Mean2': mean2_results,'25%per2': p25_2_results,'75%per2': p75_2_results,
             'PSNR': psnr_results, 'SSIM': ssim_results}
df = pd.DataFrame(data_dict)
save_name = args.save+'.tsv'
df.to_csv(save_name, sep='\t', index=False)

#==============================##==============================#
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
min = int(time/60)
sec = int(time%60)
print("TIME >", "{0:02d}".format(min)+":"+"{0:02d}".format(sec))