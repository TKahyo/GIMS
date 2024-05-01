#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:22:24 2023
Imaging of specific m/z ion
python this.py --file [tsv file] --coln 3 --size 24,28  --mz 888.63 --save d2_XXX (--down )
* --size height,width
[PID:v4]
@author: tk
"""

import cv2
e1 = cv2.getTickCount()
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import pdb # pdb.set_trace()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--file', '-f', help = 'File path *')
parser.add_argument('--coln', '-c', default=3, type=int, help = 'Start column of m/z, default=3')
parser.add_argument('--size', '-sh', help = 'Imaging size (height,width), e.g 24,28 *')
parser.add_argument('--down', '-d', type=int, help = 'Downsamplint rate e.g. --down 2 => 1/2,1/2, default=0')
parser.add_argument('--mz', '-mz', help = 'm/z e.g. --mz 798.00 *')
parser.add_argument('--comp', '-cm', default=0, type=float, help = 'm/z value for comparison m/z. e.g. --comp 888.63, default=0 (unused)')
parser.add_argument('--weight', '-w', default=1, type=float, help = 'Weight of --comp. e.g. 0.5 => comparison m/z intensity x 0.5, default=1 (unused)')
parser.add_argument('--save', '-s', type=str, help = 'basename for picture *')
args = parser.parse_args()

def ImportData(args):
    file = args.file
    coln = args.coln
    coln -= 1
    mz = str(args.mz)
    size = tuple(int(x) for x in args.size.split(','))
    height =  size[0]
    width = size[1]
    output_file = args.save
    
    # Simplifying data structure
    data_y = pd.read_csv(file, delimiter='\t', skiprows=1, header=0)
    if data_y.iloc[:, -1].isnull().any():
        data_y = data_y.dropna(axis=1, how='any')
    data_y = data_y.iloc[:, coln:]
    
    # Rounding column m/z values
    data_y.columns = [f"{float(col):.2f}" if col.replace('.', '', 1).isdigit() else col for col in data_y.columns]
    
    # Imaging
    ConvertToImage(data_y, mz, width, height, output_file, args.down)


def ConvertToImage(data_y, mz, width, height, output_file, down):
    mz2 = mz.replace(".", "")
    output_file = output_file + '_' + str(mz2) + '.png'
    data_y_mz = data_y[mz].values.reshape(height, width)    
    data_y_mz = np.where(data_y_mz < 0, 0, data_y_mz)
    
    # Normalized
    if args.comp != 0:
        comp_max = data_y[str(args.comp)].values.max() * args.weight
        print(">> Comp m/z {}, max intensity = {}".format(args.comp, comp_max))
        print(">> Target max intensity = {}".format(data_y_mz.max()))
    else:
        comp_max = data_y_mz.max() * args.weight
    
    if comp_max > 0:
        data_y_mz = 255*(data_y_mz/comp_max)
    else:
        print("**** comp_max = {} ***".format(comp_max))
        
    # plt
    fig, ax = plt.subplots(figsize=(width, height), dpi=100) # (width, height) @plt
    ax.imshow(data_y_mz, cmap='gray', vmax=255)
    ax.axis('off')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    
    # Down sampling (optional)
    if down:
        output_file = args.save + '_down' + str(down) + '_' + str(mz2) + '.png'
        data_y_mz = padding(data_y_mz, height, width, down)
        data_y_mz = data_y_mz[::down, ::down]
        fig, ax = plt.subplots(figsize=(width-width%down, height-height%down), dpi=100)
        ax.imshow(data_y_mz, cmap='gray')
        ax.axis('off')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)

def padding(data, height, width, down):
    if height%down:
        add_height = [data[-1]]*(down - height%down)
        data = np.vstack([data, add_height])
    if width%down:
        add_width = [data[:,-1]]*(down - width%down)
        data = np.hstack([data, add_width.reshape(-1,1)])
    return data

###########

ImportData(args) 

#==============================##==============================#
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
min = int(time/60)
sec = int(time%60)
print("TIME >", "{0:02d}".format(min)+":"+"{0:02d}".format(sec))




