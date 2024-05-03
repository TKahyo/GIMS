#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converting for learning data set (ignoring edges) & saving narrowed data
python this.py --file [tsv file] --z 1 --narrow narrow_list.csv --coln 3 --stride 1 --height 28 --width 30 --batch 10 --save learning_data_set
[PID:v11]
@author: tk
"""

import cv2
e1 = cv2.getTickCount()
import argparse
import sys
import pandas as pd
import dask.dataframe as dd
import os
import torch
from torch.utils.data import DataLoader
import pickle
import pdb # pdb.set_trace()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--file', '-f', help = 'IMS data file path')
parser.add_argument('--coln', '-c', default=3, type=int, help = 'Start column of m/z')
parser.add_argument('--height', '-he', type=int, help = 'The number of picture (large) height pixel')
parser.add_argument('--width', '-wi', type=int, help = 'The number of picture (large) width pixe')
parser.add_argument('--stride', '-st', default=1, type=int, help = 'stride e.g. 1')
parser.add_argument('--batch', '-b', default=10, type=int, help = 'batch size < 20 in gpu')
parser.add_argument('--save', '-s', type=str, help = 'Save basename')
parser.add_argument('--narrow', '-n', default='0,99', help = 'Exract specific m/z from list or allowable numbers after the decimal point. e.g. 0.4,0.89 => 600.42, 833.89 allowed, 600.03, 733.92 denied')
parser.add_argument('--z', '-z', default=1, type=int, help = 'zscore by m/z columns')
args = parser.parse_args()

class Index():
    def __init__(self):
        self.data_y = dd.from_pandas(pd.DataFrame(),npartitions=2) # change npartitions if necessary
        self.index_x = []
        self.index_y = []
        self.check = []

    def indexing(self, args):
        file = args.file
        KERNEL_SIZE = 2
        stride = args.stride
        coln = args.coln
        height = args.height
        width = args.width
        coln -= 1
    
        # Importing data (dask)
        self.data_y = dd.read_csv(file, delimiter='\t', comment='#', header=None, engine='python', sample=10000000000, assume_missing=True) # change sample if necessary
        if self.data_y.iloc[:, -1].isnull().any().compute():
            self.data_y = self.data_y.drop(self.data_y.columns[-1], axis=1)
        self.data_y = self.data_y.iloc[:, coln:]
        print(">> Imported.")
        
        # Processing comment lines => float m/z values
        with open(file, 'r') as ff:
            comment_lines = []
            for _ in range(2):
                line = ff.readline().strip()
                comment_lines.append(line)
        if len(comment_lines) > 1:
            comment_lines = comment_lines[1].split("\t")
            comment_lines = comment_lines[coln:]
        full_numbers = [float(str_num) for str_num in comment_lines]
        
        # Filtering m/z
        if ',' in args.narrow:
            # Processing float values of column
            float_numbers = [xx - int(xx) for xx in full_numbers]
            float_numbers = [round(x, 3) for x in float_numbers]
            self.data_y.columns = float_numbers
            # Filtering column
            low_tuple, high_tuple = tuple(float(x) for x in args.narrow.split(','))
            narrowed_numbers = [x for x in full_numbers if low_tuple <= round(x % 1, 3) < high_tuple]
            narrowed_numbers = [str(s).lstrip('\t') for s in narrowed_numbers]
            # Filtering body
            filtered = self.data_y.loc[:, self.data_y.columns.astype(float) >= low_tuple]
            self.data_y = filtered.loc[:, filtered.columns.astype(float) < high_tuple]
            del filtered
            self.data_y.columns = narrowed_numbers 
        else:
            mz_list = []
            if '.csv' in args.narrow:
                mz_list = pd.read_csv(args.narrow, comment='#', header=None)
            elif '.tsv' in args.narrow:
                mz_list = pd.read_csv(args.narrow, delimiter='\t', comment='#', header=None)
            mz_list = mz_list.values.tolist()
            self.data_y.columns = full_numbers
            self.data_y = self.data_y[mz_list[0]]
        print(">> Filtered.")
        
        # Save narrowed data
        file_path = args.save + '_narrowed.tsv'
        file_path2 = 'tmp2'
        file_path3 = 'tmp3'
        self.data_y.compute().to_csv(file_path2, sep='\t', index=False)
        
        # Inserting 2 columns (a step during the development process)
        pixel_num = self.data_y.shape[0].compute()
        insertD = ""
        insertD += "#0\t0\n"
        for _ in range(pixel_num):
            insertD += "0\t0\n"
        with open(file_path3, "w") as file:
            file.write(insertD)
        
        with open(file_path2, 'r') as file_2:
            lines_2 = file_2.readlines()
        with open(file_path3, 'r') as file_3:
            lines_3 = file_3.readlines()
        combined_lines = [line_3.strip() + '\t' + line_2.strip() + '\n' for line_3, line_2 in zip(lines_3, lines_2)]    
        combined_lines.insert(0, "#comments\n")
        with open(file_path, 'w') as combined_file: 
            combined_file.writelines(combined_lines)   
        print(">> Saved to .tsv")
        os.remove(file_path2)
        os.remove(file_path3)
        print("Pixel_num: {}, mz_num: {}\n".format(pixel_num, self.data_y.shape[1]))
        if not pixel_num == height*width:
            print("*** Unmatched pixel size *** row {}, height {}, width {}".format(pixel_num, height, width))
            sys.exit()
            
        
        """ Zscore or Normalize """
        if args.z == 1: # default
            print(">> Zscore by m/z columns")
            self.data_y = (self.data_y - self.data_y.mean(axis=0)) / self.data_y.std(axis=0)
            self.data_y = self.data_y.fillna(0)
        if args.z == 2:
            print(">> 0-1 normalized by m/z columns")
            self.data_y = self.data_y / self.data_y.max()
            self.data_y = self.data_y.fillna(0)
        self.data_y = self.data_y.to_dask_array(lengths=True)
        self.data_y = torch.tensor(self.data_y.compute(), dtype=torch.float)
        
        """ Indexing """
        for i in range(width*KERNEL_SIZE+KERNEL_SIZE, pixel_num, stride):
            # x (input)
            inx1 = i - width*KERNEL_SIZE - KERNEL_SIZE
            inx2 = inx1 + KERNEL_SIZE
            inx3 = inx2 + KERNEL_SIZE
            inx4 = i - KERNEL_SIZE
            inx5 = i
            inx6 = i + KERNEL_SIZE
            inx7 = i + width*KERNEL_SIZE - KERNEL_SIZE
            inx8 = inx7 + KERNEL_SIZE
            inx9 = inx8 + KERNEL_SIZE
            group_x = [inx1, inx2, inx3, inx4, inx5, inx6, inx7, inx8, inx9]
            
            # Skip point
            if max(group_x) > pixel_num or  not i//width - KERNEL_SIZE == inx1//width or not i//width - KERNEL_SIZE == inx3//width or not i//width + KERNEL_SIZE == inx7//width or not i//width + KERNEL_SIZE == inx9//width:
                continue
            self.index_x.append(group_x)
            
            # y (output)
            iny1 = i + 1
            iny2 = i + width
            iny3 = iny2 + 1
            group_y = [iny1, iny2, iny3]
            self.index_y.append(group_y)
            
            # rotate & flip (x8)
            group_x2 = group_x
            for nn in range(0, 3):
                group_x, group_y = rotate(i, width, group_x, nn)
                self.index_x.append(group_x)
                self.index_y.append(group_y)
            group_x2, group_y2 = flip_x(i, width, group_x2)
            self.index_x.append(group_x2)
            self.index_y.append(group_y2)
            for nn in range(0, 3):
                group_x2, group_y2 = rotate2(i, width, group_x2, nn)
                self.index_x.append(group_x2)
                self.index_y.append(group_y2)           
        
        # Tensor
        self.index_x = torch.tensor(self.index_x, dtype=torch.int)
        self.index_y = torch.tensor(self.index_y, dtype=torch.int)
        # Check of data size
        self.check = [self.data_y.shape, self.index_x.shape, self.index_y.shape]    
        
    def save(self, args):
        basename = args.save
        os.mkdir(basename)
        # Do not change '*data_y*.pkl', '*index_x*.pkl', and '*index_y*.pkl'
        with open(basename+'/d3_data_y_'+basename+'.pkl', 'wb') as f:
            pickle.dump(self.data_y, f, protocol=4)
        with open(basename+'/d3_data_index_x_'+basename+'.pkl', 'wb') as finx:
            pickle.dump(self.index_x, finx, protocol=4)
        with open(basename+'/d3_data_index_y_'+basename+'.pkl', 'wb') as finy:
            pickle.dump(self.index_y, finy, protocol=4)

# Left rotation   
def rotate(i, width, group_x, nn):
    new_order = [2, 5, 8, 1, 4, 7, 0, 3, 6]
    group_x = [group_x[i] for i in new_order]
    if nn == 0:
        group_y = [i + width, i - 1, i - 1 + width]
    elif nn == 1:
        group_y = [i - 1, i - width, i - width - 1]
    elif nn == 2:
        group_y = [i - width, i + 1, i + 1 - width] 
    return group_x, group_y

# Flip
def flip_x(i, width, group_x):
    new_order = [2, 1, 0, 5, 4, 3, 8, 7, 6]
    group_x = [group_x[i] for i in new_order]
    group_y = [i - 1, i + width, i +  width - 1]
    return group_x, group_y

# Rotation of flip
def rotate2(i, width, group_x, nn):
    new_order = [2, 5, 8, 1, 4, 7, 0, 3, 6]
    group_x = [group_x[i] for i in new_order]
    if nn == 0:
        group_y = [i + width, i + 1, i + 1 + width]
    elif nn == 1:
        group_y = [i + 1, i - width, i - width + 1]
    elif nn == 2:
        group_y = [i - width, i - 1, i - 1 - width] 
    return group_x, group_y


#==============================##==============================#

print("#{}\n".format(sys.argv))

# Devided into indexes of input (X) and ground truth (Y)
index_data = Index()
index_data.indexing(args)
print(">> Size check => data_y:{}, index_x:{}, index_y:{}\n".format(*index_data.check))

# Dataloader check
mydataset = torch.utils.data.TensorDataset(index_data.index_x, index_data.index_y)
data_loader = DataLoader(mydataset, batch_size=args.batch, shuffle=True)
for index_x, index_y in data_loader:
    print("*** CHECK *** index_x from loader: {}, shape {}\n".format(index_x, index_x.shape))
    print("*** CHECK *** index_y from loader: {}, shape {}\n".format(index_y, index_y.shape))
    break

# Save
index_data.save(args)

#==============================##==============================#
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
min = int(time/60)
sec = int(time%60)
print("TIME >", "{0:02d}".format(min)+":"+"{0:02d}".format(sec))




