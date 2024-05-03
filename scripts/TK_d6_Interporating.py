#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolating by LINEAR(inter 1) and Cubic(inter 3)
python this.py --data d2B_xxx.tsv --header 2 --coln 3 --height X --width Y --save Z --inter 3
@author: tk
"""
import argparse
import pandas as pd
import cv2
e1 = cv2.getTickCount()
import pdb # pdb.set_trace()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', '-d', help = 'Path of 1_ims.tsv data')
parser.add_argument('--coln', '-c', default=3, type=int, help = 'Start column of m/z')
parser.add_argument('--shape', '-sh', help = 'tensor shape of mass spectrum e.g 100,300')
parser.add_argument('--height', '-he', type=int, help = 'The number of origin picture (small) height pixel')
parser.add_argument('--width', '-wi', type=int, help = 'The number of origin picture ()small) width pixel')
parser.add_argument('--odd', '-o', type=str, help = '--odd height, --odd width, or --odd both if the number of original pixels is odd.')
parser.add_argument('--header', '-hd', type=int, default=2, help = 'Line number with header')
parser.add_argument('--inter', '-in', type=int, default=3, help = 'Interpolation index [0-4]  = ["INTER_NEAREST","INTER_LINEAR","INTER_AREA","INTER_CUBIC","INTER_LANCZOS4"]')
parser.add_argument('--save', '-s', type=str, help = 'Save basename')
args = parser.parse_args()    

interpolation_list = ["INTER_NEAREST","INTER_LINEAR","INTER_AREA","INTER_CUBIC","INTER_LANCZOS4"]    

# Importing asPandas DataFrame
input_data = pd.read_csv(args.data, delimiter='\t', comment='#', header=None)

# Stocking comments
comments = []
with open(args.data, 'r') as f:
    for line in f:
        if len(comments) <= args.header:
            comments.append(line.strip())
            if len(comments) == args.header:
                break

# Formatting
if input_data.iloc[:, -1].isnull().any():
    input_data = input_data.dropna(axis=1, how='any')
input_data = input_data.iloc[:, args.coln-1:]

# Interporating by m/z
processed_df = []
input_data = input_data.values
hh = 0
ww = 0
if args.odd == 'height':
    hh = 1
elif args.odd == 'width':
    ww = 1

for column in range(0,input_data.shape[1]):
    img = input_data[:, column].reshape((args.height, args.width))
    try:
        if args.inter == 0:
            img = cv2.resize(img, (args.width*2 - ww, args.height*2 - hh), interpolation=cv2.INTER_NEAREST)
        elif args.inter == 1:
            img = cv2.resize(img, (args.width*2 - ww, args.height*2 - hh), interpolation=cv2.INTER_LINEAR)
        elif args.inter == 2:
            img = cv2.resize(img, (args.width*2 - ww, args.height*2 - hh), interpolation=cv2.INTER_AREA)
        elif args.inter == 3:
            img = cv2.resize(img, (args.width*2 - ww, args.height*2 - hh), interpolation=cv2.INTER_CUBIC)
        elif args.inter == 4:
            img = cv2.resize(img, (args.width*2 - ww, args.height*2 - hh), interpolation=cv2.INTER_LANCZOS4)    
        img[img<0] = 0
        print("> column: {}, max: {}".format(column, img.max()))
    except:
        print("Interporation failed > {}".format(column))
    processed_df.append(img.reshape((img.size)))
processed_df = pd.DataFrame(processed_df)
if input_data.shape[1] > 1:
    processed_df = processed_df.T

# Saving adding 2 columns and comments
new_column_label = pd.Series([0] * len(processed_df), name='#Label')
new_column_spot = pd.Series(range(1, len(processed_df) + 1), name='Spot')
processed_df = pd.concat([new_column_label, new_column_spot, processed_df], axis=1)
save_name = args.save+'.tsv'
with open(save_name, 'w') as f:
    for com in comments:
        f.write(com + "\n")
    processed_df.to_csv(f, sep='\t', index=False, header=None)
    
#==============================##==============================#
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
min = int(time/60)
sec = int(time%60)
print("TIME >", "{0:02d}".format(min)+":"+"{0:02d}".format(sec))




