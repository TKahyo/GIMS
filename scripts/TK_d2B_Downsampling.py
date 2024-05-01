#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:22:24 2023
Spatially downsampled
python this.py --file [.tsv] --coln 3 --size 24,28  --donwn 2 --save d2B.tsv
*--down 2 => height 1/2, width 1/2
(PID:v3)
@author: tk
"""

import cv2
e1 = cv2.getTickCount()
import argparse
import numpy as np
import subprocess
import pdb # pdb.set_trace()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--file', '-f', help = 'IMS data file path *')
parser.add_argument('--coln', '-c', default=3, type=int, help = 'Start column of m/z, default=3')
parser.add_argument('--size', '-sh', help = 'Imaging size of original data (height,width) e.g 24,28 *')
parser.add_argument('--down', '-d', default=1, type=int, help = 'Downsamplint rate e.g. --down 2 => 1/2,1/2, default=2')
parser.add_argument('--save', '-s', type=str, help = 'basename for picture')
args = parser.parse_args()

def downsampling(args):
    size = tuple(int(x) for x in args.size.split(','))
    height =  size[0]
    width = size[1]
    down_size = args.down
    
    # Treating pd index as numpy data
    file_path = args.file
    command = f"grep -v '^#' {file_path} | wc -l"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line_count = int(result.stdout.split()[0])
    np_data = np.array(list(range(line_count)))
    np_data = np_data.reshape(height, width)
    
    # Downsampling
    np_data = np_data[::down_size, ::down_size]
    print(">> Downsampled")
    
    # Comments
    command = f"grep -c '^#' {file_path}"
    result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE)
    comment_lines_count = int(result.stdout)
    print(">> Counted comment lines: {}".format(comment_lines_count))
    
    # Saving
    save_file = args.save + '_down' + str(down_size) + '.tsv'
    np_data = np_data.reshape(np_data.shape[0]*np_data.shape[1]) + comment_lines_count + 1 # increasing index number by comment lines
    np_data = np.insert(np_data, 0, [1, 2]).tolist()
    with open('line_numbers.txt', 'w') as file:
        for x in np_data:
            file.write(str(x) + '\n')
    command = f"awk 'FNR == NR {{ lines[$1] = 1; next }} FNR in lines' line_numbers.txt {file_path} > {save_file}"
    subprocess.run(command, shell=True)
    print(">> Saved")

###########
downsampling(args)

#==============================##==============================#
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
min = int(time/60)
sec = int(time%60)
print("TIME >", "{0:02d}".format(min)+":"+"{0:02d}".format(sec))




