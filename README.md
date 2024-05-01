# IMSG
This is a tool developed for interpolating Imaging Mass Spectrometry (IMS) data using generative deep machine learning method.

# DEMO
 

 
# Features
 The CVAE model trained on surrounding nine pixel data to interpolate center three pixel data.

 
# Requirement
<details>
 Pytorch is used with CUDA.   
 
 It is recommended to run in anaconda environment.    
  
 ```bash   
 conda info
 ```
 ```bash   
           conda version : 23.9.0
    conda-build version : 3.27.0
         python version : 3.10.9.final.0
 ```
--- 
 ```bash   
 conda list | grep pytorch
 ```
 ```bash   
 pytorch                   2.0.0           cpu_generic_py310h3496f23_1    conda-forge
 ```
---
 ```bash   
 nvcc --version
 ```
 ```bash
 Build cuda_12.4.r12.4/compiler.34097967_0
 ```
 *Other libraries: pandas, dask   

 *The hardware environment is described below for reference.   
 CPU: Intel(R) Xeon(R) CPU E5-2603 v4 @1.70GHz    
 GPU: NVIDIA TITAN X (Pascal) 12GB  
 System Mem: DDR4 64GB   
</details>
 
# Usage
**[1] Converting .csv file and Making datasets for training/test**
   ```bash
   ./converter.sh --help
   ```
   ```bash
　　-r: lower _m/z_ value, upper _m/z_ value   
　　-e: height (in pixels)   
　　-w: width (in pixels)   
　　-l: list of selected _m/z_ values (.csv)   
　　-s: save name
   ```
---
   ```bash
   ./converter.sh -f DEMO_DATA1.csv -r 600,899,99 -e 26 -w 27 -l DEMO_LIST.csv -s DEMO_DATA1_conv
   ```
  *output    
 ```bash
 DEMO_DATA1_conv_dir    
  |--- DEMO_DATA1_conv/   
  |    |--- d3_data_index_x_DEMO_DATA1_conv.pkl    
  |    |--- d3_data_index_y_DEMO_DATA1_conv.pkl    
  |    └--- d3_data_y_DEMO_DATA1_conv.pkl   
  └--- DEMO_DATA1_conv_narrowed.tsv   
 ```

  *Do the same for DEMO_DATA_test (-e 28 -w 27)


**[2] Making path_file.txt**
```bash
cat path_list.txt
```
```bash
./DEMO_DATA1_dir/DEMO_DATA1_conv/
```
---
```bash
cat path_list_test.txt
```
```bash
./DEMO_DATA_test_dir/DEMO_DATA_test_conv/
```

  
**[3] Leaning**    
   ```bash
   python ./scripts/TK_d4_learning.py --help
   ```
   ```bash
    -h, --help            show this help message and exit   
    --data DATA, -d DATA  Path file of directories containing the pickle data *   
    --batch BATCH, -b BATCH   
                          Batch size, default=10   
    --epoch EPOCH, -e EPOCH   
                          Epoch, default=100   
    --start START, -st START   
                          Restarging epoch in using --modelG   
    --modelG MODELG, -mg MODELG   
                          Relearning using modelG   
    --modelD MODELD, -md MODELD   
                          Relearning using modelD   
    --dlr DLR, -dlr DLR   Learning rate, default 0.00015   
    --glr GLR, -glr GLR   Learning rate, default 0.00015   
    --shape SHAPE, -sh SHAPE   
                          Tensor shape of mass spectrum e.g 2,11. Needed to be mathed to the total   
                          number of m/z peaks, default=2,11   
    --fbatch FBATCH, -fb FBATCH   
                          Batch size of files. e.g. fbatch=3 => 3 files import once from --data, default=2     
    --beta1g BETA1G, -b1g BETA1G   
                          Adam parameter:beta1 of G   
    --beta1d BETA1D, -b1d BETA1D   
                          Adam parameter:beta1 of D   
    --noise NOISE, -n NOISE   
                          Rate of 1 in noise, default 0 = no noize   
    --noisemin NOISEMIN, -nmi NOISEMIN   
                          Min of noize   
    --noisemax NOISEMAX, -nma NOISEMAX   
                          Max of noize   
    --test TEST, -t TEST  Path file of directories containing the pickle for test. If not necessary, put the same file as --data *   
    --save SAVE, -s SAVE  pth save directory *   
   ```
   ```bash
   python ./scripts/TK_d4_learning.py --data path_file.txt --test path_list_test.txt --shape 2,11 --dlr 0.00000001 --glr 0.002 --beta1g 0.99 --batch 22224 –-fbatch 2 --epoch 1000--save d4_learning
   ```
*output  
 ```bash
 d4_learning_BEST_weight/
  |--- D_XXXX.pth
  |     ...
  |--- G_XXXX.pth
  └     ...
 d4_learning_D_weight/
  |--- D_YYYY.pth
  |    ...
  └--- ...
 d4_learning_G_weight/
  |--- G_YYYY.pth
  |    ...
  └--- ...
 summary.txt
 ```


**[4] Making low resolution data for demo**
 ```bash
python ./scripts/TK_d2B_Downsampling.py --file DEMO_DATA_test_conv_dir/DEMO_DATA_test_conv_narrowed.tsv --size 28,27 --donwn 2 --save d2B_downsampling
 ```
*output
 ```bash
d2B_downsampling.tsv
 ```
   
**[5] Applying a model**
 ```bash
python ./scripts/TK_d5_Applying.py --help
 ```
 ```bash
  -h, --help            show this help message and exit
  --data DATA, -d DATA  Path of target data (.tsv) *
  --coln COLN, -c COLN  Start column of m/z, default=3
  --shape SHAPE, -sh SHAPE
                        Tensor shape of mass spectrum e.g 2,11, default=2,11
  --height HEIGHT, -he HEIGHT
                        The number of origin picture height pixel *
  --width WIDTH, -wi WIDTH
                        The number of origin picture width pixel *
  --modelG MODELG, -mg MODELG
                        Applied modelG.pkl *
  --header HEADER, -hd HEADER
                        Line number of header, default=3
  --thresh THRESH, -th THRESH
                        Threshold for zero value, default=0
  --odd ODD, -o ODD     The number of original pixels is odd => --odd height or --odd width or --odd both, default=none
  --save SAVE, -s SAVE  Save basename
 ```
---
 ```bash
python ./scripts/TK_d5_Applying.py --data d2B_downsampling.tsv --model G_YYY.pth  --height 14 --width 14--odd width --shape 2,11 --save d5_applying
 ```
*output
 ```bash
d5_applying.tsv
 ```


**[6] Imaging**    
 ```bash
python ./scripts/TK_d2_Imaging.py --file d5_applying.tsv --size 28,27  --mz 888.63 --save d2_imaging_d5
 ```
 ```bash
python ./scripts/TK_d2_Imaging.py --file d2B_downsampling.tsv --size 14,14  --mz 888.63 --save d2_imaging_d2B
 ```
*output
 ```bash
d2_imaging_d5_88863.png
d2_imaging_d2B_88863.png
 ```


# Note
Experience with IMS analysis is desirable.


# Flow
![Flow chart](https://github.com/TKahyo/GIMS/GIMS_flow_chart.png)

 
# Author
Tomoaki Kahyo

 
# License
GNU General Public License v3.0 
