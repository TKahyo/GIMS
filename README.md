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
   ./converter.sh -f DEMO_DATA1.csv -r 600,899,99 -e 26 -w 27 -l DEMO_LIST.csv -s DEMO_DATA1_conv
   ```   
　　-r: lower _m/z_ value, upper _m/z_ value   
　　-e: height (in pixels)   
　　-w: width (in pixels)   
　　-l: list of selected _m/z_ values (.csv)   
　　-s: save name   
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
   python ./scripts/TK_d4_learning.py --data path_file.txt --test path_list_test.txt --shape 2,11 --dlr 0.00000001 --glr 0.002 --beta1g 0.99 --batch 22224 –-fbatch 2 --epoch 1000--save learning_result
   ```
 
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
                          Batch size of files. e.g. fbatch=3 => 3 files import once from --data,
                          default=2
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
    --test TEST, -t TEST  Path file of directories containing the pickle for test. If not necessary,
                          put the same file as --data *
    --save SAVE, -s SAVE  pth save directory *

*output  
 ```bash
 d4_learning_result_BEST_weight/
  |--- D_XXXX.pth
  |     ...
  |--- G_XXXX.pth
  └     ...
 d4_learning_result_D_weight/
  |--- D_YYYY.pth
  |    ...
  └--- ...
 d4_learning_result_G_weight/
  |--- G_YYYY.pth
  |    ...
  └--- ...
 summary.txt
 ```
   
**[4] Applying a model**



**[5] Imaging**    



# Note
Experience with IMS analysis is desirable.
 
# Author
Tomoaki Kahyo
 
# License
GNU General Public License v3.0 
