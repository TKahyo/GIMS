# IMSG
This is a tool developed for interpolating Imaging Mass Spectrometry (IMS) data using generative deep machine learning method.

# DEMO
 

 
# Features
 The CVAE model trained on surrounding nine pixel data to interpolate center three pixel data.

 
# Requirement
<details>
CPU: Intel(R) Xeon(R) CPU E5-2603 v4 @ 1.70GHz  
 
GPU: NVIDIA TITAN X (Pascal) 12GB  

System Mem: DDR4 64GB  

Pytorch is used with CUDA.  
It is recommended to run in anaconda environment.  
___
```bash
conda info
```
```bash
           conda version : 23.9.0
    conda-build version : 3.27.0
         python version : 3.10.9.final.0
```
***
```bash
 conda list | grep pytorch
```
```bash
pytorch                   2.0.0           cpu_generic_py310h3496f23_1    conda-forge
```
***
```bash
nvcc --version
```
```bash
Build cuda_12.4.r12.4/compiler.34097967_0
```
</details>
 
# Usage
1) Converting .csv file
   ```bash
   ./converter.sh -f DEMO_DATA1.csv -r 600,899,99 -e 26 -w 27 -l DEMO_LIST.csv -s DEMO_DATA1_conv
   ```
2) Making datasets for training and test

3) Training

4) Imaging
 

# Note
Experience with IMS analysis is desirable.
 
# Author
Tomoaki Kahyo
 
# License
GNU General Public License v3.0 
