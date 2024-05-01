#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:22:24 2023
python thispy --data paths_file.txt --dlr 0.00000002 --glr 0.00002 --beta1g 0.99 --batch 22224 –-fbatch 2 --epoch 1000 --save my_result
[PID:v58]
@author: tk
"""

import cv2
e1 = cv2.getTickCount()
import argparse
import sys
import numpy as np

import os
import glob
from operator import itemgetter

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pickle
import pdb # pdb.set_trace()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', '-d', help = 'Path file of directories containing the pickle data *')
parser.add_argument('--batch', '-b', default=10, type=int, help = 'Batch size, default=10')
parser.add_argument('--epoch', '-e', default=100, type=int, help = 'Epoch, default=100')
parser.add_argument('--start', '-st', default=1, type=int, help = 'Restarging epoch in using --modelG')
parser.add_argument('--modelG', '-mg', help = 'Relearning using modelG')
parser.add_argument('--modelD', '-md', help = 'Relearning using modelD')
parser.add_argument('--dlr', '-dlr',default=0.00015,type=float,  help = 'Learning rate, default 0.00015')
parser.add_argument('--glr', '-glr',default=0.00015, type=float, help = 'Learning rate, default 0.00015')
parser.add_argument('--shape', '-sh',default='2,11', type=str, help = 'Tensor shape of mass spectrum e.g 2,11. Needed to be mathed to the total number of m/z peaks, default=2,11')
parser.add_argument('--fbatch', '-fb', default=2, type=int, help = 'Batch size of files. e.g. fbatch=3 => 3 files import once from --data, default=2')
parser.add_argument('--beta1g', '-b1g', default=0.9, type=float, help = 'Adam parameter:beta1 of G')
parser.add_argument('--beta1d', '-b1d', default=0.9, type=float, help = 'Adam parameter:beta1 of D')
parser.add_argument('--noise', '-n',default=0, type=float, help = 'Rate of 1 in noise, default 0 = no noize')
parser.add_argument('--noisemin', '-nmi', default=0.5, type=float, help = 'Min of noize')
parser.add_argument('--noisemax', '-nma', default=2.0, type=float, help = 'Max of noize')
parser.add_argument('--test', '-t', help = 'Path file of directories containing the pickle for test. If not necessary, put the same file as --data *') ######
parser.add_argument('--save', '-s', type=str, help = 'pth save directory *')
args = parser.parse_args()    

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


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
          torch.nn.Conv2d(12, 64, kernel_size=[2,2], stride=1, padding=0),
          torch.nn.BatchNorm2d(64),
          torch.nn.LeakyReLU(0.2, True),
          self.__layer(64, 128),
          self.__layer(128, 512),
          torch.nn.Conv2d(512, 1, kernel_size=[1,2], stride=1),
          torch.nn.Tanh(),
        )

    def __layer(self, input, output, stride=1):
        # Downsampling
        return torch.nn.Sequential(
          torch.nn.Conv2d(input, output, kernel_size=[1,2], stride=stride, padding=0),
          torch.nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        return self.model(x)

# Initialization of weights for G and D
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
class train_vae():
    def __init__(self):
        self.sum_Loss_G_real = 'NA'
        self.sum_loss_G_mse = 'NA'
        self.sum_Loss_G_total = 'NA'
        self.sum_op1 = 'NA'
        self.sum_op2 = 'NA'
        self.sum_op3 = 'NA'
        self.sum_Loss_D_fake = 'NA'
        self.sum_Loss_D_real = 'NA'
        self.sum_Loss_D_total = 'NA'
        self.sum_real_batchsize = 0
        self.loader_count = 0
        
    def vae(self, model_G, model_D, params_G, params_D, data_loader, data_y):
        list_Loss_G_real = []
        list_loss_G_mse = []
        list_Loss_G_total = []
        list_op1 = []
        list_op2 = []
        list_op3 = []
        list_Loss_D_fake = []
        list_Loss_D_real = []
        list_Loss_D_total = []
        list_real_batchsize = []
        
        for index_x, index_y in data_loader: # index_x.shape = (batchsize, 9)
            self.real_batchsize = len(index_x)
            list_real_batchsize.append(self.real_batchsize)
            self.loader_count += 1

            index_x = index_x.int()
            index_y = index_y.int()
            train_x = data_y[index_x]
            train_y = data_y[index_y]

            if args.noise > 0: # percentage of zero    
                shape = (self.real_batchsize, 9, shape_tuple[0]*shape_tuple[1])
                min_value = args.noisemin
                max_value = args.noisemax
                random_tensor = min_value + (max_value - min_value) * torch.rand((shape))
                mask = torch.rand(shape) >= args.noise
                random_tensor[mask] = 1
                train_x = train_x*random_tensor

            train_x = train_x.view(len(index_x), 9, shape_tuple[0], shape_tuple[1])
            train_y = train_y.view(len(index_y), 3, shape_tuple[0], shape_tuple[1])       
            train_x = train_x.to(deviceGC) 
            train_y = train_y.to(deviceGC)
            
            """ D """
            # D(x+fake); G detached
            fake_xout = model_G(train_x)
            cat_fake = torch.cat((train_x, fake_xout.detach()), dim=1)
            out_fake = model_D(cat_fake)
            option2 = out_fake.cpu().detach()
            option2 = torch.mean(option2, dim=(1,2,3)).numpy()
            # Hinge loss of false (-0.95)
            loss_D_fake = loss_f(out_fake, torch.ones_like(out_fake)*(-0.95))
            
            # D(x+real)
            cat_real = torch.cat((train_x, train_y), dim=1)
            out_real = model_D(cat_real)
            option3 = out_real.cpu().detach()
            option3 = torch.mean(option3,dim=(1,2,3)).numpy()            
            # Hinge los  of true (0.95)
            loss_D_real = loss_f(out_real, torch.ones_like(out_real)*0.95)
            
            # Learning for D
            loss_D_total = 0.5*(loss_D_fake + loss_D_real*10)
            params_D.zero_grad()
            loss_D_total.backward(retain_graph=True)
            params_D.step()
            
            # Record
            list_Loss_D_fake.append(np.mean(loss_D_fake.cpu().detach().numpy()))
            list_Loss_D_real.append(np.mean(loss_D_real.cpu().detach().numpy()))
            list_Loss_D_total.append(np.mean(loss_D_total.cpu().detach().numpy()))
            
            """ G """
            # D(x+fake); D detached
            cat_fake = torch.cat((train_x, fake_xout), dim=1)
            with torch.no_grad():
                out_fake = model_D(cat_fake)
            option1 = out_fake.cpu().detach()
            option1 = torch.mean(option1, dim=(1,2,3)).numpy()
            # Hinge loss of true (1)
            loss_G_real = loss_f(out_fake, torch.ones_like(out_fake))
            # MSE Loss of G
            loss_G_mse = criterion(fake_xout, train_y)
            
            # Learning for G
            loss_G_total = loss_G_real + loss_G_mse
            params_G.zero_grad()
            loss_G_total.backward(retain_graph=True)
            params_G.step()
            
            # Record
            list_Loss_G_real.append(np.mean(loss_G_real.cpu().detach().numpy()))
            list_loss_G_mse.append(np.mean(loss_G_mse.cpu().detach().numpy()))
            list_Loss_G_total.append(np.mean(loss_G_total.cpu().detach().numpy()))
            list_op1.append(option1)
            list_op2.append(option2)
            list_op3.append(option3)
            
        # Assignment
        self.sum_Loss_G_real = np.sum(np.hstack(list_Loss_G_real))
        self.sum_loss_G_mse = np.sum(np.hstack(list_loss_G_mse))
        self.sum_Loss_G_total = np.sum(np.hstack(list_Loss_G_total))
        self.sum_op1 = np.sum(np.hstack(list_op1))
        self.sum_op2 = np.sum(np.hstack(list_op2))
        self.sum_op3 = np.sum(np.hstack(list_op3))
        self.sum_Loss_D_fake = np.sum(np.hstack(list_Loss_D_fake))
        self.sum_Loss_D_real = np.sum(np.hstack(list_Loss_D_real))
        self.sum_Loss_D_total = np.sum(np.hstack(list_Loss_D_total))
        self.sum_real_batchsize = np.sum(list_real_batchsize)

######
def test_eval(criterion2, model_G, model_D, data_y2, data_loader2, TEST_BATCH_SIZE):
    with torch.no_grad():
        test_loss_list = []
        test_score_list = []
        for index_x, index_y in data_loader2:    
            index_x = index_x.int()
            index_y = index_y.int()
            train_x = data_y2[index_x] 
            train_y = data_y2[index_y]
            train_x = train_x.to(deviceGC) 
            train_y = train_y.to(deviceGC)
            train_x = train_x.view(TEST_BATCH_SIZE, 9, shape_tuple[0], shape_tuple[1])
            train_y = train_y.view(TEST_BATCH_SIZE, 3, shape_tuple[0], shape_tuple[1])
            
            test_out = model_G(train_x)
            cat_x = torch.cat((train_x, test_out), dim=1)
            test_score = torch.mean(model_D(cat_x))
            
            # Record
            test_loss_list.append(criterion2(test_out, train_y).cpu().numpy())
            test_score_list.append(float(test_score.cpu().numpy()))
        
        return np.mean(test_loss_list), np.mean(test_score_list)

#==============================#
""" PREPARATION """
# Parameters
BATCH_SIZE = args.batch
EPOCH_START = args.start
EPOCH_MAX = args.epoch+1
SAVE_INTERVAL = 1
TEST_BATCH_SIZE = 1
deviceGC = "cuda:0" if torch.cuda.is_available() else "cpu"

# Files
summary = "summary.txt"
fp = open(summary, "w")
fp.write("#{}\n".format(sys.argv))
fp.write(">Device: "+str(deviceGC)+"\n")
if not args.data:
   print("*** ERROR *** --data necessary\n")
   sys.exit()
with open(args.data, 'r') as file:
        paths_array = [line.strip() for line in file.readlines()]
fp.write(">Paths of data files: {}\n".format(paths_array))
with open(args.test, 'r') as file:
        paths_array2 = [line.strip() for line in file.readlines()]
fp.write(">Paths of test data files: {}\n".format(paths_array2))

# Shape
shape_tuple = tuple(int(x) for x in args.shape.split(','))

# Models
kparam = kernel_param()
kparam.set_parm(args)
model_G = Generator(kparam).to(deviceGC)
model_G.apply(weights_init)
model_D = Discriminator().to(deviceGC)
model_D.apply(weights_init)

# Optimizer and scheduler
learning_rate_G = args.glr
learning_rate_D = args.dlr
params_G = optim.Adam(model_G.parameters(), lr=learning_rate_G, betas=(args.beta1g, 0.999)) 
params_D = optim.Adam(model_D.parameters(), lr=learning_rate_D, betas=(args.beta1d, 0.999))
fp.write(">Optimizer rate: G-> {}\tD-> {}\n".format(params_G.param_groups[0]['lr'], params_D.param_groups[0]['lr']))
fp.write(">Optimizer betas: G-> {}\tD-> {}\n".format(params_G.param_groups[0]['betas'], params_D.param_groups[0]['betas']))

# Model inheritance (parameters, optimizer, loss)
if args.modelG:
    checkpoint_G = torch.load(args.modelG)
    model_G.load_state_dict(checkpoint_G)
if args.modelD:
    checkpoint_D = torch.load(args.modelD)
    model_D.load_state_dict(checkpoint_D)
    
# Loss function
#loss_f = nn.BCEWithLogitsLoss()
loss_f = nn.HingeEmbeddingLoss(margin=0.5)
criterion = nn.MSELoss()
criterion2 = nn.MSELoss() ######

# Saved pth
current_path = os.getcwd()
weightG_dir = current_path+'/'+str(args.save)+"_G_weight"
weightD_dir = current_path+'/'+str(args.save)+"_D_weight"
weight_BEST_dir = current_path+'/'+str(args.save) + "_BEST_weight"
os.mkdir(weightG_dir)
os.mkdir(weightD_dir)
os.mkdir(weight_BEST_dir)

fp.write("Epoch\tLearning_Rate_G\t"
         "Loss_G_real\tloss_G_mse\tLoss_G_total\toption1\toption2\toption3\t"
         "Learning_Rate_D\t"
         "Loss_D_fake\tLoss_D_real\t"
         "Loss_test\tScore_test"
         "\n")

""" RUN """
######
shuffled_data_y_list2 = torch.tensor([])
shuffled_index_x_list2 = torch.tensor([])
shuffled_index_y_list2 = torch.tensor([])
if args.test:
    for path in paths_array2:
        data_y2, index_x2, index_y2 = index_loading(path)
        shuffled_data_y_list2 = torch.cat([shuffled_data_y_list2, data_y2], dim=0)
        shuffled_index_x_list2 = torch.cat([shuffled_index_x_list2, index_x2], dim=0)
        shuffled_index_y_list2 = torch.cat([shuffled_index_y_list2, index_y2], dim=0)
    myDataset2 = torch.utils.data.TensorDataset(shuffled_index_x_list2, shuffled_index_y_list2)
    data_loader2 = DataLoader(myDataset2, batch_size=TEST_BATCH_SIZE)

# Learning
BEST_loss_G = 1000000000
BEST_test_loss =10000000
for epoch in range(EPOCH_START, EPOCH_MAX):
    fp.write(">>Epoch{}\t".format(epoch))
    stack_Loss_G_real = []
    stack_loss_G_mse = []
    stack_Loss_G_total = []
    stack_op1 = []
    stack_op2 = []
    stack_op3 = []
    stack_Loss_D_fake = []
    stack_Loss_D_real = []
    stack_Loss_D_total = []
    stack_real_batchsize = []

    # Different order of read files in each epoch
    training_obj = train_vae()
    if args.fbatch != len(paths_array):
        np.random.shuffle(paths_array) 
        for i in range(0, len(paths_array), args.fbatch):
            # Load supervised data
            shuffled_data_y_list = torch.tensor([])
            shuffled_index_x_list = torch.tensor([])
            shuffled_index_y_list = torch.tensor([])
            paths = paths_array[i:i+args.fbatch]
            for path in paths:
                data_y, index_x, index_y = index_loading(path)
                shuffled_data_y_list = torch.cat([shuffled_data_y_list, data_y], dim=0)
                shuffled_index_x_list = torch.cat([shuffled_index_x_list, index_x], dim=0)
                shuffled_index_y_list = torch.cat([shuffled_index_y_list, index_y], dim=0)
            
            # Load to Learning architecture
            myDataset = torch.utils.data.TensorDataset(shuffled_index_x_list, shuffled_index_y_list)
            data_loader = DataLoader(myDataset, batch_size=BATCH_SIZE, shuffle=True)        
            training_obj.vae(model_G, model_D, params_G, params_D, data_loader, shuffled_data_y_list) ###
            
            # Record of results
            stack_Loss_G_real.append(training_obj.sum_Loss_G_real)
            stack_loss_G_mse.append(training_obj.sum_loss_G_mse)
            stack_Loss_G_total.append(training_obj.sum_Loss_G_total)
            stack_op1.append(training_obj.sum_op1)
            stack_op2.append(training_obj.sum_op2)
            stack_op3.append(training_obj.sum_op3)
            stack_Loss_D_fake.append(training_obj.sum_Loss_D_fake)
            stack_Loss_D_real.append(training_obj.sum_Loss_D_real)
            stack_Loss_D_total.append(training_obj.sum_Loss_D_total)
            stack_real_batchsize.append(training_obj.sum_real_batchsize)
    else: # Process all files at once
        # Load supervised data
        shuffled_data_y_list = torch.tensor([])
        shuffled_index_x_list = torch.tensor([])
        shuffled_index_y_list = torch.tensor([])
        paths = paths_array
        for path in paths:
            data_y, index_x, index_y = index_loading(path)
            shuffled_data_y_list = torch.cat([shuffled_data_y_list, data_y], dim=0)
            shuffled_index_x_list = torch.cat([shuffled_index_x_list, index_x], dim=0)
            shuffled_index_y_list = torch.cat([shuffled_index_y_list, index_y], dim=0)
        
        # Load to Learning architecture
        myDataset = torch.utils.data.TensorDataset(shuffled_index_x_list, shuffled_index_y_list)
        data_loader = DataLoader(myDataset, batch_size=BATCH_SIZE, shuffle=True)
        training_obj.vae(model_G, model_D, params_G, params_D, data_loader, shuffled_data_y_list) ###
        
        # Record
        stack_Loss_G_real.append(training_obj.sum_Loss_G_real)
        stack_loss_G_mse.append(training_obj.sum_loss_G_mse)
        stack_Loss_G_total.append(training_obj.sum_Loss_G_total)
        stack_op1.append(training_obj.sum_op1)
        stack_op2.append(training_obj.sum_op2)
        stack_op3.append(training_obj.sum_op3)
        stack_Loss_D_fake.append(training_obj.sum_Loss_D_fake)
        stack_Loss_D_real.append(training_obj.sum_Loss_D_real)
        stack_Loss_D_total.append(training_obj.sum_Loss_D_total)
        stack_real_batchsize.append(training_obj.sum_real_batchsize)
        
    # Save model
    total_num = sum(stack_real_batchsize)
    if epoch == 1 or epoch == EPOCH_MAX or epoch%5 == 0 or sum(stack_Loss_G_total)/total_num < BEST_loss_G: 
        torch.save(model_G.state_dict(), weightG_dir+"/G_{:05d}.pth".format(epoch), pickle_protocol=4)
        torch.save(model_D.state_dict(), weightD_dir+"/D_{:05d}.pth".format(epoch), pickle_protocol=4)
        if sum(stack_Loss_G_total)/total_num < BEST_loss_G:
            BEST_loss_G = sum(stack_Loss_G_total)/total_num
            os.rename(weightG_dir+"/G_{:05d}.pth".format(epoch), weight_BEST_dir+"/G_{:05d}.pth".format(epoch))
            os.rename(weightD_dir+"/D_{:05d}.pth".format(epoch), weight_BEST_dir+"/D_{:05d}.pth".format(epoch))
            
    ######
    #model_G = model_G.to("cpu")
    model_G.eval()
    test_loss, test_score = test_eval(criterion2, model_G, model_D, shuffled_data_y_list2, data_loader2, TEST_BATCH_SIZE)
    model_G.train()
    if BEST_test_loss > test_loss:
        torch.save(model_G.state_dict(), weight_BEST_dir+"/G_{:05d}.pth".format(epoch), pickle_protocol=4)
        torch.save(model_D.state_dict(), weight_BEST_dir+"/D_{:05d}.pth".format(epoch), pickle_protocol=4)
    BEST_test_loss = test_loss
    
    fp.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(learning_rate_G,
                                                   sum(stack_Loss_G_real)/total_num,
                                                   sum(stack_loss_G_mse)/total_num,
                                                   sum(stack_Loss_G_total)/total_num,
                                                   sum(stack_op1)/total_num,
                                                   sum(stack_op2)/total_num,
                                                   sum(stack_op3)/total_num,
                                                   learning_rate_D,
                                                   sum(stack_Loss_D_fake)/total_num,
                                                   sum(stack_Loss_D_real)/total_num,
                                                   test_loss,
                                                   test_score,
                                                   ))    
    fp.flush()
fp.close()

#==============================#
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
min = int(time/60)
sec = int(time%60)
print("TIME >", "{0:02d}".format(min)+":"+"{0:02d}".format(sec))
