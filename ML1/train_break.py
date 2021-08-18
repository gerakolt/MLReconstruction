from dataloader import BreakDataset
from torch.utils.data import Dataset, DataLoader
from model import Spot, Recon, Break
import sys
import torch.nn as nn
import torch.optim as optim
from fun import grade_break
import torch as th
import time
import numpy as np
from matplotlib import pyplot as plt

try:
    import psutil
    import os
    def mem_usage():
        pid = os.getpid()
        return psutil.Process(pid).memory_info().rss / 1e6
except:
    def mem_usage():
        return 'No mem_usage on this machine'





pmt=4
try:
    path='/home/gerak/Desktop/DireXeno/150621/f0/'
    data=np.load(path+'SPE{}.npz'.format(pmt))
    area_sigma=data['area_sigma']
    SPE=data['spe']
    SPE=-th.from_numpy(SPE[np.argmin(SPE)-20:np.argmin(SPE)+80])
    path_to_training_data=path+'train_sim_events/PMT{}'.format(pmt)
    path_to_validation_data=path+'valid_sim_events/PMT{}'.format(pmt)
    training_ds = BreakDataset(path_to_training_data)
    validation_ds = BreakDataset(path_to_validation_data)
    cluster=False
except:
    path='/storage/xenon/gerak/150621/f0/'
    data=np.load(path+'SPE{}.npz'.format(pmt))
    area_sigma=data['area_sigma']
    SPE=data['spe']
    SPE=-th.from_numpy(SPE[np.argmin(SPE)-20:np.argmin(SPE)+80])
    path_to_training_data=path+'train_sim_events/PMT{}_UpTo30PEs'.format(pmt)
    path_to_validation_data=path+'valid_sim_events/PMT{}_UpTo30PEs'.format(pmt)
    training_ds = BreakDataset(path_to_training_data)
    validation_ds = BreakDataset(path_to_validation_data)
    cluster=True
    
batch_size=500
training_dataloader = DataLoader(training_ds,batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(validation_ds,batch_size=batch_size, shuffle=False)

net = Break().double()
net.load_state_dict(th.load('Break.pt',map_location=th.device('cpu')))

if th.cuda.is_available():
    net.cuda()

w1=4/5
w0=1/5
loss_func=nn.CrossEntropyLoss(weight=th.tensor([w1, w0]).double())

optimizer = optim.Adam(net.parameters(), lr=1e-4) 

n_epochs = 5550


Train_Loss = []
Valid_Loss = []
Train_Corr_Signal = []
Valid_Corr_Signal = []
Train_Corr_no_Signal = []
Valid_Corr_no_Signal = []

time0=time.time()

for epoch in range(n_epochs):
    
#     net.eval() #put the net into evaluation mode
# #     train_loss, train_corr_signal, train_corr_no_signal=grade_break(training_dataloader, net,loss_func, cluster)
#     valid_loss, valid_corr_signal, valid_corr_no_signal=grade_break(valid_dataloader, net,loss_func,
#                                                                     cluster)
         
# #     Train_Loss.append(train_loss)    
#     Valid_Loss.append(valid_loss)
    
# #     Train_Corr_Signal.append(train_corr_signal)
#     Valid_Corr_Signal.append(valid_corr_signal)
# #     Train_Corr_no_Signal.append(train_corr_no_signal)
#     Valid_Corr_no_Signal.append(valid_corr_no_signal)
    
#     np.savez('Break', Train_Loss=Train_Loss, Valid_Loss=Valid_Loss, 
#             Train_Corr_Signal=Train_Corr_Signal, Valid_Corr_Signal=Valid_Corr_Signal,
#             Train_Corr_no_Signal=Train_Corr_no_Signal, Valid_Corr_no_Signal=Valid_Corr_no_Signal)
#     time0=time.time()
#     if len(Valid_Loss)==1 or Valid_Loss[-1]<=min(Valid_Loss):
#         th.save(net.state_dict(), 'Break.pt')        
        
    net.train() # put the net into "training mode"
    for i, (wf, go, n) in enumerate(training_dataloader):
        print('Train, batch', i, 'out of', len(training_dataloader), 'epoch', epoch, 'mem usage: ', mem_usage())
        if th.cuda.is_available():
            wf = wf.cuda()
            go = go.cuda()
        
        optimizer.zero_grad()
        pred_go=net(wf.unsqueeze(dim=1).double(), False, cluster, 0)
        loss = loss_func(pred_go, go.long())
        if th.isnan(loss) or th.isinf(loss):
            print('loss is nan or inf')
            sys.exit()
        loss.backward()
        optimizer.step()

    
