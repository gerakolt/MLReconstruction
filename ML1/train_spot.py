from dataloader import SpotDataset
from torch.utils.data import Dataset, DataLoader
from model import Spot, Recon
import sys
import torch.nn as nn
import torch.optim as optim
from fun import grade_spot
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
    cluster=False
except:
    path='/storage/xenon/gerak/150621/f0/'
    data=np.load(path+'SPE{}.npz'.format(pmt))
    area_sigma=data['area_sigma']
    SPE=data['spe']
    SPE=-th.from_numpy(SPE[np.argmin(SPE)-20:np.argmin(SPE)+80])
    path_to_training_data=path+'train_sim_events/PMT{}'.format(pmt)
    path_to_validation_data=path+'valid_sim_events/PMT{}'.format(pmt)
    cluster=True
    
batch_size=500


net = Spot().double()
net.load_state_dict(th.load('Spot.pt',map_location=th.device('cpu')))
# net.roll.weight=th.nn.Parameter(th.tensor([[[1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]).double())
# net.fit_x.weight=th.nn.Parameter(th.tensor([[[-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]]).double())
# net.fit_div.weight=th.nn.Parameter(th.tensor([[[-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]]).double())
# net.fit_cum_area.weight=th.nn.Parameter(th.tensor([[[-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]]).double())

if th.cuda.is_available():
    net.cuda()

loss_func=nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=1e-2) 

n_epochs = 5550


Train_Loss = []
Valid_Loss = []


time0=time.time()
training_ds = SpotDataset(path_to_training_data, SPE)
validation_ds = SpotDataset(path_to_validation_data, SPE)
training_dataloader = DataLoader(training_ds,batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(validation_ds,batch_size=batch_size, shuffle=False)

for epoch in range(n_epochs):
    net.eval() #put the net into evaluation mode
    train_loss, train_dis, bins=grade_spot(training_dataloader, net,loss_func, SPE, cluster)
    valid_loss, valid_dis, bins=grade_spot(valid_dataloader, net,loss_func, SPE, cluster)
         
    Train_Loss.append(train_loss)    
    Valid_Loss.append(valid_loss)  
    Valid_Dis=valid_dis
    Train_Dis=train_dis
    
    np.savez('Spot', Valid_Dis=Valid_Dis, Train_Dis=Train_Dis, Valid_Loss=Valid_Loss,
             Train_Loss=Train_Loss, bins=bins)
    
    if len(Valid_Loss)==1 or Valid_Loss[-1]<=min(Valid_Loss):
        th.save(net.state_dict(), 'Spot.pt')
        
        
    time0=time.time()
    net.train() # put the net into "training mode"
    for i, (wf, mask, t, n_time) in enumerate(training_dataloader):
        print('Train, batch', i, 'out of', len(training_dataloader),
              'epoch', epoch, 'mem usage: ', mem_usage())
        if th.cuda.is_available():
            wf = wf.cuda()
            mask = mask.cuda()
            t = t.cuda()
        
        optimizer.zero_grad()
        pred_time=net(wf.unsqueeze(dim=1).double(), mask, SPE, False, cluster, 0)
        loss = loss_func(pred_time, t)
        if th.isnan(loss) or th.isinf(loss):
            print('loss is nan or inf')
            sys.exit()
        loss.backward()
        optimizer.step()

    
