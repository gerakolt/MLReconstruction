from dataloader import ReconDataset
from torch.utils.data import Dataset, DataLoader
from model import Spot, Recon
import sys
import torch.nn as nn
import torch.optim as optim
from fun import grade_recon
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


net = Recon().double()
# net.load_state_dict(th.load('Recon.pt',map_location=th.device('cpu')))

if th.cuda.is_available():
    net.cuda()

loss_func=nn.L1Loss()

optimizer = optim.Adam(net.parameters(), lr=1e-4) 

n_epochs = 5550


Train_Loss = []
Valid_Loss = []


time0=time.time()
training_ds = ReconDataset(path_to_training_data, SPE)
validation_ds = ReconDataset(path_to_validation_data, SPE)
training_dataloader = DataLoader(training_ds,batch_size=batch_size, shuffle=False)
valid_dataloader = DataLoader(validation_ds,batch_size=batch_size, shuffle=False)
for epoch in range(n_epochs):
        
    net.eval() #put the net into evaluation mode
    train_loss=grade_recon(training_dataloader, net,loss_func, cluster, SPE)
    valid_loss=grade_recon(valid_dataloader, net,loss_func, cluster, SPE)
         
    Train_Loss.append(train_loss)    
    Valid_Loss.append(valid_loss)  
    
    np.savez('Recon', Train_Loss=Train_Loss, Valid_Loss=Valid_Loss)
    
    if len(Valid_Loss)==1 or Valid_Loss[-1]<=min(Valid_Loss):
        th.save(net.state_dict(), 'Recon.pt')
        
        
    time0=time.time()
    net.train() # put the net into "training mode"
    for i, (wf, area, n, norm_factor, full_wf, spot) in enumerate(training_dataloader):
        print('Train, batch', i, 'out of', len(training_dataloader), 'epoch', epoch, 'mem usage: ',
              mem_usage())
        if th.cuda.is_available():
            wf = wf.cuda()
            area=area.cuda()
            norm_factor=norm_factor.cuda()
        
        I=293
        optimizer.zero_grad()
        pred_area=net(wf.unsqueeze(dim=1).double(), SPE, False, cluster, I)
        dif=(pred_area*norm_factor/area-area/area).abs()
        
#         plt.figure()
#         spe=th.zeros(1000)
#         real_spe=th.zeros(1000)
#         real_spe[spot[I]:spot[I]+100]=area[I]*SPE
#         spe[spot[I]:spot[I]+100]+=(pred_area[I]*norm_factor[I])*SPE
#         plt.plot(full_wf[I], 'k.-')
#         plt.plot(spe.detach().numpy(), 'r.-', label='pred')
#         plt.plot(real_spe.detach().numpy(), 'c.-', label='real')
#         plt.legend()
#         plt.show()
        
#         print(dif.argmax())
#         sys.exit()
        loss = loss_func(pred_area*norm_factor, area)
        if th.isnan(loss) or th.isinf(loss):
            print('loss is nan or inf')
            sys.exit()
        loss.backward()
        optimizer.step()

    
