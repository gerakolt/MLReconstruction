import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import pickle as pkl
import time
import torch as th
import sys
try:
    import psutil
    import os
    def mem_usage():
        pid = os.getpid()
        return psutil.Process(pid).memory_info().rss / 1e6
except:
    def mem_usage():
        return 'No mem_usage on this machine'




def grade_spot(dataloader, net, loss_func, SPE, cluster):
    loss = 0
    dis=np.zeros(100)
    bins=np.arange(-50,51)
    
    if th.cuda.is_available():
        net.cuda()
    net.eval()
    
    with th.no_grad():
        for i, (wf, mask, t, n_time) in enumerate(dataloader):
            print('Eval, batch', i, 'out of', len(dataloader), 'mem usage: ', mem_usage())
            if th.cuda.is_available():
                wf = wf.cuda()
                mask = mask.cuda()
                t = t.cuda()
        
            pred_time=net(wf.unsqueeze(dim=1).double(), mask, SPE, False, cluster, 0)
            loss+=loss_func(pred_time, t).item()
            dis+=np.histogram((pred_time.argmax(dim=1)-t).detach().numpy(), bins=bins)[0]
            
    return loss/len(dataloader), dis, bins


def grade_break(dataloader, net, loss_func, cluster):
    loss = 0
    corr_sig=0
    corr_no_sig=0
    
    if th.cuda.is_available():
        net.cuda()
    net.eval()
    
    with th.no_grad():
        for i, (wf, go, n) in enumerate(dataloader):
            print('Eval, batch', i, 'out of', len(dataloader), 'mem usage: ', mem_usage())
            if th.cuda.is_available():
                wf = wf.cuda()
                go = go.cuda()
        
            pred_go=net(wf.unsqueeze(dim=1).double(), False, cluster, 0)
            loss+=loss_func(pred_go, go.long()).item()
            corr_sig+=len(th.nonzero(th.logical_and(pred_go.argmax(dim=1)==1, go==1))[:,0])/\
                    len(th.nonzero(th.logical_or(pred_go.argmax(dim=1)==1, go==1))[:,0])
            corr_no_sig+=len(th.nonzero(th.logical_and(pred_go.argmax(dim=1)==0, go==0))[:,0])/\
                    len(th.nonzero(th.logical_or(pred_go.argmax(dim=1)==0, go==0))[:,0])
            
            
    return loss/len(dataloader), corr_sig/len(dataloader), corr_no_sig/len(dataloader)


def grade_recon(dataloader, net, loss_func, cluster,SPE):
    loss = 0
    
    if th.cuda.is_available():
        net.cuda()
    net.eval()
    
    with th.no_grad():
        for i, (wf, area, n, norm_factor, real_wf, spot) in enumerate(dataloader):
            print('Eval, batch', i, 'out of', len(dataloader), 'mem usage: ', mem_usage())
            if th.cuda.is_available():
                wf = wf.cuda()
                area=area.cuda()
                norm_factor=norm_factor.cuda()
        
            pred_area=net(wf.unsqueeze(dim=1).double(), SPE, False, cluster, 0)
            loss+= loss_func(pred_area*norm_factor/area, area/area).item()
            
    return loss/len(dataloader)


def make_ts(i):
    if i%5==0:
        return []
    else:
        n=np.random.randint(1,100)
        R=0.1
        n1=np.random.binomial(n, R)
        n2=n-n1
        ts=np.concatenate((np.random.exponential(5, size=n1), np.random.exponential(30, size=n2)))
        ind=np.nonzero(np.random.choice(2, size=len(ts), p=[0.7, 0.3])==1)[0]
        ts=ts[ind]
    return ts



