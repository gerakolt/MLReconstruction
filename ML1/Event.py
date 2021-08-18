import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import sys
import dgl
import torch as th

class Event:
    
    def __init__(self, type, SPE, area_sigma, pmt):
        self.type=type   
        self.SPE=SPE
        self.area_sigma=area_sigma
        self.pmt=pmt
    
    def find_groups(self, ts):
        groups=[]
        inits=[]
        fins=[]
        for i in range(len(ts)):
            if i==0:
                inits.append(ts[0])
            elif np.all(ts[i]-ts[:i]>100/5):
                inits.append(ts[i])
            if i==len(ts)-1:
                fins.append(ts[i]+100/5)
            elif np.all(ts[i+1:]-ts[i]>100/5):
                fins.append(ts[i]+100/5)
        if not len(inits)==len(fins):
            colors=['r', 'b', 'g', 'y', 'c', 'pink']
            plt.figure()
            plt.title('len inits not eq to len fins ({}, {})')
            x=np.arange(1000)/5
            plt.plot(x, self.clean_wf, 'k--')
            for i in range(len(inits)):
                print(inits[i])
                j=i%len(colors)
                plt.plot(x[np.argmin(np.abs(x-inits[j]))], self.clean_wf[np.argmin(np.abs(x-inits[j]))],
                         marker='o', color=colors[j])
            for i in range(len(fins)):
                print(fins[i])
                j=i%len(colors)
                plt.plot(x[np.argmin(np.abs(x-fins[j]))], self.clean_wf[np.argmin(np.abs(x-fins[j]))],
                         marker='*', color=colors[j])
            plt.show()
            sys.exit()
        x=np.arange(1000)
        for i in range(len(inits)):
            if not (x[np.argmin(np.abs(x/5-inits[i]))]==998 and x[np.argmin(np.abs(x/5-fins[i]))]==999):
                grp=Group(x[np.argmin(np.abs(x/5-inits[i]))], x[np.argmin(np.abs(x/5-fins[i]))])
                groups.append(grp)
        return groups
            
    
    def make_clean_wf(self, spe, area_sigma):
        self.times+=np.random.randint(20,40)
        self.hist=np.histogram(self.times, np.arange(1001)/5)[0]
        if np.any(self.hist>0):
            self.init=np.nonzero(self.hist>0)[0][0]
        else:
            self.init=-1000
        areas=np.zeros(1000)
        inds=np.nonzero(self.hist>0)[0]
        while any(areas[inds]<0.1):
            areas[inds]=np.random.normal(self.hist[inds], area_sigma*np.sqrt(self.hist[inds]))
        self.clean_wf=signal.convolve(areas, spe, mode='full')[:1000]
        self.groups=self.find_groups(self.times)
        self.areas=areas
        
    def make_wf(self, noise):
        self.noise=noise
        self.aminSPE=np.amin(self.SPE)
        self.argminSPE=np.argmin(self.SPE)
        self.make_clean_wf(self.SPE, self.area_sigma)
        self.wf=self.clean_wf+self.noise
        
    
    def show_clean_wf(self, ax):
        ts=self.times+self.argminSPE/5
        x=np.arange(1000)/5
        
        ax.plot(x, self.clean_wf, 'b.-')
        ax.axhline(0, color='k')
        ax.xlabel('Time [ns]')
        if len(self.times)>0:
            ax.vlines(ts[ts<200], ymin=self.aminSPE, ymax=0, color='k')
        if len(self.groups)>0:
            for group in self.groups:
                ax.fill_between(x[group.init:group.fin], y1=self.clean_wf[group.init:group.fin], alpha=0.5)
        
    def show_wf(self):
        ts=self.times+self.argminSPE/5
        x=np.arange(1000)/5
        fig, ax=plt.subplots(1,1)
#         plt.plot(x, self.clean_wf, 'b.-', label='clean wf')
        ax.plot(x, self.wf, 'r.-', label='wf ')
#         plt.plot(x, self.noise, 'c.-', label='noise')
        ax.axhline(0, color='k')
        ax.set_xlabel('Time [ns]')
        ax.legend()
        for group in self.groups:
            ax.fill_between(x[group.init:group.fin], y1=self.wf[group.init:group.fin], alpha=0.5)
        return fig, ax
        
    def add_pred_groups(self):
        pred_groups_map=self.pred_groups_map.clone().detach()
        while th.any(pred_groups_map==1):
            init=th.amin(th.nonzero(pred_groups_map==1)[0])
            if th.any(pred_groups_map[init:]==0):
                fin=init+th.amin(th.nonzero(pred_groups_map[init:]==0)[0])-1
            else:
                fin=999
            grp=Group(init,fin)
            self.pred_groups.append(grp)
            pred_groups_map[init:fin+1]=0
        
        
        
        
    
        
        
        
class Group:
    
    def __init__(self, init, fin):
        self.init=init
        self.fin=fin
        
    
        
        