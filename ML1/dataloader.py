from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from scipy import signal
import torch as th
import numpy as np
import pickle as pkl
import sys
try:
    import psutil
    import os
    def mem_usage():
        pid = os.getpid()
        return psutil.Process(pid).memory_info().rss / 1e6
except:
    pass
  
    
    
    
class SpotDataset(Dataset):
    def __init__(self, sim_events_file):
        with open(sim_events_file, 'br') as f:     
            events = pkl.load(f)
            self.events=list(filter(lambda ev: np.sum(ev.hist)>0, events))
               
    def __len__(self): 
        return len(self.events)
    
    def __getitem__(self, ind):
        event=self.events[ind]
        wf=-th.from_numpy(event.wf)
        blw=th.sqrt((wf[:100]**2).mean())
        mask=th.roll(th.heaviside(wf-blw, values=th.tensor([0.0]).double()), -20)
        return wf, mask, np.nonzero(event.hist>0)[0][0]
    
    
class ReconDataset(Dataset):
    def __init__(self, sim_events_file, SPE):
        with open(sim_events_file, 'br') as f:     
            events = pkl.load(f)
            self.events=list(filter(lambda ev: np.sum(ev.hist)>0, events))
            self.SPE=SPE
            
    def __len__(self): 
        return len(self.events)
    
    def __getitem__(self, ind):
        event=self.events[ind]
        wf=-th.from_numpy(event.wf)
        spot=th.tensor(np.nonzero(event.hist>0)[0][0])
        sub_wf=th.roll(wf, -spot.item())[:100]
        norm_factor=sub_wf[self.SPE.argmax()]/self.SPE.amax()
        return sub_wf, event.areas[np.nonzero(event.hist>0)[0][0]], norm_factor, wf, spot
    
    
    
class BreakDataset(Dataset):
    def __init__(self, sim_events_file):
        with open(sim_events_file, 'br') as f:     
            events = pkl.load(f)
            self.events=events
            
               
    def __len__(self): 
        return len(self.events)
    
    def __getitem__(self, ind):
        event=self.events[ind]
        wf=-th.from_numpy(event.wf)
        return wf, len(np.nonzero(event.hist>0)[0])>0, 0
    
    
class CombDataset(Dataset):
    def __init__(self, sim_events_file):
        with open(sim_events_file, 'br') as f:     
            events = pkl.load(f)
            self.events=events
                  
    def __len__(self): 
        return len(self.events)
    
    def __getitem__(self, ind):
        event=self.events[ind]
        wf=-th.from_numpy(event.wf)
        clean_wf=-th.from_numpy(event.clean_wf)
        time_hist=th.from_numpy(event.hist)
        area_hist=th.from_numpy(event.areas)
        blw=th.sqrt((wf[:100]**2).mean())
        mask=th.roll(th.heaviside(wf-blw, values=th.tensor([0.0]).double()), -20)
        return wf, mask, blw, clean_wf, time_hist, area_hist