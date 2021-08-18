import torch.nn as nn
import sys
import torch as th
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
try:
    import psutil
    import os
    def mem_usage():
        pid = os.getpid()
        return psutil.Process(pid).memory_info().rss / 1e6
except:
    def mem_usage():
        return 'No mem_usage on this machine'

    

class Spot(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fit_cum_area=nn.Conv1d(1,1,121, padding=100, bias=False, padding_mode='reflect')
        self.fit_x=nn.Conv1d(1,1,121, padding=100, bias=False, padding_mode='reflect')
        self.fit_div=nn.Conv1d(1,1,121, padding=100, bias=False, padding_mode='reflect')
        self.relu=nn.ReLU()
        self.roll=nn.Conv1d(1,1,101, padding=100, bias=False, padding_mode='reflect')
        self.softmax=nn.Softmax(dim=1)
        self.sub_features=nn.Conv1d(3,15,21, padding=10, bias=False, padding_mode='reflect')
        self.comb_sub_features=nn.Conv3d(1,1,[5,101,3], padding=[0,50,0], bias=False)                 
        self.W = th.nn.Parameter(th.tensor([0.0,5.0,10.0],  requires_grad=True).double())
        self.div_x_w=th.tensor([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105,
                                                -1/280]).unsqueeze(0).unsqueeze(0).double()
        self.run_mean_w=(th.ones(51)/51).unsqueeze(0).unsqueeze(0).double()
        self.softplus=nn.Softplus()
        self.Max=nn.MaxPool1d(1000)
    
    def forward(self, wf, mask, SPE, plot, cluster, I):
        h=self.Max(wf).squeeze(dim=1).tile(1,self.W.shape[0])/SPE.amax()
        W=self.softmax(-self.relu(h-self.W))
        cum_area=wf.cumsum(dim=-1)
        div_x_smd=th.nn.functional.conv1d(th.nn.functional.conv1d(wf,
                                        self.div_x_w, padding=4), self.run_mean_w, padding=25)
        BL=self.roll(cum_area)[:,:,:1000]*self.fit_cum_area.weight.sum()     
        fit_cum_area=self.relu(self.fit_cum_area(cum_area)[:,:,:1000]-BL)
        fit_x=self.relu(self.fit_x(wf))[:,:,:1000]
        fit_div_x=self.relu(self.fit_div(div_x_smd))[:,:,:1000]
            
        features1=th.cat((fit_cum_area, fit_x, fit_div_x), dim=1)
        S=features1.sum(dim=2)+1e-10
        features1_norm=(features1.permute(2,0,1)/S).permute(1,2,0)

        features2=self.sub_features(features1_norm).permute(0,2,1)
        features2_broken=features2.reshape(features2.shape[0],
                                                    features2.shape[1],5,3).permute(1,2,0,3)
        features_chosen=(features2_broken*W).permute(2,1,0,3)
        out=self.softplus(self.comb_sub_features(features_chosen.unsqueeze(dim=1)))\
                            .squeeze(dim=4).squeeze(dim=2).squeeze(dim=1)
        amax=out.amax(dim=1)+1e-10
        out_norm=(out.t()/amax).t()*mask 
        soft_out=self.softmax(10000*out_norm)
        if plot and not cluster:
            plt.figure()
            plt.plot(soft_out[I].detach().numpy(), 'k.-')
            plt.figure()
            plt.plot(out[I].detach().numpy(), 'k.-')
            plt.show()
            fig, ax=plt.subplots(2,3)
            ax[0,0].plot(wf[I,0].detach().numpy(), 'k.-', label='Signal')
            ax[0,0].plot((wf[I,0].amax()*soft_out[I]/soft_out[I].amax()).detach().numpy(), 'r.-')
#             ax[0,0].fill_between(np.arange(1000), 0,
#                                  (wf[I,0].amax()*soft_out[0]/soft_out[0].amax()).detach().numpy())
            ax[0,1].plot(cum_area[I,0].detach().numpy(), 'k.-', label='Integral')
            ax[0,2].plot(div_x_smd[I,0].detach().numpy(), 'k.-', label='Derivative')

            plt.show()
            
        return soft_out
    
    


class Recon(nn.Module):
    def __init__(self):
        super().__init__()
        self.Classify=nn.Sequential(nn.Conv1d(1,5,21, padding=10, bias=False, padding_mode='zeros'),
                                    nn.ReLU(),
                                    nn.Conv1d(5,10,11, padding=5, bias=False, padding_mode='zeros'),
                                    nn.ReLU(),
                                    nn.Conv1d(10,15,5, padding=2, bias=False, padding_mode='zeros'),
                                    nn.ReLU())
        self.Combine=nn.Conv2d(1,1, [15,99], padding=[0,49], bias=False, padding_mode='zeros')
        self.softmax=nn.Softmax(dim=1)       
        self.scale=nn.Sequential(nn.Conv2d(1,1,[100,2], padding=0, bias=False, padding_mode='zeros'),
                                 nn.Sigmoid())
        self.sigmoid=nn.Sigmoid()
        self.relu=nn.ReLU()

    
    
    def forward(self, wf, SPE, plot, cluster, I):
        features=self.Combine(self.Classify(wf).unsqueeze(dim=1)).squeeze(dim=1).squeeze(dim=1)
        S=features.amax(dim=1)
        features_norm=(features.t()/S).t()
        c=self.softmax(5*features)
        y=th.cat(((wf.squeeze(dim=1)*c).unsqueeze(dim=2), (SPE*c).unsqueeze(dim=2)), dim=2)
        a=self.scale(y.unsqueeze(dim=1)).squeeze(dim=3).squeeze(dim=2).squeeze(dim=1)

        
        
        if plot and not cluster:
            fig, ax=plt.subplots(1,1)
            w=(c.t()/c.amax(dim=1)*wf[:,0].amax(dim=1)).t()
            z=self.Combine(self.Classify(wf).unsqueeze(dim=1)).squeeze(dim=1).squeeze(dim=1)
            z=(z.t()/z.amax(dim=1)).t()*wf[I,0].amax()
            ax.plot(wf[I,0].detach().numpy(), 'k.-', label='signal')
            ax.plot(w[I].detach().numpy(), 'r.-', label='mask')
            ax.legend()
#             ax.plot(z[I].detach().numpy(), 'c.-')
#             ax.plot(c[I].detach().numpy(), '.-', color='pink')
            plt.figure()
            plt.plot(y[I,:,0].detach().numpy(), label='Signal')
            plt.plot(y[I,:,1].detach().numpy(), label='SPE')
            plt.show()
        return a+0.1


    
class Break(nn.Module):

    def __init__(self):
        super().__init__()
        self.WF_Features=nn.Conv1d(1,3,51, padding=25, bias=False, padding_mode='zeros')
        self.Cum_Area_Features=nn.Conv1d(1,3,51, padding=25, bias=False, padding_mode='zeros')
        self.Div_Features=nn.Conv1d(1,3,51, padding=25, bias=False, padding_mode='zeros')
        self.relu=nn.ReLU()
        self.Mix=nn.Conv1d(1,2,[9,21], padding=[0,10], bias=False, padding_mode='zeros')
        self.Max=nn.MaxPool1d(1000)
        self.softmax=nn.Softmax(dim=3)
        self.div_x_w=th.tensor([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105,
                                                -1/280]).unsqueeze(0).unsqueeze(0).double()
        self.run_mean_w=(th.ones(51)/51).unsqueeze(0).unsqueeze(0).double()
        self.Sum20=nn.Conv2d(1,1, [1,21], padding=[0,10], bias=False, padding_mode='zeros')

    
    def forward(self, wf, plot, cluster, I):
        div=th.nn.functional.conv1d(th.nn.functional.conv1d(wf,
                                    self.div_x_w, padding=4), self.run_mean_w, padding=25)
        cum_area=wf.cumsum(dim=-1)
        wf_features=self.relu(self.WF_Features(wf))
        cum_area_features=self.relu(self.Cum_Area_Features(wf))
        div_features=self.relu(self.Div_Features(div))
        features=th.cat((wf_features, cum_area_features, div_features), dim=1)
        features2=self.relu(self.Mix(features.unsqueeze(dim=1)))
        features2_max=self.softmax(features2).squeeze(dim=2)
        features2_max_sum20=self.Sum20(features2_max.unsqueeze(dim=1)).squeeze(dim=1)
        out=self.Max(features2_max_sum20).squeeze(dim=2)
        
        if plot:
            y=features2.squeeze()
            z=features2_max.squeeze()
            fig, ax=plt.subplots(4,3)
            ax[0,0].plot(wf[I,0].detach().numpy(), 'k.')
            ax[0,1].plot(cum_area[I,0].detach().numpy(), 'k.')
            ax[0,2].plot(div[I,0].detach().numpy(), 'k.')
            ax[1,0].plot(y[0,0].detach().numpy(), 'k.', label='No signal P')
            ax[1,1].plot(y[1,0].detach().numpy(), 'k.', label='signal P')
            ax[2,0].plot(z[0,0].detach().numpy(), 'k.', label='No signal P')
            ax[2,1].plot(z[1,0].detach().numpy(), 'k.', label='signal P')
            ax[3,0].plot(features2_max_sum20[I,0].detach().numpy(), 'k.', label='No signal P')
            ax[3,1].plot(features2_max_sum20[I,1].detach().numpy(), 'k.', label='signal P')
            ax[1,0].legend()
            ax[1,1].legend()
            
            
            plt.figure()
            plt.plot(self.Sum20.weight.squeeze().detach().numpy(), 'k.-')
            plt.show()
            
        return out
    
    
    
class Comb(nn.Module):
    def __init__(self, SPE, bias, Spot, Recon, Break):
        super().__init__()
        self.Recon=Recon
        self.Break=Break
        self.Spot=Spot
        self.bias=bias
        self.SPE=SPE
        
    def update_mask(self, mask, spot):
        temp=mask.clone()
        temp[spot]=0
        return temp
    def add_area(self, area, spot):
        temp=th.zeros(1000)
        temp[spot]+=area
        return temp
        
    def forward(self, WF, Mask, BLW, plot, cluster, I):
        SPE=self.SPE
        Narea=WF.sum(dim=1)/SPE.sum()
        Niter=0
        Residual_WF=WF.clone()
        Recon_WF=th.zeros_like(WF)
        Areas=th.zeros_like(WF)
        go=self.Break(Residual_WF.unsqueeze(dim=1), False, cluster, I).argmax(dim=1)
        Inds=th.nonzero(th.logical_and(go==1, Niter<2*Narea+1000))[:,0]
        while len(Inds)>0:
            Niter+=1
            residual_wf=Residual_WF[Inds]
            mask=Mask[Inds]
            blw=BLW[Inds]
            spots=self.Spot(residual_wf.unsqueeze(dim=1), mask, SPE, False, cluster, I).argmax(dim=1)
            sub_res_wf=th.stack(list(map(lambda x,i: th.roll(x, -i)[:SPE.shape[0]], residual_wf.unbind(),
                                         spots.tolist())))
            
            inds=th.nonzero(sub_res_wf[:,SPE.argmax()]<blw)[:,0]
            if len(inds)>0:
                mask[inds]=th.stack(list(map(lambda m,i: self.update_mask(m,i), mask[inds].unbind(),
                                         spots[inds].tolist())))
            
            inds=th.nonzero(sub_res_wf[:, SPE.argmax()]>=blw)[:,0]
            if len(inds)>0:
                norm_factor=sub_res_wf[inds, SPE.argmax()].amax()/SPE.amax()
                areas=self.Recon(sub_res_wf[inds].unsqueeze(dim=1), SPE, False, cluster, I)*norm_factor
                spe=(areas*SPE.tile(areas.shape[0],1).t()).t()
                correct_spe=th.cat((spe.unsqueeze(dim=2), sub_res_wf[inds].unsqueeze(dim=2)), dim=2).amin(dim=2)
                ln=th.heaviside(spe-correct_spe, values=th.tensor([0.0]).double()).sum(dim=1)           
                correct_areas=(correct_spe.sum(dim=1)+self.bias*ln)/SPE.sum()
                padd_spe=th.cat((correct_spe, th.zeros(correct_spe.shape[0], WF.shape[1]-correct_spe.shape[1])),
                                dim=1)
                roll_spe=th.stack(list(map(lambda x,i: th.roll(x, i), padd_spe.unbind(), spots[inds].tolist())))
                Recon_WF[Inds[inds]]+=roll_spe
                Residual_WF[Inds[inds]]-=roll_spe
                Areas[Inds[inds]]+=th.stack(list(map(lambda a,i: self.add_area(a,i), correct_areas.tolist(),
                                                     spots[inds].tolist())))
            go=self.Break(Residual_WF.unsqueeze(dim=1), False, cluster, I).argmax(dim=1)
            Inds=th.nonzero(th.logical_and(go==1, Niter<2*Narea+1000))[:,0]
            
        if plot and not cluster:
            plt.figure()
            x=np.arange(1000)/5
            plt.plot(x, WF[I].detach().numpy(), 'k.-')
            plt.plot(x, Recon_WF[I].detach().numpy(), 'r.-')
            plt.plot(x, Residual_WF[I].detach().numpy(), 'g.-')
            plt.legend()
            plt.show()
            
        return Recon_WF, Areas
            
            
            
            
            
            
            
            
            

    
