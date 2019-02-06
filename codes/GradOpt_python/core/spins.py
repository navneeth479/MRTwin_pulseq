import numpy as np
import torch
import cv2

# WHAT we measure
class SpinSystem():
    
    def __init__(self,sz,NVox,NSpins,use_gpu):
        
        self.sz = sz                                             # image size
        self.NVox = sz[0]*sz[1]                                 # voxel count
        self.NSpins = NSpins              # number of spin sims in each voxel
        
        self.PD = None                        # proton density tensor (NVox,)
        self.T1 = None                          # T1 relaxation times (NVox,)
        self.T2 = None                          # T2 relaxation times (NVox,)
        self.dB0 = None                        # spin off-resonance (NSpins,)
        
        self.M0 = None     # initial magnetization state (NSpins,NRep,NVox,4)
        self.M = None       # curent magnetization state (NSpins,NRep,NVox,4)
        
        # aux
        self.img = None
        self.R2 = None
        self.use_gpu = use_gpu
        
    # device setter
    def setdevice(self,x):
        if self.use_gpu:
            x = x.cuda(0)
            
        return x
    
    # get magnitude image
    def magimg(self, x):
      return np.sqrt(np.sum(np.abs(x)**2,2))    
    
    def set_system(self):
        
        # load image
        m = np.load('../../data/phantom.npy')
        m = cv2.resize(m, dsize=(self.sz[0], self.sz[1]), interpolation=cv2.INTER_CUBIC)
        m = m / np.max(m)
        self.img = m.copy()
        
        # set relaxations (unit - seconds) and proton density
        PD = torch.from_numpy(self.magimg(m).reshape([self.NVox])).float()
        T1 = torch.ones(self.NVox, dtype=torch.float32)*4
        T2 = torch.ones(self.NVox, dtype=torch.float32)*2
        T2[0:self.NVox/2] = 0.09
        
        # set NSpins offresonance (from R2)
        factor = (0*1e0*np.pi/180) / self.NSpins
        dB0 = torch.from_numpy(factor*np.arange(0,self.NSpins).reshape([self.NSpins])).float()
        
        self.T1 = self.setdevice(T1)
        self.T2 = self.setdevice(T2)
        self.PD = self.setdevice(PD)
        self.dB0 = self.setdevice(dB0)
        
    def set_initial_magnetization(self,NRep=None):
        
        if NRep == None:      # allow magnetization transfer over repetitions
            NRep = 1
            
        M0 = torch.zeros((self.NSpins,NRep,self.NVox,4), dtype=torch.float32)
        
        M0 = self.setdevice(M0)
        
        # set initial longitudinal magnetization value
        M0[:,:,:,2:] = 1
        M0[:,:,:,2:] = M0[:,:,:,2:] * self.PD.view([self.NVox,1])    # weight by proton density
        
        M = M0.clone().view([self.NSpins,NRep,self.NVox,4,1])
        
        self.M0 = M0
        
        self.M = self.setdevice(M)
        
        
        
