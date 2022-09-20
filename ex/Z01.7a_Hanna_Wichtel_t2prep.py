experiment_id = 'exE01_FLASH_2D_em42efah'

# %% S0. SETUP env
import sys,os
os.chdir(os.path.abspath(os.path.dirname(__file__)))  #  makes the ex folder your working directory
sys.path.append(os.path.dirname(os.getcwd()))         #  add required folders to path
mpath=os.path.dirname(os.getcwd())
c1=r'codes'; c2=r'codes\GradOpt_python'; c3=r'codes\scannerloop_libs' #  add required folders to path
sys.path += [rf'{mpath}\{c1}',rf'{mpath}\{c2}',rf'{mpath}\{c3}']

## imports for simulation
from GradOpt_python.pulseq_sim_external import sim_external
from GradOpt_python.new_core.util import plot_kspace_trajectory
import math
import numpy as np
import torch
from matplotlib import pyplot as plt

## imports for pypulseq
from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.opts import Opts


# %% S1. SETUP sys

# choose the scanner limits
system = Opts(
    max_grad=28,
    grad_unit='mT/m',
    max_slew=150,
    slew_unit='T/m/s',
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=20e-6,
    grad_raster_time=50e-6
)

# %% S2. DEFINE the sequence 
N = 5 # number of iterations / number of images produced
sequences = []   # array for the different sequences
for i in range(N):
    seq = Sequence()
    sequences.append(seq)

prep_time = np.round(np.linspace(0.005,0.4,N),3)  # create different delays for different T2 weighting

# Define FOV and resolution
fov = 220e-3
slice_thickness = 8e-3
sz = (32, 32)  # spin system size / resolution
Nread = 64  # frequency encoding steps/samples
Nphase = 64  # phase encoding steps/samples

# Define rf events
rf1,gz,gzr = make_sinc_pulse(
    flip_angle=5 * np.pi/180,
    duration=1e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system
)
rf90,_,_ = make_sinc_pulse(
    flip_angle=90 * np.pi/180,
    duration=1e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system
)
rf180,_,_ = make_sinc_pulse(
    flip_angle=180 * np.pi/180,
    duration=1e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system
)
zoom = 1
# Define other gradients and ADC events
gx = make_trapezoid(channel='x', flat_area=Nread/fov*zoom, flat_time=10e-3, system=system)
adc = make_adc(num_samples=Nread, duration=10e-3, delay=gx.rise_time, system=system)
gx_pre = make_trapezoid(channel='x', area=-0.5*gx.area, duration=5e-3, system=system)
gx_spoil = make_trapezoid(channel='x', area=1.5*gx.area, duration=5e-3, system=system)
# iterate over all the sequences with corresponding T2 prep times
rf_phase = 0
rf_inc = 0
rf_spoiling_inc = 117

# ======
# CONSTRUCT SEQUENCE
# ======

for (seq,tau) in zip(sequences,prep_time):   
    # add the T2 preparation
    seq.add_block(rf90)
    seq.add_block(make_delay(tau))
    seq.add_block(rf180)
    seq.add_block(make_delay(tau))
    seq.add_block(rf90)
    seq.add_block(gx_spoil)
    
    for ii in range(-Nphase//2, Nphase//2):  # e.g. -64:63
    
        rf1.phase_offset = rf_phase / 180 * np.pi   # set current rf phase
        
        adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC
    
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]  # increase increment
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]  # increment phase
    
        seq.add_block(rf1,gz)
        seq.add_block(gzr)
        gy_pre = make_trapezoid(channel='y', area=ii/fov*zoom, duration=5e-3, system=system)
        seq.add_block(gx_pre,gy_pre)
        seq.add_block(adc, gx)
        gy_spoil = make_trapezoid(channel='y', area=-ii/fov*zoom, duration=5e-3, system=system)
        seq.add_block(gx_spoil,gy_spoil)
        if ii < Nphase-1:
            seq.add_block(make_delay(0.01))

# %% S3. CHECK, PLOT and WRITE the sequence  as .seq
for (seq,tau,ii) in zip(sequences,prep_time,range(N)):
    ok, error_report = seq.check_timing()  # Check whether the timing of the sequence is correct
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]
    
    # PLOT sequence
    # sp_adc, t_adc = seq.plot(clear=False)
    
    # Prepare the sequence output for the scanner
    seq.set_definition('FOV', [fov*1000, fov*1000, slice_thickness*1000])  # m -> mm
    seq.set_definition('Name', 'gre')
    seq.write('out/t2prep' + str(ii+1) +'.seq')

# %% S4: SETUP SPIN SYSTEM/object on which we can run the MR sequence external.seq from above
from new_core.sim_data import SimData

# (i) load a phantom object from file
if 1:
    obj_p = SimData.load('../data/phantom2D.mat')
    obj_p = SimData.load('../data/numerical_brain_cropped.mat')
    obj_p.T2dash[:] = 30e-3
    obj_p = obj_p.resize(64,64,1)
else:
# or (ii) set phantom  manually to a pixel phantom
    obj_p = torch.zeros((sz[0],sz[1],5)); 
    
    obj_p[7,25,:]=torch.tensor([1, 1, 0.1, 0.1, 0, 1]) # dimensions: PD, T1 T2, T2dash, dB0 rB1
    obj_p[7,23:29,:]=torch.tensor([1, 1, 0.1,0.1, 0, 1]) # dimensions: PD, T1 T2,T2dash, dB0 rB1
    
    obj_p=obj_p.permute(2,0,1)[:,:,:,None]
    obj_p= SimData(obj_p[0,:],obj_p[1,:],obj_p[2,:],obj_p[3,:],obj_p[4,:],obj_p[5,:]*torch.ones(1,obj_p.shape[1],obj_p.shape[2],1),torch.ones(1,obj_p.shape[1],obj_p.shape[2],1),normalize_B0_B1= False)
    obj_p = obj_p.resize(sz[0],sz[1],1)
    
# manipulate obj and plot it
obj_p.B0*=1;
obj_p.plot_sim_data()

# %% S5:. SIMULATE  the external.seq file and add acquired signal to ADC plot
signals = []   # initiate array to save all the signals
for (seq,tau,ii) in zip(sequences,prep_time,range(N)):
    path = 'out/t2prep' + str(ii+1) +'.seq'
    signal, _= sim_external(path,obj=obj_p,plot_seq_k=[0,1])
    # plot the result into the ADC subplot
    # sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
    # sp_adc.plot(t_adc,np.abs(signal.numpy()))
    # seq.plot(signal=signal.numpy())
    
    # additional noise as simulation is perfect
    z = 1e-5*np.random.randn(signal.shape[0], 2).view(np.complex128) 
    #signal+=z
    
    # append the signal to signals array
    signals.append(signal)

# %% S6: MR IMAGE RECON of signal ::: #####################################
spaces = [] # initiate array to save space information
for signal in signals:
    #fig=plt.figure(); # fig.clf()
    #plt.subplot(411); plt.title('ADC signal')
    spectrum=torch.reshape((signal),(Nphase,Nread)).clone().t()
    kspace=spectrum
    #plt.plot(torch.real(signal),label='real')
    #plt.plot(torch.imag(signal),label='imag')
    
    
    #major_ticks = np.arange(0, Nphase*Nread, Nread) # this adds ticks at the correct position szread
    #ax=plt.gca(); ax.set_xticks(major_ticks); ax.grid()
    
    space = torch.zeros_like(spectrum)
    
    # fftshift
    spectrum=torch.fft.fftshift(spectrum,0); spectrum=torch.fft.fftshift(spectrum,1)
    #FFT
    space = torch.fft.ifft2(spectrum)
    # fftshift
    space=torch.fft.ifftshift(space,0); space=torch.fft.ifftshift(space,1)
    
    
# =============================================================================
#     plt.subplot(345); plt.title('k-space')
#     plt.imshow(np.abs(kspace.numpy()))
#     plt.subplot(349); plt.title('k-space_r')
#     plt.imshow(np.log(np.abs(kspace.numpy())))
#     
#     plt.subplot(346); plt.title('FFT-magnitude')
#     plt.imshow(np.abs(space.numpy())); plt.colorbar()
#     plt.subplot(3,4,10); plt.title('FFT-phase')
#     plt.imshow(np.angle(space.numpy()),vmin=-np.pi,vmax=np.pi); plt.colorbar()
#     
#     # % compare with original phantom obj_p.PD
#     from new_core import util
#     PD = util.to_full(obj_p.PD, obj_p.mask).squeeze(0)
#     B0 = util.to_full(obj_p.B0, obj_p.mask).squeeze(0)
#     plt.subplot(348); plt.title('phantom PD')
#     plt.imshow(PD)
#     plt.subplot(3,4,12); plt.title('phantom B0')
#     plt.imshow(B0)
# =============================================================================
    
    # save space in spaces array
    spaces.append(space)
    
# %% S7: T2-Mapping

from scipy import optimize

# define the function of exponential decay
def exponential (tau,T2,A,c):
    return A * np.exp(-tau/T2) + c

# compute the absolute value for each pixel
S = []
for space in spaces:
    space = np.abs(space.numpy())
    S.append(space)
S = np.array(S)

p0 = [0.05,0,0]

# calculate the T2 value for each pixel
T2 = np.zeros(shape=(Nread,Nphase))
for i in range(Nread):
    for j in range(Nphase):
        if S[N-1,i,j] < 5e-7:
            T2[i,j] = 0 # threshold to identify the voxels in the background
        else:
            popt,pcov = optimize.curve_fit(exponential, prep_time, S[:,i,j],p0=p0)
            val = popt[0]
            T2[i,j] = val

T2[T2>100] = 0 # define a threshold to avoid large values due to fit errors

fig=plt.figure("""T2-Mapping""")
plt.imshow(T2)
plt.colorbar()
plt.title('T2-prep')
plt.show()

# %% S8: COMPARISON
def linear(x,a,b):
    return a*x + b

fig = plt.figure('''T2 Phantom''')

data = util.to_numpy(util.to_full(obj_p.T2, obj_p.mask).squeeze(0))
plt.imshow(data[:, :, data.shape[2]//2], interpolation='none')
plt.colorbar()
plt.title('T2 Phantom')
plt.show()

x = data.flatten()
y = T2.flatten()
x_plot = np.linspace(0,np.max(x),100)

popt,pcov = optimize.curve_fit(linear,x,y)

fig = plt.figure('''Comparison''')

plt.scatter(x,y,color='C0',label='data')
plt.plot(x_plot,linear(x_plot,*popt),lw=2,color='C1',label='fit')
plt.text(0.15,20,f'f(x) = {popt[0]:.2f} $\cdot$ x + {popt[1]:.2f}')
plt.xlabel('T2 Phantom')
plt.ylabel('T2 mapped')
plt.legend()
plt.show()

#%% Plotting

# =============================================================================
# fig = plt.figure()
# i = 9
# plt.imshow(S[i,:,:])
# plt.title(f'{prep_time[i]} s')
# =============================================================================

# =============================================================================
# x_time = np.linspace(0,0.4,100)
# popt,pcov = optimize.curve_fit(exponential, prep_time, S[:,36,36], p0=p0)
# print(popt)
# fig=plt.figure("""fit""")
# ax1=plt.subplot(131)
# ax=plt.plot(prep_time,S[:,36,36],'x',label='data')
# plt.plot(x_time,exponential(x_time,*popt),label='fit')
# plt.text(0,0,f'T2 = {popt[0]:.3f}, A = {popt[1]:.3f}, c = {popt[2]:.3f}')
# plt.title('fit')
# plt.legend()
# 
# plt.show()
# print(f'T2 = {popt[0]:.3f}, A = {popt[1]:.3f}, c = {popt[2]:.3f}')
# =============================================================================
