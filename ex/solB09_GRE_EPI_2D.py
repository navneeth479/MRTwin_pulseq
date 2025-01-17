experiment_id = 'exB09_GRE_EPI_2D_approach2'

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

## choose the scanner limits
system = Opts(max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s', rf_ringdown_time=20e-6,
                 rf_dead_time=100e-6, adc_dead_time=20e-6,grad_raster_time=50*10e-6)

# %% S2. DEFINE the sequence 
seq = Sequence()

# Define FOV and resolution
fov = 1000e-3 
slice_thickness=8e-3

Nread = 32   # frequency encoding steps/samples
Nphase = 32    # phase encoding steps/samples
n_slices = 1

# ======
# CREATE EVENTS
# ======
# Create 90 degree slice selection pulse and gradient
rf, gz, gzr = make_sinc_pulse(flip_angle=90 * math.pi / 180, duration=1e-3,slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4, system=system)

# Define other gradients and ADC events
delta_k = 1 / fov
k_width = Nread * delta_k
dwell_time = 4e-6
readout_time = Nread * dwell_time
flat_time = np.ceil(readout_time * 1e5) * 1e-5  # round-up to the gradient raster
gx = make_trapezoid(
    channel="x",
    system=system,
    amplitude=k_width / readout_time,
    flat_time=flat_time
)
adc = make_adc(
    num_samples=Nread,
    duration=readout_time,
    delay=gx.rise_time + flat_time / 2 - (readout_time - dwell_time) / 2
)


# Pre-phasing gradients
pre_time = 8e-4
gx_pre = make_trapezoid(
    channel="x", system=system, area=-gx.area / 2, duration=pre_time
)

gy_pre = make_trapezoid(
    channel="y", system=system, area=-Nphase / 2 * delta_k, duration=pre_time
)

# Phase blip in the shortest possible time
dur = np.ceil(2 * np.sqrt(delta_k / system.max_slew) / 10e-6) * 10e-6
gy = make_trapezoid(channel='y', area=delta_k, duration=1e-3, system=system)

# ======
# CONSTRUCT SEQUENCE
# ======
# Define sequence blocks
for s in range(n_slices):
    rf.freq_offset = gz.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
    seq.add_block(rf, gz)
    seq.add_block(gzr)
    seq.add_block(gx_pre, gy_pre)
    for i in range(Nphase):
        seq.add_block(gx, adc)  # Read one line of k-space
        seq.add_block(gy)  # Phase blip
        gx.amplitude = -gx.amplitude  # Reverse polarity of read gradient


# %% S3. CHECK, PLOT and WRITE the sequence  as .seq
ok, error_report = seq.check_timing()  # Check whether the timing of the sequence is correct
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]

# PLOT sequence
sp_adc,t_adc =seq.plot(clear=False)
#   
if 0:
    sp_adc,t_adc =seq.plot(clear=True)


# Prepare the sequence output for the scanner
seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'gre')
seq.write('out/external.seq')
seq.write('out/' + experiment_id +'.seq')


# %% S4: SETUP SPIN SYSTEM/object on which we can run the MR sequence external.seq from above
from new_core.sim_data import SimData
sz=[64,64]
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
signal, _= sim_external(obj=obj_p,plot_seq_k=[0,1],M_threshold=-1e-3)
# plot the result into the ADC subplot
sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
sp_adc.plot(t_adc,np.abs(signal.numpy()))
# seq.plot(signal=signal.numpy())

# additional noise as simulation is perfect
z = 1e-5*np.random.randn(signal.shape[0], 2).view(np.complex128) 
signal+=z
# %% S6: MR IMAGE RECON of signal ::: #####################################
fig=plt.figure(); # fig.clf()
plt.subplot(411); plt.title('ADC signal')
spectrum=torch.reshape((signal),(Nphase,Nread)).clone().t()
kspace=spectrum.numpy()
plt.plot(torch.real(signal),label='real')
plt.plot(torch.imag(signal),label='imag')


major_ticks = np.arange(0, Nphase*Nread, Nread) # this adds ticks at the correct position szread
ax=plt.gca(); ax.set_xticks(major_ticks); ax.grid()

space = np.zeros_like(kspace)

#kspace[:,::2]=kspace[::-1, ::2]
#kspace[1:,::2]=kspace[:-1,::2]

# fftshift
kspace=np.fft.fftshift(kspace)
#FFT
space = np.fft.ifft2(kspace)
# fftshift
space= np.fft.ifftshift(space)


plt.subplot(345); plt.title('k-space')
plt.imshow(np.abs(kspace))
plt.subplot(349); plt.title('k-space_r')
plt.imshow(np.log(np.abs(kspace)))

plt.subplot(346); plt.title('FFT-magnitude')
plt.imshow(np.abs(space)); plt.colorbar()
plt.subplot(3,4,10); plt.title('FFT-phase')
plt.imshow(np.angle(space),vmin=-np.pi,vmax=np.pi); plt.colorbar()

# % compare with original phantom obj_p.PD
from new_core import util
PD = util.to_full(obj_p.PD, obj_p.mask).squeeze(0)
B0 = util.to_full(obj_p.B0, obj_p.mask).squeeze(0)
plt.subplot(348); plt.title('phantom PD')
plt.imshow(PD)
plt.subplot(3,4,12); plt.title('phantom B0')
plt.imshow(B0)

