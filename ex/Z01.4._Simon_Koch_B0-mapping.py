experiment_id = 'B0_mapping_Simon_Koch'


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

# phase unwraping
from skimage import data, img_as_float, color, exposure
from skimage.restoration import unwrap_phase

# further import for plotting and simulation
from new_core import util
from new_core.sim_data import SimData

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

# delays for B0-phaseshift
delay1 = 0.0007
delay2 = 0.001
delay3 = 0.0013

delay4 = 0.05

# gamma-factor
gamma = 267.522*1e6 # rad*Hz/T
gamma_2pi = 42.577*1e6 # Hz/T

# Define FOV and resolution
fov = 220e-3
slice_thickness = 8e-3
sz = (32, 32)  # spin system size / resolution
Nread = 64  # frequency encoding steps/samples
Nphase = 64  # phase encoding steps/samples

# %% S2.0 FLASH-sequence without delay
seq = Sequence()

# Define rf events
rf1, gz, gzr = make_sinc_pulse(
    flip_angle=5 * np.pi/180,
    duration=1e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system
)

zoom=1
# Define other gradients and ADC events
gx = make_trapezoid(channel='x', flat_area=Nread/fov*zoom, flat_time=10e-3, system=system)
adc = make_adc(num_samples=Nread, duration=10e-3, delay=gx.rise_time, system=system)
gx_pre = make_trapezoid(channel='x', area=-0.5*gx.area, duration=5e-3, system=system)
gx_spoil = make_trapezoid(channel='x', area=1.5*gx.area, duration=5e-3, system=system)

rf_phase = 0
rf_inc = 0
rf_spoiling_inc = 117

# calculate TE1 
#TE1 = calc_duration(gz) + calc_duration(gzr) + calc_duration(gx_pre) + calc_duration(gx)/2
#print(TE1)

# ======
# CONSTRUCT SEQUENCE
# ======

for ii in range(-Nphase//2, Nphase//2):  # e.g. -64:63

    rf1.phase_offset = rf_phase / 180 * np.pi   # set current rf phase
    
    adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC

    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]  # increase increment
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]  # increment phase

    seq.add_block(rf1,gz) 
    seq.add_block(gzr)
    gy_pre = make_trapezoid(channel='y', area=ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_pre, gy_pre) 
  
    seq.add_block(adc, gx)
    gy_spoil = make_trapezoid(channel='y', area=-ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_spoil, gy_spoil)
    if ii < Nphase-1:
        seq.add_block(make_delay(0.001))
        
        
# %% S3.0 CHECK
ok, error_report = seq.check_timing()  # Check whether the timing of the sequence is correct
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]

# PLOT sequence
sp_adc, t_adc = seq.plot(clear=False)

# Prepare the sequence output for the scanner
seq.set_definition('FOV', [fov*1000, fov*1000, slice_thickness*1000])  # m -> mm
seq.set_definition('Name', 'gre')
seq.write('out/external.seq')
seq.write('out/' + experiment_id +'.seq')


# %% S4.0 SETUP SPIN SYSTEM/object on which we can run the MR sequence external.seq from above

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


# %% S5.0 simulation 1

use_simulation = 1

if use_simulation:
    signal, _= sim_external(obj=obj_p,plot_seq_k=[0,1])
else:
    signal = util.get_signal_from_real_system('out/' + experiment_id +'.seq.dat', Nphase, Nread)

# plot the result into the ADC subplot
sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
sp_adc.plot(t_adc,np.abs(signal.numpy()))
# seq.plot(signal=signal.numpy())


# %% S6.0 MR image 1
fig=plt.figure(); # fig.clf()
plt.subplot(411); plt.title('ADC signal')
spectrum=torch.reshape((signal),(Nphase,Nread)).clone().t()
kspace=spectrum
plt.plot(torch.real(signal),label='real')
plt.plot(torch.imag(signal),label='imag')


major_ticks = np.arange(0, Nphase*Nread, Nread) # this adds ticks at the correct position szread
ax=plt.gca(); ax.set_xticks(major_ticks); ax.grid()

space = torch.zeros_like(spectrum)

# fftshift
spectrum=torch.fft.fftshift(spectrum,0); spectrum=torch.fft.fftshift(spectrum,1)
#FFT
space = torch.fft.ifft2(spectrum)
# fftshift
space=torch.fft.ifftshift(space,0); space=torch.fft.ifftshift(space,1)

space1 = space
mask = (space1.abs().numpy()>1e-5)

phase1 = np.angle(space1.numpy())

plt.subplot(345); plt.title('k-space')
plt.imshow(np.abs(kspace.numpy()))
plt.subplot(349); plt.title('k-space_r')
plt.imshow(np.log(np.abs(kspace.numpy())))

plt.subplot(346); plt.title('FFT-magnitude')
plt.imshow(np.abs(space.numpy())); plt.colorbar()
plt.subplot(3,4,10); plt.title('FFT-phase')
plt.imshow(phase1); plt.colorbar()


# % compare with original phantom obj_p.PD
PD = util.to_full(obj_p.PD, obj_p.mask).squeeze(0)
B0 = util.to_full(obj_p.B0, obj_p.mask).squeeze(0)
plt.subplot(348); plt.title('PD-Phantom')
plt.imshow(PD)
plt.subplot(3,4,12); plt.title('B0-Phantom')
plt.imshow(B0)


# %% S2.1 FLASH-sequence 2 with delay1

seq = Sequence()

# Define rf events
rf1, gz, gzr = make_sinc_pulse(
    flip_angle=5 * np.pi/180,
    duration=1e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system
)

zoom=1
# Define other gradients and ADC events
gx = make_trapezoid(channel='x', flat_area=Nread/fov*zoom, flat_time=10e-3, system=system)
adc = make_adc(num_samples=Nread, duration=10e-3, delay=gx.rise_time, system=system)
gx_pre = make_trapezoid(channel='x', area=-0.5*gx.area, duration=5e-3, system=system)
gx_spoil = make_trapezoid(channel='x', area=1.5*gx.area, duration=5e-3, system=system)

rf_phase = 0
rf_inc = 0
rf_spoiling_inc = 117


# ======
# CONSTRUCT SEQUENCE
# ======

for ii in range(-Nphase//2, Nphase//2):  # e.g. -64:63

    rf1.phase_offset = rf_phase / 180 * np.pi   # set current rf phase
    
    adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC

    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]  # increase increment
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]  # increment phase

    seq.add_block(rf1,gz) 
    seq.add_block(gzr)
    gy_pre = make_trapezoid(channel='y', area=ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_pre, gy_pre)
    seq.add_block(make_delay(delay1)) 
    seq.add_block(adc, gx)
    gy_spoil = make_trapezoid(channel='y', area=-ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_spoil, gy_spoil)
    if ii < Nphase-1:
        seq.add_block(make_delay(0.001))
             
# %% S3.1 CHECK 2
ok, error_report = seq.check_timing()  # Check whether the timing of the sequence is correct
if ok:
    print('Timing check 2 passed successfully')
else:
    print('Timing check 2 failed. Error listing follows:')
    [print(e) for e in error_report]

# PLOT sequence
sp_adc, t_adc = seq.plot(clear=False)

# Prepare the sequence output for the scanner
seq.set_definition('FOV', [fov*1000, fov*1000, slice_thickness*1000])  # m -> mm
seq.set_definition('Name', 'gre')
seq.write('out/external.seq')
seq.write('out/' + experiment_id +'.seq')


# %% S5.1 simulation 2

use_simulation = 1

if use_simulation:
    signal, _= sim_external(obj=obj_p,plot_seq_k=[0,1])
else:
    signal = util.get_signal_from_real_system('out/' + experiment_id +'.seq.dat', Nphase, Nread)

# plot the result into the ADC subplot
sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
sp_adc.plot(t_adc,np.abs(signal.numpy()))
# seq.plot(signal=signal.numpy())


# %% S6.1 MR image 2

fig2=plt.figure(); # fig.clf()
plt.subplot(411); plt.title('ADC signal')
spectrum=torch.reshape((signal),(Nphase,Nread)).clone().t()
kspace=spectrum
plt.plot(torch.real(signal),label='real')
plt.plot(torch.imag(signal),label='imag')


major_ticks = np.arange(0, Nphase*Nread, Nread) # this adds ticks at the correct position szread
ax=plt.gca(); ax.set_xticks(major_ticks); ax.grid()

space = torch.zeros_like(spectrum)

# fftshift
spectrum=torch.fft.fftshift(spectrum,0); spectrum=torch.fft.fftshift(spectrum,1)
#FFT
space = torch.fft.ifft2(spectrum)
# fftshift
space=torch.fft.ifftshift(space,0); space=torch.fft.ifftshift(space,1)

space2 = space

phase2 = np.angle(space2.numpy())


plt.subplot(345); plt.title('k-space')
plt.imshow(np.abs(kspace.numpy()))
plt.subplot(349); plt.title('k-space_r')
plt.imshow(np.log(np.abs(kspace.numpy())))

plt.subplot(346); plt.title('FFT-magnitude')
plt.imshow(np.abs(space.numpy())); plt.colorbar()
plt.subplot(3,4,10); plt.title('FFT-phase')
plt.imshow(phase2); plt.colorbar()

   
plt.subplot(348); plt.title('PD-Phantom')
plt.imshow(PD)
plt.subplot(3,4,12); plt.title('B0-Phantom')
plt.imshow(B0)
              
                 
# %%  S2.2 FLASH-sequence 3 with delay2

seq = Sequence()

# Define rf events
rf1, gz, gzr = make_sinc_pulse(
    flip_angle=5 * np.pi/180,
    duration=1e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system
)

zoom=1
# Define other gradients and ADC events
gx = make_trapezoid(channel='x', flat_area=Nread/fov*zoom, flat_time=10e-3, system=system)
adc = make_adc(num_samples=Nread, duration=10e-3, delay=gx.rise_time, system=system)
gx_pre = make_trapezoid(channel='x', area=-0.5*gx.area, duration=5e-3, system=system)
gx_spoil = make_trapezoid(channel='x', area=1.5*gx.area, duration=5e-3, system=system)

rf_phase = 0
rf_inc = 0
rf_spoiling_inc = 117


# ======
# CONSTRUCT SEQUENCE
# ======

for ii in range(-Nphase//2, Nphase//2):  # e.g. -64:63

    rf1.phase_offset = rf_phase / 180 * np.pi   # set current rf phase
    
    adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC

    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]  # increase increment
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]  # increment phase

    seq.add_block(rf1,gz) 
    seq.add_block(gzr)
    gy_pre = make_trapezoid(channel='y', area=ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_pre, gy_pre) 
    seq.add_block(make_delay(delay2)) 
    seq.add_block(adc, gx)
    gy_spoil = make_trapezoid(channel='y', area=-ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_spoil, gy_spoil)
    if ii < Nphase-1:
        seq.add_block(make_delay(0.001))
        
# %% S3.2 CHECK 3
ok, error_report = seq.check_timing()  # Check whether the timing of the sequence is correct
if ok:
    print('Timing check 3 passed successfully')
else:
    print('Timing check 3 failed. Error listing follows:')
    [print(e) for e in error_report]

# PLOT sequence
#sp_adc, t_adc = seq.plot(clear=False)

# Prepare the sequence output for the scanner
seq.set_definition('FOV', [fov*1000, fov*1000, slice_thickness*1000])  # m -> mm
seq.set_definition('Name', 'gre')
seq.write('out/external.seq')
seq.write('out/' + experiment_id +'.seq')    

# %% S5.2 simulation 3 

use_simulation = 1

if use_simulation:
    signal, _= sim_external(obj=obj_p,plot_seq_k=[0,1])
else:
    signal = util.get_signal_from_real_system('out/' + experiment_id +'.seq.dat', Nphase, Nread)

# plot the result into the ADC subplot
#sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
#sp_adc.plot(t_adc,np.abs(signal.numpy()))
# seq.plot(signal=signal.numpy())

# %% S6.2 MR Image 3

fig3=plt.figure(); # fig.clf()
plt.subplot(411); plt.title('ADC signal')
spectrum=torch.reshape((signal),(Nphase,Nread)).clone().t()
kspace=spectrum
plt.plot(torch.real(signal),label='real')
plt.plot(torch.imag(signal),label='imag')


major_ticks = np.arange(0, Nphase*Nread, Nread) # this adds ticks at the correct position szread
ax=plt.gca(); ax.set_xticks(major_ticks); ax.grid()

space = torch.zeros_like(spectrum)

# fftshift
spectrum=torch.fft.fftshift(spectrum,0); spectrum=torch.fft.fftshift(spectrum,1)
#FFT
space = torch.fft.ifft2(spectrum)
# fftshift
space=torch.fft.ifftshift(space,0); space=torch.fft.ifftshift(space,1)

space3 = space

phase3 = np.angle(space3.numpy())

plt.subplot(345); plt.title('k-space')
plt.imshow(np.abs(kspace.numpy()))
plt.subplot(349); plt.title('k-space_r')
plt.imshow(np.log(np.abs(kspace.numpy())))

plt.subplot(346); plt.title('FFT-magnitude')
plt.imshow(np.abs(space.numpy())); plt.colorbar()
plt.subplot(3,4,10); plt.title('FFT-phase')
plt.imshow(phase3); plt.colorbar()


plt.subplot(348); plt.title('PD-Phantom')
plt.imshow(PD)
plt.subplot(3,4,12); plt.title('B0-Phantom')
plt.imshow(B0)

# %% S2.3 FLASH-sequence 4 with delay3

seq = Sequence()

# Define rf events
rf1, gz, gzr = make_sinc_pulse(
    flip_angle=5 * np.pi/180,
    duration=1e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system
)

zoom=1
# Define other gradients and ADC events
gx = make_trapezoid(channel='x', flat_area=Nread/fov*zoom, flat_time=10e-3, system=system)
adc = make_adc(num_samples=Nread, duration=10e-3, delay=gx.rise_time, system=system)
gx_pre = make_trapezoid(channel='x', area=-0.5*gx.area, duration=5e-3, system=system)
gx_spoil = make_trapezoid(channel='x', area=1.5*gx.area, duration=5e-3, system=system)

rf_phase = 0
rf_inc = 0
rf_spoiling_inc = 117


# ======
# CONSTRUCT SEQUENCE
# ======

for ii in range(-Nphase//2, Nphase//2):  # e.g. -64:63

    rf1.phase_offset = rf_phase / 180 * np.pi   # set current rf phase
    
    adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC

    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]  # increase increment
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]  # increment phase

    seq.add_block(rf1,gz) 
    seq.add_block(gzr)
    gy_pre = make_trapezoid(channel='y', area=ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_pre, gy_pre) 
    seq.add_block(make_delay(delay3)) 
    seq.add_block(adc, gx)
    gy_spoil = make_trapezoid(channel='y', area=-ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_spoil, gy_spoil)
    if ii < Nphase-1:
        seq.add_block(make_delay(0.001))
    
# %% S3.3 CHECK 4

ok, error_report = seq.check_timing()  # Check whether the timing of the sequence is correct
if ok:
    print('Timing check 4 passed successfully')
else:
    print('Timing check 4 failed. Error listing follows:')
    [print(e) for e in error_report]

# PLOT sequence
#sp_adc, t_adc = seq.plot(clear=False)

# Prepare the sequence output for the scanner
seq.set_definition('FOV', [fov*1000, fov*1000, slice_thickness*1000])  # m -> mm
seq.set_definition('Name', 'gre')
seq.write('out/external.seq')
seq.write('out/' + experiment_id +'.seq') 

# %% S5.3 simulation 4

use_simulation = 1

if use_simulation:
    signal, _= sim_external(obj=obj_p,plot_seq_k=[0,1])
else:
    signal = util.get_signal_from_real_system('out/' + experiment_id +'.seq.dat', Nphase, Nread)

# plot the result into the ADC subplot
#sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
#sp_adc.plot(t_adc,np.abs(signal.numpy()))
# seq.plot(signal=signal.numpy())

# %% S6.3 MR image 4

fig4=plt.figure(); # fig.clf()
plt.subplot(411); plt.title('ADC signal')
spectrum=torch.reshape((signal),(Nphase,Nread)).clone().t()
kspace=spectrum
plt.plot(torch.real(signal),label='real')
plt.plot(torch.imag(signal),label='imag')


major_ticks = np.arange(0, Nphase*Nread, Nread) # this adds ticks at the correct position szread
ax=plt.gca(); ax.set_xticks(major_ticks); ax.grid()

space = torch.zeros_like(spectrum)

# fftshift
spectrum=torch.fft.fftshift(spectrum,0); spectrum=torch.fft.fftshift(spectrum,1)
#FFT
space = torch.fft.ifft2(spectrum)
# fftshift
space=torch.fft.ifftshift(space,0); space=torch.fft.ifftshift(space,1)

space4 = space

phase4 = np.angle(space4.numpy())


plt.subplot(345); plt.title('k-space')
plt.imshow(np.abs(kspace.numpy()))
plt.subplot(349); plt.title('k-space_r')
plt.imshow(np.log(np.abs(kspace.numpy())))

plt.subplot(346); plt.title('FFT-magnitude')
plt.imshow(np.abs(space.numpy())); plt.colorbar()
plt.subplot(3,4,10); plt.title('FFT-phase')
plt.imshow(phase4); plt.colorbar()


plt.subplot(348); plt.title('PD-Phantom')
plt.imshow(PD)
plt.subplot(3,4,12); plt.title('B0-Phantom')
plt.imshow(B0)
# %% phase- and lamor-frequency-plots

# calculate Phase-Difference
delta_phase1 = (phase2 - phase1)
delta_phase2 = (phase3 - phase1)
delta_phase3 = (phase4 - phase1)
delta_phase = np.array([delta_phase1,delta_phase2,delta_phase3])


delta_TE1 = np.arange(64.0**2).reshape(np.shape(delta_phase1)[0],np.shape(delta_phase1)[1])
for i in np.arange(64):
    for j in np.arange(64):
        delta_TE1[i][j] = delay1
        
delta_TE2 = np.arange(64.0**2).reshape(np.shape(delta_phase2)[0],np.shape(delta_phase2)[1])
for i in np.arange(64):
    for j in np.arange(64):
        delta_TE2[i][j] = delay2
        
delta_TE3 = np.arange(64.0**2).reshape(np.shape(delta_phase3)[0],np.shape(delta_phase3)[1])
for i in np.arange(64):
    for j in np.arange(64):
        delta_TE3[i][j] = delay3

delta_TE = np.array([delta_TE1,delta_TE2,delta_TE3])

    
fig5 = plt.figure()    

#r'$\Delta$'+r'$\Phi_{1}$'

#B0-Phantom-plot
plt.subplot(337); plt.title('B$_{0}$'+'-Phantom [Hz]', fontsize=20)
plt.imshow(B0.numpy().reshape(Nread,Nread)); plt.colorbar()


# phase-plots
plt.subplot(331); plt.title(r'$\Delta$' + r'$\Phi_{1}$' + ' [rad]', fontsize=20)
plt.imshow(delta_phase[0]*mask,vmin=-0.35,vmax=0.35, cmap='inferno'); plt.colorbar() 

plt.subplot(332); plt.title(r'$\Delta$' + r'$\Phi_{2}$' + ' [rad]', fontsize=20)
plt.imshow(delta_phase[1]*mask,vmin=-0.35,vmax=0.35, cmap='inferno'); plt.colorbar()

plt.subplot(333); plt.title(r'$\Delta$' + r'$\Phi_{3}$' + ' [rad]', fontsize=20)
plt.imshow(delta_phase[2]*mask,vmin=-0.35,vmax=0.35, cmap='inferno'); plt.colorbar()


#frequency-plots
plt.subplot(334); plt.title(r'$\Delta$' + r'$\nu_{larmor,1}$' + ' [Hz]', fontsize=20)
plt.imshow((delta_phase[0]/(delay1*267.522*1e6))*42.577*1e6*mask); plt.colorbar()
 
plt.subplot(335); plt.title(r'$\Delta$' + r'$\nu_{larmor,2}$' + ' [Hz]', fontsize=20)
plt.imshow((delta_phase[1]/(delay2*267.522*1e6))*42.577*1e6*mask); plt.colorbar()

plt.subplot(336); plt.title(r'$\Delta$' + r'$\nu_{larmor,3}$' + ' [Hz]', fontsize=20)
plt.imshow((delta_phase[2]/(delay3*267.522*1e6))*42.577*1e6*mask); plt.colorbar()


#%%  FITTING BLOCK 

from scipy import optimize

def fit_func(dTE, dB0freq):
    return dB0freq*dTE  
  

delta_B0freq_array = np.arange(64.0**2).reshape(64,64)
delta_B0_array2 = np.arange(64.0**2).reshape(64,64)

for i in np.arange(np.shape(delta_phase)[1]):
    for j in np.arange(np.shape(delta_phase)[2]):
        
        fit_TE = np.array([delta_TE[0][i][j], delta_TE[1][i][j], delta_TE[2][i][j]])
        
        fit_phase = np.array([delta_phase[0][i][j], delta_phase[1][i][j], delta_phase[2][i][j]])
        
        parameters, covariance_matrix = optimize.curve_fit(fit_func, fit_TE, fit_phase)
        delta_B0freq = parameters
      
        delta_B0freq_array[i][j] = delta_B0freq
     

delta_B0freq_array = delta_B0freq_array*mask*1/(2*np.pi)

plt.subplot(338); plt.title(r'$\Delta$' + r'$\nu_{larmor}$' + ' from fit [Hz]', fontsize=20)
plt.imshow(delta_B0freq_array); plt.colorbar() 

#B0_field = delta_B0freq_array/gamma_2pi
#plt.subplot(339); plt.title(r'$\Delta$' + 'B$_{0}$' + ' [Hz]', fontsize=20)
#plt.imshow(B0_field); plt.colorbar() 


# %% S2.4 FLASH-sequence 2 with high delay4

seq = Sequence()

# Define rf events
rf1, gz, gzr = make_sinc_pulse(
    flip_angle=5 * np.pi/180,
    duration=1e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system
)

zoom=1
# Define other gradients and ADC events
gx = make_trapezoid(channel='x', flat_area=Nread/fov*zoom, flat_time=10e-3, system=system)
adc = make_adc(num_samples=Nread, duration=10e-3, delay=gx.rise_time, system=system)
gx_pre = make_trapezoid(channel='x', area=-0.5*gx.area, duration=5e-3, system=system)
gx_spoil = make_trapezoid(channel='x', area=1.5*gx.area, duration=5e-3, system=system)

rf_phase = 0
rf_inc = 0
rf_spoiling_inc = 117


# ======
# CONSTRUCT SEQUENCE
# ======

for ii in range(-Nphase//2, Nphase//2):  # e.g. -64:63

    rf1.phase_offset = rf_phase / 180 * np.pi   # set current rf phase
    
    adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC

    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]  # increase increment
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]  # increment phase

    seq.add_block(rf1,gz) 
    seq.add_block(gzr)
    gy_pre = make_trapezoid(channel='y', area=ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_pre, gy_pre)
    seq.add_block(make_delay(delay4)) 
    seq.add_block(adc, gx)
    gy_spoil = make_trapezoid(channel='y', area=-ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_spoil, gy_spoil)
    if ii < Nphase-1:
        seq.add_block(make_delay(0.001))
             
# %% S3.4 CHECK 5
ok, error_report = seq.check_timing()  # Check whether the timing of the sequence is correct
if ok:
    print('Timing check 5 passed successfully')
else:
    print('Timing check 5 failed. Error listing follows:')
    [print(e) for e in error_report]

# PLOT sequence
#sp_adc, t_adc = seq.plot(clear=False)

# Prepare the sequence output for the scanner
seq.set_definition('FOV', [fov*1000, fov*1000, slice_thickness*1000])  # m -> mm
seq.set_definition('Name', 'gre')
seq.write('out/external.seq')
seq.write('out/' + experiment_id +'.seq')


# %% S5.4 simulation 5

use_simulation = 1

if use_simulation:
    signal, _= sim_external(obj=obj_p,plot_seq_k=[0,1])
else:
    signal = util.get_signal_from_real_system('out/' + experiment_id +'.seq.dat', Nphase, Nread)

# plot the result into the ADC subplot
#sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
#sp_adc.plot(t_adc,np.abs(signal.numpy()))
# seq.plot(signal=signal.numpy())


# %% S6.4 MR image 5 phase-wrapping

spectrum=torch.reshape((signal),(Nphase,Nread)).clone().t()
kspace=spectrum

space = torch.zeros_like(spectrum)

# fftshift
spectrum=torch.fft.fftshift(spectrum,0); spectrum=torch.fft.fftshift(spectrum,1)
#FFT
space = torch.fft.ifft2(spectrum)
# fftshift
space=torch.fft.ifftshift(space,0); space=torch.fft.ifftshift(space,1)

space5 = space

phase5 = np.angle(space5.numpy())


delta_phase4 = (phase5 - phase1)

fig6=plt.figure();
plt.subplot(131); plt.title('FFT-phase with wrap-around [rad]', fontsize=20)
plt.imshow(phase5*mask, cmap='inferno'); plt.colorbar()

plt.subplot(132); plt.title(r'$\Delta$' + r'$\Phi$' + ' with wrap-around [rad]', fontsize=20)
plt.imshow(delta_phase4*mask, cmap='inferno'); plt.colorbar()

plt.subplot(133); plt.title(r'unwrapped $\Delta$' + r'$\Phi$' + ' [rad]', fontsize=20)
plt.imshow(unwrap_phase(delta_phase4)*mask, cmap='inferno'); plt.colorbar()

print('done')
   

              

