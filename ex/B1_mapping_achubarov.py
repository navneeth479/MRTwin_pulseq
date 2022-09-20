experiment_id = 'exE01_FLASH_2D_user_tag_fruit#'


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
seq = Sequence()

# Define FOV and resolution
fov = 220e-3
slice_thickness = 8e-3
sz = (32, 32)  # spin system size / resolution
Nread = 64  # frequency encoding steps/samples
Nphase = 64  # phase encoding steps/samples

a = 10
# Define rf events
rf1, gz, gzr = make_sinc_pulse(
    flip_angle= a * np.pi/180,
    duration=1e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system
)

zoom=1
# Define other gradients and ADC events
gx = make_trapezoid(channel='x', flat_area=Nread/fov*zoom, flat_time=5e-3, system=system)
gx_pre = make_trapezoid(channel='x', area=-0.5*gx.area, duration=5e-3, system=system)
gx_spoil = make_trapezoid(channel='x', area=1.5*gx.area, duration=5e-3, system=system)

adc = make_adc(num_samples=Nread, duration=5e-3, delay=gx.rise_time, system=system)
#adc2 = make_adc(num_samples=Nread, duration=2e-3, delay=gx.rise_time, system=system)

rf_phase = 0
rf_inc = 0
rf_spoiling_inc = 117

# ======
# CONSTRUCT SEQUENCE
# ======

#rfprep, oberer for loop 
#rfprep
rf_prep, _, _ = make_sinc_pulse(flip_angle= 180 * np.pi/180,duration=1e-3,slice_thickness=slice_thickness,apodization=0.5,time_bw_product=4, system=system)

seq.add_block(rf_prep)
#seq.add_block(make_delay(2.7))
seq.add_block(gx_spoil)

for ii in range(-Nphase//2, Nphase//2):  # e.g. -64:63
        
    rf1.phase_offset = rf_phase / 180 * np.pi   # set current rf phase
    
    adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC

    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]  # increase increment
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]  # increment phase
    
    #imaging 1a
    seq.add_block(rf1) #,gz
    #seq.add_block(gzr)
    gy_pre = make_trapezoid(channel='y', area=ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_pre, gy_pre)
    seq.add_block(adc,gx)
    gy_spoil = make_trapezoid(channel='y', area=-ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_spoil, gy_spoil)
    #seq.add_block(make_delay(0))
    
    if ii < Nphase-1:
            seq.add_block(make_delay(0.001))
            
seq.add_block(make_delay(10))

      
# %% S3. CHECK, PLOT and WRITE the sequence  as .seq
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
from new_core import util

use_simulation = 1

if use_simulation:
    signal, _= sim_external(obj=obj_p,plot_seq_k=[0,1])
else:
    signal = util.get_signal_from_real_system('out/' + experiment_id +'.seq.dat', Nphase, Nread)

# plot the result into the ADC subplot
sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
sp_adc.plot(t_adc,np.abs(signal.numpy()))
# seq.plot(signal=signal.numpy())


# %% S6: MR IMAGE RECON of signal ::: #####################################
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

space10 = space

plt.subplot(345); plt.title('k-space')
plt.imshow(np.abs(kspace.numpy()))
plt.subplot(349); plt.title('k-space_r')
plt.imshow(np.log(np.abs(kspace.numpy())))

plt.subplot(346); plt.title('FFT-magnitude')
plt.imshow(np.abs(space.numpy())); plt.colorbar()
plt.subplot(3,4,10); plt.title('FFT-phase')
plt.imshow(np.angle(space.numpy()),vmin=-np.pi,vmax=np.pi); plt.colorbar()


# % compare with original phantom obj_p.PD
from new_core import util
PD = util.to_full(obj_p.PD, obj_p.mask).squeeze(0)
B0 = util.to_full(obj_p.B0, obj_p.mask).squeeze(0)
plt.subplot(348); plt.title('phantom PD')
plt.imshow(PD)
plt.subplot(3,4,12); plt.title('phantom B0')
plt.imshow(B0)

#####
#fig=plt.figure("""B1 map 1""")
#plt.subplot(121); plt.title('FFT-magnitude')
#plt.imshow(np.abs(space.numpy())); plt.colorbar()
#plt.subplot(122); plt.title('FFT-phase')
#plt.imshow(np.angle(space.numpy()),vmin=-np.pi,vmax=np.pi); plt.colorbar()

#%% S2.2 repeat 20

seq = Sequence()

a = 20
# Define rf events
rf1, gz, gzr = make_sinc_pulse(
    flip_angle= a * np.pi/180,
    duration=1e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system
)

# ======
# CONSTRUCT SEQUENCE
# ======

seq.add_block(rf_prep)
seq.add_block(gx_spoil)

for ii in range(-Nphase//2, Nphase//2):  # e.g. -64:63
        
    rf1.phase_offset = rf_phase / 180 * np.pi   # set current rf phase
    
    adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC

    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]  # increase increment
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]  # increment phase
    
    #imaging 1a
    seq.add_block(rf1) #,gz
    #seq.add_block(gzr)
    gy_pre = make_trapezoid(channel='y', area=ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_pre, gy_pre)
    seq.add_block(adc,gx)
    gy_spoil = make_trapezoid(channel='y', area=-ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_spoil, gy_spoil)
    #seq.add_block(make_delay(0))
    
    if ii < Nphase-1:
            seq.add_block(make_delay(0.001))
            
seq.add_block(make_delay(10))

# %% S3.2
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

# %% S5.2
from new_core import util

use_simulation = 1

if use_simulation:
    signal, _= sim_external(obj=obj_p,plot_seq_k=[0,1])
else:
    signal = util.get_signal_from_real_system('out/' + experiment_id +'.seq.dat', Nphase, Nread)

# plot the result into the ADC subplot
sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
sp_adc.plot(t_adc,np.abs(signal.numpy()))
# seq.plot(signal=signal.numpy())


# %% S6.2 20
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

space20 = space

plt.subplot(345); plt.title('k-space')
plt.imshow(np.abs(kspace.numpy()))
plt.subplot(349); plt.title('k-space_r')
plt.imshow(np.log(np.abs(kspace.numpy())))

plt.subplot(346); plt.title('FFT-magnitude')
plt.imshow(np.abs(space.numpy())); plt.colorbar()
plt.subplot(3,4,10); plt.title('FFT-phase')
plt.imshow(np.angle(space.numpy()),vmin=-np.pi,vmax=np.pi); plt.colorbar()


# % compare with original phantom obj_p.PD
from new_core import util
PD = util.to_full(obj_p.PD, obj_p.mask).squeeze(0)
B0 = util.to_full(obj_p.B0, obj_p.mask).squeeze(0)
plt.subplot(348); plt.title('phantom PD')
plt.imshow(PD)
plt.subplot(3,4,12); plt.title('phantom B0')
plt.imshow(B0)

#####
#fig=plt.figure("""B1 map 2""")
#plt.subplot(121); plt.title('FFT-magnitude')
#plt.imshow(np.abs(space.numpy())); plt.colorbar()
#plt.subplot(122); plt.title('FFT-phase')
#plt.imshow(np.angle(space.numpy()),vmin=-np.pi,vmax=np.pi); plt.colorbar()

#%% S2.3 repeat

seq = Sequence()

a = 30
# Define rf events
rf1, gz, gzr = make_sinc_pulse(
    flip_angle= a * np.pi/180,
    duration=1e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system
)

# ======
# CONSTRUCT SEQUENCE
# ======

seq.add_block(rf_prep)
seq.add_block(gx_spoil)

for ii in range(-Nphase//2, Nphase//2):  # e.g. -64:63
        
    rf1.phase_offset = rf_phase / 180 * np.pi   # set current rf phase
    
    adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC

    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]  # increase increment
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]  # increment phase
    
    #imaging 1a
    seq.add_block(rf1) #,gz
    #seq.add_block(gzr)
    gy_pre = make_trapezoid(channel='y', area=ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_pre, gy_pre)
    seq.add_block(adc,gx)
    gy_spoil = make_trapezoid(channel='y', area=-ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_spoil, gy_spoil)
    #seq.add_block(make_delay(0))
    
    if ii < Nphase-1:
            seq.add_block(make_delay(0.001))
            
seq.add_block(make_delay(10))

# %% S3.3
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

# %% S5.3
from new_core import util

use_simulation = 1

if use_simulation:
    signal, _= sim_external(obj=obj_p,plot_seq_k=[0,1])
else:
    signal = util.get_signal_from_real_system('out/' + experiment_id +'.seq.dat', Nphase, Nread)

# plot the result into the ADC subplot
sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
sp_adc.plot(t_adc,np.abs(signal.numpy()))
# seq.plot(signal=signal.numpy())


# %% S6.3
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

space30 = space

plt.subplot(345); plt.title('k-space')
plt.imshow(np.abs(kspace.numpy()))
plt.subplot(349); plt.title('k-space_r')
plt.imshow(np.log(np.abs(kspace.numpy())))

plt.subplot(346); plt.title('FFT-magnitude')
plt.imshow(np.abs(space.numpy())); plt.colorbar()
plt.subplot(3,4,10); plt.title('FFT-phase')
plt.imshow(np.angle(space.numpy()),vmin=-np.pi,vmax=np.pi); plt.colorbar()


# % compare with original phantom obj_p.PD
from new_core import util
PD = util.to_full(obj_p.PD, obj_p.mask).squeeze(0)
B0 = util.to_full(obj_p.B0, obj_p.mask).squeeze(0)
plt.subplot(348); plt.title('phantom PD')
plt.imshow(PD)
plt.subplot(3,4,12); plt.title('phantom B0')
plt.imshow(B0)

#####
#fig=plt.figure("""B1 map 3""")
#plt.subplot(121); plt.title('FFT-magnitude')
#plt.imshow(np.abs(space.numpy())); plt.colorbar()
#plt.subplot(122); plt.title('FFT-phase')
#plt.imshow(np.angle(space.numpy()),vmin=-np.pi,vmax=np.pi); plt.colorbar()

#%% S2.4 repeat 40

seq = Sequence()

a = 40
# Define rf events
rf1, gz, gzr = make_sinc_pulse(
    flip_angle= a * np.pi/180,
    duration=1e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system
)

# ======
# CONSTRUCT SEQUENCE
# ======

seq.add_block(rf_prep)
seq.add_block(gx_spoil)

for ii in range(-Nphase//2, Nphase//2):  # e.g. -64:63
        
    rf1.phase_offset = rf_phase / 180 * np.pi   # set current rf phase
    
    adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC

    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]  # increase increment
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]  # increment phase
    
    #imaging 1a
    seq.add_block(rf1) #,gz
    #seq.add_block(gzr)
    gy_pre = make_trapezoid(channel='y', area=ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_pre, gy_pre)
    seq.add_block(adc,gx)
    gy_spoil = make_trapezoid(channel='y', area=-ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_spoil, gy_spoil)
    #seq.add_block(make_delay(0))
    
    if ii < Nphase-1:
            seq.add_block(make_delay(0.001))
            
seq.add_block(make_delay(10))

# %% S3.4
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

# %% S5.4
from new_core import util

use_simulation = 1

if use_simulation:
    signal, _= sim_external(obj=obj_p,plot_seq_k=[0,1])
else:
    signal = util.get_signal_from_real_system('out/' + experiment_id +'.seq.dat', Nphase, Nread)

# plot the result into the ADC subplot
sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
sp_adc.plot(t_adc,np.abs(signal.numpy()))
# seq.plot(signal=signal.numpy())


# %% S6.4
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

space40 = space

plt.subplot(345); plt.title('k-space')
plt.imshow(np.abs(kspace.numpy()))
plt.subplot(349); plt.title('k-space_r')
plt.imshow(np.log(np.abs(kspace.numpy())))

plt.subplot(346); plt.title('FFT-magnitude')
plt.imshow(np.abs(space.numpy())); plt.colorbar()
plt.subplot(3,4,10); plt.title('FFT-phase')
plt.imshow(np.angle(space.numpy()),vmin=-np.pi,vmax=np.pi); plt.colorbar()


# % compare with original phantom obj_p.PD
from new_core import util
PD = util.to_full(obj_p.PD, obj_p.mask).squeeze(0)
B0 = util.to_full(obj_p.B0, obj_p.mask).squeeze(0)
plt.subplot(348); plt.title('phantom PD')
plt.imshow(PD)
plt.subplot(3,4,12); plt.title('phantom B0')
plt.imshow(B0)

#####

#fig=plt.figure("""B1 map 4""")
#plt.subplot(121); plt.title('FFT-magnitude')
#plt.imshow(np.abs(space.numpy())); plt.colorbar()
#plt.subplot(122); plt.title('FFT-phase')
#plt.imshow(np.angle(space.numpy()),vmin=-np.pi,vmax=np.pi); plt.colorbar()

#%% S2.5 repeat 5

seq = Sequence()

a = 5
# Define rf events
rf1, gz, gzr = make_sinc_pulse(
    flip_angle= a * np.pi/180,
    duration=1e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system
)

# ======
# CONSTRUCT SEQUENCE
# ======

seq.add_block(rf_prep)
seq.add_block(gx_spoil)

for ii in range(-Nphase//2, Nphase//2):  # e.g. -64:63
        
    rf1.phase_offset = rf_phase / 180 * np.pi   # set current rf phase
    
    adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC

    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]  # increase increment
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]  # increment phase
    
    #imaging 1a
    seq.add_block(rf1) #,gz
    #seq.add_block(gzr)
    gy_pre = make_trapezoid(channel='y', area=ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_pre, gy_pre)
    seq.add_block(adc,gx)
    gy_spoil = make_trapezoid(channel='y', area=-ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_spoil, gy_spoil)
    #seq.add_block(make_delay(0))
    
    if ii < Nphase-1:
            seq.add_block(make_delay(0.001))
            
seq.add_block(make_delay(10))

# %% S3.5
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

# %% S5.5
from new_core import util

use_simulation = 1

if use_simulation:
    signal, _= sim_external(obj=obj_p,plot_seq_k=[0,1])
else:
    signal = util.get_signal_from_real_system('out/' + experiment_id +'.seq.dat', Nphase, Nread)

# plot the result into the ADC subplot
sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
sp_adc.plot(t_adc,np.abs(signal.numpy()))
# seq.plot(signal=signal.numpy())


# %% S6.5
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

space5 = space

plt.subplot(345); plt.title('k-space')
plt.imshow(np.abs(kspace.numpy()))
plt.subplot(349); plt.title('k-space_r')
plt.imshow(np.log(np.abs(kspace.numpy())))

plt.subplot(346); plt.title('FFT-magnitude')
plt.imshow(np.abs(space.numpy())); plt.colorbar()
plt.subplot(3,4,10); plt.title('FFT-phase')
plt.imshow(np.angle(space.numpy()),vmin=-np.pi,vmax=np.pi); plt.colorbar()


# % compare with original phantom obj_p.PD
from new_core import util
PD = util.to_full(obj_p.PD, obj_p.mask).squeeze(0)
B0 = util.to_full(obj_p.B0, obj_p.mask).squeeze(0)
plt.subplot(348); plt.title('phantom PD')
plt.imshow(PD)
plt.subplot(3,4,12); plt.title('phantom B0')
plt.imshow(B0)

#%% S2.6 repeat 15

seq = Sequence()

a = 15
# Define rf events
rf1, gz, gzr = make_sinc_pulse(
    flip_angle= a * np.pi/180,
    duration=1e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system
)

# ======
# CONSTRUCT SEQUENCE
# ======

seq.add_block(rf_prep)
seq.add_block(gx_spoil)

for ii in range(-Nphase//2, Nphase//2):  # e.g. -64:63
        
    rf1.phase_offset = rf_phase / 180 * np.pi   # set current rf phase
    
    adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC

    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]  # increase increment
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]  # increment phase
    
    #imaging 1a
    seq.add_block(rf1) #,gz
    #seq.add_block(gzr)
    gy_pre = make_trapezoid(channel='y', area=ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_pre, gy_pre)
    seq.add_block(adc,gx)
    gy_spoil = make_trapezoid(channel='y', area=-ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_spoil, gy_spoil)
    #seq.add_block(make_delay(0))
    
    if ii < Nphase-1:
            seq.add_block(make_delay(0.001))
            
seq.add_block(make_delay(10))

# %% S3.6
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

# %% S5.6
from new_core import util

use_simulation = 1

if use_simulation:
    signal, _= sim_external(obj=obj_p,plot_seq_k=[0,1])
else:
    signal = util.get_signal_from_real_system('out/' + experiment_id +'.seq.dat', Nphase, Nread)

# plot the result into the ADC subplot
sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
sp_adc.plot(t_adc,np.abs(signal.numpy()))
# seq.plot(signal=signal.numpy())


# %% S6.6
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

space15 = space

plt.subplot(345); plt.title('k-space')
plt.imshow(np.abs(kspace.numpy()))
plt.subplot(349); plt.title('k-space_r')
plt.imshow(np.log(np.abs(kspace.numpy())))

plt.subplot(346); plt.title('FFT-magnitude')
plt.imshow(np.abs(space.numpy())); plt.colorbar()
plt.subplot(3,4,10); plt.title('FFT-phase')
plt.imshow(np.angle(space.numpy()),vmin=-np.pi,vmax=np.pi); plt.colorbar()


# % compare with original phantom obj_p.PD
from new_core import util
PD = util.to_full(obj_p.PD, obj_p.mask).squeeze(0)
B0 = util.to_full(obj_p.B0, obj_p.mask).squeeze(0)
plt.subplot(348); plt.title('phantom PD')
plt.imshow(PD)
plt.subplot(3,4,12); plt.title('phantom B0')
plt.imshow(B0)

#%% S2.7 repeat 90

seq = Sequence()

a = 90
# Define rf events
rf1, gz, gzr = make_sinc_pulse(
    flip_angle= a * np.pi/180,
    duration=1e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system
)

# ======
# CONSTRUCT SEQUENCE
# ======

seq.add_block(rf_prep)
seq.add_block(gx_spoil)

for ii in range(-Nphase//2, Nphase//2):  # e.g. -64:63
        
    rf1.phase_offset = rf_phase / 180 * np.pi   # set current rf phase
    
    adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC

    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]  # increase increment
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]  # increment phase
    
    #imaging 1a
    seq.add_block(rf1) #,gz
    #seq.add_block(gzr)
    gy_pre = make_trapezoid(channel='y', area=ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_pre, gy_pre)
    seq.add_block(adc,gx)
    gy_spoil = make_trapezoid(channel='y', area=-ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_spoil, gy_spoil)
    #seq.add_block(make_delay(0))
    
    if ii < Nphase-1:
            seq.add_block(make_delay(0.001))
            
seq.add_block(make_delay(10))

# %% S3.7
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

# %% S5.7
from new_core import util

use_simulation = 1

if use_simulation:
    signal, _= sim_external(obj=obj_p,plot_seq_k=[0,1])
else:
    signal = util.get_signal_from_real_system('out/' + experiment_id +'.seq.dat', Nphase, Nread)

# plot the result into the ADC subplot
sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
sp_adc.plot(t_adc,np.abs(signal.numpy()))
# seq.plot(signal=signal.numpy())


# %% S6.7
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

space90 = space

plt.subplot(345); plt.title('k-space')
plt.imshow(np.abs(kspace.numpy()))
plt.subplot(349); plt.title('k-space_r')
plt.imshow(np.log(np.abs(kspace.numpy())))

plt.subplot(346); plt.title('FFT-magnitude')
plt.imshow(np.abs(space.numpy())); plt.colorbar()
plt.subplot(3,4,10); plt.title('FFT-phase')
plt.imshow(np.angle(space.numpy()),vmin=-np.pi,vmax=np.pi); plt.colorbar()


# % compare with original phantom obj_p.PD
from new_core import util
PD = util.to_full(obj_p.PD, obj_p.mask).squeeze(0)
B0 = util.to_full(obj_p.B0, obj_p.mask).squeeze(0)
plt.subplot(348); plt.title('phantom PD')
plt.imshow(PD)
plt.subplot(3,4,12); plt.title('phantom B0')
plt.imshow(B0)

#%% S2.8 repeat 180

seq = Sequence()

a = 180
# Define rf events
rf1, gz, gzr = make_sinc_pulse(
    flip_angle= a * np.pi/180,
    duration=1e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system
)

# ======
# CONSTRUCT SEQUENCE
# ======

seq.add_block(rf_prep)
seq.add_block(gx_spoil)

for ii in range(-Nphase//2, Nphase//2):  # e.g. -64:63
        
    rf1.phase_offset = rf_phase / 180 * np.pi   # set current rf phase
    
    adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC

    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]  # increase increment
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]  # increment phase
    
    #imaging 1a
    seq.add_block(rf1) #,gz
    #seq.add_block(gzr)
    gy_pre = make_trapezoid(channel='y', area=ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_pre, gy_pre)
    seq.add_block(adc,gx)
    gy_spoil = make_trapezoid(channel='y', area=-ii/fov*zoom, duration=5e-3, system=system)
    seq.add_block(gx_spoil, gy_spoil)
    #seq.add_block(make_delay(0))
    
    if ii < Nphase-1:
            seq.add_block(make_delay(0.001))
            
seq.add_block(make_delay(10))

# %% S3.8
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

# %% S5.8
from new_core import util

use_simulation = 1

if use_simulation:
    signal, _= sim_external(obj=obj_p,plot_seq_k=[0,1])
else:
    signal = util.get_signal_from_real_system('out/' + experiment_id +'.seq.dat', Nphase, Nread)

# plot the result into the ADC subplot
sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
sp_adc.plot(t_adc,np.abs(signal.numpy()))
# seq.plot(signal=signal.numpy())


# %% S6.6
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

space180 = space

plt.subplot(345); plt.title('k-space')
plt.imshow(np.abs(kspace.numpy()))
plt.subplot(349); plt.title('k-space_r')
plt.imshow(np.log(np.abs(kspace.numpy())))

plt.subplot(346); plt.title('FFT-magnitude')
plt.imshow(np.abs(space.numpy())); plt.colorbar()
plt.subplot(3,4,10); plt.title('FFT-phase')
plt.imshow(np.angle(space.numpy()),vmin=-np.pi,vmax=np.pi); plt.colorbar()


# % compare with original phantom obj_p.PD
from new_core import util
PD = util.to_full(obj_p.PD, obj_p.mask).squeeze(0)
B0 = util.to_full(obj_p.B0, obj_p.mask).squeeze(0)
plt.subplot(348); plt.title('phantom PD')
plt.imshow(PD)
plt.subplot(3,4,12); plt.title('phantom B0')
plt.imshow(B0)
#%% MATH

fig=plt.figure("""B1 map all""")
plt.subplot(231); plt.title('FFT-magnitude 5')
plt.imshow(np.abs(space5.numpy())); plt.colorbar()
plt.subplot(232); plt.title('FFT-magnitude 10')
plt.imshow(np.abs(space10.numpy())); plt.colorbar()
plt.subplot(233); plt.title('FFT-magnitude 15')
plt.imshow(np.abs(space15.numpy())); plt.colorbar()
plt.subplot(234); plt.title('FFT-magnitude 20')
plt.imshow(np.abs(space20.numpy())); plt.colorbar()
plt.subplot(235); plt.title('FFT-magnitude 30')
plt.imshow(np.abs(space30.numpy())); plt.colorbar()
#plt.subplot(236); plt.title('FFT-magnitude 40')
#plt.imshow(np.abs(space40.numpy())); plt.colorbar()
plt.subplot(236); plt.title('FFT-magnitude 90')
plt.imshow(np.abs(space90.numpy())); plt.colorbar()

#ratio formula: rat = arccos(S2/2*S1) ?
rat1 = np.arccos(np.abs(space10.numpy())/np.abs(2*space5.numpy()))
print (rat1)

rat2 = np.arccos(np.abs(space20.numpy())/np.abs(2*space10.numpy()))
print (rat2)

rat3 = np.arccos(np.abs(space30.numpy())/np.abs(2*space15.numpy()))
print (rat3)

rat4 = np.arccos(np.abs(space40.numpy())/np.abs(2*space20.numpy()))
print (rat4)

rat5 = np.arccos(np.abs(space90.numpy())/np.abs(2*space180.numpy()))
print (rat5)

fig=plt.figure("""ratios""")
plt.subplot(231); plt.title('rat1')
plt.imshow(rat1); plt.colorbar()
plt.subplot(232); plt.title('rat2')
plt.imshow(rat2); plt.colorbar()
plt.subplot(233); plt.title('rat3')
plt.imshow(rat3); plt.colorbar()
plt.subplot(234); plt.title('rat4')
plt.imshow(rat4); plt.colorbar()
plt.subplot(235); plt.title('rat5')
plt.imshow(rat5); plt.colorbar()

#Plot vorbereitung


#%%  FITTING BLOCK - work in progress
from scipy import optimize

# choose echo tops and flatten extra dimensions
S=signal[63::128,:].abs().ravel()
t=t_adc[63::128].ravel()

S=S.numpy()
def fit_func(t, a, R,c):
    return a*np.cos(-R*t) + c   


p=optimize.curve_fit(fit_func,t,S,p0=(1, 1,0))
print(p[0][1])

fig=plt.figure("""fit""")
ax1=plt.subplot(111)
ax=plt.plot(t_adc,np.abs(signal.numpy()),label='fulldata')
ax=plt.plot(t,S,'x',label='data')
plt.plot(t,fit_func(t,p[0][0],p[0][1],p[0][2]),label="f={:.2}*exp(-{:.2}*t)+{:.2}".format(p[0][0], p[0][1],p[0][2]))
plt.title('fit')
plt.legend()
plt.ion()

plt.show()