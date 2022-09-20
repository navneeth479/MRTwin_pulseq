experiment_id = 'exD01_bSSFP_2D'

# %% S0. SETUP env
import sys,os
os.chdir(os.path.abspath(os.path.dirname(__file__)))  #  makes the ex folder your working directory
sys.path.append(os.path.dirname(os.getcwd()))         #  add required folders to path
mpath=os.path.dirname(os.getcwd())
c1=r'codes'; c2=r'codes\GradOpt_python'; c3=r'codes\scannerloop_libs' #  add required folders to path
sys.path += [rf'{mpath}\{c1}',rf'{mpath}\{c2}',rf'{mpath}\{c3}']

## imports for simulation
from GradOpt_python.pulseq_sim_external import sim_external
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
from pypulseq.make_gauss_pulse import make_gauss_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.opts import Opts
from pypulseq.traj_to_grad import traj_to_grad

# %% S1. SETUP sys

import time
start = time.time()

## choose the scanner limits
system = Opts(max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s', rf_ringdown_time=20e-6,
                 rf_dead_time=100e-6, adc_dead_time=20e-6, grad_raster_time=10e-6)

# %% S2. DEFINE the sequence
seq = Sequence()

# Define FOV and resolution
fov = 1000e-3
slice_thickness=8e-3
sz=(64,64)   # spin system size / resolution
Nread = 64   # frequency encoding steps/samples
Nphase = 64    # phase encoding steps/samples

method = 'traj'

if method == 'traj': 
    # # Define trajectory
    
    #plt.ion()
    
    deltak = 1/fov
    OverSampling = 2
    k_radius = np.round(Nread / 2)
    k_samples = np.round(2 * np.pi * k_radius) * OverSampling
    
    ka = np.zeros(int(k_radius * k_samples + 1), dtype=np.complex128)
    
    for c in range(len(ka)):
        r = deltak * c / k_samples;
        a = np.mod(c, k_samples) * 2 * np.pi / k_samples;
        ka[c] = r * np.exp(1j * a);
        #plt.clf()
        #plt.scatter(ka.real, ka.imag)
        #plt.pause(0.001)
        
# =============================================================================
#     # another kinds of trajectory
#     deltak = 1 / fov
#     OverSampling = 2
#     
#     
#     for k_radius in range(1, Nread//2+1):
#         k_samples = np.round(2 * np.pi * k_radius) * OverSampling
#         for c in range(int(k_samples)):
#     
#             r = k_radius-1 + deltak * c / k_samples;
#             a = c * 2 * np.pi / k_samples;
#     
#             if k_radius == 1 and c == 0:
#                 ka = np.array([r * np.exp(1j * a)], dtype=np.complex128)
#             else:
#                 ka = np.append(ka, r * np.exp(1j * a));
#     
#             #plt.clf()
#             #plt.scatter(ka.real, ka.imag)
#             #plt.pause(0.001)
#     ka = np.append(ka, Nread//2+0j)
# =============================================================================
    
    plt.ioff()
    plt.show()
    
    ka2 = np.asarray([ka.real, ka.imag])
    ga, sa = traj_to_grad(ka, system.grad_raster_time)
    # ga = np.asarray([ga.real, ga.imag])
    # sa = np.asarray([sa.real, sa.imag])
    plt.figure()
    plt.plot(ga.real)
    plt.plot(ga.imag)
    #plt.plot(sa)
    plt.show()
    # =================
    
    safety_margin = 0.74
    dt_gcomp = np.abs([ga.real, ga.imag]) / (system.max_grad * safety_margin) * system.grad_raster_time
    dt_gabs = np.abs(ga) / (system.max_grad * safety_margin) * system.grad_raster_time
    dt_scomp =  np.sqrt(np.abs([sa.real, sa.imag]) / (system.max_slew * safety_margin)) * system.grad_raster_time
    dt_sabs = np.sqrt(np.abs(sa) / (system.max_slew * safety_margin)) * system.grad_raster_time
    
    plt.figure()
    plt.plot(dt_gabs)
    plt.plot(np.max(dt_gcomp,axis=0))
    plt.plot(dt_sabs)
    plt.plot(np.max(dt_scomp,axis=0))
    plt.show()
    
    dt_smooth = np.max(np.vstack([dt_gabs, dt_sabs]), axis=0)
    dt_rough = np.max(np.vstack([dt_gcomp, dt_scomp]), axis=0)
    # apply the lower limit
    dt_min = 2*system.grad_raster_time/k_samples;
    dt_smooth0 = dt_smooth.copy();
    dt_rough0 = dt_rough.copy();
    dt_smooth[dt_smooth<dt_min]=dt_min;
    dt_rough[dt_rough<dt_min]=dt_min;
    
    plt.figure()
    plt.plot(dt_smooth0)
    plt.plot(dt_smooth)
    plt.plot(dt_rough0)
    plt.plot(dt_rough)
    plt.show()
    

    t_smooth = np.insert(np.cumsum(dt_smooth), 0, 0)
    t_rough = np.insert(np.cumsum(dt_rough), 0, 0)
    
    
    kopt_smooth = np.interp(np.arange(0, np.floor(t_smooth[-1]/system.grad_raster_time)+1)*system.grad_raster_time, t_smooth, ka.T)
    kopt_rough = np.interp(np.arange(0, np.floor(t_rough[-1]/system.grad_raster_time)+1)*system.grad_raster_time, t_rough, ka.T)
    
    print('duration orig %d us', round(1e6*system.grad_raster_time*len(ka)));
    print('duration smooth %d us', round(1e6*system.grad_raster_time*len(kopt_smooth)));
    print('duration rough %d us', round(1e6*system.grad_raster_time*len(kopt_rough)));
    
    gos, sos = traj_to_grad(kopt_smooth)
    gor, sor = traj_to_grad(kopt_rough)
    
    figure, ax = plt.subplots(2,2)
    ax[0][0].plot(np.asarray([gos.real, gos.imag, np.abs(gos)]).T)
    #ax[0][0].title = 'gradient smooth(abs) constraint'
    ax[0][1].plot(np.asarray([gor.real, gor.imag, np.abs(gor)]).T)
    #ax[0][1].title = 'gradient rough(component) constraint'
    ax[1][0].plot(np.asarray([sos.real, sos.imag, np.abs(sos)]).T)
    #ax[0][0].title = 'gradient smooth(abs) constraint'
    ax[1][1].plot(np.asarray([sor.real, sor.imag, np.abs(sor)]).T)
    #ax[0][1].title = 'gradient rough(component) constraint'
    figure.show()
    plt.show()
    

    #Define gradients and ADC events
    spiral_grad_shape = gos
    
    # extend spiral_grad_shape by repeating the last sample
    # this is needed to accomodate for the ADC tuning delay
    spiral_grad_shape = np.insert(spiral_grad_shape, -1, spiral_grad_shape[-1])
    
    # readout grad
    gx = make_arbitrary_grad(channel='x', waveform=spiral_grad_shape.real)
    gy = make_arbitrary_grad(channel='y', waveform=spiral_grad_shape.imag)

if method == 'equations':
    OverSampling = 1
    gamma = 1
    n_r = Nphase//2
    n_theta = np.round(2 * np.pi * n_r) * OverSampling
    det_T = system.grad_raster_time
    det_r = np.pi / n_r
    eta = np.pi / (gamma * n_theta * n_r * det_T * det_r)
    epsilon = 2*np.pi / (n_theta*det_T)
    
    time_seq = np.arange(n_r*n_theta) * system.grad_raster_time
    
    wave_x = eta * np.cos(epsilon * time_seq) - eta*epsilon * time_seq * np.sin(epsilon * time_seq) 
    wave_y = eta * np.sin(epsilon * time_seq) + eta*epsilon * time_seq * np.cos(epsilon * time_seq)
    gx = make_arbitrary_grad(channel='x', waveform=wave_x)
    gy = make_arbitrary_grad(channel='y', waveform=wave_y)

adc = make_adc(Nread*Nphase, duration=calc_duration(gx))
adc_show_echo = make_adc(Nread, duration = 1e-3)
# ======
# CONSTRUCT SEQUENCE
# ======

rf_fs, _,_ = make_sinc_pulse(flip_angle=90 * math.pi / 180, duration=1e-3, phase_offset=90*math.pi / 180, slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4, system=system)
rf, _, _ = make_sinc_pulse(flip_angle=180 * math.pi / 180, duration=1e-3,slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4, system=system)

seq.add_block(rf_fs)
seq.add_block(rf)

seq.add_block(make_delay(calc_duration(rf)))
seq.add_block(gx, gy, adc)

# show spin echo
#seq.add_block(adc_show_echo)



# S3. CHECK, PLOT and WRITE the sequence  as .seq
ok, error_report = seq.check_timing()  # Check whether the timing of the sequence is correct
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]

# PLOT sequence
sp_adc,t_adc =seq.plot(clear=False)
#   
if 1:
    sp_adc,t_adc =seq.plot(clear=True)

# Prepare the sequence output for the scanner
seq.set_definition('FOV', [fov, fov, slice_thickness]*1000)
seq.set_definition('Name', 'gre')
seq.write('out/external.seq')
seq.write('out/' + experiment_id +'.seq')


#S4: SETUP SPIN SYSTEM/object on which we can run the MR sequence external.seq from above
from new_core.sim_data import SimData

# (i) load a phantom object from file
if 1:
    obj_p = SimData.load('../data/phantom2D.mat')
    obj_p = SimData.load('../data/numerical_brain_cropped.mat')
    obj_p.T2dash[:] = 30e-3
    obj_p = obj_p.resize(64,64,1)
else:
# or (ii) set phantom  manually to a pixel phantom
    obj_p = torch.zeros((sz[0],sz[1],6)); 
    
    obj_p[7,25,:]=torch.tensor([1, 1, 0.1, 0.1, 0, 1]) # dimensions: PD, T1 T2, T2dash, dB0 rB1
    obj_p[7,23:29,:]=torch.tensor([1, 1, 0.1,0.1, 0, 1]) # dimensions: PD, T1 T2,T2dash, dB0 rB1
    
    obj_p=obj_p.permute(2,0,1)[:,:,:,None]
    obj_p= SimData(obj_p[0,:],obj_p[1,:],obj_p[2,:],obj_p[3,:],obj_p[4,:],obj_p[5,:]*torch.ones(1,obj_p.shape[1],obj_p.shape[2],1),torch.ones(1,obj_p.shape[1],obj_p.shape[2],1),normalize_B0_B1= False)
    obj_p = obj_p.resize(sz[0],sz[1],1)
    
# manipulate obj and plot it
obj_p.B0*=1;
obj_p.plot_sim_data()


#S5:. SIMULATE  the external.seq file and add acquired signal to ADC plot
signal, kspace_loc,= sim_external(obj=obj_p,plot_seq_k=[0,1])   

# plot the result into the ADC subplot
sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
sp_adc.plot(t_adc,np.abs(signal.numpy()))
seq.plot(signal=signal.numpy())

# %% S6: MR IMAGE RECON of signal ::: #####################################
fig=plt.figure(); # fig.clf()
plt.subplot(411); plt.title('ADC signal')
spectrum=torch.reshape((signal),(Nphase,Nread)).clone().t()
kspace=spectrum
kspace_adc=spectrum
plt.plot(torch.real(signal),label='real')
plt.plot(torch.imag(signal),label='imag')

major_ticks = np.arange(0, Nphase*Nread, Nread)*adc.num_samples//(Nphase*Nread) # this adds ticks at the correct position szread
ax=plt.gca(); ax.set_xticks(major_ticks); ax.grid()

space = torch.zeros_like(spectrum)

if 0: # FFT
    # fftshift
    spectrum=torch.fft.fftshift(spectrum,0); spectrum=torch.fft.fftshift(spectrum,1)
    #FFT
    space = torch.fft.ifft2(spectrum)
    # fftshift
    space=torch.fft.ifftshift(space,0); space=torch.fft.ifftshift(space,1)


if 1: # NUFFT
    import scipy.interpolate
    grid = kspace_loc[:,:2]
    Nx=Nphase
    Ny=Nread
    
    X, Y = np.meshgrid(np.linspace(0,Nx-1,Nx) - Nx / 2, np.linspace(0,Ny-1,Ny) - Ny/2)
    grid = np.double(grid.numpy())
    grid[np.abs(grid) < 1e-3] = 0
    
    plt.subplot(347); plt.plot(grid[:,0].ravel(),grid[:,1].ravel(),'rx',markersize=3);  plt.plot(X,Y,'k.',markersize=2);
    plt.show()
    
    spectrum_resampled_x = scipy.interpolate.griddata((grid[:,0].ravel(), grid[:,1].ravel()), np.real(signal.ravel()), (X, Y), method='cubic')
    spectrum_resampled_y = scipy.interpolate.griddata((grid[:,0].ravel(), grid[:,1].ravel()), np.imag(signal.ravel()), (X, Y), method='cubic')

    kspace_r=spectrum_resampled_x+1j*spectrum_resampled_y
    kspace_r[np.isnan(kspace_r)] = 0
    
    # fftshift
    # kspace_r = np.roll(kspace_r,Nx//2,axis=0)
    # kspace_r = np.roll(kspace_r,Ny//2,axis=1)
    kspace_r_shifted=np.fft.ifftshift(kspace_r,0); kspace_r_shifted=np.fft.ifftshift(kspace_r_shifted,1)
             
    space = np.fft.ifft2(kspace_r_shifted)
    space=np.fft.ifftshift(space,0); space=np.fft.ifftshift(space,1)

end=time.time()
print(f'cost:{end -start}')

space=np.transpose(space)
plt.subplot(345); plt.title('k-space')
plt.imshow(np.abs(kspace))
plt.subplot(349); plt.title('k-space_r')
plt.imshow(np.abs(kspace_r))

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