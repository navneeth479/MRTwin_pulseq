# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:56:16 2020

@author: zaissmz
"""
import numpy as np
import scipy.io as sio
from math import pi,ceil, sqrt, pow
import sys

sys.path.append("../")
from pypulseq.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import makeadc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc import make_sinc_pulse
from pypulseq.make_trap import make_trapezoid
from pypulseq.make_block import make_block_pulse
from pypulseq.opts import Opts

# for trap and sinc
from pypulseq.holder import Holder

def append_header(seq_fn, FOV,slice_thickness):
    # append version and definitions
    with open(seq_fn, 'r') as fin:
        lines = fin.read().splitlines(True)
        
    updated_lines = []
    updated_lines.append("# Pulseq sequence file\n")
    updated_lines.append("# Created by MRIzero/IMR/GPI pulseq converter\n")
    updated_lines.append('#' + seq_fn + "\n")
    updated_lines.append("\n")
    updated_lines.append("[VERSION]\n")
    updated_lines.append("major 1\n")
    updated_lines.append("minor 2\n")   
    updated_lines.append("revision 1\n")  
    updated_lines.append("\n")    
    updated_lines.append("[DEFINITIONS]\n")
    updated_lines.append("FOV "+str(round(FOV*1e3))+" "+str(round(FOV*1e3))+" "+str(round(slice_thickness*1e3))+" \n")   
    updated_lines.append("\n")    
    
    updated_lines.extend(lines[3:])

    with open(seq_fn, 'w') as fout:
        fout.writelines(updated_lines)    


def FOV():
    #FOV = 0.110
    return 0.200

def toK(gradm_event): # gradm_event_numpy = deltak*gradm_event_numpy_input  # this is required to adjust for FOV
    deltak = 1.0 / FOV()
    return deltak*gradm_event

nonsel = 0
if nonsel==1:
    slice_thickness = 200*1e-3
else:
    slice_thickness = 8e-3
   
# save pulseq definition
MAXSLEW = 140
kwargs_for_opts = {"rf_ring_down_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 36, "grad_unit": "mT/m", "max_slew": MAXSLEW, "slew_unit": "T/m/s"}
system = Opts(kwargs_for_opts)
seq = Sequence(system)  
### actual sequence starts here: first: non-selective GRE
seq.add_block(make_delay(5.0))
NRep=16
for rep in range(NRep):
    ### first action excitation pulse
    rf = make_block_pulse({"flip_angle": 90.0, "system": system, "duration": 1e-3, "phase_offset": 0.0}, 1)            
    seq.add_block(rf)  
    seq.add_block(make_delay(1e-4))         
    ### second action rewinder gradient
    gxpre = make_trapezoid({"channel": 'x', "system": system, "flat_area": toK(-8.0), "flat_time": 1e-3})    
    gypre = make_trapezoid({"channel": 'y', "system": system, "flat_area": toK(rep-NRep/2), "flat_time": 1e-3})    
    seq.add_block(gxpre,gypre) 
    ###  ADC line acquisition, later this is NEvnt(5:end-2)
    gx_gradmom = 8 # only readout in x for now   
    gx = make_trapezoid({"channel": 'x', "system": system, "flat_area": toK(gx_gradmom), "flat_time": 1e-3})    
    adc = makeadc({"num_samples": 16, "duration": gx.flat_time, "delay": (gx.rise_time), "phase_offset": rf.phase_offset - np.pi/4})    
    seq.add_block(gx,adc)
    ###     second last extra event  
    gx_post = make_trapezoid({"channel": 'x', "system": system, "area": toK(8.0), "duration": 1e-3})  
    gy_post = make_trapezoid( {"channel": 'y', "system": system, "area": toK(-(rep-NRep/2)), "duration": 1e-3})  
    seq.add_block(gx_post, gy_post)    
    ###     last extra event  T(end)
    seq.add_block(make_delay(2*1e-3))
## end sequence definition
seq.plot()
seq_fn='out/test.seq'
seq.write(seq_fn)
append_header(seq_fn, FOV(),slice_thickness)
    
#%%## actual sequence starts here: second: slice-selective GRE
seq.add_block(make_delay(5.0)) 
NRep=16
for rep in range(NRep):
    ### first action excitation pulse
    kwargs_for_block = {"flip_angle": 90.0, "system": system, "duration": 1e-3, "phase_offset": 0.0}
    rf = make_block_pulse(kwargs_for_block, 1)            
    seq.add_block(rf)  
    seq.add_block(make_delay(1e-4))

##slice selective        
#        kwargs_for_sinc = {"flip_angle": 90.0, "system": system, "duration": 1e-3, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": rf_event_numpy[idx_T,rep,1]}
#        rf, gz, gzr = make_sinc_pulse(kwargs_for_sinc, 3)
#        seq.add_block(rf, gz)
#        seq.add_block(gzr)            
#        RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time
            
    ### second action rewinder gradient
    kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": toK(-8.0), "flat_time": 1e-3}
    gxpre = make_trapezoid(kwargs_for_gx)    
    
    kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": toK(rep-NRep/2), "flat_time": 1e-3}
    gypre = make_trapezoid(kwargs_for_gy)    
    seq.add_block(gxpre,gypre) 
                
    ###############################
    ###  line acquisition, later this is NEvnt(5:end-2)
    
    gx_gradmom = 8
    kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": toK(gx_gradmom), "flat_time": 1e-3}
    gx = make_trapezoid(kwargs_for_gx)    
    
    gy_gradmom = 0  # only readout in x for now
    kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": gy_gradmom, "flat_time": 1e-3}
    gy = make_trapezoid(kwargs_for_gy)
    
    kwargs_for_adc = {"num_samples": 16, "duration": gx.flat_time, "delay": (gx.rise_time), "phase_offset": rf.phase_offset - np.pi/4}
    adc = makeadc(kwargs_for_adc)    
    
    # dont play zero grads (cant even do FID otherwise)
    if np.abs(gx_gradmom) > 0 or np.abs(gy_gradmom) > 0:
        seq.add_block(gx,gy,adc)
    else:
        seq.add_block(adc)
    
#        #update rewinder for gxgy ramp times, from second event
#        kwargs_for_gxpre = {"channel": 'x', "system": system, "area": -gradm_event_numpy[idx_T[0],rep,0]/2 + gradmom_rewinder[0]-gx.amplitude*gx.rise_time/2, "duration": eventtime_rewinder}
#        gx_pre = make_trapezoid(kwargs_for_gxpre)
#        
#        kwargs_for_gypre = {"channel": 'y', "system": system, "area": gradmom_rewinder[1]-gy.amplitude*gy.rise_time/2, "duration": eventtime_rewinder}
#        gy_pre = make_trapezoid(kwargs_for_gypre)
#        
#        seq.add_block(gx_pre, gy_pre)
#        
#        seq.add_block(make_delay(1e-3))
    

    ###############################
    ###     second last extra event  T(end)  # adjusted also for fallramps of ADC
    
    kwargs_for_gxpost = {"channel": 'x', "system": system, "area": toK(8.0), "duration": 1e-3}
    gx_post = make_trapezoid(kwargs_for_gxpost)  
    
    kwargs_for_gypost = {"channel": 'y', "system": system, "area": toK(-(rep-NRep/2)), "duration": 1e-3}
    gy_post = make_trapezoid(kwargs_for_gypost)  
    
    seq.add_block(gx_post, gy_post)
    
    ###############################
    ###     last extra event  T(end)
    seq.add_block(make_delay(2*1e-3))
    
    
seq.plot()

seq_fn='out/test.seq'

seq.write(seq_fn)
append_header(seq_fn, FOV(),slice_thickness)