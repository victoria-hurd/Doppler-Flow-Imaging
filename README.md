# Doppler-Flow-Imaging
This work transforms rudimentary radio frequency (RF) data into basic color Doppler and power Doppler data, as would be displayed on conventional ultrasound devices in clinical practice. 

This code operates via running a wrapper script (wrapper.m) to pull in RF data and repeatedly run through core calculations (core.m). This program performs basic B-Mode visualization and animation, RF signal demodulation, signal baseband shifting, wall filter application, phase-based velocity estimation (via Kasai's), power Doppler flow magnitude estimation, and color Doppler angle corrections. 

# Input Data

flow_data.mat:

rf bmode: RF data corresponding to pixels in the high quality 2-D image (axial x lateral)
rf:  RF data corresponding to pixels in the low quality 2-D image through time (axial x lateral x time x angle) x Lateral coordinates in meters
z: Axial coordinates in meters
angles: Beam angles for low quality data through time
c: Speed of sound
f0: Transmit center frequency in Hz
prf: Pulse repetition rate for the sequence (Hz)
cmap_cf: Colormap to be used with color flow Doppler
cmap_pd: Colormap to be used with color power Doppler

mask_data.mat:
Logical-type polygon mask to create the power/color Doppler overlay

# Output Data
Outputs include basic B-Mode, wall filtered B-Mode, color Doppler, and velocity-corrected color Doppler animations for each of the three steering angles. These outputs are placed into a movieOutputs folder. 

Thanks to Prof. Nick Bottenus for putting this project together!
