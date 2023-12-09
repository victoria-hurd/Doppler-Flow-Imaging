% AUTHOR: Victoria Hurd
% DATE CREATED: 12/8/2023
% DATE LAST MODIFIED: 12/9/2023
% PROJECT: MCEN 5127 Final Project
% DESCRIPTION: Wrapper function to perform calculation for all beam angles.
% Performs velocity estimation angle corrections. Overlays power and color
% Doppler results in a single plot. Per beam angle, outputs include an
% animation of unprocessed B-Mode over time, axial frequency spectrum
% before and after baseband demodulation wand shifting, wall filter effects,
% color Doppler flow data, and power Doppler data. 
% (Created on Mac M1 ARM chip)

%% Housekeeping
clear;clc;close all
OS = ispc;

%% Data Read
if OS == 0
    load("./data/flow_data.mat")
elseif OS == 1
    load(".\data\flow_data.mat")
end

%% High Quality vs Low Quality 
% Displaying B-Mode Images
% Perform envelope detection to convert RF data to pressure field
% Hilbert's Transform - absolute value of complex hilbert's gives envelope
% High-Quality data
envHigh = abs(hilbert(rf_bmode));
% Low-Quality data per steering angle
envLow_n12 = abs(hilbert(rf(:,:,1,1)));
envLow_0 = abs(hilbert(rf(:,:,1,2)));
envLow_12 = abs(hilbert(rf(:,:,1,3)));

% Display the log-compressed image using proper axes (with labels) and 
% proportions with a 60 dB dynamic range, normalized to the mean point
% in the image. Provide a colorbar.
figure
hold on 
h = surf(x*1e3,z*1e3,20*log10(envHigh/max(envHigh(:))));
set(h,'LineStyle','none')
title("High Quality B-mode")
xlabel("X Position [mm]")
ylabel("Z Position [mm]")
colormap(gray)
colorbar
ylim([min(z*1e3),max(z*1e3)])
xlim([min(x*1e3),max(x*1e3)])
set(gca, 'YDir','reverse')
clim([-60 0])
hold off

figure
hold on 
h = surf(x*1e3,z*1e3,20*log10(envLow_n12/max(envLow_n12(:))));
set(h,'LineStyle','none')
title("Low Quality B-mode: -12deg Steering Angle")
xlabel("X Position [mm]")
ylabel("Z Position [mm]")
colormap(gray)
colorbar
ylim([min(z*1e3),max(z*1e3)])
xlim([min(x*1e3),max(x*1e3)])
set(gca, 'YDir','reverse')
clim([-60 0])
hold off

figure
hold on 
h = surf(x*1e3,z*1e3,20*log10(envLow_0/max(envLow_0(:))));
set(h,'LineStyle','none')
title("Low Quality B-mode: -0deg Steering Angle")
xlabel("X Position [mm]")
ylabel("Z Position [mm]")
colormap(gray)
colorbar
ylim([min(z*1e3),max(z*1e3)])
xlim([min(x*1e3),max(x*1e3)])
set(gca, 'YDir','reverse')
clim([-60 0])
hold off

figure
hold on 
h = surf(x*1e3,z*1e3,20*log10(envLow_12/max(envLow_12(:))));
set(h,'LineStyle','none')
title("Low Quality B-mode: 12deg Steering Angle")
xlabel("X Position [mm]")
ylabel("Z Position [mm]")
colormap(gray)
colorbar
ylim([min(z*1e3),max(z*1e3)])
xlim([min(x*1e3),max(x*1e3)])
set(gca, 'YDir','reverse')
clim([-60 0])
hold off

%% Function Call
% Preallocate space for color Doppler and power Doppler (primary outputs)
colorDoppler = ones(1,1,1,length(angles));
powerDoppler = ones(1,1,length(angles));
% Function call outputs 
for i = 1:length(angles)
    [colorDoppler(:,:,:,i),powerDoppler(:,:,i)] = ...
        core(rf,f0,x,z,c,prf,cmap_cf,cmap_pd,angles,i,OS);
end

%% Angle Estimations

%% Power/Color Overlay