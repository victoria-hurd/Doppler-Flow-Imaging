% AUTHOR: Victoria Hurd
% DATE CREATED: 11/28/2023
% DATE LAST MODIFIED: 11/28/2023
% PROJECT: MCEN 5127 Final Project
% DESCRIPTION: Core code for MCEN 5127 Final Project. Eventually will be
% run from wrapper.m as a function

%% Housekeeping
clear;clc;close all

%% Data Read
load("./data/flow_data.mat")

%% Display B-Mode Images
% Perform envelope detection to convert RF data to pressure field
% Hilbert's Transform - absolute value of complex hilbert's gives envelope
% High-Quality data
env = abs(hilbert(rf_bmode));
% Low-Quality data per steering angle
envLow_n12 = abs(hilbert(rf(:,:,1,1)));
envLow_0 = abs(hilbert(rf(:,:,1,2)));
envLow_12 = abs(hilbert(rf(:,:,1,3)));

% Display the log-compressed image using proper axes (with labels) and 
% proportions with a 60 dB dynamic range, normalized to the brightest point
% in the image. Provide a colorbar.
figure
hold on 
h = surf(x,z,20*log10(env/max(env(:))));
set(h,'LineStyle','none')
title("High Quality B-mode")
xlabel("X Position [m]")
ylabel("Z Position [m]")
colormap(gray)
colorbar
ylim([min(z),max(z)])
xlim([min(x),max(x)])
set(gca, 'YDir','reverse')
clim([-60 0])
hold off

figure
hold on 
h = surf(x,z,20*log10(envLow_n12/max(envLow_n12(:))));
set(h,'LineStyle','none')
title("Low Quality B-mode: -12deg Steering Angle")
xlabel("X Position [m]")
ylabel("Z Position [m]")
colormap(gray)
colorbar
ylim([min(z),max(z)])
xlim([min(x),max(x)])
set(gca, 'YDir','reverse')
clim([-60 0])
hold off

figure
hold on 
h = surf(x,z,20*log10(envLow_0/max(envLow_0(:))));
set(h,'LineStyle','none')
title("Low Quality B-mode: -0deg Steering Angle")
xlabel("X Position [m]")
ylabel("Z Position [m]")
colormap(gray)
colorbar
ylim([min(z),max(z)])
xlim([min(x),max(x)])
set(gca, 'YDir','reverse')
clim([-60 0])
hold off

figure
hold on 
h = surf(x,z,20*log10(envLow_12/max(envLow_12(:))));
set(h,'LineStyle','none')
title("Low Quality B-mode: 12deg Steering Angle")
xlabel("X Position [m]")
ylabel("Z Position [m]")
colormap(gray)
colorbar
ylim([min(z),max(z)])
xlim([min(x),max(x)])
set(gca, 'YDir','reverse')
clim([-60 0])
hold off

%% B-Mode Movie
%loops = size(rf,3); % Get number of timestamps within rf
loops = 10;
angleInd = 3;
F(loops) = struct('cdata',[],'colormap',[]);
for j = 1:loops
    figure
    hold on 
    %h = surf(x,z,20*log10(rf_bmode/max(rf_bmode(:))));
    h = surf(x,z,rf(:,:,j,angleInd));
    set(h,'LineStyle','none')
    %title("Log-Compressed Phantom Image")
    title("Averaged Phantom Image")
    xlabel("X Position [m]")
    ylabel("Z Position [m]")
    colormap(gray)
    colorbar
    ylim([min(z),max(z)])
    xlim([min(x),max(x)])
    set(gca, 'YDir','reverse')
    %clim([-60 0])
    axis tight manual
    ax = gca;
    ax.NextPlot = 'replaceChildren';
    hold off
    drawnow
    F(j) = getframe(gcf);
end
% Playback the movie two times.
fig = figure;
movie(fig,F,2)

%% Baseband Demodulation

%% Wall Filter

%% Color Flow Doppler

