% AUTHOR: Victoria Hurd
% DATE CREATED: 11/28/2023
% DATE LAST MODIFIED: 11/30/2023
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
envHigh = abs(hilbert(rf_bmode));
% Low-Quality data per steering angle
envLow_n12 = abs(hilbert(rf(:,:,1,1)));
envLow_0 = abs(hilbert(rf(:,:,1,2)));
envLow_12 = abs(hilbert(rf(:,:,1,3)));

% Display the log-compressed image using proper axes (with labels) and 
% proportions with a 60 dB dynamic range, normalized to the brightest point
% in the image. Provide a colorbar.
figure
hold on 
h = surf(x,z,20*log10(envHigh/max(envHigh(:))));
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
% Perform envelope detection to convert RF data to pressure field
% Hilbert's Transform - absolute value of complex hilbert's gives envelope
env = abs(hilbert(rf));
% Define number of frames for animation
%loops = size(rf,3); % Get number of timestamps within rf
loops = 10;
angleInd = 3;
% Create struct to store frames for animation
frames(loops) = struct('cdata',[],'colormap',[]);
for i = 1:loops
    env_i = env(:,:,i,angleInd);
    figure
    hold on 
    h = surf(x,z,20*log10(env_i/max(env_i(:))));
    set(h,'LineStyle','none')
    title("Flow over Time")
    xlabel("X Position [m]")
    ylabel("Z Position [m]")
    colormap(gray)
    colorbar
    ylim([min(z),max(z)])
    xlim([min(x),max(x)])
    set(gca, 'YDir','reverse')
    clim([-60 0])
    hold off
    %drawnow
    frames(i) = getframe(gcf);
end
% Create animation figure with relevant name and make figure fullscreen
%figure('Name', 'Flow Over Time: Animation','units','normalized','outerposition',[0 0.006 1 1])
figure('Name','Animation')
% Switch hold to on to add helpful attributes to the plot
hold on
% Play animation twice at 3 frames per second
% Movie syntax: movie(gcf,[variable name],[number times to play], [fps])
movie(gcf,frames,2,1);
% Switch hold to off
hold off

% To save the animation, open an object under a relevant name
writerObj = VideoWriter('DopplerAnimation','MPEG-4');
% Set the framerate to 2 fps
writerObj.FrameRate = 2;
% Open the video writer
open(writerObj);
% Write the saved frames to the video
for i=1:length(frames)
    % Convert the image to a frame
    frame = frames(i) ;    
    drawnow
    % Write the frame for the video
    writeVideo(writerObj, frame);
end
% Close the writer object
close(writerObj);

%% Baseband Demodulation

%% Wall Filter

%% Color Flow Doppler

