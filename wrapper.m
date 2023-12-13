% AUTHOR: Victoria Hurd
% DATE CREATED: 12/8/2023
% DATE LAST MODIFIED: 12/12/2023
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
    load("./data/mask.mat")
    path = "./movieOutputs/";
elseif OS == 1
    load(".\data\flow_data.mat")
    load(".\data\mask.mat")
    path = ".\movieOutputs\";
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
colorDoppler = ones(800,300,47,length(angles));
powerDoppler = ones(800,300,length(angles));
% Function call outputs 
for i = 1:length(angles)
    [colorDoppler(:,:,:,i),powerDoppler(:,:,i)] = ...
        core(rf,f0,x,z,c,prf,cmap_cf,cmap_pd,angles,i,OS);
end

%% Angle Estimations
% Estimate the angle of the vessel relative to the wave propagation 
% direction, considering the difference between the vessel angle and the 
% beam angle
theta = acosd(dot(colorDoppler(500,250,:,1),colorDoppler(500,250,:,3)));
thetaCorrect = theta+angles;

%% Angle Correction 
% Compare averaged velocity at specific points over time
for i = 1:length(angles)
    angle = num2str(angles(i));
    colorDoppler(:,:,:,i) = colorDoppler(:,:,:,i)./cosd(thetaCorrect(i));
% Doppler Flow Movie
    % Define number of frames for animation
    loops = size(colorDoppler(:,:,:,i),3); % Get number of timestamps within rf
    % Create video object
    str = strcat(path,"ColorDopplerCorrected_",angle,"deg");
    v = VideoWriter(str,'MPEG-4');
    % Open object
    open(v)
    % Create movie
    figure
    for j = 1:loops
        clf
        imagesc(colorDoppler(:,:,j,i)*1e2)
        colormap(cmap_cf)
        title("Doppler Flow")
        xlabel("X Position [m]")
        ylabel("Z Position [m]")
        cb = colorbar(); 
        ylabel(cb,'Flow Velocity [cm/s]','FontSize',14)
        clim([-40 40])
        drawnow
        frame = getframe(gcf);
       writeVideo(v,frame)
    end
    % close object
    close(v)
end

%% Power/Color Overlay
% Credit to Subhadeep Koley:
% https://www.mathworks.com/matlabcentral/answers/476715-superimposing-two-imagesc-graphs-over-each-other
%plot first data 
figure
ax1 = axes; 
im = imagesc(powerDoppler(:,:,2)); 
im.AlphaData = 0.5; % change this value to change the background image transparency 
axis square; 
hold all; 
%plot second data 
ax2 = axes; 
im1 = imagesc(mask.*colorDoppler(:,:,25,2)*1e2); 
im1.AlphaData = 0.5; % change this value to change the foreground image transparency 
axis square; 
title('Color and Power Doppler Overlay')
xlabel("X Position [m]")
ylabel("Z Position [m]")
%link axes 
linkaxes([ax1,ax2]) 
% Hide the top axes 
ax2.Visible = 'off'; 
% ax2.XTick = []; 
% ax2.YTick = []; 
% add differenct colormap to different data if you wish 
colormap(ax1,cmap_pd) 
colormap(ax2,cmap_cf) 
%set the axes and colorbar position 
set([ax1,ax2],'Position',[.17 .11 .685 .815]); 
cb1 = colorbar(ax1,'Position',[.05 .11 .0675 .815]); 
cb2 = colorbar(ax2,'Position',[.88 .11 .0675 .815]);
ylabel(cb2,'Flow Velocity [cm/s]')
title('Color and Power Doppler Overlay')
xlabel("X Position [m]")
ylabel("Z Position [m]")


