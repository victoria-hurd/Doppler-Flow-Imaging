% AUTHOR: Victoria Hurd
% DATE CREATED: 11/28/2023
% DATE LAST MODIFIED: 12/6/2023
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

% Check single high quality axial beam envelope to verify methods
figure
hold on
plot(envHigh(:,1))
plot(rf_bmode(:,1))
hold off

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

%% B-Mode Movie
% Perform envelope detection to convert RF data to pressure field
% Hilbert's Transform - absolute value of complex hilbert's gives envelope
Hdata = abs(hilbert(rf));
% Define number of frames for animation
loops = size(rf,3); % Get number of timestamps within rf
%loops = 10;
angleInd = 3;
% Create struct to store frames for animation
frames(loops) = struct('cdata',[],'colormap',[]);
for i = 1:loops
    env_i = Hdata(:,:,i,angleInd);
    figure
    hold on 
    h = surf(x*1e3,z*1e3,20*log10(env_i/max(env_i(:))));
    set(h,'LineStyle','none')
    title("Flow over Time")
    xlabel("X Position [mm]")
    ylabel("Z Position [mm]")
    colormap(gray)
    colorbar
    ylim([min(z*1e3),max(z*1e3)])
    xlim([min(x*1e3),max(x*1e3)])
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
% Separate out the angle of interest - makes 4D data into 3D data
rf_angle = rf(:,:,:,3); % 12 deg angle only for now

% Perform Hilbert Transform to convert RF to analytical signal
% https://www.mathworks.com/help/signal/ug/envelope-extraction-using-the-analytic-signal.html
Hdata = hilbert(rf_angle);

% Prep for FFT
% https://www.mathworks.com/help/matlab/ref/fft.html
Fs = 1/(mean(diff(z/c*2)));
T = 1/Fs;
[M, ~, ~, ~] = size(rf);
L = M; % number of rows - represents signal length
t = (0:L-1)*T; % time vector
f = Fs/L*(0:(L/2)); % one-sided frequency vector

% Perform baseband shifting - multiply signal by complex exponential to 
% shift towards f0
ybefore = abs(fft(rf_angle));
ypreshift = abs(fft(Hdata));
y = abs(fft(Hdata.*exp(-1i*2*pi*f0*t')));

% Time averaging - time is dimension 3 - average to make into 2D data
before_timeavg = mean(ybefore,3);
preshift_timeavg = mean(ypreshift,3);
after_timeavg = mean(y,3);

% Average laterally  
sig_before = mean(before_timeavg,2);
sig_preshift = mean(preshift_timeavg,2);
sig_after = mean(after_timeavg,2);

% Power spectrum results
figure
hold on
title('Single-Sided Power Spectra')
xlabel('Frequency [Hz]')
ylabel('Magnitude')
grid minor
plot(f,sig_before(1:L/2+1))
plot(f,sig_preshift(1:L/2+1))
plot(f,sig_after(1:L/2+1))
xline(f0)
xlim([min(f) max(f)])
legend('Real Data','Hilbert Transformed Data',...
    'Hilbert Transformed Data After Baseband Shifting',...
    'Known Center Frequency')
hold off

%% Wall Filter
% These analyses should be done of demodulated & baseband shifted data (y)
% Permute to make time dimension 1 instead of 3
mat = permute(y,[3 2 1]); % time, axial, lateral
% Cast as double
mat = double(mat);

% Prep FFT
% https://www.mathworks.com/help/matlab/ref/fft.html
Fs = prf; 
T = 1/Fs;
[M, ~, ~, ~] = size(mat);
L = M; % number of rows - represents signal length
t = (0:L-1)*T; % time vector

% Define Nyquist 
Fn = Fs/2;

% Design filter
% from Starstrider on Matlab Answers
% https://www.mathworks.com/matlabcentral/answers/412443-butterworth-filtfilt-and-fft-ifft-problem
fhc = 7e6/Fn;
flc = 3e6/Fn;
Wn=[flc fhc];
n = 4;
[b,a] = butter(n,Wn,'bandpass');

% Apply filter
matFiltered = filtfilt(b, a, mat);

% Permute again to make time dimension 3 again
matFinal = permute(matFiltered,[3 2 1]); % lateral, axial, time

% Perform ffts on filtered and unfiltered
yfiltered = abs(fft(matFinal));
yunfiltered = abs(fft(y));

% Time averaging - time is dimension 3 - average to make into 2D data
filtered_timeavg = mean(ypreshift,3);
unfiltered_timeavg = mean(y,3);

% Average laterally  
sig_filtered = mean(filtered_timeavg,2);
sig_unfiltered = mean(unfiltered_timeavg,2);

% Plot spectra results
figure
hold on 
grid minor
plot(sig_unfiltered)
plot(sig_filtered)
legend('Unfiltered Signal','Filtered Signal')
hold off

% Plot frequency response
% https://www.mathworks.com/help/signal/ref/freqz.html
%trapz();

%% Color Flow Doppler
M = 5;
N = 50; % change this if we remove frames in wall filter step

I = real(Hdata);
Q = imag(Hdata);

num = 0;
denom = 0;
% I'm confused by the bounds given in the project document. 
% We're told to sum from element n=-1 which doesn't exist?
for i = 1:M
    for j=1:N
        num = 1; % Change
        denom = 1; % Change
    end
end

% This gives a single value?
v = (-c*prf/(4*pi*f0))*atan2(num/denom);

