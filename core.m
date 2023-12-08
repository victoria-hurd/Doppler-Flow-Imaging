% AUTHOR: Victoria Hurd
% DATE CREATED: 11/28/2023
% DATE LAST MODIFIED: 12/8/2023
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
env = abs(hilbert(rf));
% Define number of frames for animation
loops = size(rf,3); % Get number of timestamps within rf
angleInd = 3;
% Create video object
v = VideoWriter("newfile",'MPEG-4');
% Open object
open(v)
% Create movie
for i = 1:loops
    env_i = env(:,:,i,angleInd);
    env_i = 20*log10(env_i/max(env_i(:)));
    clf
    imagesc(env_i)
    colormap(gray)
    title("Flow over Time")
    xlabel("X Position [m]")
    ylabel("Z Position [m]")
    colorbar
    drawnow
    frame = getframe(gcf);
   writeVideo(v,frame)
end
% close object
close(v)

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
y = Hdata.*exp(-1i*2*pi*f0*t');
yFFT = abs(fft(Hdata.*exp(-1i*2*pi*f0*t')));

% Time averaging - time is dimension 3 - average to make into 2D data
before_timeavg = mean(ybefore,3);
preshift_timeavg = mean(ypreshift,3);
after_timeavg = mean(yFFT,3);

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
f = Fs/L*(0:(L/2)); % one-sided frequency vector

% FFT in Doppler Freq
ffty = abs(fft(mat));

% Find cutoffs from time data
% Lateral averaging - lateral is dimension 3
mat_lateralavg = mean(ffty,3);
% Average axially  
mat_axialavg = mean(mat_lateralavg,2);
% Plot
figure
hold on
grid minor
plot(f,mat_axialavg(1:L/2+1))
xlim([min(f) max(f)])
xline(720)
title('Single-Sided Frequency Spectrum')
xlabel('Doppler Frequency (Hz)')
ylabel('|FFT|')
hold off

% Design filter
% Define Nyquist 
Fn = Fs/2;
% https://www.mathworks.com/matlabcentral/answers/412443-butterworth-filtfilt-and-fft-ifft-problem
flc = 200/Fn;
n = 4;
[b,a] = butter(n,flc,'high');

% Apply filter
matFiltered = filtfilt(b, a, mat);

% Show effect of frequency spectrum cutoff
% Average over axial and lateral
filteredSpectra = mean(abs(fft(matFiltered)),[2 3]);
unfilteredSpectra = mean(ffty,[2 3]);

% Plot spectra results
figure
hold on 
grid minor
plot(f,unfilteredSpectra(1:L/2+1))
plot(f,filteredSpectra(1:L/2+1))
legend('Unfiltered Signal','Filtered Signal')
title('Filter Cutoff Effects')
xlabel('Doppler Frequency (Hz)')
ylabel('|FFT|')
hold off

% Permute again to make time dimension 3 again
matFinal = permute(matFiltered,[3 2 1]); % lateral, axial, time

% Displaying filtered B mode
wallBMode = abs(matFinal);
figure
imagesc(wallBMode(:,:,25))


%% Color Flow Doppler
M = 5;
N = 50; % change this if we remove frames in wall filter step

% https://www.biomecardio.com/files/Color_Doppler.pdf
% dim #1-2 = space; dim #3 = slow-time
y1 = matFinal(:,:,1:end-1);
y2 = matFinal(:,:,2:end); % lag-1 temporal shift
%-- lag-1 I/Q auto-correlation
R1 = sum(y1.*conj(y2),3);
% R1 = y1.*conj(y2);
%-- Doppler velocity (Vd)
VN = c*prf/4/f0; % Nyquist velocity
Vd = -VN*imag(log(R1))/pi;
% Vd = -c*prf/(4*pi*f0)*atan(real(R1)); % Nyquist velocity

figure
imagesc(Vd(:,:)*1e2)
colormap(cmap_cf)
c = colorbar;
c.Label.String = 'Flow Velocity (cm/s)';


%% Color Flow Vicki's first attempt
I = real(matFinal);
Q = imag(matFinal);

num = 0;
denom = 0;
% I'm confused by the bounds given in the project document. 
% We're told to sum from element n=-1 which doesn't exist?
for i = 1:M
    for j=2:N
        numnew = Q(:,i,j).*I(:,i,j-1)-I(:,i,j).*Q(:,i,j-1); 
        denomnew = I(:,i,j).*I(:,i,j-1)+Q(:,i,j).*Q(:,i,j-1); 
        num = num+numnew;
        denom = denom+denomnew;
    end
end

% This gives a single vector? In project doc it gives scalar. What should
% dimensions really be here? 800*300*50? Show flow over time?
v = (-c*prf/(4*pi*f0))*atan2(num,denom);

%% Power Doppler
s_bb = 1;
power = abs(matFinal*s_bb)^2;

