% AUTHOR: Victoria Hurd
% DATE CREATED: 11/28/2023
% DATE LAST MODIFIED: 12/9/2023
% PROJECT: MCEN 5127 Final Project
% DESCRIPTION: Core code for MCEN 5127 Final Project. Eventually will be
% run from wrapper.m as a function

function [vD,power] = core(rf,f0,x,z,c,prf,cmap_cf,cmap_pd,angles,idx,OS)
    %% Movie Naming String
    % Get operating system
    if OS == 0
        path = "./movieOutputs/";
    elseif OS == 1
        path = ".\movieOutputs\";
    end
    % Concatenate path and string
    angle = num2str(angles(idx));

    %% B-Mode Movie
    % Perform envelope detection to convert RF data to pressure field
    % Hilbert's Transform - absolute value of complex hilbert's gives envelope
    env = abs(hilbert(rf(:,:,:,idx)));
    % Normalize frames
    env = 20*log10(env/max(env(:)));
    % Define number of frames for animation
    loops = size(rf,3); % Get number of timestamps within rf
    % Create video object
    str = strcat(path,"BModeMovie_",angle,"deg");
    v = VideoWriter(str,'MPEG-4');
    % Open object
    open(v)
    % Create movie
    figure
    pause(0.01)
    for i = 1:loops
        env_i = env(:,:,i);
        clf
        imagesc(env_i)
        colormap(gray)
        title("Flow over Time")
        xlabel("X Position [m]")
        ylabel("Z Position [m]")
        colorbar
        clim([-60 0])
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
    [M, ~, ~, ~] = size(mat);
    L = M; % number of rows - represents signal length
    f = Fs/L*(0:(L/2)); % one-sided frequency vector
    
    % FFT in Doppler Freq
    ffty = abs(fft(mat));

    % Design filter
    % Define Nyquist 
    Fn = Fs/2;
    % https://www.mathworks.com/matlabcentral/answers/412443-butterworth-filtfilt-and-fft-ifft-problem
    flc = 300/Fn;
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
    
    % Remove first and last frames since they're dark
    matFinal(:,:,1) = [];
    matFinal(:,:,end) = [];
    
    % Wall filter movie
    wallBMode = abs(matFinal);
    % Define number of frames for animation
    loops = size(wallBMode,3); % Get number of timestamps within rf
    % Create video object
    str = strcat(path,"WallFilter_",angle,"deg");
    v = VideoWriter(str,'MPEG-4');
    % Open object
    open(v)
    % Create movie
    figure
    pause(0.01)
    for i = 1:loops
        clf
        imagesc(wallBMode(:,:,i))
        colormap(gray)
        title("Flow over Time")
        xlabel("X Position [m]")
        ylabel("Z Position [m]")
        colorbar
        clim([0 150])
        drawnow
        frame = getframe(gcf);
       writeVideo(v,frame)
    end
    % close object
    close(v)
    
    %% Color Flow Doppler
    % separate in phase and quadrature
    I = real(matFinal);
    Q = imag(matFinal);
    
    % Kasai's Calculations part 1 - arctan argument
    % Preallocations
    num = zeros(size(I,1),size(I,2),size(I,3)-1);
    denom = zeros(size(I,1),size(I,2),size(I,3)-1);
    % Calculate number of frames
    nFrames = size(I,3);
    for i = 2:nFrames
        num(:,:,i-1) = Q(:,:,i).*I(:,:,i-1)-I(:,:,i).*Q(:,:,i-1); 
        denom(:,:,i-1) = I(:,:,i).*I(:,:,i-1)+Q(:,:,i).*Q(:,:,i-1); 
    end
    % Kernel convolution
    kernel = ones(5,1,3);
    numConv = convn(num,kernel,'same');
    denomConv = convn(denom,kernel,'same');
    
    %  Kasai's Calculations part 2
    vD = (1*c*prf/(4*pi*f0))*atan2(numConv,denomConv);
    
    % Plotting single frame
    figure
    imagesc(imgaussfilt(vD(:,:,25),2))
    colormap(cmap_cf)
    cb = colorbar(); 
    ylabel(cb,'Flow Velocity [cm/s]','FontSize',14)

    % Doppler Flow Movie
    % Define number of frames for animation
    loops = size(vD,3); % Get number of timestamps within rf
    % Create video object
    str = strcat(path,"ColorDoppler_",angle,"deg");
    v = VideoWriter(str,'MPEG-4');
    % Open object
    open(v)
    % Create movie
    figure
    for i = 1:loops
        clf
        imagesc(vD(:,:,i)*1e2)
        colormap(cmap_cf)
        title("Doppler Flow")
        xlabel("X Position [m]")
        ylabel("Z Position [m]")
        cb = colorbar(); 
        ylabel(cb,'Flow Velocity [cm/s]','FontSize',14)
        drawnow
        frame = getframe(gcf);
       writeVideo(v,frame)
    end
    % close object
    close(v)

%     % Loupas'Calculations part 1 - arctan argument
%     % Preallocations
%     Lnum = zeros(size(I,1),size(I,2),size(I,3)-1);
%     Ldenom = zeros(size(I,1),size(I,2),size(I,3)-1);
%     % Calculate number of frames
%     nFrames = size(I,3);
%     for i = 2:nFrames
%         Lnum(:,:,i-1) = Q(:,:,i).*I(:,:,i-1)-I(:,:,i).*Q(:,:,i-1); 
%         Ldenom(:,:,i-1) = I(:,:,i).*I(:,:,i-1)+Q(:,:,i).*Q(:,:,i-1); 
%     end
%     % Kernel convolution
%     Lkernel = ones(5,1,3);
%     LnumConv = convn(num,Lkernel,'same');
%     LdenomConv = convn(denom,Lkernel,'same');
%     
%     %  Kasai's Calculations part 2
%     LvD = (1*c*prf/(4*pi*f0))*atan2(LnumConv,LdenomConv);
%     
%     % Plotting single frame
%     figure
%     imagesc(imgaussfilt(vD(:,:,25),2))
%     colormap(cmap_cf)
%     cb = colorbar(); 
%     ylabel(cb,'Flow Velocity [cm/s]','FontSize',14)
% 
%     % Doppler Flow Movie
%     % Define number of frames for animation
%     loops = size(LvD,3); % Get number of timestamps within rf
%     % Create video object
%     str = strcat(path,"LoupasColorDoppler_",angle,"deg");
%     v = VideoWriter(str,'MPEG-4');
%     % Open object
%     open(v)
%     % Create movie
%     figure
%     for i = 1:loops
%         clf
%         imagesc(LvD(:,:,i)*1e2)
%         colormap(cmap_cf)
%         title("Doppler Flow")
%         xlabel("X Position [m]")
%         ylabel("Z Position [m]")
%         cb = colorbar(); 
%         ylabel(cb,'Flow Velocity [cm/s]','FontSize',14)
%         drawnow
%         frame = getframe(gcf);
%        writeVideo(v,frame)
%     end
%     % close object
%     close(v)
    
    %% Power Doppler
    power = sum(abs(matFinal).^2,3);
    % Plotting single frame
    figure
    imagesc(x*1e3,z*1e3,power(:,:))
    colormap(cmap_pd)
    cb = colorbar(); 
    ylabel(cb,'units?','FontSize',14)
    xlabel("X Position [mm]")
    ylabel("Z Position [mm]")
    title('Power Doppler')
end
