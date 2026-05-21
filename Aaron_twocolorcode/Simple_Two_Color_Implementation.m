clear all; close all; clc;

% Steps needed before code can be run

% Make sure LinearLocalDemosaic_BlackFly23S6C.m is also in the current
% Matlab folder

% Import camera recording (Should be imported as bFLY_recording)
% Import homography transformation (Should be imported as h)
% Import camera calibration data (Should be imported as
% TemperatureRGnofilter)

load('WFS_8_TS_6_camera.mat'); % This is where you load each of nine data files. 
load('ESAB_homography_transform.mat');
TemperatureRGnofilter = readtable('Temperature_RG_nofilter.xlsx');

%% Determine the average pixel intesity of each frame

data = bFLY_recording./16;
APV = squeeze(mean(data,[1,2]));
frame_range = 1:size(data,4);

%% Plot average pixel intensity for all frames

figure(3)
plot(frame_range, APV,'-k','LineWidth',1.25,'DisplayName','t = 350 us')
xlabel('Frame Number','FontSize',16)
ylabel('Average Pixel Intensity','FontSize',16)
set(gca,'FontSize',16)

%% Plot individual images to check for overexposure
% Change the number to select which frame is shown.

figure(1)
imshow(bFLY_recording(:,:,:,205))

%% Seperate out the calibration data components (only needs to be run once)

bfly_data = TemperatureRGnofilter; % R/G to Temp calibration table
theoretical_temp = table2array(bfly_data(:,1));% Temperature from RG calibration table
theoretical_signal = table2array(bfly_data(:,2));% Signal R/G from RG calibration table

%% Set upper and lower noise limits, solidus and liquidus temperatures, and pixel size (only needs to be run once)

upper_signal_noise = 3500; %upper bound limit for saturation ~4095
lower_signal_noise = 150; % lower bound noise without accurat temp from calib
T_solidus = 1670;
T_liquidus = 1750;

pixelsize = 0.0305667702; % (Not like correct for ESAB experiments)

%% Load and demosaic a chosen image
% Select the image by frame number in the following line of code

id = 125;

current_frame = bFLY_recording(:,:,:,id);
idemosaic = LinearLocalDemosaic_BlackFly23S6C(current_frame./16);

%% Plot red channel of the demosaiced image (just for checking it worked)

figure(5)
imshow(uint16(idemosaic(:,:,1).*16))

%% Perform the homography transformation

image_template = imref2d([1200,1920,1]);
% idemosaic = imwarp(idemosaic, h,'OutputView',image_template);
idemosaic = imwarp(idemosaic, h, 'OutputView', imref2d(size(idemosaic)));
%, 'OutputView', imref2d(size(idemosaic))

%% Take the red green ratio and apply noise limits

redgreen_ratio = squeeze(double(idemosaic(:,:,1))./double(idemosaic(:,:,2)));
redgreen_ratio(idemosaic(:,:,1)>upper_signal_noise) = NaN; %denoising
redgreen_ratio(idemosaic(:,:,2)<lower_signal_noise) = NaN; %denoising

%% Convert the red green ratio to temperature using the calibration data

Temp_RG = interp1(theoretical_signal,theoretical_temp,redgreen_ratio,'linear',0);
y_pixels = 1:size(Temp_RG,1);
x_pixels = 1:size(Temp_RG,2);
y_distancemm = y_pixels*pixelsize;
x_distancemm = x_pixels*pixelsize;

Temp_RG = flipud(Temp_RG);

%% Plot the red green ration (just for checking it worked)

contourf(x_distancemm, y_distancemm, redgreen_ratio, linspace(0, 3, 100),'LineStyle','none')
title('R/G')
colormap jet
clim([0, 3])
%colormap jet
xlabel('x (mm)')  % Original y-axis becomes x-axis
ylabel('y (mm)')  % Original x-axis becomes y-axis
colorbar
% xlim([40 90])
% ylim([25 55])

 
%% Plot the final temperature distribution

figure(2)
contourf(x_distancemm, y_distancemm, Temp_RG, 1100:50:2600, 'LineStyle', 'none')
%x_distancemm, y_distancemm, 
%title('T (K)')
colormap jet
%clim([1500, 2500])
xlabel('x (mm)')  % Original y-axis becomes x-axis
ylabel('y (mm)')  % Original x-axis becomes y-axis
c = colorbar;
c.Label.String = 'Temperature (K)';
axis equal
%ax = gca; ax.DataAspectRatio = [1 1 1];
% xlim([40 90])
% ylim([25 55])
set(gcf, 'color', 'w'); set(gca, 'fontname', 'helvetica', 'fontsize', 16, 'color', 'white');

