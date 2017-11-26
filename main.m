clc;
clear all;
close all;
rng default;
addpath helpers;

N = 1000; % particle count
input = 'data/board.mp4'; % video file
positionSigma = 15*[1 1 1]; % mm, in XYZ axes
rotationSigma = 0.2*[1 1 1]; % radians, in Euler axes
groundTruth = ggt(input); % ground truth 2d and 3d data
[positionNoise, rotationNoise] = grn(N, positionSigma, rotationSigma);
frameCount = length(groundTruth); % this is how many total frames we have

% Initialization
vr = VideoReader(input);
load('cameraParams.mat');
readFrame(vr); % discard the first frame, it's used for initialization
% tracking (localization) begins in the second frame (post initialization)
z = zeros(frameCount,7); % all estimates of quaternion and translation (means)

index = 1;
while hasFrame(vr)
    frame = rgb2gray(readFrame(vr));
    frameUndistorted = undistortImage(frame, cameraParams);
    x = groundTruth(index);
    
    if index == 1
        % first frame is for initialization
        q = x.RotationExtrinsics;
        t = x.TranslationExtrinsics;
        [xp,wp] = createParticles(q,t,N);
    else
        % second frame onward is for tracking (localization)
        xp = updateParticles(xp,positionNoise,rotationNoise);
        wp = updateWeights(wp,xp,cameraParams,x.ImagePoints,x.WorldPoints);
        xp = resampleParticles(xp,wp);
        [q,t] = extractEstimates(xp);
    end
    
    disp(num2str(index));
    index = index + 1;
end

% _________________________________________________________________________
% Extratcs the mean of particles as system's estimate.
function [q,t] = extractEstimates(xp)
    Qe = mean(xp(:,4:7)); % estimated quaternion
    Te = mean(xp(:,1:3)); % estimated translation
    [Qp,t] = cameraPoseToExtrinsics(quat2rotm(Qe),Te);
    q = rotm2quat(Qp);
end

% Systematic resmapling of particles.
function xp = resampleParticles(xp,wp)
    R = cumsum(wp);
    N = size(wp,1);
    T = rand(1, N);
    [~, I] = histc(T, R);
    xp = xp(I + 1, :);
end

% Updates particle weights based on their liklihood (measurement model).
function wp = updateWeights(wp,xp,cameraParams,imagePoints,worldPoints)
    N = size(wp,1);
    featureCount = size(worldPoints, 1);
    observationPoints = imagePoints * 0;
    
    for i=1:N
        t = xp(i,1:3);
        R = quat2rotm(xp(i,4:7));
        [Rx,tx] = cameraPoseToExtrinsics(R,t);
        camMatrix = cameraMatrix(cameraParams,Rx,tx);

        for j=1:featureCount
            projection = [worldPoints(j,:) 1] * camMatrix;
            projection = projection ./ projection(3);
            observationPoints(j,:) = projection(1:2);
        end
        
        C = cov(observationPoints(:,:));
        D = zeros(featureCount,1);
        
        for j=1:featureCount
            d = observationPoints(j,:)-imagePoints(j,:);
            D(j) = (d/C)*d';
        end
        
        wp(i) = (1/(2*pi*sqrt(det(C))))*exp(-sum(D)/2);
    end
    
    wp = wp ./ sum(wp);
end

% Adds noise to particles (system model)
function xp = updateParticles(xp,posNoise,rotNoise)
    xp(:,1:3) = xp(:,1:3) + posNoise;
    xp(:,4:7) = quatmultiply(xp(:,4:7), rotNoise);
end

% Creates the initial particle cloud
function [xp,wp] = createParticles(q0,t0,N)
    xp = ones(N,7);
    xp(:,1) = xp(:,1) * t0(1); % x
    xp(:,2) = xp(:,2) * t0(2); % y
    xp(:,3) = xp(:,3) * t0(3); % z
    xp(:,4) = xp(:,4) * q0(1); % q-r
    xp(:,5) = xp(:,5) * q0(2); % q-i
    xp(:,6) = xp(:,6) * q0(3); % q-j
    xp(:,7) = xp(:,7) * q0(4); % q-k
    wp = ones(N,1) / N;
end
