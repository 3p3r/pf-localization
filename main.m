clc;
clear all;
close all;
rng default;
addpath bin;
addpath helpers;
if ~libisloaded('p2c'); loadlibrary p2c; end;

N = 1000000; % particle count
input = 'data/pixel-xl.mp4'; % video file
policy = 'gpu'; % 'gpu', 'cpu', or 'matlab'
positionSigma = [100 200 100]; % mm, in XYZ axes
rotationSigma = [0.1 0.05 0.5 1]; % unitless
groundTruth = ggt(input); % ground truth 2d and 3d data
[positionNoise, rotationNoise] = createNoise(N, positionSigma, rotationSigma);
frameCount = length(groundTruth); % this is how many total frames we have
load('cameraParams.mat'); % previously calculated camera intrinsics (K-mat)

% video IO
vr = VideoReader(input);
vw = VideoWriter('result.avi');
vw.FrameRate = vr.FrameRate;
open(vw);

% Initialization
System.wp = zeros(N,1); % particle weights
System.xp = zeros(N,1); % particles
System.cameraParams = cameraParams.IntrinsicMatrix;
System.imagePoints = []; % 2D feature locations in the image
System.worldPoints = []; % corresponding 3D world points
System.N = N; % static constant, number of particles
System.F = 0; % updated every frame, number of features found in image

index = 1;
while index <= length(groundTruth) && hasFrame(vr)
    entry = groundTruth(index);
    frame = rgb2gray(readFrame(vr));
    System.F = length(entry.WorldPoints);
    System.imagePoints = entry.ImagePoints;
    System.worldPoints = entry.WorldPoints;
    
    if index == 1
        % first frame is for initialization
        q = entry.PoseRotationTrue;
        t = entry.PoseTranslationTrue;
        [System.xp,System.wp] = createParticles(q,t,N);
    else
        % second frame onward is for tracking (localization)
        System.xp = updateParticles(System,positionNoise,rotationNoise);
        System.wp = updateWeights(System,cameraParams,policy);
        System.xp = resampleParticles(System);
        [q,t] = extractEstimates(System);
        plotSystem(vw,cameraParams,System,frame,entry,t,q,index);
        System.wp = resetWeights(System);
    end
    
    index = index + 1;
end
unloadlibrary p2c;
close(vw);

% _________________________________________________________________________
% Extratcs the mean of particles as system's estimate.
function [q,t] = extractEstimates(System)
    q = sum(System.wp .* System.xp(:,4:7)); % estimated quaternion
    t = sum(System.wp .* System.xp(:,1:3)); % estimated translation
    [q,t] = cameraPoseToExtrinsics(quat2rotm(q),t);
    q = rotm2quat(q)';
end

% Systematic resmapling of particles.
function xp = resampleParticles(System)
    R = cumsum(System.wp);
    T = rand(1, System.N);
    [~, I] = histc(T, R);
    xp = System.xp(I + 1, :);
end

% Systematic resmapling of particles.
function wp = resetWeights(System)
    wp = ones(System.N,1) / System.N;
end

% Updates particle weights based on their liklihood (measurement model).
function wp = updateWeights(System,cameraParams,policy)
    if isequal(policy, 'gpu')
        % Execude on a Cuda capable device (GPU)
        wp = nativeUpdateWeightsGpu(System);
    elseif isequal(policy, 'cpu')
        % Execude with native code (C++11 runtime)
        wp = nativeUpdateWeightsCpu(System);
    else
        % Execude with matlab (matlab runtime)
        N = size(System.wp,1);
        observationPoints = System.imagePoints * 0;
        for i=1:N
            t = System.xp(i,1:3);
            R = quat2rotm(System.xp(i,4:7));
            [Rx,tx] = cameraPoseToExtrinsics(R,t);
            camMatrix = cameraMatrix(cameraParams,Rx,tx);

            for j=1:System.F
                projection = [System.worldPoints(j,:) 1] * camMatrix;
                projection = projection ./ projection(3);
                observationPoints(j,:) = projection(1:2);
            end

            C = cov(observationPoints(:,:));
            D = zeros(System.F,1);

            for j=1:System.F
                d = observationPoints(j,:)-System.imagePoints(j,:);
                D(j) = (d/C)*d';
            end

            System.wp(i) = System.wp(i) * (1/(2*pi*sqrt(det(C))))*exp(-sum(D)/2);
        end
        wp = System.wp ./ sum(System.wp);        
    end
end

% Adds noise to particles (system model)
function xp = updateParticles(System,posNoise,rotNoise)
    xp(:,1:3) = System.xp(:,1:3) + posNoise;
    xp(:,4:7) = quatmultiply(System.xp(:,4:7), rotNoise);
end

% Creates the initial particle cloud
function [xp,wp] = createParticles(q0,t0,N)
    xp = ones(N,7);
    xp(:,1) = t0(1); % x
    xp(:,2) = t0(2); % y
    xp(:,3) = t0(3); % z
    xp(:,4) = q0(1); % q-r
    xp(:,5) = q0(2); % q-i
    xp(:,6) = q0(3); % q-j
    xp(:,7) = q0(4); % q-k
    wp = ones(N,1) / N;
end

% genrate Random Noise.
function [positionNoise, rotationNoise] = createNoise(N, positionSigma, rotationSigma)
    assert(isequal(size(positionSigma),[1,3]), 'position standard deviation msut be 3x1');
    assert(isequal(size(rotationSigma),[1 4]), 'rotation standard deviation msut be 1x4');
    v = rotationSigma(2:4) .* randn(N,3);
    r = rotationSigma(1) * randn(N,1);
    rotationNoise = zeros(N,4);
    for i=1:N
        rotationNoise(i,:) = [cos(r(i)) sin(r(i)) * v(i,:) / norm(v(i,:))]';
    end
    positionNoise = repmat(positionSigma, N, 1) .* randn(N,3);
end
