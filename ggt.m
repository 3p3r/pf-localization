function out = ggt(videoFilePath)
% GGT  loads ground truth data from input video file. The input video file
% must contain a valid Matlab-checkerboard (odd-by-even sized) and be
% visible in ALL frames compeletely.
%   C = GGT('file.mp4') loads ground truth camera path from 'file.mp4'.
addpath helpers;

cacheDir = '.cache';
fileKey = DataHash(videoFilePath);
cacheKey = [cacheDir '/' fileKey '.mat'];
if ~exist(cacheDir, 'dir'); mkdir('.cache'); end;

if exist(cacheKey, 'file')
    disp(['Cache exists for "' videoFilePath '" in: "' cacheKey '"']);
    load(cacheKey, 'out');
    return;
end

disp(['Creating cache for "' videoFilePath '" in: "' cacheKey '"']);
vr = VideoReader(videoFilePath);
load('cameraParams.mat');
total = round(vr.Duration * vr.FrameRate);

n=0;
out = [];
index = 1;
while(hasFrame(vr))
    msg = ['Creating cache entry: ' num2str(index) '/' num2str(total)];
    fprintf(repmat('\b',1,n));
    fprintf(msg);
    n=numel(msg);
    
    entry = {};
    % frame's index
    entry.Index = index;
    % color frame
    frame = rgb2gray(readFrame(vr));
    % grayscale frame
    frameUndistorted = undistortImage(frame, cameraParams);
    % checkerboard pattern 2d points and detected board size
    [entry.ImagePoints, entry.BoardSize] = detectCheckerboardPoints(frameUndistorted);
    % checkerboard 3d points
    entry.WorldPoints = zeros(prod(entry.BoardSize - 1),3);
    entry.WorldPoints(:,1:2) = generateCheckerboardPoints(entry.BoardSize, 30); % board's blocks are 30mm
    % camera's extrinsics which represent the coordinate system transformation from world coordinates to camera coordinates.
    [entry.RotationExtrinsics, entry.TranslationExtrinsics] = ...
        extrinsics(entry.ImagePoints, entry.WorldPoints(:,1:2), cameraParams);
    % camera's orientation and location, represent the 3-D camera pose in the world coordinates
    [entry.Rotation, entry.Translation] = ...
        extrinsicsToCameraPose(entry.RotationExtrinsics, entry.TranslationExtrinsics);
    % convert to quaternions, since we estimate quaternions in this implementation
    entry.Rotation = rotm2quat(entry.Rotation)';
    entry.RotationExtrinsics = rotm2quat(entry.RotationExtrinsics)';
    % we use Nx1 sizes for everything by convention in this implementation
    entry.Translation = entry.Translation';
    entry.TranslationExtrinsics = entry.TranslationExtrinsics';
    out = [out entry];
    index = index + 1;
end
fprintf('\n');
save(cacheKey, 'out');
disp(['Cache created for "' videoFilePath '" in: "' cacheKey '"']);

rmpath helpers;
end