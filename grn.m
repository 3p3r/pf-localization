function [positionNoise, rotationNoise] = grn(N, positionSigma, rotationSigma)
% GRN  Genrate Random Noise. This function generates a set of static random
% numbers to be used in our particle filter with 'N' particles.
%   [P,Q] = GRN(1000) generates noise for 1000 particles.
    assert(isequal(size(positionSigma),[1,3]), 'position standard deviation msut be 3x1');
    assert(isequal(size(rotationSigma),[1,3]), 'rotation standard deviation msut be 3x1');
    positionNoise = repmat(positionSigma, N, 1) .* randn(N,3);
    rotationNoise = eul2quat(rotationSigma .* randn(N,3));
end
