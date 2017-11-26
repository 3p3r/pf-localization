function detectorParams = createDetectorParams()
    % marker detector parameters
    detectorParams = struct();
    if false
        %detectorParams.nMarkers = 1024;
        detectorParams.adaptiveThreshWinSizeMin = 3;
        detectorParams.adaptiveThreshWinSizeMax = 23;
        detectorParams.adaptiveThreshWinSizeStep = 10;
        detectorParams.adaptiveThreshConstant = 7;
        detectorParams.minMarkerPerimeterRate = 0.03;
        detectorParams.maxMarkerPerimeterRate = 4.0;
        detectorParams.polygonalApproxAccuracyRate = 0.05;
        detectorParams.minCornerDistanceRate = 0.05;
        detectorParams.minDistanceToBorder = 3;
        detectorParams.minMarkerDistanceRate = 0.05;
        detectorParams.cornerRefinementMethod = 'None';
        detectorParams.cornerRefinementWinSize = 5;
        detectorParams.cornerRefinementMaxIterations = 30;
        detectorParams.cornerRefinementMinAccuracy = 0.1;
        detectorParams.markerBorderBits = 1;
        detectorParams.perspectiveRemovePixelPerCell = 8;
        detectorParams.perspectiveRemoveIgnoredMarginPerCell = 0.13;
        detectorParams.maxErroneousBitsInBorderRate = 0.04;
        detectorParams.minOtsuStdDev = 5.0;
        detectorParams.errorCorrectionRate = 0.6;
    end
    detectorParams.cornerRefinementMethod = 'Subpix';  % do corner refinement in markers
end
