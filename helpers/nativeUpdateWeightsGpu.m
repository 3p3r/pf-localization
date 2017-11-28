% Updates particle weights on GPU
function wp = nativeUpdateWeightsGpu(System)
    System.wp = libpointer('doublePtr',System.wp);
    calllib('p2c','updateWeights_gpu',System);
    wp = get(System.wp,'Value');
end
