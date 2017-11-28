% Updates particle weights on CPU
function wp = nativeUpdateWeightsCpu(System)
    System.wp = libpointer('doublePtr',System.wp);
    calllib('p2c','updateWeights_cpu',System);
    wp = get(System.wp,'Value');
end
