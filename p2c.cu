#include "p2c.h"

#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

struct updateWeightsOp {
    updateWeightsOp(const System& input) :
        input(input) { /* no-op */ }

    System input;

    __host__ __device__
    void operator()(std::size_t i) {
        glm::dvec3 translation(input.xp[i], input.xp[input.N+i], input.xp[2*input.N+i]);
        glm::dvec4 quaternion(input.xp[3*input.N+i], input.xp[4*input.N+i], input.xp[5*input.N+i], input.xp[6*input.N+i]);

        input.wp[i] = i;
    }
};

extern "C" {

    DLL_PUBLIC void updateWeights_cpu(System input) {
        const thrust::counting_iterator<std::size_t> first(0);
        const auto last = first + input.N;
        thrust::for_each(thrust::host, first, last, updateWeightsOp(input));
    }

    DLL_PUBLIC void updateWeights_gpu(System input) {
        using vec = thrust::device_vector<double>;

        // transfer Matlab data >> to GPU (device)
        vec d_wp(input.wp, input.wp + input.N);
        vec d_xp(input.xp, input.xp + input.N*7);
        vec d_cameraParams(input.cameraParams, input.cameraParams + 3*3);
        vec d_imagePoints(input.imagePoints, input.imagePoints + input.F * 2);
        vec d_worldPoints(input.worldPoints, input.worldPoints + input.F * 2);

        // this will be available to every thread on GPU (data)
        const System d_system{
            raw_pointer_cast(d_wp.data()),
            raw_pointer_cast(d_xp.data()),
            raw_pointer_cast(d_cameraParams.data()),
            raw_pointer_cast(d_imagePoints.data()),
            raw_pointer_cast(d_worldPoints.data()),
            input.N,
            input.F
        };

        const thrust::counting_iterator<std::size_t> first(0);
        const auto last = first + input.N;
        thrust::for_each(thrust::device, first, last, updateWeightsOp(d_system));

        // transfer GPU data >> to Matlab (host)
        ::cudaMemcpy(input.wp, thrust::raw_pointer_cast(d_wp.data()),
                     input.N * sizeof(double), cudaMemcpyDeviceToHost);
    }

}
