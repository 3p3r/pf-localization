#include "p2c.h"

#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/tuple.h>

#include <vector>
#include <cmath>

#ifdef BUILDING_GPU
using vec = thrust::device_vector<double>;
#else
using vec = std::vector<double>;
#endif

struct updateWeightsOp {
    updateWeightsOp(
        double *wp,
        double *xp,
        double *cameraParams,
        double *imagePoints,
        double *worldPoints,
        unsigned int N,
        unsigned int F) :
        wp(wp),
        xp(xp),
        cameraParams(cameraParams),
        imagePoints(imagePoints),
        worldPoints(worldPoints),
        N(N), F(F) { /* no-op */ }

    double *wp;
    double *xp;
    double *cameraParams;
    double *imagePoints;
    double *worldPoints;
    unsigned int N;
    unsigned int F;

    __host__ __device__
    void operator()(const thrust::tuple<std::size_t, const double>& particle) {
        auto idx = thrust::get<0>(particle);
        auto data = thrust::get<1>(particle);
    }
};

extern "C" {

    /**
     * wp is Nx1
     * xp is Nx7
     * cameraParams is 3x3
     * imagePoints is Fx2
     * worldPoints is Fx3
     */
    DLL_PUBLIC DLL_PUBLIC void p2c(
        double *wp,
        double *xp,
        double *cameraParams,
        double *imagePoints,
        double *worldPoints,
        unsigned int N,
        unsigned int F) {

        vec d_wp(wp,wp + N);
        vec d_xp(xp, xp + N*7);
        vec d_cameraParams(cameraParams, cameraParams + 3*3);
        vec d_imagePoints(imagePoints, imagePoints + F * 2);
        vec d_worldPoints(worldPoints, worldPoints + F * 2);

        thrust::counting_iterator<std::size_t> first(0);
        thrust::counting_iterator<std::size_t> last = first + N;

        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(first, d_wp.begin())),
            thrust::make_zip_iterator(
                thrust::make_tuple(last, d_wp.end())),
            updateWeightsOp(
                thrust::raw_pointer_cast(d_wp.data()),
                thrust::raw_pointer_cast(d_xp.data()),
                thrust::raw_pointer_cast(d_cameraParams.data()),
                thrust::raw_pointer_cast(d_imagePoints.data()),
                thrust::raw_pointer_cast(d_worldPoints.data()),
                N, F
            )
        );

#ifdef BUILDING_GPU
        ::cudaMemcpy(wp, thrust::raw_pointer_cast(d_wp.data()), N * sizeof(double), cudaMemcpyDeviceToHost);
#else
        ::cudaMemcpy(wp, thrust::raw_pointer_cast(d_wp.data()), N * sizeof(double), cudaMemcpyHostToHost);
#endif
    }
}
