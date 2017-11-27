#include "p2c.h"

#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/matrix.hpp>
#include <glm/gtc/quaternion.hpp>

// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
class onlineCovariance {
  public:
    __host__ __device__
    void update(double x, double y) {
        n += 1;
        auto dx = x - meanx;
        meanx += dx / n;
        meany += (y - meany) / n;
        C += dx * (y - meany);
    }

    __host__ __device__
    double getPopulationCov() {
        return C / n;
    }

    __host__ __device__
    // Bessel's correction for sample variance
    double getSampleCov() {
        return C / (n - 1);
    }

  private:
    double meanx = 0;
    double meany = 0;
    double C = 0;
    double n = 0;
};

struct updateWeightsOp {
    updateWeightsOp(const System& input, double* observationPoints) :
        input(input), observationPoints(observationPoints) { /* no-op */ }

    System input;
    double *observationPoints;

    __host__ __device__
    void operator()(std::size_t i) {
        using namespace glm;
        dvec3 t(input.xp[i], input.xp[input.N+i], input.xp[2*input.N+i]);
        dmat3x3 R = mat3_cast(dquat(input.xp[3 * input.N + i], input.xp[4 * input.N + i], input.xp[5 * input.N + i], input.xp[6 * input.N + i]));
        cameraPoseToExtrinsics(t, R);
        dmat4x3 camMatrix = cameraMatrix(R, t);

        onlineCovariance cov_xx;
        onlineCovariance cov_yy;
        onlineCovariance cov_xy;
        for (unsigned j = 0; j < input.F; ++j) {
            dvec3 projection{
                camMatrix * dvec4(input.worldPoints[j], input.worldPoints[input.F + j], input.worldPoints[2 * input.F + j], 1)
            };
            auto x = projection.x / projection.z;
            auto y = projection.y / projection.z;
            cov_xx.update(x, x);
            cov_yy.update(y, y);
            cov_xy.update(x, y);
            observationPoints[2 * i*input.F + 2 * j] = x;
            observationPoints[2 * i*input.F + 2 * j+1] = y;
        }

        dmat2x2 C(cov_xx.getSampleCov(), cov_xy.getSampleCov(),
                  cov_xy.getSampleCov(), cov_yy.getSampleCov());
        dmat2x2 invC = glm::inverse(C);

        double D = 0;
        for (unsigned j = 0; j < input.F; ++j) {
            glm::dvec2 d {
                observationPoints[2 * i*input.F + 2 * j] - input.imagePoints[j],
                observationPoints[2 * i*input.F + 2 * j + 1] - input.imagePoints[input.F + j]
            };
            D += glm::dot(d, invC*d);
        }

        input.wp[i] = (1 / (2 * glm::pi<double>()*sqrt(glm::determinant(C))))*exp(-D / 2);
    }

    __host__ __device__
    // https://www.mathworks.com/help/vision/ref/cameraposetoextrinsics.html
    void cameraPoseToExtrinsics(glm::dvec3& t, glm::dmat3x3& R) {
        R = glm::transpose(R);
        t = -t * R;
    }

    __host__ __device__
    // https://www.mathworks.com/help/vision/ref/cameramatrix.html
    glm::dmat4x3 cameraMatrix(const glm::dmat3x3& R, const glm::dvec3& t) {
        glm::dmat3x3 K(input.cameraParams[0], input.cameraParams[3], input.cameraParams[6],
                       input.cameraParams[1], input.cameraParams[4], input.cameraParams[7],
                       input.cameraParams[2], input.cameraParams[5], input.cameraParams[8]);

        glm::dmat4x3 Rt(R[0].x, R[1].x, R[2].x,
                        R[0].y, R[1].y, R[2].y,
                        R[0].z, R[1].z, R[2].z,
                        t.x, t.y, t.z);

        return K * Rt;
    }
};

extern "C" {

    DLL_PUBLIC void updateWeights_cpu(System input) {
        static std::vector<double> observation_points;
        const auto observation_size = input.N * input.F * 2;

        if (observation_points.size() != observation_size)
            observation_points.resize(observation_size);

        const thrust::counting_iterator<std::size_t> first(0);
        const auto last = first + input.N;
        thrust::for_each(
            thrust::host, first, last,
            updateWeightsOp(input, observation_points.data()));
    }

    DLL_PUBLIC void updateWeights_gpu(System input) {
        using vec = thrust::device_vector<double>;

        static vec d_observation_points;
        const auto observation_size = input.N * input.F * 2;

        if (d_observation_points.size() != observation_size)
            d_observation_points.resize(observation_size);

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
        thrust::for_each(
            thrust::device, first, last,
            updateWeightsOp(d_system, raw_pointer_cast(d_observation_points.data())));

        // transfer GPU data >> to Matlab (host)
        ::cudaMemcpy(input.wp, thrust::raw_pointer_cast(d_wp.data()),
                     input.N * sizeof(double), cudaMemcpyDeviceToHost);
    }

}
