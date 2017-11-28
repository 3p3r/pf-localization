#include "pf.cuh"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/random.h>

#include <glm/gtc/type_ptr.hpp>

struct kern_randn {
    __host__ __device__
    double operator()(unsigned int n) const {
        thrust::default_random_engine rng;
        thrust::normal_distribution<double> dist;
        rng.discard(n);
        return dist(rng);
    }
};

struct kern_rotation_noise {
    double* rotation_noise;
    const double* randn;
    const double rotation_sigma;
    const unsigned N;

    __host__ __device__
    explicit kern_rotation_noise(double *rotation_noise, const double* randn, double rotation_sigma, unsigned N)
        : rotation_noise(rotation_noise), randn(randn), rotation_sigma(rotation_sigma), N(N) { /* no-op */ }

    __host__ __device__
    void operator()(unsigned int n) const {
        glm::dvec3 v(randn[n],
                     randn[N + n],
                     randn[2 * N + n]);
        // random rotation angle
        const auto r = rotation_sigma * randn[n];
        // random unit vector
        v = sin(r) * glm::normalize(v);

        rotation_noise[n] = cos(r);      // q.w
        rotation_noise[N + n] = v.x;     // q.x
        rotation_noise[2 * N + n] = v.y; // q.y
        rotation_noise[3 * N + n] = v.z; // q.z
    }
};

struct kern_position_noise {
    double* position_noise;
    const glm::dvec3 position_sigma;
    const unsigned N;

    __host__ __device__
    explicit kern_position_noise(double *position_noise, const glm::dvec3& position_sigma, unsigned N)
        : position_noise(position_noise), position_sigma(position_sigma), N(N) { /* no-op */ }

    __host__ __device__
    void operator()(unsigned int n) const {
        position_noise[n] *= position_sigma.x;
        position_noise[N + n] *= position_sigma.y;
        position_noise[2 * N + n] *= position_sigma.z;
    }
};

struct kern_create_particles {
    const glm::dvec3 initial_translation;
    const glm::dquat initial_rotation;
    double* states;
    double* weights;
    const unsigned N;

    __host__ __device__
    explicit kern_create_particles(const glm::dvec3& initial_translation, const glm::dquat& initial_rotation, double* states, double* weights, unsigned N)
        : initial_translation(initial_translation), initial_rotation(initial_rotation), states(states), weights(weights), N(N) { /* no-op */ }

    __host__ __device__
    void operator()(unsigned int n) const {
        weights[n] = 1. / N;
        states[n] = initial_translation.x;
        states[N + n] = initial_translation.y;
        states[2 * N + n] = initial_translation.z;
        states[3 * N + n] = initial_rotation.w;
        states[4 * N + n] = initial_rotation.x;
        states[5 * N + n] = initial_rotation.y;
        states[6 * N + n] = initial_rotation.z;
    }
};

ParticleFilter::ParticleFilter(const Options& opts) : m_options_(opts) {
    const auto N = opts.particle_count();
    const auto F = opts.features_count();

    m_position_noise_.resize(N * 3); // array of 3D position gaussian noises
    m_rotation_noise_.resize(N * 4); // array of 4D quaternion gaussian noises
    m_particles_.resize(N * 7); // array of 3D position and 4D quaternion states
    m_weights_.resize(N * 1); // array of states assciated with weights
    m_camera_intrinsics_.resize(3 * 3); // on-gpu pre-calibrated camera intrinscis
    m_camera_extrinsics_.resize(4 * 4); // on-gpu estimated camera extrinscis
    m_image_points_.resize(F * 2); // array of 2D features in the image
    m_world_points_.resize(F * 3); // 3D points in world corresponding to 2D features
    m_observation_points_.resize(m_image_points_.size()); // projected particle cloud

    // Generate and cache some guassian noise
    // For position noise, just populate all the vectors with random numbers.
    const thrust::counting_iterator<unsigned int> counter(0);
    thrust::transform(counter, counter + N,
                      m_position_noise_.begin() + 0 * N, kern_randn());
    thrust::transform(counter, counter + N,
                      m_position_noise_.begin() + 1 * N, kern_randn());
    thrust::transform(counter, counter + N,
                      m_position_noise_.begin() + 2 * N, kern_randn());

    // For quaternion noise, we first generate Nx3 random vectors and Nx1 random
    // angles. We then proceed to create unit quaternions based on the formula in
    // the following paper:
    // Kraft, E. (2003). "A quaternion-based unscented Kalman filter for orientation tracking."
    // NOTE: we reuse the same noises generated for position for performance reasons.
    thrust::for_each(counter, counter + N, kern_rotation_noise(
                         thrust::raw_pointer_cast(m_rotation_noise_.data()),
                         thrust::raw_pointer_cast(m_position_noise_.data()),
                         opts.rotation_standard_deviation(), N));
    thrust::for_each(counter, counter + N, kern_position_noise(
                         thrust::raw_pointer_cast(m_position_noise_.data()),
                         opts.position_standard_deviation(), N));

    // create the initial particles and weights
    thrust::for_each(counter, counter + m_particles_.size(),
                     kern_create_particles(
                         opts.initial_translation(),
                         opts.initial_rotation(),
                         thrust::raw_pointer_cast(m_particles_.data()),
                         thrust::raw_pointer_cast(m_weights_.data()), N));

    // copy camera intrinsics to gpu
    thrust::copy(glm::value_ptr(opts.camera_intrinsics()),
                 glm::value_ptr(opts.camera_intrinsics()) + m_camera_intrinsics_.size(),
                 m_camera_intrinsics_.begin());
}

void ParticleFilter::update(const std::vector<std::pair<glm::dvec2, glm::dvec3>>& points) {
    if (m_image_points_.size() < 2 * points.size()) {
        m_image_points_.resize(points.size() * 2);
        m_world_points_.resize(points.size() * 3);
        m_observation_points_.resize(m_image_points_.size());
    }

    // ...
}

void ParticleFilter::resample() {
    // ...
}

void ParticleFilter::extract() {
    // ...
}

glm::dvec3 ParticleFilter::translation() const {
    return m_extracted_position_;
}

glm::dmat3 ParticleFilter::rotation() const {
    return m_extracted_rotation_;
}

glm::dmat4 ParticleFilter::extrinsics() const {
    return m_extracted_extrinsics_;
}
