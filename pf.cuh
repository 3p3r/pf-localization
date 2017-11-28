#pragma once

#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif//GLM_FORCE_CUDA

#include <vector> // std::pair
#include <utility> // std::vector
#include <glm/glm.hpp> // glm::stuff
#include <glm/gtc/quaternion.hpp> // glm::dquat
#include <thrust/device_vector.h> // thrust::stuff

/**
 * @class ParticleFilter
 * @brief A particle filter that estimates camera extrinsics based on a set of
 * 2D observations and their corresponding 3D world points.
 * @note resampling method is systematic.
 * @note estimation method is state mean.
 * @note You may have seen other implementations separate the correction and
 * prediction steps (e.g. Matlab's robotics.ParticleFilter). In this library
 * both correction and prediction happen inside one single function call; to
 * increase the overall perforamce and decrease CUDA's kernel call overhead.
 */
class ParticleFilter {
  public:
    class Options;
    // Constructs the filter with the given options (GPU memory allocations).
    explicit ParticleFilter(const Options& opts);

    /// Runs one pass of the filter based on the process model (random walk) and
    /// subsequently adjusts particles based on measurement model (homography).
    /// @param points pair of 2D feature points and their corresponding 3D point
    void update(const std::vector<std::pair<glm::dvec2, glm::dvec3>>& points);

    // Resamples weights based on the reampling method (e.g. systematic)
    void resample();

    // Extracts state estimates based on the estimation method (e.g. mean)
    void extract();

    // Returns estimated camera translation (valid after call to extract())
    glm::dvec3 translation() const;

    // Returns estimated camera rotation (valid after call to extract())
    glm::dmat3 rotation() const;

    // Returns estimated camera extrinsics (valid after call to extract())
    glm::dmat4 extrinsics() const;

    /**
    * @class ParticleFilter::Options
    * @brief Construct options to pass to Particle Filter
    */
    class Options {
      public:
        glm::dvec3 initial_translation() const {
            return m_initial_translation_;
        }

        Options& set_initial_translation(const glm::dvec3& highp_dvec3) {
            m_initial_translation_ = highp_dvec3;
            return *this;
        }

        glm::dquat initial_rotation() const {
            return m_initial_rotation_;
        }

        Options& set_initial_rotation(const glm::dquat& highp_dquat) {
            m_initial_rotation_ = highp_dquat;
            return *this;
        }

        glm::dmat3 camera_intrinsics() const {
            return m_camera_intrinsics_;
        }

        Options& set_camera_intrinsics(const glm::dmat3& highp_dmat3) {
            m_camera_intrinsics_ = highp_dmat3;
            return *this;
        }

        glm::dvec3 position_standard_deviation() const {
            return m_position_standard_deviation_;
        }

        Options& set_position_standard_deviation(const glm::dvec3& highp_dvec3) {
            m_position_standard_deviation_ = highp_dvec3;
            return *this;
        }

        double rotation_standard_deviation() const {
            return m_rotation_standard_deviation_;
        }

        Options& set_rotation_standard_deviation(const double rotation_standard_deviation) {
            m_rotation_standard_deviation_ = rotation_standard_deviation;
            return *this;
        }

        unsigned particle_count() const {
            return m_particle_count_;
        }

        Options& set_particle_count(const unsigned particle_count) {
            m_particle_count_ = particle_count;
            return *this;
        }

        unsigned features_count() const {
            return m_features_count_;
        }

        Options& set_features_count(const unsigned features_count) {
            m_features_count_ = features_count;
            return *this;
        }

      private:
        glm::dvec3 m_initial_translation_;
        glm::dquat m_initial_rotation_;
        glm::dmat3 m_camera_intrinsics_;
        glm::dvec3 m_position_standard_deviation_ = glm::dvec3(100.);
        double m_rotation_standard_deviation_ = 0.01;
        unsigned m_particle_count_ = 100000;
        unsigned m_features_count_ = 28;
    };

  private:
    const Options m_options_;
    thrust::device_vector<double> m_position_noise_;
    thrust::device_vector<double> m_rotation_noise_;
    thrust::device_vector<double> m_particles_;
    thrust::device_vector<double> m_weights_;
    thrust::device_vector<double> m_camera_intrinsics_;
    thrust::device_vector<double> m_camera_extrinsics_;
    thrust::device_vector<double> m_image_points_;
    thrust::device_vector<double> m_world_points_;
    thrust::device_vector<double> m_observation_points_;
    glm::dmat4 m_extracted_extrinsics_;
    glm::dmat3 m_extracted_rotation_;
    glm::dvec3 m_extracted_position_;
};
