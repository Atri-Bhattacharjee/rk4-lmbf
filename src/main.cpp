#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include "datatypes.h"
#include "models.h"
#include "adaptive_birth_model.h"
#include "smc_lmb_tracker.h"
#include "in_orbit_sensor_model.h"
#include "assignment.h"
#include "metrics.h"

#include "two_body_propagator.h"
#include "validation.h"

namespace {

Eigen::VectorXd cartesian_to_measurement_validated(const Eigen::VectorXd& cartesian_state,
                                                   const Eigen::VectorXd& sensor_state) {
    validation::require_state_vector(cartesian_state, "cartesian_state");
    validation::require_state_vector(sensor_state, "sensor_state");
    return Measurement::cartesianToMeasurement(cartesian_state, sensor_state);
}

Eigen::VectorXd measurement_to_cartesian_validated(const Eigen::VectorXd& measurement,
                                                   const Eigen::VectorXd& sensor_state) {
    if (measurement.size() != validation::MEAS_DIM) {
        throw std::invalid_argument(
            "measurement: expected size " + std::to_string(validation::MEAS_DIM) +
            ", got " + std::to_string(measurement.size()));
    }
    validation::require_state_vector(sensor_state, "sensor_state");
    return Measurement::measurementToCartesian(measurement, sensor_state);
}

std::shared_ptr<AdaptiveBirthModel> make_adaptive_birth_model(int particles_per_track,
                                                              double initial_existence_probability,
                                                              const Eigen::MatrixXd& initial_covariance) {
    validation::require_particles_per_track(particles_per_track);
    validation::require_positive_definite(initial_covariance, "initial_covariance");
    return std::make_shared<AdaptiveBirthModel>(
        particles_per_track, initial_existence_probability, initial_covariance);
}

std::shared_ptr<TwoBodyPropagator> make_two_body_propagator(const Eigen::MatrixXd& process_noise_covariance) {
    validation::require_covariance_6x6(process_noise_covariance, "process_noise_covariance");
    return std::make_shared<TwoBodyPropagator>(process_noise_covariance);
}

Particle propagate_validated(const TwoBodyPropagator& propagator,
                             const Particle& particle,
                             double dt,
                             double current_time,
                             double noise_scale) {
    validation::require_state_vector(particle.state_vector, "particle.state_vector");
    return propagator.propagate(particle, dt, current_time, noise_scale);
}

double calculate_likelihood_validated(const InOrbitSensorModel& sensor_model,
                                      const Particle& particle,
                                      const Measurement& measurement) {
    validation::require_state_vector(particle.state_vector, "particle.state_vector");
    validation::require_measurement(measurement);
    return sensor_model.calculate_likelihood(particle, measurement);
}

Eigen::VectorXd convert_particle_to_measurement_validated(const InOrbitSensorModel& sensor_model,
                                                          const Particle& particle,
                                                          const Eigen::VectorXd& sensor_state) {
    validation::require_state_vector(particle.state_vector, "particle.state_vector");
    validation::require_state_vector(sensor_state, "sensor_state");
    return sensor_model.convertParticleToMeasurement(particle, sensor_state);
}

std::shared_ptr<SMC_LMB_Tracker> make_smc_lmb_tracker(
    std::shared_ptr<IOrbitPropagator> propagator,
    std::shared_ptr<ISensorModel> sensor_model,
    std::shared_ptr<IBirthModel> birth_model,
    double survival_probability,
    int k_best,
    double prune_threshold,
    double clutter_intensity,
    double p_detection,
    double noise_decay_rate,
    double noise_min_scale) {
    return std::make_shared<SMC_LMB_Tracker>(
        std::move(propagator),
        std::move(sensor_model),
        std::move(birth_model),
        survival_probability,
        k_best,
        prune_threshold,
        clutter_intensity,
        p_detection,
        noise_decay_rate,
        noise_min_scale);
}

std::vector<Track> get_tracks_copy(const SMC_LMB_Tracker& tracker) {
    return tracker.get_tracks();
}

TrackLabel get_track_label_copy(const Track& track) {
    return track.label();
}

std::vector<Particle> get_track_particles_copy(const Track& track) {
    return track.particles();
}

std::vector<Track> get_filter_state_tracks_copy(const FilterState& state) {
    return state.tracks();
}

}  // namespace

PYBIND11_MODULE(lmb_engine, m) {
    m.doc() = "High-performance C++ engine for SMC-LMB space debris tracking";
    
    // Bind core data structures
    pybind11::class_<TrackLabel>(m, "TrackLabel")
        .def(pybind11::init<>())
        .def_readwrite("birth_time", &TrackLabel::birth_time)
        .def_readwrite("index", &TrackLabel::index);
    
    pybind11::class_<Particle>(m, "Particle")
        .def(pybind11::init<>())
        .def_readwrite("state_vector", &Particle::state_vector)
        .def_readwrite("weight", &Particle::weight);
    
    pybind11::class_<Measurement>(m, "Measurement")
        .def(pybind11::init<>())
        .def_readwrite("timestamp_", &Measurement::timestamp_)
        .def_readwrite("value_", &Measurement::value_)
        .def_readwrite("covariance_", &Measurement::covariance_)
        .def_readwrite("sensor_id_", &Measurement::sensor_id_)
        .def_readwrite("sensor_state_", &Measurement::sensor_state_)
        .def_static("cartesianToMeasurement", &cartesian_to_measurement_validated, 
                   pybind11::arg("cartesian_state"), 
                   pybind11::arg("sensor_state"),
                   "Convert Cartesian state to measurement space (range, range rate, azimuth, elevation)")
        .def_static("measurementToCartesian", &measurement_to_cartesian_validated,
                   pybind11::arg("measurement"),
                   pybind11::arg("sensor_state"),
                   "Convert measurement space to Cartesian state");
    
    pybind11::class_<Track>(m, "Track")
        .def(pybind11::init<>())
        .def(pybind11::init<const TrackLabel&, double, const std::vector<Particle>&>())
        .def("label", &get_track_label_copy)
        .def("existence_probability", &Track::existence_probability)
        .def("particles", &get_track_particles_copy)
        .def("set_existence_probability", &Track::set_existence_probability)
        .def("set_particles", &Track::set_particles);
    
    pybind11::class_<FilterState>(m, "FilterState")
        .def(pybind11::init<>())
        .def(pybind11::init<double, const std::vector<Track>&>())
        .def("timestamp", &FilterState::timestamp)
        .def("tracks", &get_filter_state_tracks_copy)
        .def("set_timestamp", &FilterState::set_timestamp)
        .def("set_tracks", &FilterState::set_tracks);
    
    // Bind abstract base interfaces
    pybind11::class_<IOrbitPropagator, std::shared_ptr<IOrbitPropagator>>(m, "IOrbitPropagator");
    pybind11::class_<ISensorModel, std::shared_ptr<ISensorModel>>(m, "ISensorModel");
    pybind11::class_<IBirthModel, std::shared_ptr<IBirthModel>>(m, "IBirthModel");
    
    // Bind concrete model implementations    
    pybind11::class_<AdaptiveBirthModel, IBirthModel, std::shared_ptr<AdaptiveBirthModel>>(m, "AdaptiveBirthModel")
        .def(pybind11::init(&make_adaptive_birth_model),
             pybind11::arg("particles_per_track"),
             pybind11::arg("initial_existence_probability"),
             pybind11::arg("initial_covariance"),
             "Create adaptive birth model with physics-based tangent fan velocity sampling.\n\n"
             "For each unassociated measurement, generates particles with:\n"
             "  - Position computed from range/azimuth/elevation\n"
             "  - Radial velocity fixed from range-rate measurement\n"
             "  - Tangent velocity sampled uniformly in direction, with magnitude\n"
             "    computed from circular orbital velocity at target altitude\n"
             "  - Gaussian noise from initial_covariance added to all components\n\n"
             "The covariance matrix should have velocity components sized to capture\n"
             "expected eccentricity variation (e.g., 500 m/s std for moderate eccentricity).")
        .def("generate_new_tracks", &AdaptiveBirthModel::generate_new_tracks);

    pybind11::class_<TwoBodyPropagator, IOrbitPropagator, std::shared_ptr<TwoBodyPropagator>>(m, "TwoBodyPropagator")
        .def(pybind11::init(&make_two_body_propagator), pybind11::arg("process_noise_covariance"))
        .def("propagate", &propagate_validated,
             pybind11::arg("particle"),
             pybind11::arg("dt"),
             pybind11::arg("current_time"),
             pybind11::arg("noise_scale") = 1.0);
    
    pybind11::class_<InOrbitSensorModel, ISensorModel, std::shared_ptr<InOrbitSensorModel>>(m, "InOrbitSensorModel")
        .def(pybind11::init<>(), "Default constructor for InOrbitSensorModel with default noise parameters")
        .def(pybind11::init<double, double, double, double>(), 
             pybind11::arg("range_var"), 
             pybind11::arg("range_rate_var"), 
             pybind11::arg("azimuth_var"), 
             pybind11::arg("elevation_var"),
             "Constructor for InOrbitSensorModel with specified noise parameters")
        .def("calculate_likelihood", &calculate_likelihood_validated)
        .def("convertParticleToMeasurement", &convert_particle_to_measurement_validated);
    
    // Bind the main tracker class with direct constructor support
    pybind11::class_<SMC_LMB_Tracker, std::shared_ptr<SMC_LMB_Tracker>>(m, "SMC_LMB_Tracker")
        .def(pybind11::init(&make_smc_lmb_tracker),
             pybind11::arg("propagator"),
             pybind11::arg("sensor_model"),
             pybind11::arg("birth_model"),
             pybind11::arg("survival_probability"),
             pybind11::arg("k_best") = 100,
             pybind11::arg("prune_threshold") = 0.01,
             pybind11::arg("clutter_intensity") = 1.0e-6,
             pybind11::arg("p_detection") = 0.99,
             pybind11::arg("noise_decay_rate") = 0.0,
             pybind11::arg("noise_min_scale") = 1.0,
             "Constructor for SMC_LMB_Tracker with model dependencies")
        .def("predict", &SMC_LMB_Tracker::predict, "Runs the predict step for a given time delta")
        .def("update", &SMC_LMB_Tracker::update, "Runs the update step with measurements")
        .def("get_tracks", &get_tracks_copy, "Gets the current list of tracks")
        .def("set_tracks", &SMC_LMB_Tracker::set_tracks, "Sets the initial list of tracks for the filter")
        .def("compute_association_likelihood", &SMC_LMB_Tracker::compute_association_likelihood, "Compute the association likelihood for a track and measurement");
    
    // Bind the Hypothesis struct
    pybind11::class_<Hypothesis>(m, "Hypothesis")
        .def(pybind11::init<>())
        .def_readwrite("associations", &Hypothesis::associations)
        .def_readwrite("weight", &Hypothesis::weight);

    // Bind the solve_assignment function
    m.def("solve_assignment", &solve_assignment, pybind11::arg("cost_matrix"), pybind11::arg("k_best"),
          "Solves the assignment problem and returns K-best hypotheses.");

    // Bind the calculate_ospa_distance function
    m.def("calculate_ospa_distance", &calculate_ospa_distance,
          "Calculates the OSPA distance between estimated tracks and ground truths",
          pybind11::arg("tracks"),
          pybind11::arg("ground_truths"),
          pybind11::arg("cutoff"));
}