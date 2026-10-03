#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include "datatypes.h"
#include "models.h"
#include "adaptive_birth_model.h"
#include "smc_lmb_tracker.h"
#include "fast_random.h"
#include "in_orbit_sensor_model.h"
#include "sensor_fov.h"
#include "assignment.h"
#include "metrics.h"
#include "particle_statistics.h"

#include "two_body_propagator.h"
#include "validation.h"
#include "los_geometry.h"

namespace {

StateVector require_state_vector_fixed(const Eigen::VectorXd& vector, const char* context) {
    validation::require_state_vector(vector, context);
    return StateVector(vector);
}

Eigen::Vector3d require_vector3_fixed(const Eigen::VectorXd& vector, const char* context) {
    if (vector.size() != 3) {
        throw std::invalid_argument(std::string(context) + ": expected 3 elements, got " +
                                    std::to_string(vector.size()));
    }
    return Eigen::Vector3d(vector);
}

//! A position, taken either as a 3-vector or as the first half of a 6-D state. Visibility depends
//! only on position, so both spellings are accepted wherever a target is named.
Eigen::Vector3d require_position_fixed(const Eigen::VectorXd& vector, const char* context) {
    if (vector.size() != 3 && vector.size() != 6) {
        throw std::invalid_argument(std::string(context) + ": expected 3 or 6 elements, got " +
                                    std::to_string(vector.size()));
    }
    if (!vector.allFinite()) {
        throw std::invalid_argument(std::string(context) + ": must be finite");
    }
    return Eigen::Vector3d(vector.head(3));
}

sensor::SensorFovConfig make_sensor_fov_config(double min_range, double max_range,
                                               double half_width, double half_height) {
    sensor::SensorFovConfig fov;
    fov.min_range = min_range;
    fov.max_range = max_range;
    fov.half_width = half_width;
    fov.half_height = half_height;
    sensor::require_fov_config(fov);
    return fov;
}

//! Rows of `values` assigned one per sensor; the row count must match the array exactly so a
//! mis-shaped per-step update fails loudly instead of silently re-pointing a subset.
void require_row_count(const Eigen::MatrixXd& values, size_t expected_rows, int expected_cols,
                       const char* context) {
    if (values.rows() != static_cast<Eigen::Index>(expected_rows) || values.cols() != expected_cols) {
        throw std::invalid_argument(std::string(context) + ": expected a (" +
                                    std::to_string(expected_rows) + ", " + std::to_string(expected_cols) +
                                    ") array, got (" + std::to_string(values.rows()) + ", " +
                                    std::to_string(values.cols()) + ")");
    }
}

pybind11::array_t<double> to_numpy(const std::vector<double>& values) {
    pybind11::array_t<double> out(static_cast<pybind11::ssize_t>(values.size()));
    std::copy(values.begin(), values.end(), out.mutable_data());
    return out;
}

Eigen::Vector3d require_vector3(const Eigen::VectorXd& vector, const char* context) {
    if (vector.size() != 3) {
        throw std::invalid_argument(
            std::string(context) + ": expected a 3-vector, got size " + std::to_string(vector.size()));
    }
    return Eigen::Vector3d(vector);
}

LocalMeasVector require_local_vector(const Eigen::VectorXd& vector, const char* context) {
    if (vector.size() != validation::MEAS_DIM) {
        throw std::invalid_argument(
            std::string(context) + ": expected a local tangent-frame 6-vector "
            "[d_range, d_range_rate, d_theta1, d_theta2, d_omega1, d_omega2], got size " +
            std::to_string(vector.size()));
    }
    return LocalMeasVector(vector);
}

MeasCovariance require_meas_covariance(const Eigen::MatrixXd& covariance, const char* context) {
    validation::require_covariance_6x6(covariance, context);
    return MeasCovariance(covariance);
}

Measurement measurement_from_cartesian_validated(const Eigen::VectorXd& target_state,
                                                 const Eigen::VectorXd& sensor_state) {
    return Measurement::fromCartesian(require_state_vector_fixed(target_state, "target_state"),
                                      require_state_vector_fixed(sensor_state, "sensor_state"));
}

Measurement measurement_from_angles_validated(double range, double range_rate,
                                              double azimuth, double elevation,
                                              double azimuth_rate, double elevation_rate,
                                              const Eigen::VectorXd& sensor_state) {
    return Measurement::fromAnglesAndRates(range, range_rate, azimuth, elevation, azimuth_rate, elevation_rate,
                                           require_state_vector_fixed(sensor_state, "sensor_state"));
}

Measurement measurement_perturbed_validated(const Measurement& measurement, const Eigen::VectorXd& eps) {
    return measurement.perturbed(require_local_vector(eps, "eps"));
}

los::LosObservation observe_validated(const Eigen::VectorXd& target_state, const Eigen::VectorXd& sensor_state) {
    return los::observe(require_state_vector_fixed(target_state, "target_state"),
                        require_state_vector_fixed(sensor_state, "sensor_state"));
}

StateVector to_cartesian_validated(const los::LosObservation& observation, const Eigen::VectorXd& sensor_state) {
    return los::toCartesian(observation, require_state_vector_fixed(sensor_state, "sensor_state"));
}

los::LosObservation perturbed_validated(const los::LosObservation& observation, const Eigen::VectorXd& eps) {
    return los::perturbed(observation, require_local_vector(eps, "eps"));
}

std::shared_ptr<AdaptiveBirthModel> make_adaptive_birth_model(int particles_per_track,
                                                              double initial_existence_probability,
                                                              const Eigen::MatrixXd& birth_covariance_local,
                                                              std::optional<uint64_t> seed) {
    validation::require_particles_per_track(particles_per_track);
    validation::require_covariance_6x6_positive_definite(birth_covariance_local,
                                                         "birth_covariance (local tangent frame)");
    return std::make_shared<AdaptiveBirthModel>(
        particles_per_track, initial_existence_probability, birth_covariance_local, seed);
}

std::vector<Track> generate_new_tracks_validated(const AdaptiveBirthModel& birth_model,
                                                 const std::vector<Measurement>& unused_measurements,
                                                 double current_time) {
    for (const Measurement& measurement : unused_measurements) {
        validation::require_measurement(measurement);
    }
    return birth_model.generate_new_tracks(unused_measurements, current_time);
}

std::shared_ptr<TwoBodyPropagator> make_two_body_propagator(const Eigen::MatrixXd& process_noise_covariance,
                                                            std::optional<uint64_t> seed,
                                                            std::optional<double> noise_reference_dt) {
    validation::require_covariance_6x6(process_noise_covariance, "process_noise_covariance");
    return std::make_shared<TwoBodyPropagator>(process_noise_covariance, seed, noise_reference_dt);
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

los::LosObservation predict_observation_validated(const InOrbitSensorModel& sensor_model,
                                                  const Particle& particle,
                                                  const Eigen::VectorXd& sensor_state) {
    validation::require_state_vector(particle.state_vector, "particle.state_vector");
    return sensor_model.predictObservation(particle, require_state_vector_fixed(sensor_state, "sensor_state"));
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
    double noise_min_scale,
    std::optional<uint64_t> seed) {
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
        noise_min_scale,
        seed);
}

std::vector<Track> get_tracks_copy(const SMC_LMB_Tracker& tracker) {
    return tracker.get_tracks();
}

//! Per-track scalars (and optionally means) as NumPy arrays, without copying any particle cloud.
pybind11::dict track_summary(const SMC_LMB_Tracker& tracker, bool with_means, bool with_covariances) {
    const std::vector<Track>& tracks = tracker.get_tracks();
    const auto count = static_cast<pybind11::ssize_t>(tracks.size());
    pybind11::array_t<uint64_t> birth_time(count);
    pybind11::array_t<int64_t> index(count);
    pybind11::array_t<double> existence(count);
    pybind11::array_t<double> propagated_time(count);
    pybind11::array_t<int64_t> particle_count(count);
    auto birth_view = birth_time.mutable_unchecked<1>();
    auto index_view = index.mutable_unchecked<1>();
    auto existence_view = existence.mutable_unchecked<1>();
    auto time_view = propagated_time.mutable_unchecked<1>();
    auto count_view = particle_count.mutable_unchecked<1>();
    for (pybind11::ssize_t i = 0; i < count; ++i) {
        const Track& track = tracks[static_cast<size_t>(i)];
        birth_view(i) = track.label().birth_time;
        index_view(i) = static_cast<int64_t>(track.label().index);
        existence_view(i) = track.existence_probability();
        time_view(i) = track.propagated_time();
        count_view(i) = static_cast<int64_t>(track.particles().size());
    }
    pybind11::dict out;
    out["birth_time"] = birth_time;
    out["index"] = index;
    out["existence"] = existence;
    out["propagated_time"] = propagated_time;
    out["particle_count"] = particle_count;
    if (with_means) {
        pybind11::array_t<double> means(std::vector<pybind11::ssize_t>{count, 6});
        auto mean_view = means.mutable_unchecked<2>();
        for (pybind11::ssize_t i = 0; i < count; ++i) {
            const StateVector mean = particle_stats::mean_state(tracks[static_cast<size_t>(i)]);
            for (int k = 0; k < 6; ++k) {
                mean_view(i, k) = mean(k);
            }
        }
        out["mean"] = means;
    }
    if (with_covariances) {
        pybind11::array_t<double> covariances(std::vector<pybind11::ssize_t>{count, 6, 6});
        auto cov_view = covariances.mutable_unchecked<3>();
        for (pybind11::ssize_t i = 0; i < count; ++i) {
            const StateCovariance cov = particle_stats::covariance(tracks[static_cast<size_t>(i)]);
            for (int r = 0; r < 6; ++r) {
                for (int c = 0; c < 6; ++c) {
                    cov_view(i, r, c) = cov(r, c);
                }
            }
        }
        out["covariance"] = covariances;
    }
    return out;
}

//! `count` particle states from each requested track, taken evenly through its weights (the k-th
//! is the particle at cumulative weight (k + 0.5) / count), without copying any whole cloud.
//! Deterministic: no random number is drawn, so reading a cloud cannot perturb the filter.
pybind11::dict sample_particles(const SMC_LMB_Tracker& tracker, const std::vector<int64_t>& track_indices,
                                int count) {
    if (count <= 0) {
        throw std::invalid_argument("sample_particles: count must be positive, got " + std::to_string(count));
    }
    const std::vector<Track>& tracks = tracker.get_tracks();
    const auto rows = static_cast<pybind11::ssize_t>(track_indices.size());
    pybind11::array_t<double> states(std::vector<pybind11::ssize_t>{rows, count, 6});
    pybind11::array_t<double> propagated_time(rows);
    auto states_view = states.mutable_unchecked<3>();
    auto time_view = propagated_time.mutable_unchecked<1>();
    for (pybind11::ssize_t r = 0; r < rows; ++r) {
        const int64_t index = track_indices[static_cast<size_t>(r)];
        if (index < 0 || static_cast<size_t>(index) >= tracks.size()) {
            throw std::out_of_range("sample_particles: track index " + std::to_string(index) +
                                    " out of range for " + std::to_string(tracks.size()) + " track(s)");
        }
        const Track& track = tracks[static_cast<size_t>(index)];
        const std::vector<Particle>& particles = track.particles();
        if (particles.empty()) {
            throw std::invalid_argument("sample_particles: track " + std::to_string(index) +
                                        " has no particles");
        }
        time_view(r) = track.propagated_time();
        double total = 0.0;
        for (const Particle& particle : particles) {
            total += particle.weight;
        }
        const size_t num_particles = particles.size();
        size_t p = 0;
        double cumulative = particles[0].weight;
        for (int k = 0; k < count; ++k) {
            if (total > 0.0) {
                const double target = (static_cast<double>(k) + 0.5) * total / static_cast<double>(count);
                while (cumulative < target && p + 1 < num_particles) {
                    ++p;
                    cumulative += particles[p].weight;
                }
            } else {
                // No usable weights: spread evenly over the cloud by index.
                p = (static_cast<size_t>(k) * num_particles) / static_cast<size_t>(count);
            }
            for (int c = 0; c < 6; ++c) {
                states_view(r, k, c) = particles[p].state_vector(c);
            }
        }
    }
    pybind11::dict out;
    out["states"] = states;
    out["propagated_time"] = propagated_time;
    return out;
}

//! GOSPA of a subset of the tracker's own tracks against truth rows, without copying any cloud.
GospaComponents tracker_gospa_components(const SMC_LMB_Tracker& tracker,
                                         const std::vector<int64_t>& track_indices,
                                         const Eigen::MatrixXd& truths, double cutoff) {
    const std::vector<Track>& tracks = tracker.get_tracks();
    if (truths.rows() > 0 && truths.cols() != 6) {
        throw std::invalid_argument("tracker_gospa_components: truths must be (N, 6)");
    }
    std::vector<StateVector> means;
    means.reserve(track_indices.size());
    for (int64_t index : track_indices) {
        if (index < 0 || static_cast<size_t>(index) >= tracks.size()) {
            throw std::out_of_range("tracker_gospa_components: track index " + std::to_string(index) +
                                    " out of range for " + std::to_string(tracks.size()) + " track(s)");
        }
        means.push_back(particle_stats::weighted_mean(tracks[static_cast<size_t>(index)]));
    }
    std::vector<Eigen::VectorXd> truth_list;
    truth_list.reserve(static_cast<size_t>(truths.rows()));
    for (Eigen::Index r = 0; r < truths.rows(); ++r) {
        truth_list.push_back(truths.row(r).transpose());
    }
    return calculate_gospa_components_from_means(means, truth_list, cutoff);
}

TrackLabel get_track_label_copy(const Track& track) {
    return track.label();
}

std::vector<Particle> get_track_particles_copy(const Track& track) {
    return track.particles();
}

// A std::vector<Particle> is a contiguous array, so a NumPy array can address the state components
// and the weights in place with a stride of sizeof(Particle). That replaces a per-particle pybind11
// object conversion (measured at ~4.4 ms for three 10,000-particle clouds) with a pointer.
//
// The views are read-only and keep the owning Track alive through the array's base object. They are
// still invalidated by anything that reallocates the cloud -- set_particles (copy or move),
// mutable_particles followed by a reallocation, or a further update that replaces the cloud on the
// tracker the track came from -- exactly like a C++ iterator would be. predict currently overwrites
// particles in place without reallocating, so addresses stay valid but the values change.
static_assert(sizeof(Particle) == 64,
              "the zero-copy particle views assume a 64-byte Particle; re-check the strides below");
static_assert(sizeof(StateVector) == 6 * sizeof(double),
              "the zero-copy particle views assume six contiguous doubles per state vector");

pybind11::array_t<double> make_readonly_view(const std::vector<pybind11::ssize_t>& shape,
                                             const std::vector<pybind11::ssize_t>& strides,
                                             const double* data,
                                             pybind11::object owner) {
    pybind11::array_t<double> view(shape, strides, data, std::move(owner));
    pybind11::detail::array_proxy(view.ptr())->flags &=
        ~pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_;
    return view;
}

pybind11::array_t<double> get_track_particle_states(pybind11::object self) {
    const std::vector<Particle>& particles = self.cast<const Track&>().particles();
    const auto count = static_cast<pybind11::ssize_t>(particles.size());
    if (count == 0) {
        return pybind11::array_t<double>(std::vector<pybind11::ssize_t>{0, 6});
    }
    // Taken from a live object rather than offsetof, which is only conditionally supported for a
    // type with an Eigen member.
    return make_readonly_view({count, 6},
                              {static_cast<pybind11::ssize_t>(sizeof(Particle)),
                               static_cast<pybind11::ssize_t>(sizeof(double))},
                              particles[0].state_vector.data(),
                              std::move(self));
}

pybind11::array_t<double> get_track_particle_weights(pybind11::object self) {
    const std::vector<Particle>& particles = self.cast<const Track&>().particles();
    const auto count = static_cast<pybind11::ssize_t>(particles.size());
    if (count == 0) {
        return pybind11::array_t<double>(std::vector<pybind11::ssize_t>{0});
    }
    return make_readonly_view({count},
                              {static_cast<pybind11::ssize_t>(sizeof(Particle))},
                              &particles[0].weight,
                              std::move(self));
}

std::vector<Track> get_filter_state_tracks_copy(const FilterState& state) {
    return state.tracks();
}

}  // namespace

PYBIND11_MODULE(lmb_engine, m) {
    m.doc() = "High-performance C++ engine for SMC-LMB space debris tracking";
    m.attr("VALIDATION_ENABLED") = static_cast<bool>(LMB_VALIDATION_ENABLED);
    m.attr("MEAS_DIM") = validation::MEAS_DIM;

    // Line-of-sight geometry primitives (see los_geometry.h for the frame conventions)
    pybind11::class_<los::LosObservation>(m, "LosObservation",
        "Geometric part of a measurement: range [m], range_rate [m/s], unit line-of-sight direction los (ECI)\n"
        "and its angular rate los_rate [rad/s] (orthogonal to los). Exact inverse: r = r_s + range*los,\n"
        "v = v_s + range_rate*los + range*los_rate.")
        .def(pybind11::init<>())
        .def_readwrite("range", &los::LosObservation::range)
        .def_readwrite("range_rate", &los::LosObservation::range_rate)
        .def_readwrite("los", &los::LosObservation::los)
        .def_readwrite("los_rate", &los::LosObservation::los_rate);
    m.def("tangent_basis", &los::tangentBasis, pybind11::arg("u"),
          "3x2 right-handed orthonormal basis [e1 e2] of the tangent plane at unit vector u");
    m.def("exp_map", &los::expMap, pybind11::arg("u"), pybind11::arg("tangent"),
          "Sphere exponential map from unit u along a tangent vector (|tangent| radians)");
    m.def("log_map", &los::logMap, pybind11::arg("from_dir"), pybind11::arg("to_dir"),
          "Sphere logarithmic map: tangent vector at from_dir pointing to to_dir with |.| = geodesic angle");
    m.def("parallel_transport", &los::parallelTransport,
          pybind11::arg("from_dir"), pybind11::arg("to_dir"), pybind11::arg("tangent"),
          "Parallel transport of a tangent vector at from_dir to the tangent plane at to_dir");
    m.def("observe", &observe_validated, pybind11::arg("target_state"), pybind11::arg("sensor_state"),
          "Forward model: LosObservation of a 6-D ECI target seen from a 6-D ECI sensor");
    m.def("to_cartesian", &to_cartesian_validated, pybind11::arg("observation"), pybind11::arg("sensor_state"),
          "Exact inverse model: 6-D ECI state from a LosObservation and the sensor state");
    m.def("perturbed", &perturbed_validated, pybind11::arg("observation"), pybind11::arg("eps"),
          "Apply a 6-D local tangent-frame perturbation [d_range, d_range_rate, d_theta1, d_theta2, d_omega1, d_omega2]\n"
          "expressed in tangent_basis(observation.los)");
    m.def("local_residual",
          static_cast<los::Vector6 (*)(const los::LosObservation&, const los::LosObservation&)>(
              &los::localResidual),
          pybind11::arg("measured"), pybind11::arg("predicted"),
          "Residual (measured - predicted) in the 6-D local tangent frame of the measured direction:\n"
          "[d_range, d_range_rate, d_theta1, d_theta2, d_omega1, d_omega2]")
     .def("local_residual",
          static_cast<los::Vector6 (*)(const los::LosObservation&, const los::LosObservation&,
                                       const los::TangentBasis&)>(&los::localResidual),
          pybind11::arg("measured"), pybind11::arg("predicted"), pybind11::arg("measured_basis"),
          "Same residual, but with a precomputed tangent_basis(measured.los). Bit-identical to the\n"
          "two-argument form when measured_basis == tangent_basis(measured.los).");
    m.def("angular_coordinates", &los::angularCoordinates, pybind11::arg("observation"),
          "Derived ECI-axis [azimuth, elevation, azimuth_rate, elevation_rate] (display/interop only; singular on the z-axis)");
    m.def("from_angles_and_rates", &los::fromAnglesAndRates,
          pybind11::arg("range"), pybind11::arg("range_rate"), pybind11::arg("azimuth"),
          pybind11::arg("elevation"), pybind11::arg("azimuth_rate"), pybind11::arg("elevation_rate"),
          "Exact LosObservation from ECI-axis spherical angles and their rates");
    
    // Bind core data structures
    pybind11::class_<TrackLabel>(m, "TrackLabel")
        .def(pybind11::init<>())
        .def_readwrite("birth_time", &TrackLabel::birth_time)
        .def_readwrite("index", &TrackLabel::index);
    
    pybind11::class_<Particle>(m, "Particle")
        .def(pybind11::init<>())
        .def_readwrite("state_vector", &Particle::state_vector)
        .def_readwrite("weight", &Particle::weight);
    
    pybind11::class_<Measurement>(m, "Measurement",
        "A sensor detection: range_ [m], range_rate_ [m/s], unit line-of-sight direction los_ (ECI) and its\n"
        "angular rate los_rate_ [rad/s], plus a 6x6 covariance_ in the local tangent frame at los_ with the\n"
        "basis tangent_basis(los_) and ordering\n"
        "  [d_range (m), d_range_rate (m/s), d_theta1 (rad), d_theta2 (rad), d_omega1 (rad/s), d_omega2 (rad/s)].\n"
        "sensor_state_ is the 6-D ECI state of the sensor. The representation is free of angular singularities.")
        .def(pybind11::init<>())
        .def_readwrite("timestamp_", &Measurement::timestamp_)
        .def_readwrite("range_", &Measurement::range_)
        .def_readwrite("range_rate_", &Measurement::range_rate_)
        .def_property("los_",
                      [](const Measurement& self) { return self.los_; },
                      [](Measurement& self, const Eigen::VectorXd& value) {
                          self.los_ = require_vector3(value, "Measurement.los_");
                      },
                      "Unit line-of-sight direction from the sensor to the object (ECI)")
        .def_property("los_rate_",
                      [](const Measurement& self) { return self.los_rate_; },
                      [](Measurement& self, const Eigen::VectorXd& value) {
                          self.los_rate_ = require_vector3(value, "Measurement.los_rate_");
                      },
                      "Line-of-sight angular rate [rad/s] (ECI, orthogonal to los_)")
        .def_property("covariance_",
                      [](const Measurement& self) { return self.covariance_; },
                      [](Measurement& self, const Eigen::MatrixXd& value) {
                          self.covariance_ = require_meas_covariance(value, "Measurement.covariance_");
                      },
                      "6x6 noise covariance in the local tangent frame at los_ (see class docstring for ordering)")
        .def_readwrite("sensor_id_", &Measurement::sensor_id_)
        .def_property("sensor_state_",
                      [](const Measurement& self) { return self.sensor_state_; },
                      [](Measurement& self, const Eigen::VectorXd& value) {
                          self.sensor_state_ = require_state_vector_fixed(value, "Measurement.sensor_state_");
                      },
                      "6-D ECI state of the sensor [x, y, z, vx, vy, vz]")
        .def_static("fromCartesian", &measurement_from_cartesian_validated,
                    pybind11::arg("target_state"), pybind11::arg("sensor_state"),
                    "Noise-free measurement of a 6-D ECI target state from a 6-D ECI sensor state\n"
                    "(covariance_ left at zero, sensor_state_ stored)")
        .def_static("fromAnglesAndRates", &measurement_from_angles_validated,
                    pybind11::arg("range"), pybind11::arg("range_rate"), pybind11::arg("azimuth"),
                    pybind11::arg("elevation"), pybind11::arg("azimuth_rate"), pybind11::arg("elevation_rate"),
                    pybind11::arg("sensor_state"),
                    "Measurement from ECI-axis spherical angles and their rates (interop convention;\n"
                    "singular on the ECI z-axis, never used internally)")
        .def("observation", &Measurement::observation, "Geometric part as a LosObservation")
        .def("setObservation", &Measurement::setObservation, pybind11::arg("observation"),
             "Overwrite range_, range_rate_, los_, los_rate_ from a LosObservation")
        .def("toCartesian", &Measurement::toCartesian,
             "Exact 6-D ECI state implied by the measurement and sensor_state_")
        .def("perturbed", &measurement_perturbed_validated, pybind11::arg("eps"),
             "Copy with the local tangent-frame perturbation eps = [d_range, d_range_rate, d_theta1, d_theta2,\n"
             "d_omega1, d_omega2] applied to the geometry (covariance_, sensor and timestamp unchanged)")
        .def("angularCoordinates", &Measurement::angularCoordinates,
             "Derived ECI-axis [azimuth, elevation, azimuth_rate, elevation_rate] (display/interop only)");
    
    pybind11::class_<Track>(m, "Track")
        .def(pybind11::init<>())
        .def(pybind11::init<const TrackLabel&, double, const std::vector<Particle>&>())
        .def("label", &get_track_label_copy)
        .def("existence_probability", &Track::existence_probability)
        .def("propagated_time", &Track::propagated_time,
             "Time [s] the particle cloud is valid at, or NaN if no tracker has stamped it. Under\n"
             "lazy propagation this can lag the tracker's clock; see SMC_LMB_Tracker.synchronize.")
        .def("particles", &get_track_particles_copy)
        .def("particle_states", &get_track_particle_states,
             "Read-only (N, 6) view of the particle state vectors, aliasing the track's own memory.\n"
             "Invalidated by anything that reallocates the cloud (set_particles, or an update that\n"
             "replaces the cloud on the tracker this track came from). predict overwrites values in\n"
             "place without reallocating, so addresses stay valid but observed values change.")
        .def("particle_weights", &get_track_particle_weights,
             "Read-only (N,) view of the particle weights, aliasing the track's own memory.\n"
             "Same invalidation rules as particle_states().")
        .def("mean_state", &particle_stats::mean_state,
             "Weighted mean state. Falls back to the unweighted mean when the total weight is at or\n"
             "below 1e-12, and returns zeros for an empty cloud.")
        .def("covariance", &particle_stats::covariance,
             "6x6 weighted covariance about mean_state(), normalized by the total weight.")
        .def("weight_sum", &particle_stats::weight_sum, "Sum of the particle weights")
        .def("set_existence_probability", &Track::set_existence_probability)
        .def("set_particles",
             static_cast<void (Track::*)(const std::vector<Particle>&)>(&Track::set_particles),
             "Replace the particle cloud (copy). Reallocates and invalidates particle_states/\n"
             "particle_weights views.");
    
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
             pybind11::arg("birth_covariance_local"),
             pybind11::arg("seed") = pybind11::none(),
             "Birth model: one new track per unused measurement, particles sampled by adding Gaussian\n"
             "noise to the measurement in its local tangent frame and mapping each sample exactly to ECI\n"
             "(r = r_s + range*los, v = v_s + range_rate*los + range*los_rate).\n\n"
             "birth_covariance_local: 6x6 symmetric positive-definite covariance in the local tangent frame\n"
             "at the measured direction, basis tangent_basis(los_), ordered as\n"
             "  [d_range (m), d_range_rate (m/s), d_theta1 (rad), d_theta2 (rad), d_omega1 (rad/s), d_omega2 (rad/s)].\n"
             "The particle spread is therefore range-, angle- and rate-wise, never in ECI x/y/z.\n"
             "seed: optional integer for a reproducible particle stream (default: std::random_device).")
        .def("generate_new_tracks", &generate_new_tracks_validated,
             pybind11::arg("unused_measurements"), pybind11::arg("current_time"),
             "One track per measurement; label.index is the measurement index, weights are 1/N")
        .def("birth_covariance_local", &AdaptiveBirthModel::birthCovarianceLocal,
             "The 6x6 birth covariance in the local tangent frame");

    pybind11::class_<TwoBodyPropagator, IOrbitPropagator, std::shared_ptr<TwoBodyPropagator>>(m, "TwoBodyPropagator")
        .def(pybind11::init(&make_two_body_propagator),
             pybind11::arg("process_noise_covariance"),
             pybind11::arg("seed") = pybind11::none(),
             pybind11::arg("noise_reference_dt") = pybind11::none(),
             "RK4 two-body propagator with optional additive Gaussian process noise.\n\n"
             "seed: optional integer for a reproducible noise stream (default: std::random_device).\n"
             "noise_reference_dt: None (default) adds process_noise_covariance in full on every\n"
             "propagate() call, whatever dt is -- correct only for a fixed step. A value T reads the\n"
             "matrix as the covariance accumulated over T seconds of continuous white noise on\n"
             "[r, v] with r' = v, so any step length injects the consistent amount; velocity noise\n"
             "then also diffuses into position within a step. Lazy propagation requires it.")
        .def("propagate", &propagate_validated,
             pybind11::arg("particle"),
             pybind11::arg("dt"),
             pybind11::arg("current_time"),
             pybind11::arg("noise_scale") = 1.0)
        .def_property_readonly("noise_reference_dt", &TwoBodyPropagator::noise_reference_dt,
             "Reference interval of the time-consistent noise model [s], or None for per-call noise")
        .def("step_noise_covariance", &TwoBodyPropagator::step_noise_covariance, pybind11::arg("dt"),
             "6x6 covariance of the noise one propagate(dt) call adds at noise_scale = 1")
        .def("noise_displacement_bound", &TwoBodyPropagator::noise_displacement_bound,
             pybind11::arg("interval"),
             "Conservative bound [m] on how far process noise can move a position over `interval`\n"
             "seconds, or None when the noise model is per-call (no such bound exists)");
    
    pybind11::class_<InOrbitSensorModel, ISensorModel, std::shared_ptr<InOrbitSensorModel>>(m, "InOrbitSensorModel",
        "6-D Gaussian likelihood of a measurement given a particle, evaluated in the local tangent frame of\n"
        "the measured direction with los::localResidual (log map for the direction, parallel transport for\n"
        "the angular rate). Measurement.covariance_ is authoritative; the six variances passed to the\n"
        "constructor define defaultCovariance() in the same frame and ordering.")
        .def(pybind11::init<>(), "Default variances: 50 m, 1 m/s, 1e-4 rad, 1e-4 rad, 1e-5 rad/s, 1e-5 rad/s (squared)")
        .def(pybind11::init<double, double, double, double, double, double>(),
             pybind11::arg("range_var"),
             pybind11::arg("range_rate_var"),
             pybind11::arg("angle_var_1"),
             pybind11::arg("angle_var_2"),
             pybind11::arg("angle_rate_var_1"),
             pybind11::arg("angle_rate_var_2"),
             "Variances [m^2, (m/s)^2, rad^2, rad^2, (rad/s)^2, (rad/s)^2] along\n"
             "[range, range_rate, e1, e2, e1 rate, e2 rate] of the local tangent frame")
        .def("calculate_likelihood", &calculate_likelihood_validated,
             pybind11::arg("particle"), pybind11::arg("measurement"),
             "Likelihood density of the measurement given the particle state")
        .def("predictObservation", &predict_observation_validated,
             pybind11::arg("particle"), pybind11::arg("sensor_state"),
             "Noise-free LosObservation of the particle from the given 6-D sensor state")
        .def("defaultCovariance", &InOrbitSensorModel::defaultCovariance,
             "diag(range_var, range_rate_var, angle_var_1, angle_var_2, angle_rate_var_1, angle_rate_var_2)");
    
    m.attr("SENSOR_MAX_HALF_ANGLE") = sensor::kMaxHalfAngle;
    m.attr("SENSOR_DEFAULT_HALF_ANGLE") = sensor::kDefaultHalfAngle;

    pybind11::class_<sensor::SensorFovConfig>(m, "SensorFovConfig",
        "Range bounds and angular half-widths, shared by every sensor in a SensorArray.\n\n"
        "A pointed sensor sees a target at relative position d when\n"
        "  min_range <= |d| <= max_range,  d.boresight > 0,\n"
        "  |d.width| <= (d.boresight) tan(half_width),  |d.up| <= (d.boresight) tan(half_height)\n"
        "-- a rectangular pyramid in the tangent-plane (pinhole) convention. An unpointed sensor\n"
        "ignores the angular test and is bounded only by range.\n\n"
        "The defaults ([0, inf) in range) paired with SensorArray.add_unpointed reproduce the\n"
        "omniscient single sensor the filter had before fields of view existed.")
        .def(pybind11::init(&make_sensor_fov_config),
             pybind11::arg("min_range") = 0.0,
             pybind11::arg("max_range") = std::numeric_limits<double>::infinity(),
             pybind11::arg("half_width") = sensor::kDefaultHalfAngle,
             pybind11::arg("half_height") = sensor::kDefaultHalfAngle,
             "Half-angles are in radians and must lie in (0, pi/2).")
        .def_readwrite("min_range", &sensor::SensorFovConfig::min_range, "Inclusive lower range bound [m]")
        .def_readwrite("max_range", &sensor::SensorFovConfig::max_range, "Inclusive upper range bound [m]")
        .def_readwrite("half_width", &sensor::SensorFovConfig::half_width,
                       "Half-angle about the frame's width axis [rad], in (0, pi/2)")
        .def_readwrite("half_height", &sensor::SensorFovConfig::half_height,
                       "Half-angle about the frame's height axis [rad], in (0, pi/2)")
        .def("__repr__", [](const sensor::SensorFovConfig& fov) {
            return "SensorFovConfig(min_range=" + std::to_string(fov.min_range) +
                   ", max_range=" + std::to_string(fov.max_range) +
                   ", half_width=" + std::to_string(fov.half_width) +
                   ", half_height=" + std::to_string(fov.half_height) + ")";
        });

    pybind11::class_<sensor::SensorArray, std::shared_ptr<sensor::SensorArray>>(m, "SensorArray",
        "An ordered set of sensors sharing one SensorFovConfig.\n\n"
        "Sensors are addressed by the index add() returns, and looked up by id when a Measurement\n"
        "has to be traced back to the sensor that produced it, so ids must be unique and must match\n"
        "Measurement.sensor_id_. Pointing is expected to be rewritten every timestep.\n\n"
        "sees() is the single source of truth for visibility: use it to decide whether a truth\n"
        "object produces a measurement at all, and the filter will score that measurement with the\n"
        "same predicate.\n\n"
        "Sensor volumes are assumed disjoint. The filter is not built to have one object reported\n"
        "by two sensors in the same step.")
        .def(pybind11::init([](const sensor::SensorFovConfig& fov) {
                 return std::make_shared<sensor::SensorArray>(fov);
             }),
             pybind11::arg("fov_config") = sensor::SensorFovConfig(),
             "Empty array with the given global field-of-view configuration.")
        .def_property("fov_config",
                      [](const sensor::SensorArray& self) { return self.fov(); },
                      [](sensor::SensorArray& self, const sensor::SensorFovConfig& fov) { self.set_fov(fov); },
                      "Range bounds and half-angles shared by every sensor here. The getter returns a\n"
                      "copy, so mutating it does nothing until it is assigned back.")
        .def("add",
             [](sensor::SensorArray& self, std::string id, const Eigen::VectorXd& state,
                const Eigen::VectorXd& boresight) {
                 return self.add(std::move(id), require_state_vector_fixed(state, "state"),
                                 require_vector3_fixed(boresight, "boresight"));
             },
             pybind11::arg("id"), pybind11::arg("state"), pybind11::arg("boresight"),
             "Add a pointed sensor and return its index. The boresight is normalised; the roll is\n"
             "initialised deterministically and can be set with set_pointing().")
        .def("add_unpointed",
             [](sensor::SensorArray& self, std::string id, const Eigen::VectorXd& state) {
                 return self.add_unpointed(std::move(id), require_state_vector_fixed(state, "state"));
             },
             pybind11::arg("id"), pybind11::arg("state"),
             "Add a range-only sensor and return its index. It ignores the field of view entirely.")
        .def("size", &sensor::SensorArray::size, "Number of sensors")
        .def("__len__", &sensor::SensorArray::size)
        .def("index_of", &sensor::SensorArray::index_of, pybind11::arg("id"),
             "Index of the sensor with this id, or -1 when there is none")
        .def("id", [](const sensor::SensorArray& self, size_t i) { return self.at(i).id; },
             pybind11::arg("index"), "Id of sensor `index`")
        .def("state", [](const sensor::SensorArray& self, size_t i) { return self.at(i).state; },
             pybind11::arg("index"), "6-D ECI state of sensor `index`")
        .def("boresight", [](const sensor::SensorArray& self, size_t i) { return self.at(i).boresight; },
             pybind11::arg("index"), "Unit ECI pointing direction of sensor `index`")
        .def("up", [](const sensor::SensorArray& self, size_t i) { return self.at(i).up; },
             pybind11::arg("index"), "Unit height axis of sensor `index`, orthogonal to its boresight")
        .def("width_axis", [](const sensor::SensorArray& self, size_t i) { return self.at(i).width; },
             pybind11::arg("index"), "Unit width axis of sensor `index`: up x boresight")
        .def("pointed", [](const sensor::SensorArray& self, size_t i) { return self.at(i).pointed; },
             pybind11::arg("index"), "False when sensor `index` is bounded by range alone")
        .def("set_state",
             [](sensor::SensorArray& self, size_t i, const Eigen::VectorXd& state) {
                 self.set_state(i, require_state_vector_fixed(state, "state"));
             },
             pybind11::arg("index"), pybind11::arg("state"), "Move one sensor")
        .def("set_states",
             [](sensor::SensorArray& self, const Eigen::MatrixXd& states) {
                 require_row_count(states, self.size(), 6, "states");
                 for (size_t i = 0; i < self.size(); ++i) {
                     self.set_state(i, StateVector(states.row(static_cast<Eigen::Index>(i)).transpose()));
                 }
             },
             pybind11::arg("states"), "Move every sensor from an (S, 6) array of ECI states")
        .def("set_boresight",
             [](sensor::SensorArray& self, size_t i, const Eigen::VectorXd& boresight) {
                 self.set_boresight(i, require_vector3_fixed(boresight, "boresight"));
             },
             pybind11::arg("index"), pybind11::arg("boresight"),
             "Re-point one sensor. The vector is normalised, and the roll is carried over by\n"
             "parallel transport so a slewing field of view does not spin about its own axis.")
        .def("set_boresights",
             [](sensor::SensorArray& self, const Eigen::MatrixXd& boresights) {
                 require_row_count(boresights, self.size(), 3, "boresights");
                 for (size_t i = 0; i < self.size(); ++i) {
                     self.set_boresight(i, Eigen::Vector3d(boresights.row(static_cast<Eigen::Index>(i)).transpose()));
                 }
             },
             pybind11::arg("boresights"), "Re-point every sensor from an (S, 3) array of ECI directions")
        .def("set_pointing",
             [](sensor::SensorArray& self, size_t i, const Eigen::VectorXd& boresight,
                const Eigen::VectorXd& up) {
                 self.set_pointing(i, require_vector3_fixed(boresight, "boresight"),
                                   require_vector3_fixed(up, "up"));
             },
             pybind11::arg("index"), pybind11::arg("boresight"), pybind11::arg("up"),
             "Re-point one sensor with an explicit roll. `up` is orthogonalised against the\n"
             "boresight and must not be parallel to it.")
        .def("set_pointings",
             [](sensor::SensorArray& self, const Eigen::MatrixXd& boresights, const Eigen::MatrixXd& ups) {
                 require_row_count(boresights, self.size(), 3, "boresights");
                 require_row_count(ups, self.size(), 3, "ups");
                 // Every row is checked before any sensor moves, so a rejected call changes nothing.
                 for (size_t i = 0; i < self.size(); ++i) {
                     const Eigen::Vector3d b = sensor::normalized_direction(
                         Eigen::Vector3d(boresights.row(static_cast<Eigen::Index>(i)).transpose()), "boresight");
                     const Eigen::Vector3d u = sensor::normalized_direction(
                         Eigen::Vector3d(ups.row(static_cast<Eigen::Index>(i)).transpose()), "up");
                     if (!((u - u.dot(b) * b).norm() > sensor::kDirectionEpsilon)) {
                         throw std::invalid_argument("up: must not be parallel to boresight");
                     }
                 }
                 for (size_t i = 0; i < self.size(); ++i) {
                     self.set_pointing(i, Eigen::Vector3d(boresights.row(static_cast<Eigen::Index>(i)).transpose()),
                                       Eigen::Vector3d(ups.row(static_cast<Eigen::Index>(i)).transpose()));
                 }
             },
             pybind11::arg("boresights"), pybind11::arg("ups"),
             "Re-point every sensor with an explicit roll, from two (S, 3) arrays of ECI directions:\n"
             "set_pointing for each row. All rows are validated first, so a rejected call leaves\n"
             "the array as it was.")
        .def("point_at",
             [](sensor::SensorArray& self, size_t i, const Eigen::VectorXd& target_position) {
                 self.point_at(i, require_position_fixed(target_position, "target_position"));
             },
             pybind11::arg("index"), pybind11::arg("target_position"),
             "Aim one sensor's boresight at an ECI position (3-vector or 6-D state), keeping its roll")
        .def("set_pointed", &sensor::SensorArray::set_pointed,
             pybind11::arg("index"), pybind11::arg("pointed"),
             "Turn the angular field-of-view test on or off for one sensor")
        .def("sees",
             [](const sensor::SensorArray& self, size_t i, const Eigen::VectorXd& target) {
                 return self.sees(i, require_position_fixed(target, "target"));
             },
             pybind11::arg("index"), pybind11::arg("target"),
             "Whether sensor `index` can observe an object at this position (3-vector or 6-D state)")
        .def("visible_sensor",
             [](const sensor::SensorArray& self, const Eigen::VectorXd& target) {
                 return self.visible_sensor(require_position_fixed(target, "target"));
             },
             pybind11::arg("target"),
             "Lowest-indexed sensor that can observe this position, or -1 when none can")
        .def("coverage_fractions",
             [](const sensor::SensorArray& self, const Track& track) {
                 std::vector<double> per_sensor;
                 self.coverage(track, per_sensor);
                 return to_numpy(per_sensor);
             },
             pybind11::arg("track"),
             "(S,) weight-fraction of the track's particle cloud inside each sensor's volume")
        .def("coverage_fraction",
             [](const sensor::SensorArray& self, const Track& track) {
                 std::vector<double> per_sensor;
                 return self.coverage(track, per_sensor);
             },
             pybind11::arg("track"),
             "Weight-fraction of the track's particle cloud inside the union of all sensor volumes.\n"
             "This is what the filter scales P_D by in its missed-detection branch.");

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
             pybind11::arg("seed") = pybind11::none(),
             "Constructor for SMC_LMB_Tracker with model dependencies.\n\n"
             "seed: optional integer for a reproducible resampler stream (default: std::random_device).")
        .def("predict", &SMC_LMB_Tracker::predict, "Runs the predict step for a given time delta")
        .def("update",
             static_cast<void (SMC_LMB_Tracker::*)(const std::vector<Measurement>&)>(&SMC_LMB_Tracker::update),
             pybind11::arg("measurements"),
             "Runs the update step with measurements, against a sensor that sees everything.\n"
             "Every track is fully observable, so the effective P_D is the configured P_D and a\n"
             "step with no measurements changes nothing.")
        .def("update",
             static_cast<void (SMC_LMB_Tracker::*)(const std::vector<Measurement>&,
                                                   const sensor::SensorArray&)>(&SMC_LMB_Tracker::update),
             pybind11::arg("measurements"), pybind11::arg("sensors"),
             "Runs the update step against a configured set of sensors.\n\n"
             "Each measurement is traced back through its sensor_id_, and every track's effective\n"
             "P_D is the configured P_D scaled by the fraction of that track's particle cloud inside\n"
             "the relevant sensor volume. A track outside every field of view keeps its existence\n"
             "probability; a step in which no sensor reported anything is applied as a pure missed\n"
             "detection. Raises ValueError if a sensor_id_ is not in the array.")
        .def("get_tracks", &get_tracks_copy, "Gets the current list of tracks")
        .def("set_lazy_propagation", &SMC_LMB_Tracker::set_lazy_propagation,
             pybind11::arg("enabled"), pybind11::arg("max_pending") = 60.0,
             "Turn lazy propagation on or off.\n\n"
             "With it on, predict() advances only the clock and the survival probability. A track's\n"
             "cloud is propagated once it lags the clock by max_pending seconds (in substeps of at\n"
             "most max_pending), when update(measurements, sensors) finds that some particle of it\n"
             "could be inside a sensor volume, or on synchronize(). Requires a propagator built\n"
             "with noise_reference_dt; raises ValueError otherwise. get_tracks() may then return\n"
             "tracks that lag the clock -- call synchronize() first when you need them current.")
        .def_property_readonly("lazy_propagation", &SMC_LMB_Tracker::lazy_propagation,
             "Whether lazy propagation is on")
        .def_property_readonly("max_pending", &SMC_LMB_Tracker::max_pending,
             "Longest a track may lag the clock under lazy propagation [s]")
        .def("set_regularization", &SMC_LMB_Tracker::set_regularization,
             pybind11::arg("enabled"), pybind11::arg("bandwidth_scale") = 1.0,
             pybind11::arg("ess_threshold") = 0.5,
             "Turn the regularization (kernel jitter) step on or off.\n\n"
             "After resampling a cloud whose posterior effective sample size fell below\n"
             "ess_threshold * N, each particle moves as x <- m + a (x - m) + h L eps, with m, L L^T the\n"
             "weighted posterior mean and covariance, a = sqrt(1 - h^2) (Liu & West shrinkage, which\n"
             "keeps the mean and covariance) and h = bandwidth_scale * (4 / (N (d + 2)))^(1/(d+4)),\n"
             "d = 6 (Musso, Oudjane & Le Gland). Restores the diversity resampling destroys.")
        .def_property_readonly("regularization", &SMC_LMB_Tracker::regularization,
             "Whether the regularization step is on")
        .def("set_fused_proposal", &SMC_LMB_Tracker::set_fused_proposal,
             pybind11::arg("enabled"), pybind11::arg("ess_min") = 20.0, pybind11::arg("neighbours") = 0,
             pybind11::arg("fallback_ess_min") = 20.0,
             "Turn the fused proposal on or off.\n\n"
             "When the ordinary particle update for a (track, measurement) pair collapses (effective\n"
             "sample size below ess_min), that pair's detection component is drawn from the overlap of\n"
             "the track's cloud near the measured state (kernel density of its `neighbours` nearest\n"
             "particles; 0 = max(30, 5% of the cloud)) and the measurement as a Gaussian in state\n"
             "space, importance-weighted with the exact likelihood. Its association likelihood replaces\n"
             "the particle sum, which underflows to zero in that regime. Clouds that reach a sensor's\n"
             "volume with no particle inside it are scored the same way. If the fused weights collapse\n"
             "too (ESS below fallback_ess_min), the component falls back to uniform draws from the\n"
             "Gaussian product with a closed-form likelihood (an approximation, kept as a last resort).")
        .def_property_readonly("fused_proposal", &SMC_LMB_Tracker::fused_proposal,
             "Whether the fused proposal is on")
        .def("set_record_diagnostics", &SMC_LMB_Tracker::set_record_diagnostics, pybind11::arg("enabled"),
             "Record one entry per track per update: ESS, detection mass, whether it was regularized")
        .def("take_diagnostics",
             [](SMC_LMB_Tracker& self) {
                 const auto records = self.take_diagnostics();
                 const auto count = static_cast<pybind11::ssize_t>(records.size());
                 pybind11::array_t<double> time(count), ess(count), detection_mass(count);
                 pybind11::array_t<uint64_t> birth_time(count);
                 pybind11::array_t<int64_t> index(count), num_particles(count);
                 pybind11::array_t<bool> regularized(count);
                 pybind11::array_t<int64_t> fused_components(count), fallback_components(count),
                     best_measurement(count);
                 pybind11::array_t<double> fused_ess(count), best_coefficient(count);
                 for (pybind11::ssize_t i = 0; i < count; ++i) {
                     const auto& r = records[static_cast<size_t>(i)];
                     time.mutable_at(i) = r.time;
                     ess.mutable_at(i) = r.ess;
                     detection_mass.mutable_at(i) = r.detection_mass;
                     birth_time.mutable_at(i) = r.birth_time;
                     index.mutable_at(i) = static_cast<int64_t>(r.index);
                     num_particles.mutable_at(i) = static_cast<int64_t>(r.num_particles);
                     regularized.mutable_at(i) = r.regularized;
                     fused_components.mutable_at(i) = r.fused_components;
                     fallback_components.mutable_at(i) = r.fallback_components;
                     fused_ess.mutable_at(i) = r.fused_ess;
                     best_measurement.mutable_at(i) = r.best_measurement;
                     best_coefficient.mutable_at(i) = r.best_coefficient;
                 }
                 pybind11::dict out;
                 out["time"] = time;
                 out["birth_time"] = birth_time;
                 out["index"] = index;
                 out["ess"] = ess;
                 out["num_particles"] = num_particles;
                 out["detection_mass"] = detection_mass;
                 out["regularized"] = regularized;
                 out["fused_components"] = fused_components;
                 out["fallback_components"] = fallback_components;
                 out["fused_ess"] = fused_ess;
                 out["best_measurement"] = best_measurement;
                 out["best_coefficient"] = best_coefficient;
                 return out;
             },
             "Posterior records collected since the last call (then cleared), as NumPy arrays")
        .def("set_particle_gate", &SMC_LMB_Tracker::set_particle_gate, pybind11::arg("enabled"),
             "Re-test a lazy cloud's 'maybe visible' particle by particle before propagating it\n"
             "(default off). Changes when clouds are propagated, never what an update learns;\n"
             "statistically, not bitwise, equivalent. Off, the filter is bit-for-bit unchanged.")
        .def_property_readonly("particle_gate", &SMC_LMB_Tracker::particle_gate,
             "Whether the particle gate is on")
        .def("set_gate_audit", &SMC_LMB_Tracker::set_gate_audit, pybind11::arg("enabled"),
             "Validation only (slow): check every 'not observable' decision of the particle gate\n"
             "against a noise-free propagation of the cloud. The run itself is unchanged.")
        .def("set_gate_sleep", &SMC_LMB_Tracker::set_gate_sleep, pybind11::arg("enabled"),
             "Sleep between particle-gate tests (default on with the gate). Bit-for-bit neutral;\n"
             "the switch exists for testing.")
        .def("sleep_audit", &SMC_LMB_Tracker::sleep_audit,
             "(sleeping skips audited, skips whose test would have said 'maybe', first such case)")
        .def("gate_audit_counts", &SMC_LMB_Tracker::gate_audit_counts,
             "(decisions audited, violations) since construction")
        .def("set_fast_mode", &SMC_LMB_Tracker::set_fast_mode, pybind11::arg("enabled"),
             "Turn fast mode on or off (default off): keyed random streams, a ziggurat normal\n"
             "sampler and batched propagation. The deterministic RK4 step is the legacy one; the noise\n"
             "streams differ, so a run is statistically, not bitwise, equivalent to a legacy run.\n"
             "Off, the filter is bit-for-bit unchanged. Raises ValueError for a propagator without\n"
             "keyed propagation.")
        .def_property_readonly("fast_mode", &SMC_LMB_Tracker::fast_mode, "Whether fast mode is on")
        .def("set_profiling", &SMC_LMB_Tracker::set_profiling, pybind11::arg("enabled"),
             "Accumulate wall-clock phase totals and work counts (off by default). Read-only\n"
             "instrumentation: a profiled run is bit-for-bit the unprofiled one.")
        .def_property_readonly("profiling", &SMC_LMB_Tracker::profiling, "Whether profiling is on")
        .def("reset_profile", &SMC_LMB_Tracker::reset_profile, "Zero every profile total")
        .def("profile",
             [](const SMC_LMB_Tracker& self) {
                 const SMC_LMB_Tracker::Profile& p = self.profile();
                 pybind11::dict seconds;
                 seconds["predict"] = p.predict;
                 seconds["predict_propagate"] = p.predict_propagate;
                 seconds["update"] = p.update;
                 seconds["refresh"] = p.refresh;
                 seconds["refresh_propagate"] = p.refresh_propagate;
                 seconds["refresh_check"] = p.refresh_check;
                 seconds["synchronize"] = p.synchronize;
                 seconds["coverage"] = p.coverage;
                 seconds["likelihood"] = p.likelihood;
                 seconds["fused"] = p.fused;
                 seconds["assignment"] = p.assignment;
                 seconds["posterior"] = p.posterior;
                 seconds["birth"] = p.birth;
                 pybind11::dict counts;
                 counts["predict_calls"] = p.predict_calls;
                 counts["update_calls"] = p.update_calls;
                 counts["updates_with_measurements"] = p.updates_with_measurements;
                 counts["predict_particle_steps"] = p.predict_particle_steps;
                 counts["refresh_particle_steps"] = p.refresh_particle_steps;
                 counts["synchronize_particle_steps"] = p.synchronize_particle_steps;
                 counts["refresh_checks"] = p.refresh_checks;
                 counts["refresh_slept"] = p.refresh_slept;
                 counts["sleep_resets"] = p.sleep_resets;
                 counts["sleep_none"] = p.sleep_histogram[0];
                 counts["sleep_under_1s"] = p.sleep_histogram[1];
                 counts["sleep_1_to_5s"] = p.sleep_histogram[2];
                 counts["sleep_5_to_20s"] = p.sleep_histogram[3];
                 counts["sleep_over_20s"] = p.sleep_histogram[4];
                 counts["tracks_refreshed"] = p.tracks_refreshed;
                 counts["refreshed_tracks_covered"] = p.refreshed_tracks_covered;
                 counts["refresh_gap_inside"] = p.refresh_gap_inside;
                 counts["refresh_gap_under_5km"] = p.refresh_gap_under_5km;
                 counts["refresh_gap_5_to_50km"] = p.refresh_gap_5_to_50km;
                 counts["refresh_gap_over_50km"] = p.refresh_gap_over_50km;
                 counts["active_tracks"] = p.active_tracks;
                 counts["likelihood_evaluations"] = p.likelihood_evaluations;
                 counts["fused_builds"] = p.fused_builds;
                 counts["fused_kernel_evaluations"] = p.fused_kernel_evaluations;
                 counts["posterior_updates"] = p.posterior_updates;
                 counts["regularizations"] = p.regularizations;
                 counts["births"] = p.births;
                 pybind11::dict out;
                 out["seconds"] = seconds;
                 out["counts"] = counts;
                 return out;
             },
             "Profile totals since the last reset_profile(): {'seconds': {...}, 'counts': {...}}")
        .def("synchronize", &SMC_LMB_Tracker::synchronize,
             "Propagate every track that lags the clock up to it. A no-op when nothing lags.")
        .def("timestamp", &SMC_LMB_Tracker::timestamp, "The filter clock [s]")
        .def("track_summary", &track_summary, pybind11::arg("with_means") = true,
             pybind11::arg("with_covariances") = false,
             "Per-track arrays without copying particle clouds: birth_time, index, existence,\n"
             "propagated_time, particle_count, (with_means) the (N, 6) weighted mean states and\n"
             "(with_covariances) the (N, 6, 6) covariances, as Track.covariance() computes them.\n"
             "Means of lagging tracks are at their propagated_time, not at the clock.")
        .def("sample_particles", &sample_particles, pybind11::arg("track_indices"), pybind11::arg("count"),
             "States of `count` particles from each track in track_indices, taken evenly through\n"
             "the track's weights, as {'states': (T, count, 6), 'propagated_time': (T,)}. The states\n"
             "are at each track's propagated_time, which may lag the clock. Copies only what it\n"
             "returns and draws no random number.")
        .def("gospa_components", &tracker_gospa_components, pybind11::arg("track_indices"),
             pybind11::arg("truths"), pybind11::arg("cutoff") = kGospaDefaultCutoff,
             "calculate_gospa_components over the tracks at track_indices (in that order) against\n"
             "an (N, 6) truth array, without copying any particle cloud. Associations index into\n"
             "track_indices. Same arithmetic as calculate_gospa_components.")
        .def("set_tracks", &SMC_LMB_Tracker::set_tracks, "Sets the initial list of tracks for the filter")
        .def("compute_association_likelihood", &SMC_LMB_Tracker::compute_association_likelihood, "Compute the association likelihood for a track and measurement");
    
    // Bind the Hypothesis struct
    pybind11::class_<Hypothesis>(m, "Hypothesis")
        .def(pybind11::init<>())
        .def_readwrite("associations", &Hypothesis::associations)
        .def_readwrite("weight", &Hypothesis::weight);

    m.def("two_body_rk4_steps",
          [](const Eigen::MatrixXd& states, double dt, int num_steps) {
              if (states.cols() != 6) {
                  throw std::invalid_argument("two_body_rk4_steps: states must be (N, 6), got (" +
                                              std::to_string(states.rows()) + ", " +
                                              std::to_string(states.cols()) + ")");
              }
              if (!std::isfinite(dt)) {
                  throw std::invalid_argument("two_body_rk4_steps: dt must be finite");
              }
              if (num_steps < 0) {
                  throw std::invalid_argument("two_body_rk4_steps: num_steps must be >= 0");
              }
              if (!states.allFinite()) {
                  throw std::invalid_argument("two_body_rk4_steps: states must be finite");
              }
              const auto rows = static_cast<pybind11::ssize_t>(states.rows());
              pybind11::array_t<double> out(std::vector<pybind11::ssize_t>{num_steps, rows, 6});
              double* data = out.mutable_data();
              const size_t row_doubles = static_cast<size_t>(rows) * 6;
              // Row-major working copy, stepped in place; each step is then copied out whole.
              std::vector<double> current(row_doubles);
              for (pybind11::ssize_t i = 0; i < rows; ++i) {
                  for (int c = 0; c < 6; ++c) {
                      current[static_cast<size_t>(i) * 6 + static_cast<size_t>(c)] =
                          states(static_cast<Eigen::Index>(i), c);
                  }
              }
              for (int k = 0; k < num_steps; ++k) {
                  two_body_rk4_step_strided(current.data(), 6, static_cast<size_t>(rows), dt);
                  std::copy(current.begin(), current.end(), data + static_cast<size_t>(k) * row_doubles);
              }
              return out;
          },
          pybind11::arg("states"), pybind11::arg("dt"), pybind11::arg("num_steps"),
          "Noise-free RK4 two-body steps of an (N, 6) array of ECI states: returns (num_steps, N, 6),\n"
          "the states after 1..num_steps steps of dt. Exactly the deterministic part of\n"
          "TwoBodyPropagator.propagate, so a truth stepped with it moves like the filter's model.");

    m.def("_fast_normals",
          [](uint64_t key, size_t count) {
              fast_random::Stream stream(key);
              std::vector<double> out(count);
              for (double& value : out) {
                  value = stream.normal();
              }
              return to_numpy(out);
          },
          pybind11::arg("key"), pybind11::arg("count"),
          "Test hook: `count` standard normals from the fast-mode keyed stream `key`.");

    // Bind the solve_assignment function
    m.def("solve_assignment", &solve_assignment, pybind11::arg("cost_matrix"), pybind11::arg("k_best"),
          "Solves the assignment problem and returns K-best hypotheses.");

    // --- GOSPA metric (see src/metrics.h) ---
    // Exposed as module attributes so every Python harness reads one number rather than
    // re-declaring its own literal.
    m.attr("GOSPA_DEFAULT_CUTOFF") = kGospaDefaultCutoff;
    m.attr("GOSPA_ORDER_P") = kGospaOrder;
    m.attr("GOSPA_ALPHA") = kGospaAlpha;

    pybind11::class_<GospaComponents>(m, "GospaComponents",
        "alpha=2 GOSPA decomposition.\n\n"
        "localisation/missed/false_positive are p-th-power costs (m^p) and sum EXACTLY to\n"
        "total**p; the roots are not additive. total is in metres.")
        .def_readonly("localisation", &GospaComponents::localisation_cost)
        .def_readonly("missed", &GospaComponents::missed_cost)
        // Not "false": that is a Python keyword, and the attribute would only be reachable
        // through getattr().
        .def_readonly("false_positive", &GospaComponents::false_cost)
        .def_readonly("total", &GospaComponents::total)
        .def_readonly("num_assigned", &GospaComponents::num_assigned)
        .def_readonly("num_missed", &GospaComponents::num_missed)
        .def_readonly("num_false", &GospaComponents::num_false)
        .def_readonly("associations", &GospaComponents::associations)
        .def("__repr__", [](const GospaComponents& g) {
            return "<GospaComponents total=" + std::to_string(g.total) +
                   " assigned=" + std::to_string(g.num_assigned) +
                   " missed=" + std::to_string(g.num_missed) +
                   " false=" + std::to_string(g.num_false) + ">";
        });

    m.def("calculate_gospa_distance", &calculate_gospa_distance,
          "Unnormalised GOSPA (p=2, alpha=2) in metres between estimated tracks and ground\n"
          "truths, position-only base distance. Not bounded by the cutoff; the tight bound is\n"
          "cutoff * sqrt((len(tracks) + len(ground_truths)) / 2).",
          pybind11::arg("tracks"),
          pybind11::arg("ground_truths"),
          pybind11::arg("cutoff") = kGospaDefaultCutoff);

    m.def("calculate_gospa_components", &calculate_gospa_components,
          "GOSPA with its localisation / missed / false-track decomposition.",
          pybind11::arg("tracks"),
          pybind11::arg("ground_truths"),
          pybind11::arg("cutoff") = kGospaDefaultCutoff);
}