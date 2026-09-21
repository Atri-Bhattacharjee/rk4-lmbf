#pragma once

/**
 * @file sensor_fov.h
 * @brief Sensor pointing, bounded field of view, and per-track coverage fractions.
 *
 * A sensor here is a 6-D ECI state plus an orientation. Range bounds and the angular half-widths
 * are held once per SensorArray (SensorFovConfig) and shared by every sensor in it; pointing is
 * per sensor and is expected to be rewritten from Python at every timestep.
 *
 * The visibility predicate defined by SensorArray::sees is the single source of truth for
 * "can this sensor observe an object at this position". The simulation side uses it to decide
 * whether a ground-truth object produces a measurement at all; the filter uses it to work out
 * what fraction of a track's particle cloud is observable. The two therefore cannot drift.
 *
 * Frame convention for a pointed sensor: (width, height, boresight) is a right-handed orthonormal
 * triad with height = up and width = up x boresight, so width x height = boresight. A target at
 * relative position d is inside the field of view when it is in front of the sensor and its
 * tangent-plane (pinhole) angles are within the half-widths:
 *
 *     z = d . boresight > 0,  |d . width| <= z tan(half_width),  |d . height| <= z tan(half_height)
 *
 * That is a rectangular pyramid, the natural shape for a focal-plane detector. There is no acos
 * and no trigonometry per particle: the two tangents are cached whenever the config changes.
 *
 * An unpointed sensor ignores the angular test entirely and is bounded only by range.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "datatypes.h"
#include "los_geometry.h"

namespace sensor {

//! A direction shorter than this is treated as degenerate rather than normalised.
constexpr double kDirectionEpsilon = 1e-12;

//! Exclusive upper bound on a half-angle. tan() diverges at pi/2 and the pyramid is only defined
//! over the forward hemisphere, so half-angles must stay strictly below it.
constexpr double kMaxHalfAngle = los::kPi / 2.0;

//! Arbitrary but documented default half-angle (45 degrees). It is only consulted by pointed
//! sensors; a default-constructed config with add_unpointed() reproduces an omniscient sensor.
constexpr double kDefaultHalfAngle = los::kPi / 4.0;

/**
 * @brief Range bounds and angular half-widths, global to every sensor in one SensorArray.
 *
 * Defaults are deliberately permissive: [0, inf) in range, so a default-constructed config paired
 * with add_unpointed() sees everything, which is what the filter did before fields of view existed.
 */
struct SensorFovConfig {
    double min_range = 0.0;                                        //!< Inclusive lower range bound [m]
    double max_range = std::numeric_limits<double>::infinity();    //!< Inclusive upper range bound [m]
    double half_width = kDefaultHalfAngle;                         //!< Half-angle about the width axis [rad]
    double half_height = kDefaultHalfAngle;                        //!< Half-angle about the height axis [rad]
};

/**
 * @brief One sensor: where it is, where it points, and whether pointing constrains it at all.
 */
struct Sensor {
    std::string id;                                          //!< Matched against Measurement::sensor_id_
    StateVector state = StateVector::Zero();                 //!< 6-D ECI state [x, y, z, vx, vy, vz]
    Eigen::Vector3d boresight = Eigen::Vector3d::UnitX();    //!< Unit ECI pointing direction
    Eigen::Vector3d up = Eigen::Vector3d::UnitY();           //!< Unit, orthogonal to boresight; fixes the roll
    //! Derived: up x boresight, cached so the coverage loop does no cross product per particle.
    //! Maintained by SensorArray; never assign it directly.
    Eigen::Vector3d width = Eigen::Vector3d::UnitZ();
    bool pointed = true;                                     //!< false: range-only, the FOV is ignored
};

//! Finite, non-degenerate direction, normalised. Throws naming the offending field otherwise.
inline Eigen::Vector3d normalized_direction(const Eigen::Vector3d& direction, const char* context) {
    if (!direction.allFinite()) {
        throw std::invalid_argument(std::string(context) + ": must be finite");
    }
    const double norm = direction.norm();
    if (!(norm > kDirectionEpsilon)) {
        throw std::invalid_argument(std::string(context) + ": must have non-zero length, got norm " +
                                    std::to_string(norm));
    }
    return direction / norm;
}

inline void require_fov_config(const SensorFovConfig& fov) {
    if (!(std::isfinite(fov.min_range) && fov.min_range >= 0.0)) {
        throw std::invalid_argument("SensorFovConfig.min_range: must be finite and >= 0, got " +
                                    std::to_string(fov.min_range));
    }
    if (std::isnan(fov.max_range) || !(fov.max_range >= fov.min_range)) {
        throw std::invalid_argument("SensorFovConfig.max_range: must be >= min_range, got " +
                                    std::to_string(fov.max_range));
    }
    if (!(std::isfinite(fov.half_width) && fov.half_width > 0.0 && fov.half_width < kMaxHalfAngle)) {
        throw std::invalid_argument("SensorFovConfig.half_width: must be in (0, pi/2) radians, got " +
                                    std::to_string(fov.half_width));
    }
    if (!(std::isfinite(fov.half_height) && fov.half_height > 0.0 && fov.half_height < kMaxHalfAngle)) {
        throw std::invalid_argument("SensorFovConfig.half_height: must be in (0, pi/2) radians, got " +
                                    std::to_string(fov.half_height));
    }
}

/**
 * @brief An ordered collection of sensors sharing one SensorFovConfig.
 *
 * Sensors are addressed by index (stable, assigned at add time) and looked up by id when a
 * Measurement has to be traced back to the sensor that produced it. Ids must be unique.
 */
class SensorArray {
public:
    explicit SensorArray(const SensorFovConfig& fov = SensorFovConfig()) { set_fov(fov); }

    const SensorFovConfig& fov() const { return fov_; }

    void set_fov(const SensorFovConfig& fov) {
        require_fov_config(fov);
        fov_ = fov;
        tan_half_width_ = std::tan(fov_.half_width);
        tan_half_height_ = std::tan(fov_.half_height);
    }

    /**
     * @brief Add a pointed sensor. The roll is initialised deterministically from tangentBasis().
     *
     * The boresight is validated before anything is inserted, so a rejected add leaves the array
     * exactly as it was -- in particular it does not burn the id.
     */
    size_t add(std::string id, const StateVector& state, const Eigen::Vector3d& boresight) {
        const Eigen::Vector3d unit_boresight = normalized_direction(boresight, "boresight");
        const size_t index = insert(std::move(id), state);
        sensors_[index].pointed = true;
        assign_pointing_from_boresight(sensors_[index], unit_boresight,
                                       /*transport_existing_up=*/false);
        return index;
    }

    //! Add a range-only sensor. Its pointing is stored but never consulted while pointed is false.
    size_t add_unpointed(std::string id, const StateVector& state) {
        const size_t index = insert(std::move(id), state);
        sensors_[index].pointed = false;
        return index;
    }

    size_t size() const { return sensors_.size(); }
    const std::vector<Sensor>& sensors() const { return sensors_; }

    const Sensor& at(size_t index) const {
        require_index(index);
        return sensors_[index];
    }

    //! Index of the sensor with this id, or -1 when there is none.
    int index_of(const std::string& id) const {
        const auto it = index_.find(id);
        return it == index_.end() ? -1 : static_cast<int>(it->second);
    }

    void set_state(size_t index, const StateVector& state) {
        require_index(index);
        if (!state.allFinite()) {
            throw std::invalid_argument("Sensor.state: must be finite");
        }
        sensors_[index].state = state;
    }

    /**
     * @brief Re-point a sensor, preserving its roll.
     *
     * The up vector is parallel-transported from the old boresight to the new one, which is the
     * minimal rotation, so a slewing sensor's field-of-view rectangle does not spin about its own
     * axis. A near-antipodal flip leaves nothing to transport and falls back to the deterministic
     * tangentBasis frame.
     */
    void set_boresight(size_t index, const Eigen::Vector3d& boresight) {
        require_index(index);
        assign_pointing_from_boresight(sensors_[index], normalized_direction(boresight, "boresight"),
                                       /*transport_existing_up=*/true);
    }

    //! Re-point with an explicit roll. `up` is orthogonalised against the boresight.
    void set_pointing(size_t index, const Eigen::Vector3d& boresight, const Eigen::Vector3d& up) {
        require_index(index);
        const Eigen::Vector3d b = normalized_direction(boresight, "boresight");
        const Eigen::Vector3d raw_up = normalized_direction(up, "up");
        Eigen::Vector3d orthogonal_up = raw_up - raw_up.dot(b) * b;
        if (!(orthogonal_up.norm() > kDirectionEpsilon)) {
            throw std::invalid_argument("up: must not be parallel to boresight");
        }
        set_frame(sensors_[index], b, orthogonal_up.normalized());
    }

    //! Point the boresight at an ECI position, preserving roll. The target must not be at the sensor.
    void point_at(size_t index, const Eigen::Vector3d& target_position) {
        require_index(index);
        if (!target_position.allFinite()) {
            throw std::invalid_argument("target_position: must be finite");
        }
        const Eigen::Vector3d offset = target_position - sensors_[index].state.head<3>();
        assign_pointing_from_boresight(sensors_[index],
                                       normalized_direction(offset, "target_position - sensor position"),
                                       /*transport_existing_up=*/true);
    }

    void set_pointed(size_t index, bool pointed) {
        require_index(index);
        sensors_[index].pointed = pointed;
    }

    //! Whether sensor `index` can observe an object at this ECI position. See the file comment.
    bool sees(size_t index, const Eigen::Vector3d& target_position) const {
        require_index(index);
        return sees_unchecked(sensors_[index], target_position);
    }

    /**
     * @brief Lowest-indexed sensor that can observe this position, or -1 when none can.
     *
     * Sensor volumes are assumed disjoint, so "lowest index" is a tie-break that should never fire.
     */
    int visible_sensor(const Eigen::Vector3d& target_position) const {
        for (size_t i = 0; i < sensors_.size(); ++i) {
            if (sees_unchecked(sensors_[i], target_position)) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    /**
     * @brief Weight-fraction of a track's particle cloud inside each sensor's volume.
     *
     * One pass over the cloud. `per_sensor_out` is resized and overwritten with q[s], the fraction
     * of the cloud's weight inside sensor s. The return value is the fraction inside the union of
     * all sensor volumes, accumulated as a genuine union rather than a sum of the per-sensor
     * fractions, so it stays correct (and <= 1) even if two volumes overlap.
     *
     * A cloud with no particles, or with non-positive total weight, reports zero coverage: there
     * is no evidence it is observable, and the filter's miss branch then leaves its existence
     * probability alone.
     */
    double coverage(const Track& track, std::vector<double>& per_sensor_out) const {
        const size_t num_sensors = sensors_.size();
        per_sensor_out.assign(num_sensors, 0.0);

        double weight_sum = 0.0;
        double union_weight = 0.0;
        for (const Particle& particle : track.particles()) {
            const double weight = particle.weight;
            weight_sum += weight;
            const Eigen::Vector3d position = particle.state_vector.head<3>();
            bool visible_anywhere = false;
            for (size_t s = 0; s < num_sensors; ++s) {
                if (sees_unchecked(sensors_[s], position)) {
                    per_sensor_out[s] += weight;
                    visible_anywhere = true;
                }
            }
            if (visible_anywhere) {
                union_weight += weight;
            }
        }

        if (!(weight_sum > 1e-12)) {
            std::fill(per_sensor_out.begin(), per_sensor_out.end(), 0.0);
            return 0.0;
        }

        const double inv_weight_sum = 1.0 / weight_sum;
        for (double& fraction : per_sensor_out) {
            fraction = clamp_unit(fraction * inv_weight_sum);
        }
        return clamp_unit(union_weight * inv_weight_sum);
    }

private:
    std::vector<Sensor> sensors_;
    std::unordered_map<std::string, size_t> index_;
    SensorFovConfig fov_;
    double tan_half_width_ = std::tan(kDefaultHalfAngle);
    double tan_half_height_ = std::tan(kDefaultHalfAngle);

    //! Fractions are ratios of accumulated weights, so rounding can put them a few ulp outside
    //! [0, 1]. The filter multiplies them by P_D and subtracts from 1, so clamp rather than trust.
    static double clamp_unit(double value) { return std::min(1.0, std::max(0.0, value)); }

    void require_index(size_t index) const {
        if (index >= sensors_.size()) {
            throw std::out_of_range("sensor index " + std::to_string(index) + " out of range for " +
                                    std::to_string(sensors_.size()) + " sensor(s)");
        }
    }

    size_t insert(std::string id, const StateVector& state) {
        if (id.empty()) {
            throw std::invalid_argument("Sensor.id: must not be empty");
        }
        if (index_.find(id) != index_.end()) {
            throw std::invalid_argument("Sensor.id: duplicate id '" + id + "'");
        }
        if (!state.allFinite()) {
            throw std::invalid_argument("Sensor.state: must be finite");
        }
        const size_t index = sensors_.size();
        sensors_.emplace_back();
        sensors_.back().id = id;
        sensors_.back().state = state;
        index_.emplace(std::move(id), index);
        return index;
    }

    static void set_frame(Sensor& sensor, const Eigen::Vector3d& boresight, const Eigen::Vector3d& up) {
        sensor.boresight = boresight;
        sensor.up = up;
        sensor.width = up.cross(boresight);
    }

    /**
     * @brief Install a new unit boresight, deriving the roll.
     *
     * With transport_existing_up the current up vector is parallel-transported onto the new
     * boresight (minimal rotation), then re-orthogonalised; otherwise, and whenever the transport
     * degenerates, the deterministic tangentBasis frame is used.
     */
    static void assign_pointing_from_boresight(Sensor& sensor, const Eigen::Vector3d& boresight,
                                               bool transport_existing_up) {
        if (transport_existing_up) {
            Eigen::Vector3d up = los::parallelTransport(sensor.boresight, boresight, sensor.up);
            up -= up.dot(boresight) * boresight;
            if (up.allFinite() && up.norm() > kDirectionEpsilon) {
                set_frame(sensor, boresight, up.normalized());
                return;
            }
            // A near-antipodal flip leaves nothing to transport; fall through.
        }
        set_frame(sensor, boresight, los::tangentBasis(boresight).col(1));
    }

    bool sees_unchecked(const Sensor& sensor, const Eigen::Vector3d& target_position) const {
        const Eigen::Vector3d offset = target_position - sensor.state.head<3>();
        const double range = offset.norm();
        // Negated comparisons so a NaN position is not visible rather than accidentally visible.
        if (!(range >= fov_.min_range) || !(range <= fov_.max_range)) {
            return false;
        }
        if (!sensor.pointed) {
            return true;
        }
        const double along = offset.dot(sensor.boresight);
        if (!(along > 0.0)) {
            return false;
        }
        return std::abs(offset.dot(sensor.width)) <= along * tan_half_width_ &&
               std::abs(offset.dot(sensor.up)) <= along * tan_half_height_;
    }
};

}  // namespace sensor
