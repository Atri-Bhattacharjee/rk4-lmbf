"""
SMC-LMB Filter Simulation Harness

This script validates the Sequential Monte Carlo Labeled Multi-Bernoulli filter
implementation by simulating a multi-object space debris tracking scenario.

Scenario:
- 3 LEO objects with staggered birth times (steps 0, 30, 50)
- Sensor at Earth's center (mathematical testing configuration)
- 100 time steps at 60-second intervals

Measurements are (range, range rate, line-of-sight unit vector, line-of-sight angular rate) with a
6x6 noise covariance in the local tangent frame of the measured direction (see the constants below).

The simulation uses a "dual-noise" strategy:
- Truth generation uses low noise (high precision sensor)
- Filter model uses inflated noise (wide acceptance gate for birth convergence)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from lmb_engine_loader import import_lmb_engine

lmb_engine = import_lmb_engine(verbose=os.environ.get("LMB_ENGINE_VERBOSE") == "1")

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# --- Simulation Parameters ---
NUM_STEPS = 100          # Total simulation steps
DT = 60.0                # Time step in seconds
NUM_PARTICLES = 10000     # Particles per track

# --- Probability Parameters ---
P_DETECTION = 0.999999999       # Probability of detecting an object
P_SURVIVAL = 0.999999999        # Probability of track survival per step
P_BIRTH = 0.9            # Initial existence probability for new tracks
CLUTTER_INTENSITY = 1e-15 # False alarm rate per unit measurement volume
PRUNE_THRESHOLD = 0.001  # Existence probability threshold for track pruning
K_BEST = 100             # Number of K-best assignment hypotheses

# --- Metric Parameters ---
# GOSPA cutoff c, metres. Read from the engine (src/metrics.h) rather than re-declared here:
# this harness is deliberately independent *code*, but it must not report a differently
# parameterised metric under the same name as the other drivers.
GOSPA_CUTOFF = lmb_engine.GOSPA_DEFAULT_CUTOFF
# An unnormalised GOSPA value is meaningless without its parameters, so labels carry them.
GOSPA_PARAMS = (
    f"unnormalised, c = {GOSPA_CUTOFF / 1000:.0f} km, "
    f"p = {lmb_engine.GOSPA_ORDER_P:.0f}, alpha = {lmb_engine.GOSPA_ALPHA:.0f}"
)

# --- Process Noise Annealing ---
# Exponential decay of process noise as tracks mature
# alpha(age) = NOISE_MIN_SCALE + (1 - NOISE_MIN_SCALE) * exp(-NOISE_DECAY_RATE * age)
NOISE_DECAY_RATE = 0.001  # Decay rate (lambda), per second - noise drops significantly over ~5 mins
NOISE_MIN_SCALE = 0.001  # Minimum scale factor (alpha_min) - steady-state is 0.1% of birth noise

# --- Measurement representation ---
# A measurement is (range, range rate, unit line-of-sight vector, line-of-sight angular-rate vector).
# All measurement noise lives in the local tangent frame of the measured direction, with basis
# (e1, e2) = lmb_engine.tangent_basis(los), ordered as
#   [d_range (m), d_range_rate (m/s), d_theta1 (rad), d_theta2 (rad), d_omega1 (rad/s), d_omega2 (rad/s)]
# There is no azimuth/elevation anywhere in the filter, hence no pole singularity.

# --- Truth Generation Noise (High Precision) ---
# These represent the actual sensor precision for generating measurements
TRUTH_SIGMA_RANGE = 10.0       # meters
TRUTH_SIGMA_RANGE_RATE = 1.0   # m/s
TRUTH_SIGMA_ANGLE = 1e-6       # radians (~7m at LEO)
TRUTH_SIGMA_ANGLE_RATE = 1e-7  # rad/s -- PLACEHOLDER, no sensor characterisation behind this value yet

# --- Filter Model Noise (Inflated) ---
# These are what the filter "believes" the noise is - inflated for robustness
FILTER_SIGMA_RANGE = 5000.0      # meters
FILTER_SIGMA_RANGE_RATE = 500.0  # m/s
FILTER_SIGMA_ANGLE = 1e-2      # radians (~7km gate at LEO)
FILTER_SIGMA_ANGLE_RATE = 1e-3  # rad/s -- PLACEHOLDER (~700 m/s transverse gate at 700 km)

# Six-component sigma vectors in local tangent-frame order
TRUTH_SIGMAS = np.array([
    TRUTH_SIGMA_RANGE,
    TRUTH_SIGMA_RANGE_RATE,
    TRUTH_SIGMA_ANGLE,
    TRUTH_SIGMA_ANGLE,
    TRUTH_SIGMA_ANGLE_RATE,
    TRUTH_SIGMA_ANGLE_RATE,
])
FILTER_SIGMAS = np.array([
    FILTER_SIGMA_RANGE,
    FILTER_SIGMA_RANGE_RATE,
    FILTER_SIGMA_ANGLE,
    FILTER_SIGMA_ANGLE,
    FILTER_SIGMA_ANGLE_RATE,
    FILTER_SIGMA_ANGLE_RATE,
])

# --- Process Noise Covariance ---
# Filter propagator noise (small perturbations)
Q_FILTER = np.diag([
    500.0**2,   # x position variance (m^2)
    500.0**2,   # y position variance (m^2)
    500.0**2,   # z position variance (m^2)
    50.0**2,    # vx velocity variance ((m/s)^2)
    50.0**2,    # vy velocity variance ((m/s)^2)
    50.0**2     # vz velocity variance ((m/s)^2)
])

# --- Birth Covariance (local tangent frame of the measured direction) ---
# New-track particles are the measurement plus Gaussian noise with this covariance, mapped exactly to
# ECI. The spread is therefore range-, angle- and rate-wise, never in ECI x/y/z.
BIRTH_COVARIANCE_LOCAL = np.diag([
    1000.0**2,  # range variance (m^2)
    500.0**2,   # range-rate variance ((m/s)^2)
    1e-4**2,    # angle variance along e1 (rad^2)
    1e-4**2,    # angle variance along e2 (rad^2)
    5e-5**2,    # angular-rate variance along e1 ((rad/s)^2)
    5e-5**2,    # angular-rate variance along e2 ((rad/s)^2)
])

# =============================================================================
# SCENARIO DEFINITION
# =============================================================================

# Physical constants
R_EARTH = 6.371e6        # Earth radius (m)
ALTITUDE = 400e3         # 400 km altitude (typical LEO)
MU_EARTH = 3.986004418e14  # Earth gravitational parameter (m^3/s^2)

# Orbital radius and circular velocity
ORBIT_RADIUS = R_EARTH + ALTITUDE  # ~6771 km
V_CIRCULAR = np.sqrt(MU_EARTH / ORBIT_RADIUS)  # ~7672 m/s

# --- Sensor Configuration ---
# Sensor at Earth's center (for mathematical testing)
SENSOR_STATE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Unbounded in range and, via add_unpointed below, not constrained by a field of view either.
# That is the omniscient sensor this configuration has always assumed, so the paper's numbers are
# unaffected by the sensor rework.
SENSOR_FOV = lmb_engine.SensorFovConfig()

# Define 3 ground truth objects with staggered births
# Format: (object_id, birth_step, initial_state_vector)
# Each initial state is [x, y, z, vx, vy, vz] in meters and m/s

SCENARIO = [
    # Object 1: CPE debris, a=7195.0km, e=0.0168, i=83.0°
    # Born at step 0
    (1, 0, np.array([
        +2.6544665658e+06,  # x (m)
        -1.5306571649e+06,  # y (m)
        -6.5227229294e+06,  # z (m)
        -4.5588669893e+03,  # vx (m/s)
        +5.0941118632e+03,  # vy (m/s)
        -2.9133829034e+03,  # vz (m/s)
    ])),

    # Object 2: CPE debris, a=7189.1km, e=0.0185, i=65.8°
    # Born at step 30
    (2, 30, np.array([
        +3.7638743804e+06,  # x (m)
        -6.2234498679e+05,  # y (m)
        +6.0037549884e+06,  # z (m)
        +5.0461811923e+03,  # vx (m/s)
        +4.9856212029e+03,  # vy (m/s)
        -2.5111625240e+03,  # vz (m/s)
    ])),

    # Object 3: CPE debris, a=7287.5km, e=0.0439, i=65.3°
    # Born at step 50
    (3, 50, np.array([
        -7.1242552406e+06,  # x (m)
        -2.0344233795e+06,  # y (m)
        +1.3982281842e+06,  # z (m)
        -1.3769335820e+02,  # vx (m/s)
        -3.1939023671e+03,  # vy (m/s)
        -6.3886825206e+03,  # vz (m/s)
    ])),

]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_ground_truth_propagator():
    """
    Create a deterministic propagator for ground truth simulation.
    
    Uses a near-zero process noise matrix to ensure smooth, deterministic
    orbital motion without random jitter. A tiny epsilon (1e-18) is used
    on the diagonal to keep the matrix positive definite for Eigen's
    Cholesky decomposition, while being effectively zero for simulation.
    
    Returns:
        TwoBodyPropagator: Propagator with effectively zero process noise
    """
    # Use tiny epsilon instead of pure zeros to avoid Cholesky failure
    # on positive semi-definite matrix (Eigen::LLT requires positive definite)
    epsilon_noise = np.eye(6) * 1e-18
    return lmb_engine.TwoBodyPropagator(epsilon_noise)


def propagate_truth_state(propagator, state_vector, dt):
    """
    Propagate a raw state vector forward in time using the propagator.
    
    Wraps the state vector in a Particle object, propagates it,
    and extracts the resulting state vector.
    
    Args:
        propagator: TwoBodyPropagator instance
        state_vector: 6D numpy array [x, y, z, vx, vy, vz]
        dt: Time step in seconds
        
    Returns:
        numpy array: Propagated 6D state vector
    """
    # Create a particle wrapper
    particle = lmb_engine.Particle()
    particle.state_vector = state_vector.copy()
    particle.weight = 1.0
    
    # Propagate (current_time=0 is fine since two-body doesn't use it)
    propagated = propagator.propagate(particle, dt, 0.0)
    
    # Extract and return the new state
    return np.array(propagated.state_vector)


def build_sensor_array(fov_config=None):
    """The one-sensor array this configuration uses: a single unpointed sensor at SENSOR_STATE."""
    sensors = lmb_engine.SensorArray(SENSOR_FOV if fov_config is None else fov_config)
    sensors.add_unpointed("sensor_0", SENSOR_STATE)
    return sensors


def _as_sensor_array(sensors):
    """Accept a SensorArray, or a bare 6-D sensor state for callers that predate them."""
    if isinstance(sensors, lmb_engine.SensorArray):
        return sensors
    array = lmb_engine.SensorArray(lmb_engine.SensorFovConfig())
    array.add_unpointed("sensor_0", np.asarray(sensors, dtype=np.float64).reshape(6))
    return array


def generate_measurements(active_truths, sensor_state, current_time):
    """
    Simulate sensor measurements from active ground truth objects.
    
    For each active truth:
    1. Roll for detection (skip if miss based on P_DETECTION)
    2. Form the exact (range, range rate, line of sight, line-of-sight rate) observation
    3. Add Gaussian noise using TRUTH sigmas in the local tangent frame of the true direction
    4. Attach the FILTER covariance (inflated) in the same frame and ordering
    
    An object is only reported when some sensor can observe it, and it is reported as a
    measurement of that sensor -- stamped with its id and state, which is what lets the filter
    score the association against the right field of view.
    
    Args:
        active_truths: List of (object_id, state_vector) tuples
        sensor_state: a SensorArray, or a 6D sensor state vector (wrapped into a one-sensor
            unbounded, unpointed array)
        current_time: Current simulation time
        
    Returns:
        list: List of lmb_engine.Measurement objects
    """
    sensors = build_sensor_array() if sensor_state is None else _as_sensor_array(sensor_state)
    measurements = []
    
    for obj_id, truth_state in active_truths:
        # Visibility gate: an object nobody can observe produces nothing, and does not consume a
        # detection roll either. SensorArray.sees is the same predicate the filter applies to the
        # particle clouds, so the two sides cannot disagree about what is observable.
        sensor_index = sensors.visible_sensor(truth_state)
        if sensor_index < 0:
            continue
        
        # Detection roll
        if np.random.random() > P_DETECTION:
            # Missed detection - skip this object
            continue
        
        # Exact observation of the truth, then TRUTH-sigma noise applied in the local tangent frame
        # [d_range, d_range_rate, d_theta1, d_theta2, d_omega1, d_omega2] (sphere exponential map for the
        # direction, parallel transport for the angular rate). This is where the actual sensor precision enters.
        measurement = lmb_engine.Measurement.fromCartesian(
            truth_state, sensors.state(sensor_index)
        ).perturbed(np.random.normal(size=6) * TRUTH_SIGMAS)
        measurement.timestamp_ = current_time
        measurement.sensor_id_ = sensors.id(sensor_index)
        
        # CRITICAL: Set covariance using FILTER sigmas (inflated)
        # This tells the filter "my data is rough" -> wide acceptance gate
        measurement.covariance_ = np.diag(FILTER_SIGMAS**2)
        
        measurements.append(measurement)
    
    return measurements


def compute_track_mean(track):
    """
    Compute the weighted mean state of a track's particles.

    Delegates to Track.mean_state(), which implements the same contract in C++: the weighted mean,
    falling back to the unweighted mean when the total weight is at or below 1e-12, and zeros for an
    empty cloud. The previous NumPy version cost ~13 ms per step for three 10,000-particle tracks,
    almost all of it converting each particle into a Python object.

    Args:
        track: lmb_engine.Track object

    Returns:
        numpy array: 6D mean state vector
    """
    return np.asarray(track.mean_state(), dtype=np.float64).reshape(6)


# =============================================================================
# MONTE CARLO CONFIGURATION
# =============================================================================

# Number of Monte Carlo runs; override with the LMB_NUM_RUNS environment variable (e.g. for smoke tests)
NUM_MONTE_CARLO = int(os.environ.get("LMB_NUM_RUNS", "20"))


# =============================================================================
# SINGLE SIMULATION
# =============================================================================

def run_single_simulation(verbose=False, collect_track_errors=True, seed=None):
    """
    Run a single SMC-LMB filter simulation.
    
    This function initializes all models and tracker from scratch,
    runs the full simulation, and returns the GOSPA results along with
    component-wise error history for Object 1 (for Figure 3).
    
    Args:
        verbose: If True, print detailed progress information
        collect_track_errors: When False, skip Object-1 component-error work.
        seed: Optional integer. When set, seeds NumPy and the filter/birth/resampler RNGs.
        
    Returns:
        tuple: (gospa_results, track_error_history)
            - gospa_results: GOSPA distance at each time step (length NUM_STEPS)
            - track_error_history: 6D error vectors for Object 1 (shape NUM_STEPS x 6)
    """
    if seed is not None:
        np.random.seed(seed)

    if verbose:
        print("=" * 60)
        print("SMC-LMB Filter Validation Simulation")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Steps: {NUM_STEPS}, DT: {DT}s, Particles: {NUM_PARTICLES}")
        print(f"  P_D: {P_DETECTION}, P_S: {P_SURVIVAL}, P_B: {P_BIRTH}")
        print(f"  Clutter: {CLUTTER_INTENSITY}, K-best: {K_BEST}")
        print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 1: Initialize Models and Tracker
    # -------------------------------------------------------------------------
    
    # Truth propagator (zero noise for deterministic motion). When seed is set, pin it too:
    # the epsilon noise matrix still has a positive trace, so an unseeded instance draws from
    # random_device and breaks reproducibility.
    if seed is None:
        truth_propagator = get_ground_truth_propagator()
    else:
        truth_propagator = lmb_engine.TwoBodyPropagator(np.eye(6) * 1e-18, seed=seed)
    
    # Filter propagator (with process noise)
    filter_propagator = lmb_engine.TwoBodyPropagator(Q_FILTER, seed=seed)
    
    # Sensor model (using FILTER variances - inflated). Measurement.covariance_ is authoritative for the
    # likelihood; these six variances define the model's defaultCovariance() in the same frame/order.
    sensor_model = lmb_engine.InOrbitSensorModel(*FILTER_SIGMAS**2)
    
    # Birth model: measurement + local tangent-frame Gaussian noise, mapped exactly to ECI
    birth_model = lmb_engine.AdaptiveBirthModel(
        NUM_PARTICLES,
        P_BIRTH,
        BIRTH_COVARIANCE_LOCAL,
        seed=seed,
    )
    
    # Main tracker
    tracker = lmb_engine.SMC_LMB_Tracker(
        filter_propagator,
        sensor_model,
        birth_model,
        P_SURVIVAL,
        K_BEST,
        PRUNE_THRESHOLD,
        CLUTTER_INTENSITY,
        P_DETECTION,
        NOISE_DECAY_RATE,
        NOISE_MIN_SCALE,
        seed=seed,
    )
    
    # -------------------------------------------------------------------------
    # Step 2: Initialize Simulation State
    # -------------------------------------------------------------------------
    
    # Active ground truth objects: list of (object_id, state_vector)
    active_ground_truths = []
    sensors = build_sensor_array()
    
    # Results storage
    gospa_results = []
    track_error_history = []  # 6D error vectors for Object 1 (for Figure 3)
    
    # -------------------------------------------------------------------------
    # Step 3: Simulation Loop
    # -------------------------------------------------------------------------
    
    if verbose:
        print("\nSimulation Progress:")
        print("-" * 60)
    
    for step in range(NUM_STEPS):
        current_time = step * DT
        
        # ---------------------------------------------------------------------
        # A. PROPAGATE EXISTING ground truths (move from t-DT to t)
        # ---------------------------------------------------------------------
        if step > 0:
            for i in range(len(active_ground_truths)):
                obj_id, state = active_ground_truths[i]
                new_state = propagate_truth_state(truth_propagator, state, DT)
                active_ground_truths[i] = (obj_id, new_state)
        
        # ---------------------------------------------------------------------
        # B. BIRTH CHECK (add new objects at their t=current_time position)
        # ---------------------------------------------------------------------
        for obj_id, birth_step, initial_state in SCENARIO:
            if step == birth_step:
                active_ground_truths.append((obj_id, initial_state.copy()))
                if verbose:
                    print(f"  [Step {step:3d}] Object {obj_id} BORN at t={current_time:.0f}s")
        
        # ---------------------------------------------------------------------
        # C. GENERATE MEASUREMENTS
        # ---------------------------------------------------------------------
        measurements = generate_measurements(
            active_ground_truths, 
            sensors, 
            current_time
        )
        
        # ---------------------------------------------------------------------
        # D. FILTER CYCLE: Predict then Update
        # ---------------------------------------------------------------------
        # CRITICAL FIX: The filter starts at t=0. The first measurement is at t=0.
        # We must NOT predict forward on the very first step, or the filter 
        # state (t=60) will desynchronize from the measurement (t=0).
        if step > 0:
            tracker.predict(DT)
            
        tracker.update(measurements, sensors)
        
        # ---------------------------------------------------------------------
        # E. DATA EXTRACTION
        # ---------------------------------------------------------------------
        tracks = tracker.get_tracks()
        
        # Extract truth states for GOSPA calculation
        truth_states = [state.copy() for (_, state) in active_ground_truths]
        
        # ---------------------------------------------------------------------
        # E2. TRACK ERROR FOR OBJECT 1 (Figure 3 data)
        # ---------------------------------------------------------------------
        if collect_track_errors:
            # Find Truth Object 1
            truth_obj1_state = None
            for obj_id, state in active_ground_truths:
                if obj_id == 1:
                    truth_obj1_state = state
                    break
            
            if truth_obj1_state is not None and len(tracks) > 0:
                # Find the closest track to Truth Object 1 (by position distance)
                min_dist = float('inf')
                best_track = None
                for track in tracks:
                    track_mean = compute_track_mean(track)
                    pos_dist = np.linalg.norm(track_mean[:3] - truth_obj1_state[:3])
                    if pos_dist < min_dist:
                        min_dist = pos_dist
                        best_track = track_mean
                
                # Compute component-wise error: Track - Truth
                error_vector = best_track - truth_obj1_state
                track_error_history.append(error_vector)
            else:
                # Object 1 doesn't exist yet or no tracks - append zeros
                track_error_history.append(np.zeros(6))
        
        # ---------------------------------------------------------------------
        # F. METRIC CALCULATION
        # ---------------------------------------------------------------------
        # No "if truth_states" guard: with GOSPA, m tracks against zero truths is c*sqrt(m/2) of
        # false-track cost, not zero. The engine handles n == 0 by construction.
        gospa = lmb_engine.calculate_gospa_distance(
            tracks,
            truth_states,
            GOSPA_CUTOFF,
        )
        
        gospa_results.append(gospa)
        
        # ---------------------------------------------------------------------
        # G. LOGGING (every 10 steps + birth events)
        # ---------------------------------------------------------------------
        if verbose and (step % 10 == 0 or step in [0, 30, 50]):
            # Get track existence probabilities
            track_probs = [t.existence_probability() for t in tracks]
            prob_str = ", ".join([f"{p:.2f}" for p in track_probs[:5]])  # Show first 5
            if len(track_probs) > 5:
                prob_str += ", ..."
            
            print(f"  [Step {step:3d}] t={current_time:6.0f}s | "
                  f"Tracks: {len(tracks):2d} | Truths: {len(active_ground_truths)} | "
                  f"Meas: {len(measurements)} | GOSPA: {gospa:8.1f}m | "
                  f"r=[{prob_str}]")
    
    # -------------------------------------------------------------------------
    # Step 4: Final Results (verbose only)
    # -------------------------------------------------------------------------
    
    if verbose:
        print("-" * 60)
        print("\nFinal Results:")
        print(f"  Final GOSPA: {gospa_results[-1]:.1f} m")
        print(f"  Mean GOSPA (last 20 steps): {np.mean(gospa_results[-20:]):.1f} m")
        
        # Track-by-track summary
        tracks = tracker.get_tracks()
        print(f"\nTrack Summary:")
        for i, track in enumerate(tracks):
            mean_state = compute_track_mean(track)
            pos_mag = np.linalg.norm(mean_state[:3]) / 1000  # km
            vel_mag = np.linalg.norm(mean_state[3:6]) / 1000  # km/s
            print(f"  Track {i+1}: r={track.existence_probability():.3f}, "
                  f"|pos|={pos_mag:.1f} km, |vel|={vel_mag:.2f} km/s")
    
    return gospa_results, track_error_history


# =============================================================================
# MAIN EXECUTION (MONTE CARLO DRIVER)
# =============================================================================

def _ieee_derive_run_seeds(master_seed, num_runs):
    children = np.random.SeedSequence(int(master_seed)).spawn(int(num_runs))
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def _ieee_resolve_max_workers():
    env = os.environ.get("LMB_NUM_WORKERS")
    if env is not None and str(env).strip() != "":
        return max(1, int(env))
    return max(1, int(os.cpu_count() or 1))


def _ieee_monte_carlo_worker(payload):
    """Top-level worker for ProcessPoolExecutor (picklable under spawn)."""
    run_index, seed = payload
    gospa, errors = run_single_simulation(
        verbose=False,
        collect_track_errors=(run_index == 0),
        seed=int(seed),
    )
    gospa_arr = np.asarray(gospa, dtype=np.float64)
    err_arr = np.asarray(errors, dtype=np.float64) if run_index == 0 else None
    return int(run_index), gospa_arr, err_arr


def main():
    """
    Monte Carlo simulation driver.
    
    Runs NUM_MONTE_CARLO independent simulations and generates:
    - Figure 1: All individual runs overlaid (thin cyan lines)
    - Figure 2: Average performance (thick black line)
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    master_env = os.environ.get("LMB_MC_SEED")
    if master_env is not None and str(master_env).strip() != "":
        master_seed = int(master_env)
    else:
        master_seed = int(np.random.SeedSequence().entropy)
    max_workers = _ieee_resolve_max_workers()
    run_seeds = _ieee_derive_run_seeds(master_seed, NUM_MONTE_CARLO)

    print("=" * 60)
    print("SMC-LMB Monte Carlo Analysis")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Monte Carlo Runs: {NUM_MONTE_CARLO}")
    print(f"  Steps per Run: {NUM_STEPS}, DT: {DT}s")
    print(f"  Particles: {NUM_PARTICLES}")
    print(f"  Master seed: {master_seed}")
    print(f"  Workers: {max_workers}" + (" (serial)" if max_workers == 1 else ""))
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Phase 1: Run Monte Carlo Simulations
    # -------------------------------------------------------------------------
    
    print("\nRunning Monte Carlo simulations...")
    results_by_index = [None] * NUM_MONTE_CARLO
    representative_errors = None
    finished = 0
    payloads = [(i, run_seeds[i]) for i in range(NUM_MONTE_CARLO)]

    def _store(run_index, gospa_results, track_error_history):
        nonlocal representative_errors, finished
        results_by_index[run_index] = gospa_results
        if run_index == 0:
            representative_errors = track_error_history
        finished += 1
        print(
            f"Run {run_index + 1}/{NUM_MONTE_CARLO} complete - Final GOSPA: {gospa_results[-1]:.1f}m "
            f"({finished}/{NUM_MONTE_CARLO} finished)"
        )

    if max_workers == 1:
        for payload in payloads:
            run_index, gospa_results, track_error_history = _ieee_monte_carlo_worker(payload)
            _store(run_index, gospa_results, track_error_history)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(_ieee_monte_carlo_worker, payload): payload[0]
                for payload in payloads
            }
            for future in as_completed(futures):
                run_index, gospa_results, track_error_history = future.result()
                _store(run_index, gospa_results, track_error_history)

    # Convert to 2D numpy array: shape (NUM_MONTE_CARLO, NUM_STEPS), ordered by run index
    all_run_data = np.stack(results_by_index, axis=0)
    
    # -------------------------------------------------------------------------
    # Phase 2: Statistical Calculation
    # -------------------------------------------------------------------------
    
    # Compute column-wise mean (average GOSPA at each time step)
    mean_gospa = np.mean(all_run_data, axis=0)
    
    print("\n" + "-" * 60)
    print("Monte Carlo Statistics:")
    print(f"  Mean Final GOSPA: {np.mean(all_run_data[:, -1]):.1f} m")
    print(f"  Std Final GOSPA: {np.std(all_run_data[:, -1]):.1f} m")
    print(f"  Mean GOSPA (last 20 steps, averaged): {np.mean(mean_gospa[-20:]):.1f} m")
    print("-" * 60)
    
    # -------------------------------------------------------------------------
    # Phase 3: Figure 1 - Individual Runs
    # -------------------------------------------------------------------------
    
    print("\nGenerating Figure 1 (Individual Runs)...")
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # X-axis: Time steps (0 to NUM_STEPS-1)
    time_axis = np.arange(NUM_STEPS)
    
    # Plot all individual runs
    for i, run_data in enumerate(all_run_data):
        # Only label the first run (legend trick)
        label = "Individual Runs" if i == 0 else None
        ax1.plot(time_axis, run_data, 
                 color='#00CED1',  # Cyan/Dark Turquoise
                 linewidth=0.5, 
                 alpha=0.4,
                 label=label)
    
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('GOSPA (m)', fontsize=12)
    ax1.set_title(f'GOSPA over {NUM_MONTE_CARLO} Runs\n{GOSPA_PARAMS}', fontsize=13)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, NUM_STEPS - 1])
    ax1.set_ylim([0, np.max(all_run_data) * 1.1])
    
    plt.tight_layout()
    
    # Save Figure 1
    output_path_1 = os.path.join(os.path.dirname(__file__), 'figure_1_individual_runs.png')
    plt.savefig(output_path_1, dpi=150)
    print(f"  Saved: {output_path_1}")
    
    # -------------------------------------------------------------------------
    # Phase 4: Figure 2 - Average Performance
    # -------------------------------------------------------------------------
    
    print("Generating Figure 2 (Average Performance)...")
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Plot mean GOSPA
    ax2.plot(time_axis, mean_gospa, 
             color='k',  # Black
             linewidth=2.0,
             label=f'Average of {NUM_MONTE_CARLO} Runs')
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Average GOSPA (m)', fontsize=12)
    ax2.set_title(f'Average GOSPA Across {NUM_MONTE_CARLO} Runs\n{GOSPA_PARAMS}', fontsize=13)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, NUM_STEPS - 1])
    ax2.set_ylim([0, np.max(mean_gospa) * 1.1])
    
    plt.tight_layout()
    
    # Save Figure 2
    output_path_2 = os.path.join(os.path.dirname(__file__), 'figure_2_average_performance.png')
    plt.savefig(output_path_2, dpi=150)
    print(f"  Saved: {output_path_2}")
    
    # -------------------------------------------------------------------------
    # Phase 5: Figure 3 - Component Error for Object 1
    # -------------------------------------------------------------------------
    
    print("Generating Figure 3 (Component Error for Object 1)...")
    
    fig3, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Component labels and units
    component_titles = [
        "X Error (m)", "Y Error (m)", "Z Error (m)",
        "Vx Error (m/s)", "Vy Error (m/s)", "Vz Error (m/s)"
    ]
    
    # Plot each component in its subplot
    for idx in range(6):
        row = idx // 3  # 0 for position (0,1,2), 1 for velocity (3,4,5)
        col = idx % 3   # 0, 1, 2
        ax = axes[row, col]
        
        # Plot error data (blue solid line)
        ax.plot(time_axis, representative_errors[:, idx],
                color='b', linewidth=1.0, label='Error')
        
        # Plot zero reference (red dashed line)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1.0, alpha=0.7)
        
        # Formatting
        ax.set_title(component_titles[idx], fontsize=12)
        ax.set_xlabel('Time Step', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Overall figure title
    fig3.suptitle('Filter State Component Error for Object 1 (Representative Run)', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save Figure 3
    output_path_3 = os.path.join(os.path.dirname(__file__), 'figure_3_component_error.png')
    plt.savefig(output_path_3, dpi=150)
    print(f"  Saved: {output_path_3}")
    
    # -------------------------------------------------------------------------
    # Show plots
    # -------------------------------------------------------------------------
    
    plt.show()
    
    print("\nMonte Carlo analysis complete.")
    return all_run_data, mean_gospa


if __name__ == "__main__":
    # Per-run seeds are derived from LMB_MC_SEED (or a one-shot entropy master).
    all_data, mean_data = main()
