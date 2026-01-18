"""
SMC-LMB Filter Simulation Harness

This script validates the Sequential Monte Carlo Labeled Multi-Bernoulli filter
implementation by simulating a multi-object space debris tracking scenario.

Scenario:
- 3 LEO objects with staggered birth times (steps 0, 30, 50)
- Sensor at Earth's center (mathematical testing configuration)
- 100 time steps at 60-second intervals

The simulation uses a "dual-noise" strategy:
- Truth generation uses low noise (high precision sensor)
- Filter model uses inflated noise (wide acceptance gate for birth convergence)
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Import the C++ engine
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'lmb_engine', 'Release')))
import lmb_engine
print(f"LOADED MODULE FROM: {lmb_engine.__file__}")
print(f"FILE CREATED AT: {os.path.getmtime(lmb_engine.__file__)}")

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

# --- Process Noise Annealing ---
# Exponential decay of process noise as tracks mature
# alpha(age) = NOISE_MIN_SCALE + (1 - NOISE_MIN_SCALE) * exp(-NOISE_DECAY_RATE * age)
NOISE_DECAY_RATE = 0.001  # Decay rate (lambda), per second - noise drops significantly over ~5 mins
NOISE_MIN_SCALE = 0.001  # Minimum scale factor (alpha_min) - steady-state is 0.1% of birth noise

# --- Truth Generation Noise (High Precision) ---
# These represent the actual sensor precision for generating measurements
TRUTH_SIGMA_RANGE = 10.0       # meters
TRUTH_SIGMA_RANGE_RATE = 1.0   # m/s
TRUTH_SIGMA_ANGLE = 1e-6       # radians (~7m at LEO)

# --- Filter Model Noise (Inflated) ---
# These are what the filter "believes" the noise is - inflated for robustness
FILTER_SIGMA_RANGE = 5000.0      # meters
FILTER_SIGMA_RANGE_RATE = 500.0  # m/s
FILTER_SIGMA_ANGLE = 1e-2      # radians (~7km gate at LEO)

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

# Birth model noise (high velocity variance for eccentricity)
Q_BIRTH = np.diag([
    1000.0**2,    # x position variance (m^2)
    1000.0**2,    # y position variance (m^2)
    1000.0**2,    # z position variance (m^2)
    500.0**2,   # vx velocity variance ((m/s)^2) - captures eccentricity
    500.0**2,   # vy velocity variance ((m/s)^2)
    500.0**2    # vz velocity variance ((m/s)^2)
])

# --- Sensor Configuration ---
# Sensor at Earth's center (for mathematical testing)
SENSOR_STATE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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


def generate_measurements(active_truths, sensor_state, current_time):
    """
    Simulate sensor measurements from active ground truth objects.
    
    For each active truth:
    1. Roll for detection (skip if miss based on P_DETECTION)
    2. Convert ECI position to spherical measurement
    3. Add Gaussian noise using TRUTH sigmas
    4. Create Measurement object with FILTER covariance (inflated)
    
    Args:
        active_truths: List of (object_id, state_vector) tuples
        sensor_state: 6D sensor state vector
        current_time: Current simulation time
        
    Returns:
        list: List of lmb_engine.Measurement objects
    """
    measurements = []
    
    for obj_id, truth_state in active_truths:
        # Detection roll
        if np.random.random() > P_DETECTION:
            # Missed detection - skip this object
            continue
        
        # Convert truth state to measurement space (perfect measurement)
        # Returns [range, range_rate, azimuth, elevation]
        perfect_meas = lmb_engine.Measurement.cartesianToMeasurement(
            truth_state, sensor_state
        )
        
        # Add noise using TRUTH sigmas (actual sensor precision)
        noise = np.array([
            np.random.normal(0, TRUTH_SIGMA_RANGE),
            np.random.normal(0, TRUTH_SIGMA_RANGE_RATE),
            np.random.normal(0, TRUTH_SIGMA_ANGLE),
            np.random.normal(0, TRUTH_SIGMA_ANGLE)
        ])
        noisy_meas = perfect_meas + noise
        
        # Create Measurement object
        measurement = lmb_engine.Measurement()
        measurement.timestamp_ = current_time
        measurement.value_ = noisy_meas
        measurement.sensor_state_ = sensor_state
        measurement.sensor_id_ = "sensor_0"
        
        # CRITICAL: Set covariance using FILTER sigmas (inflated)
        # This tells the filter "my data is rough" -> wide acceptance gate
        measurement.covariance_ = np.diag([
            FILTER_SIGMA_RANGE**2,
            FILTER_SIGMA_RANGE_RATE**2,
            FILTER_SIGMA_ANGLE**2,
            FILTER_SIGMA_ANGLE**2
        ])
        
        measurements.append(measurement)
    
    return measurements


def compute_track_mean(track):
    """
    Compute the weighted mean state of a track's particles.
    
    Args:
        track: lmb_engine.Track object
        
    Returns:
        numpy array: 6D mean state vector
    """
    particles = track.particles()
    if len(particles) == 0:
        return np.zeros(6)
    
    states = np.array([p.state_vector for p in particles])
    weights = np.array([p.weight for p in particles])
    
    # Normalize weights (should already sum to 1, but be robust)
    weight_sum = np.sum(weights)
    if weight_sum > 1e-12:
        weights = weights / weight_sum
    else:
        weights = np.ones(len(particles)) / len(particles)
    
    # Weighted average
    mean_state = np.average(states, weights=weights, axis=0)
    return mean_state


# =============================================================================
# MONTE CARLO CONFIGURATION
# =============================================================================

NUM_MONTE_CARLO = 20  # Number of Monte Carlo runs


# =============================================================================
# SINGLE SIMULATION
# =============================================================================

def run_single_simulation(verbose=False):
    """
    Run a single SMC-LMB filter simulation.
    
    This function initializes all models and tracker from scratch,
    runs the full simulation, and returns the OSPA results along with
    component-wise error history for Object 1 (for Figure 3).
    
    Args:
        verbose: If True, print detailed progress information
        
    Returns:
        tuple: (ospa_results, track_error_history)
            - ospa_results: OSPA distance at each time step (length NUM_STEPS)
            - track_error_history: 6D error vectors for Object 1 (shape NUM_STEPS x 6)
    """
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
    
    # Truth propagator (zero noise for deterministic motion)
    truth_propagator = get_ground_truth_propagator()
    
    # Filter propagator (with process noise)
    filter_propagator = lmb_engine.TwoBodyPropagator(Q_FILTER)
    
    # Sensor model (using FILTER variances - inflated)
    sensor_model = lmb_engine.InOrbitSensorModel(
        FILTER_SIGMA_RANGE**2,
        FILTER_SIGMA_RANGE_RATE**2,
        FILTER_SIGMA_ANGLE**2,
        FILTER_SIGMA_ANGLE**2
    )
    
    # Birth model (tangent fan initialization)
    birth_model = lmb_engine.AdaptiveBirthModel(
        NUM_PARTICLES,
        P_BIRTH,
        Q_BIRTH
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
        NOISE_MIN_SCALE
    )
    
    # -------------------------------------------------------------------------
    # Step 2: Initialize Simulation State
    # -------------------------------------------------------------------------
    
    # Active ground truth objects: list of (object_id, state_vector)
    active_ground_truths = []
    
    # Results storage
    ospa_results = []
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
            SENSOR_STATE, 
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
            
        tracker.update(measurements)
        
        # ---------------------------------------------------------------------
        # E. DATA EXTRACTION
        # ---------------------------------------------------------------------
        tracks = tracker.get_tracks()
        
        # Extract truth states for OSPA calculation
        truth_states = [state.copy() for (_, state) in active_ground_truths]
        
        # ---------------------------------------------------------------------
        # E2. TRACK ERROR FOR OBJECT 1 (Figure 3 data)
        # ---------------------------------------------------------------------
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
        if len(truth_states) > 0:
            ospa = lmb_engine.calculate_ospa_distance(
                tracks, 
                truth_states, 
                100000.0  # cutoff distance in meters
            )
        else:
            ospa = 0.0
        
        ospa_results.append(ospa)
        
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
                  f"Meas: {len(measurements)} | OSPA: {ospa:8.1f}m | "
                  f"r=[{prob_str}]")
    
    # -------------------------------------------------------------------------
    # Step 4: Final Results (verbose only)
    # -------------------------------------------------------------------------
    
    if verbose:
        print("-" * 60)
        print("\nFinal Results:")
        print(f"  Final OSPA: {ospa_results[-1]:.1f} m")
        print(f"  Mean OSPA (last 20 steps): {np.mean(ospa_results[-20:]):.1f} m")
        
        # Track-by-track summary
        tracks = tracker.get_tracks()
        print(f"\nTrack Summary:")
        for i, track in enumerate(tracks):
            mean_state = compute_track_mean(track)
            pos_mag = np.linalg.norm(mean_state[:3]) / 1000  # km
            vel_mag = np.linalg.norm(mean_state[3:6]) / 1000  # km/s
            print(f"  Track {i+1}: r={track.existence_probability():.3f}, "
                  f"|pos|={pos_mag:.1f} km, |vel|={vel_mag:.2f} km/s")
    
    return ospa_results, track_error_history


# =============================================================================
# MAIN EXECUTION (MONTE CARLO DRIVER)
# =============================================================================

def main():
    """
    Monte Carlo simulation driver.
    
    Runs NUM_MONTE_CARLO independent simulations and generates:
    - Figure 1: All individual runs overlaid (thin cyan lines)
    - Figure 2: Average performance (thick black line)
    """
    print("=" * 60)
    print("SMC-LMB Monte Carlo Analysis")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Monte Carlo Runs: {NUM_MONTE_CARLO}")
    print(f"  Steps per Run: {NUM_STEPS}, DT: {DT}s")
    print(f"  Particles: {NUM_PARTICLES}")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Phase 1: Run Monte Carlo Simulations
    # -------------------------------------------------------------------------
    
    print("\nRunning Monte Carlo simulations...")
    all_run_data = []
    representative_errors = None  # Store error history from first run for Figure 3
    
    for i in range(NUM_MONTE_CARLO):
        ospa_results, track_error_history = run_single_simulation(verbose=False)
        all_run_data.append(ospa_results)
        
        # Save error history from first run only (for Figure 3)
        if i == 0:
            representative_errors = np.array(track_error_history)
        
        print(f"Run {i+1}/{NUM_MONTE_CARLO} complete - Final OSPA: {ospa_results[-1]:.1f}m")
    
    # -------------------------------------------------------------------------
    # Phase 2: Statistical Calculation
    # -------------------------------------------------------------------------
    
    # Convert to 2D numpy array: shape (NUM_MONTE_CARLO, NUM_STEPS)
    all_run_data = np.array(all_run_data)
    
    # Compute column-wise mean (average OSPA at each time step)
    mean_ospa = np.mean(all_run_data, axis=0)
    
    print("\n" + "-" * 60)
    print("Monte Carlo Statistics:")
    print(f"  Mean Final OSPA: {np.mean(all_run_data[:, -1]):.1f} m")
    print(f"  Std Final OSPA: {np.std(all_run_data[:, -1]):.1f} m")
    print(f"  Mean OSPA (last 20 steps, averaged): {np.mean(mean_ospa[-20:]):.1f} m")
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
    ax1.set_ylabel('OSPA Distance (m)', fontsize=12)
    ax1.set_title(f'OSPA Distance Plot of {NUM_MONTE_CARLO} Runs', fontsize=14)
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
    
    # Plot mean OSPA
    ax2.plot(time_axis, mean_ospa, 
             color='k',  # Black
             linewidth=2.0,
             label=f'Average of {NUM_MONTE_CARLO} Runs')
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Average OSPA Distance (m)', fontsize=12)
    ax2.set_title(f'Average OSPA Performance Across {NUM_MONTE_CARLO} Runs', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, NUM_STEPS - 1])
    ax2.set_ylim([0, np.max(mean_ospa) * 1.1])
    
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
    return all_run_data, mean_ospa


if __name__ == "__main__":
    # No seed - allow OS entropy for true Monte Carlo randomness
    
    # Run Monte Carlo analysis
    all_data, mean_data = main()
