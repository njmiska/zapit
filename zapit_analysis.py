#!/usr/bin/env python3
"""
Zapit Optogenetics Analysis Pipeline

Analyzes behavioral data from the IBL task with laser scanning optogenetic
inhibition (Zapit system). Mice perform a 2-AFC task with prior blocks while
receiving bilateral optogenetic stimulation at one of 52 cortical locations.

Key analyses:
- Reaction time effects by stimulation location
- Bias shift effects (block-induced choice bias)

Usage:
    1. Configure paths, options, and session filters in config.py
    2. Update session metadata in metadata_zapit.py
    3. Run this script: python zapit_analysis.py

Author: Nate Miska
        Developed with AI pair-programming assistance (Claude, Anthropic)
        for code refactoring and documentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from datetime import datetime
from scipy import stats
from one.api import ONE

# Local imports
from config import (
    ZAPIT_TRIALS_LOG, ZAPIT_LOCATIONS_LOG,
    ALLEN_CCF_ANNOTATION, ALLEN_STRUCTURE_TREE,
    FIGURE_SAVE_PATH, FIGURE_PREFIX, SAVE_FIGURES,
    BASELINE_PERFORMANCE_THRESHOLD, STIM_PERFORMANCE_THRESHOLD,
    MIN_BIAS_THRESHOLD_ZAPIT, RT_THRESHOLD, MIN_NUM_TRIALS,
    NUM_STIM_LOCATIONS, TRIALS_PER_DATAPOINT,
    USE_TRIALS_AFTER_STIM,
    WHEEL_ANALYSIS_DURATION, WHEEL_TIME_INTERVAL, WHEEL_ALIGN_TO,
    ONLY_LOW_CONTRASTS, LOW_CONTRAST_THRESHOLD,
    ALL_CONTRASTS, LOW_CONTRASTS, BIAS_HEATMAP_CONTRASTS,
    ALYX_BASE_URL, FLAG_FAILED_LOADS,
    SESSION_FILTERS,
    BREGMA_CCF, CCF_SCALE_FACTOR,
    PLOT_AP_LIMITS, PLOT_ML_LIMITS,
    SHOW_RIGHT_HEMISPHERE_ONLY,
)
from metadata_zapit import sessions, find_sessions_by_advanced_criteria
from zapit_helpers import (
    load_session_data, load_laser_intervals, parse_zapit_log,
    build_stim_location_dict, load_stim_locations_coordinates,
    signed_contrast, create_trial_data_dict, get_valid_trials_range,
    check_session_accuracy, compute_session_bias_shift,
    extract_wheel_trajectory,
    compute_bias_values_by_contrast, compute_bias_values_by_cycle,
    compute_effect_sizes,
    run_condition_comparisons, run_cycle_comparisons, run_rt_analysis,
    generate_mip_with_borders,
)


# =============================================================================
# SESSION SELECTION (configured in config.py)
# =============================================================================

# Filter out None values from session filters
active_filters = {k: v for k, v in SESSION_FILTERS.items() if v is not None}

eids, trials_ranges, MouseIDs, stim_params = find_sessions_by_advanced_criteria(
    sessions, **active_filters
)

print(f"Found {len(eids)} sessions matching criteria")
if active_filters:
    print(f"Active filters: {active_filters}")


# =============================================================================
# INITIALIZE DATA STRUCTURES
# =============================================================================

# Connect to IBL database
one = ONE(base_url=ALYX_BASE_URL)

# Trial data organized by condition (0 = control, 1-52 = stim locations)
condition_data = {i: [] for i in range(NUM_STIM_LOCATIONS + 1)}

# Wheel data by condition and block
Rblock_wheel_by_condition = [[] for _ in range(NUM_STIM_LOCATIONS + 1)]
Lblock_wheel_by_condition = [[] for _ in range(NUM_STIM_LOCATIONS + 1)]

# Session tracking
num_analyzed_sessions = 0
num_unique_mice = 0
previous_mouse_id = None


# =============================================================================
# MAIN SESSION LOOP
# =============================================================================

print("\n" + "="*60)
print("Processing sessions...")
print("="*60 + "\n")

for j, eid in enumerate(eids):
    print(f"Processing session {j+1}/{len(eids)}: {eid}")
    
    # -------------------------------------------------------------------------
    # Load session data
    # -------------------------------------------------------------------------
    trials, wheel = load_session_data(one, eid, flag_failed_loads=FLAG_FAILED_LOADS)
    if trials is None:
        continue
    
    laser_intervals = load_laser_intervals(one, eid)
    current_mouse_id = MouseIDs[j]
    
    # -------------------------------------------------------------------------
    # Determine valid trial range
    # -------------------------------------------------------------------------
    num_trials = len(trials['contrastLeft'])
    trials_range = get_valid_trials_range(trials_ranges[j], num_trials)
    
    if len(trials_range) < MIN_NUM_TRIALS:
        print(f"  Skipping: Only {len(trials_range)} trials (minimum: {MIN_NUM_TRIALS})")
        continue
    
    # -------------------------------------------------------------------------
    # Parse Zapit log and build stim location mapping
    # -------------------------------------------------------------------------
    details = one.get_details(eid)
    session_start = datetime.strptime(details['start_time'][:19], '%Y-%m-%dT%H:%M:%S')
    
    relevant_events = parse_zapit_log(ZAPIT_TRIALS_LOG, session_start, eid)
    
    stimtrial_location_dict = build_stim_location_dict(
        laser_intervals, trials, relevant_events, trials_range, eid
    )
    
    # Handle "trials after stim" analysis mode
    if USE_TRIALS_AFTER_STIM:
        stimtrial_location_dict_original = stimtrial_location_dict.copy()
        stimtrial_location_dict = {
            k + 1: v for k, v in stimtrial_location_dict.items()
            if min(trials_range) <= k < max(trials_range)
        }
    
    # -------------------------------------------------------------------------
    # Session quality check: accuracy at high contrasts
    # -------------------------------------------------------------------------
    # Get control trials for quality check
    control_trial_nums = [t for t, c in stimtrial_location_dict.items() if c == 0]
    
    contrast_values = signed_contrast(trials)
    session_control_data = []
    
    for trial_num in control_trial_nums:
        trial_data = create_trial_data_dict(trials, trial_num, contrast_values)
        if trial_data['reaction_times'] <= RT_THRESHOLD and not np.isnan(trial_data['reaction_times']):
            session_control_data.append(trial_data)
    
    passes_accuracy, accuracy = check_session_accuracy(
        session_control_data, BASELINE_PERFORMANCE_THRESHOLD
    )
    if not passes_accuracy:
        print(f"  Skipping: Accuracy {accuracy:.2f} below threshold {BASELINE_PERFORMANCE_THRESHOLD}")
        continue
    
    # -------------------------------------------------------------------------
    # Session quality check: bias shift
    # -------------------------------------------------------------------------
    total_bias_shift = compute_session_bias_shift(session_control_data)
    print(f"  Accuracy: {accuracy:.2f}, Bias shift: {total_bias_shift:.2f}")
    
    if total_bias_shift < MIN_BIAS_THRESHOLD_ZAPIT:
        print(f"  Skipping: Bias shift below threshold {MIN_BIAS_THRESHOLD_ZAPIT}")
        continue
    
    # -------------------------------------------------------------------------
    # Collect trial data for all conditions
    # -------------------------------------------------------------------------
    whlpos, whlt = wheel.position, wheel.timestamps
    
    for trial_num, condition_num in stimtrial_location_dict.items():
        # Skip trials that don't meet RT criteria
        trial_data = create_trial_data_dict(trials, trial_num, contrast_values)
        
        if np.isnan(trial_data['reaction_times']):
            continue
        if trial_data['reaction_times'] > RT_THRESHOLD:
            continue
        
        # Optional: filter for low contrasts only
        if ONLY_LOW_CONTRASTS:
            if abs(trial_data['contrast']) > LOW_CONTRAST_THRESHOLD:
                continue
        
        # Add to condition data
        condition_data[condition_num].append(trial_data)
        
        # Extract wheel trajectory
        if wheel is not None:
            trajectory = extract_wheel_trajectory(
                wheel, trials, trial_num,
                align_to=WHEEL_ALIGN_TO,
                duration=WHEEL_ANALYSIS_DURATION,
                interval=WHEEL_TIME_INTERVAL
            )
            
            # Store by block type
            if trial_data['probabilityLeft'] == 0.2:  # R block
                if len(Rblock_wheel_by_condition[condition_num]) == 0:
                    Rblock_wheel_by_condition[condition_num] = trajectory.reshape(1, -1)
                else:
                    Rblock_wheel_by_condition[condition_num] = np.vstack([
                        Rblock_wheel_by_condition[condition_num], trajectory
                    ])
            else:  # L block
                if len(Lblock_wheel_by_condition[condition_num]) == 0:
                    Lblock_wheel_by_condition[condition_num] = trajectory.reshape(1, -1)
                else:
                    Lblock_wheel_by_condition[condition_num] = np.vstack([
                        Lblock_wheel_by_condition[condition_num], trajectory
                    ])
    
    # Update session counters
    num_analyzed_sessions += 1
    if current_mouse_id != previous_mouse_id:
        num_unique_mice += 1
        previous_mouse_id = current_mouse_id
    
    print(f"  ✓ Session added (Control: {len(condition_data[0])} trials)")


# =============================================================================
# CHECK THAT WE HAVE DATA
# =============================================================================

if num_analyzed_sessions == 0:
    raise RuntimeError("No sessions met criteria for analysis!")

print("\n" + "="*60)
print(f"Analysis complete: {num_analyzed_sessions} sessions, {num_unique_mice} mice")
print(f"Total control trials: {len(condition_data[0])}")
print(f"Total stim trials: {sum(len(condition_data[c]) for c in range(1, 53))}")
print("="*60 + "\n")


# =============================================================================
# COMPUTE BIAS METRICS
# =============================================================================

print("Computing bias metrics...")

# Bias by contrast level (for paired comparisons)
contrasts_to_use = LOW_CONTRASTS if ONLY_LOW_CONTRASTS else ALL_CONTRASTS
bias_values, left_block_probs, right_block_probs = compute_bias_values_by_contrast(
    condition_data, contrasts_to_use, NUM_STIM_LOCATIONS + 1
)

# Bias by trial cycles (for independent comparisons)
bias_vals_LC = compute_bias_values_by_cycle(
    condition_data, 
    trials_per_cycle=TRIALS_PER_DATAPOINT,
    low_contrast_threshold=LOW_CONTRAST_THRESHOLD,
    num_conditions=NUM_STIM_LOCATIONS + 1
)

# Effect sizes
effect_sizes = compute_effect_sizes(bias_values, only_low_contrasts=ONLY_LOW_CONTRASTS)
effect_sizes_LC = {}
ctrl_mean = np.mean(bias_vals_LC[0]) if len(bias_vals_LC[0]) > 0 else 0
for cond in range(1, 53):
    if len(bias_vals_LC[cond]) > 0 and ctrl_mean != 0:
        stim_mean = np.mean(bias_vals_LC[cond])
        effect_sizes_LC[cond] = -(stim_mean - ctrl_mean) / ctrl_mean


# =============================================================================
# STATISTICAL COMPARISONS
# =============================================================================

print("Running statistical tests...")

# Contrast-based comparisons (Mann-Whitney, paired t-test)
contrast_stats = run_condition_comparisons(bias_values, NUM_STIM_LOCATIONS + 1)

# Cycle-based comparisons (independent t-test)
cycle_stats = run_cycle_comparisons(bias_vals_LC, NUM_STIM_LOCATIONS + 1)

# RT analysis
rt_results, qp_results, lapse_results = run_rt_analysis(
    condition_data, NUM_STIM_LOCATIONS
)

print(f"  Kruskal-Wallis (contrasts): p = {contrast_stats.get('kruskal_p', 'N/A'):.4g}")
print(f"  Kruskal-Wallis (cycles): p = {cycle_stats.get('kruskal_p', 'N/A'):.4g}")


# =============================================================================
# LOAD STIM LOCATION COORDINATES
# =============================================================================

stim_locations = load_stim_locations_coordinates(ZAPIT_LOCATIONS_LOG)


# =============================================================================
# PLOTTING
# =============================================================================

print("\nGenerating figures...")

# Set up figure directory
FIGURE_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Load Allen CCF
print("  Loading Allen CCF data...")
allenCCF_data = np.load(ALLEN_CCF_ANNOTATION)
structure_tree = pd.read_csv(ALLEN_STRUCTURE_TREE)

# Generate brain surface image with borders
print("  Generating max intensity projection (this may take a moment)...")
mip, edges = generate_mip_with_borders(allenCCF_data)

# Create binary border image (white borders on black)
dorsal_mip_with_borders = np.where(edges > 0, 1, 0)

# Calculate the extents using original method (negative scale factor handles flip)
bregma = BREGMA_CCF  # [AP, DV, ML]
scale_factor = CCF_SCALE_FACTOR  # -100

left_extent = -bregma[2] / scale_factor
right_extent = (dorsal_mip_with_borders.shape[1] - bregma[2]) / scale_factor
lower_extent = (dorsal_mip_with_borders.shape[0] - bregma[0]) / scale_factor
upper_extent = -bregma[0] / scale_factor

extent = [left_extent, right_extent, lower_extent, upper_extent]


# -----------------------------------------------------------------------------
# Figure 1: RT Heatmap on Brain Atlas
# -----------------------------------------------------------------------------
print("  Generating RT heatmap...")

fig, ax = plt.subplots(figsize=(10, 8))

# Show brain borders
ax.imshow(dorsal_mip_with_borders, cmap='gray', extent=extent)

# Compute mean RT and p-values for each condition
rt_means = {}
rt_pvals = {}
ctrl_rts = [t['reaction_times'] for t in condition_data[0] if not np.isnan(t['reaction_times'])]
ctrl_mean_rt = np.mean(ctrl_rts) if ctrl_rts else 1.0

for cond in range(1, NUM_STIM_LOCATIONS + 1):
    stim_rts = [t['reaction_times'] for t in condition_data[cond] if not np.isnan(t['reaction_times'])]
    if len(stim_rts) > 0:
        rt_means[cond] = np.mean(stim_rts)
        if len(ctrl_rts) > 0:
            _, p = stats.ttest_ind(stim_rts, ctrl_rts, equal_var=False)
            rt_pvals[cond] = p
        else:
            rt_pvals[cond] = 1.0

# Set up colormap - using TwoSlopeNorm centered on control RT (matching original)
from matplotlib import colors
divnorm = colors.TwoSlopeNorm(
    vmin=0.8*(ctrl_mean_rt),#ctrl_mean_rt - 0.4 * ctrl_mean_rt,
    vcenter=ctrl_mean_rt,
    vmax=1.5*(ctrl_mean_rt)
)
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=divnorm)

# Plot each stim location (both hemispheres, control display with axis limits)
for condition, coords in stim_locations.items():
    if condition not in rt_means:
        continue
    
    mean_rt = rt_means[condition]
    p_val = rt_pvals.get(condition, 1.0)
    
    # Size based on -100 * log10(p_val) - matching original
    size = -100 * np.log10(p_val) if p_val > 0 else 300
    color = sm.to_rgba(mean_rt)
    alpha = 0.5 if p_val >= 0.05 else 1.0
    
    ml_left = coords.get('ML_left')
    ml_right = coords.get('ML_right')
    ap = coords.get('AP')
    
    if ml_left is not None and ml_right is not None and ap is not None:
        # Plot both hemispheres (axis limits will control what's visible)
        ax.scatter(ml_left, ap, color=color, alpha=alpha, edgecolors='w', s=size)
        ax.scatter(ml_right, ap, color=color, alpha=alpha, edgecolors='w', s=size)

# Set axis limits
if PLOT_AP_LIMITS is not None:
    ax.set_ylim(bottom=PLOT_AP_LIMITS[0], top=PLOT_AP_LIMITS[1])
if PLOT_ML_LIMITS is not None:
    ax.set_xlim(left=PLOT_ML_LIMITS[0], right=PLOT_ML_LIMITS[1])
elif SHOW_RIGHT_HEMISPHERE_ONLY:
    ax.set_xlim(left=0)  # Show only right hemisphere (ML > 0)

# Labels and title
ax.set_xlabel('Mediolateral Position (mm from Bregma)', fontsize=14)
ax.set_ylabel('Anteroposterior Position (mm from Bregma)', fontsize=14)
ax.set_title(f'Reaction Time by Stimulation Location\n({num_analyzed_sessions} sessions, {num_unique_mice} mice)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)

# P-value legend (matching original style)
p_values_legend = [10e-8, 0.001, 0.05]
sizes_legend = [-100 * np.log10(p) for p in p_values_legend]
for p, size in zip(p_values_legend, sizes_legend):
    ax.scatter([], [], s=size, label=f'p = {p}', edgecolors='w', color='black')
ax.legend(loc='upper right', labelspacing=1.5, fontsize=10)

# Colorbar
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04, aspect=12)
cbar.set_label('Reaction time (s)', fontsize=14, labelpad=15)
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_RT_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
else:
    plt.show()


# # -----------------------------------------------------------------------------
# # Figure 2: QP (Quiescent Period) Heatmap on Brain Atlas
# # -----------------------------------------------------------------------------
# print("  Generating QP heatmap...")

# fig, ax = plt.subplots(figsize=(10, 8))

# # Show brain borders
# ax.imshow(dorsal_mip_with_borders, cmap='gray', extent=extent)

# # Compute mean QP and p-values for each condition (matching original script logic)
# qp_means = {}
# qp_pvals = {}
# ctrl_qps = [t['qp_times'] for t in condition_data[0] if not np.isnan(t['qp_times'])]
# ctrl_mean_qp = np.mean(ctrl_qps) if ctrl_qps else 0.5

# for cond in range(1, NUM_STIM_LOCATIONS + 1):
#     stim_qps = [t['qp_times'] for t in condition_data[cond] if not np.isnan(t['qp_times'])]
#     if len(stim_qps) > 0:
#         qp_means[cond] = np.mean(stim_qps)
#         if len(ctrl_qps) > 0:
#             _, p = stats.ttest_ind(stim_qps, ctrl_qps, equal_var=False)
#             qp_pvals[cond] = p
#         else:
#             qp_pvals[cond] = 1.0

# # Set up colormap - matching original QP normalization
# norm_qp = mcolors.Normalize(
#     vmin=ctrl_mean_qp - 0.3 * ctrl_mean_qp,
#     vmax=ctrl_mean_qp + 0.3 * ctrl_mean_qp
# )
# sm_qp = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm_qp)

# # Plot each stim location
# for condition, coords in stim_locations.items():
#     if condition not in qp_means:
#         continue
    
#     mean_qp = qp_means[condition]
#     p_val = qp_pvals.get(condition, 1.0)
    
#     # Size based on -100 * log10(p_val) - matching original
#     size = -100 * np.log10(p_val) if p_val > 0 else 300
#     color = sm_qp.to_rgba(mean_qp)
#     alpha = 0.5 if p_val >= 0.05 else 1.0
    
#     ml_left = coords.get('ML_left')
#     ml_right = coords.get('ML_right')
#     ap = coords.get('AP')
    
#     if ml_left is not None and ml_right is not None and ap is not None:
#         ax.scatter(ml_left, ap, color=color, alpha=alpha, edgecolors='w', s=size)
#         ax.scatter(ml_right, ap, color=color, alpha=alpha, edgecolors='w', s=size)

# # Set axis limits
# if PLOT_AP_LIMITS is not None:
#     ax.set_ylim(bottom=PLOT_AP_LIMITS[0], top=PLOT_AP_LIMITS[1])
# if PLOT_ML_LIMITS is not None:
#     ax.set_xlim(left=PLOT_ML_LIMITS[0], right=PLOT_ML_LIMITS[1])
# elif SHOW_RIGHT_HEMISPHERE_ONLY:
#     ax.set_xlim(left=0)

# # Labels and title
# ax.set_xlabel('Mediolateral Position (mm from Bregma)', fontsize=14)
# ax.set_ylabel('Anteroposterior Position (mm from Bregma)', fontsize=14)
# ax.set_title(f'Quiescent Period by Stimulation Location\n({num_analyzed_sessions} sessions, {num_unique_mice} mice)', fontsize=14)
# ax.tick_params(axis='both', which='major', labelsize=12)

# # P-value legend (matching RT style)
# p_values_legend = [0.001, 0.05, 0.2]
# sizes_legend = [-100 * np.log10(p) for p in p_values_legend]
# for p, size in zip(p_values_legend, sizes_legend):
#     ax.scatter([], [], s=size, label=f'p = {p}', edgecolors='w', color='black')
# ax.legend(loc='upper right', labelspacing=1.5, fontsize=10)

# # Colorbar
# cbar = plt.colorbar(sm_qp, ax=ax, orientation='vertical', fraction=0.046, pad=0.04, aspect=12)
# cbar.set_label('Quiescent Period (s)', fontsize=14, labelpad=15)
# cbar.ax.tick_params(labelsize=12)

# plt.tight_layout()

# if SAVE_FIGURES:
#     plt.savefig(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_QP_heatmap.png', dpi=150, bbox_inches='tight')
#     plt.close()
# else:
#     plt.show()


# # -----------------------------------------------------------------------------
# # Figure 3: Bias Reduction Heatmap
# # -----------------------------------------------------------------------------
# print("  Generating bias reduction heatmap...")

# # Compute bias values per cycle for each condition (matching original bias_vals_LC logic)
# # Bias = mean(L block choices) - mean(R block choices) at low contrasts
# # This quantifies block bias - positive values mean more leftward choices in L block

# import math

# bias_vals_by_condition = {cond: np.array([]) for cond in range(NUM_STIM_LOCATIONS + 1)}

# for condition in range(NUM_STIM_LOCATIONS + 1):
#     # Filter for low contrast trials in each block
#     # Using absolute contrast < LOW_CONTRAST_THRESHOLD
#     data_Lblock = [t for t in condition_data[condition] 
#                    if abs(t['contrast']) < LOW_CONTRAST_THRESHOLD and t['probabilityLeft'] == 0.8]
#     data_Rblock = [t for t in condition_data[condition] 
#                    if abs(t['contrast']) < LOW_CONTRAST_THRESHOLD and t['probabilityLeft'] == 0.2]
    
#     # Determine number of complete cycles
#     num_cycles = min(len(data_Lblock), len(data_Rblock)) // TRIALS_PER_DATAPOINT
    
#     if num_cycles == 0:
#         continue
    
#     bias_vals = np.empty(num_cycles)
#     bias_vals[:] = np.nan
    
#     for k in range(num_cycles):
#         start_idx = k * TRIALS_PER_DATAPOINT
#         end_idx = (k + 1) * TRIALS_PER_DATAPOINT
        
#         choices_L = [t['choice'] for t in data_Lblock[start_idx:end_idx]]
#         choices_R = [t['choice'] for t in data_Rblock[start_idx:end_idx]]
        
#         mean_L = np.mean(choices_L)
#         mean_R = np.mean(choices_R)
#         bias_vals[k] = mean_L - mean_R  # Positive = more leftward in L block
    
#     bias_vals_by_condition[condition] = bias_vals

# # Control bias statistics (for normalization)
# ctrl_bias_vals = bias_vals_by_condition[0]
# ctrl_mean_bias = np.mean(ctrl_bias_vals) if len(ctrl_bias_vals) > 0 else 0
# ctrl_abs_mean_bias = np.abs(ctrl_mean_bias)

# # Compute effect size (bias reduction) and p-value for each stim condition vs control
# bias_reduction = {}  # Normalized reduction: 0 = no change, 1 = complete elimination of bias
# bias_pvals = {}

# for cond in range(1, NUM_STIM_LOCATIONS + 1):
#     stim_bias_vals = bias_vals_by_condition[cond]
    
#     if len(stim_bias_vals) > 0 and len(ctrl_bias_vals) > 0:
#         # Independent t-test: stim vs control bias values
#         _, p_val = stats.ttest_ind(ctrl_bias_vals, stim_bias_vals)
#         bias_pvals[cond] = p_val
        
#         # Compute bias reduction as fraction of control bias eliminated
#         # Positive value = bias reduced; 1 = bias eliminated; 0 = no change; negative = bias increased
#         stim_mean_bias = np.mean(stim_bias_vals)
        
#         if ctrl_abs_mean_bias > 0:
#             # Reduction = (ctrl - stim) / ctrl, but we want magnitude reduction
#             # If ctrl_bias is positive (leftward bias in L block), reduction means stim_bias is smaller
#             bias_reduction[cond] = (np.abs(ctrl_mean_bias) - np.abs(stim_mean_bias)) / ctrl_abs_mean_bias
#         else:
#             bias_reduction[cond] = 0

# # Plot with same style as RT/QP heatmaps
# fig, ax = plt.subplots(figsize=(10, 8))

# # Show brain borders
# ax.imshow(dorsal_mip_with_borders, cmap='gray', extent=extent)

# # Set up colormap: blue = no reduction (0), red = full reduction (1)
# # Using 'coolwarm' so blue is low values, red is high values
# cmap_bias = plt.get_cmap('coolwarm')

# # Normalize based on data range, centered appropriately
# # Clip to reasonable range (some might be negative if bias increased, or >1 if stim reverses bias)
# reduction_values = [v for v in bias_reduction.values() if not np.isnan(v)]
# if reduction_values:
#     min_red = min(min(reduction_values), -0.2)  # Allow some negative (bias increase)
#     max_red = max(max(reduction_values), 0.8)   # Cap at reasonable maximum
# else:
#     min_red, max_red = -0.2, 0.8

# norm_bias = mcolors.Normalize(vmin=min_red, vmax=max_red)

# # Plot each stim location
# for condition, coords in stim_locations.items():
#     if condition == 52:  # Skip condition 52 as in original
#         continue
#     if condition not in bias_pvals or np.isnan(bias_pvals.get(condition, np.nan)):
#         continue
    
#     reduction = bias_reduction.get(condition, 0)
#     p_val = bias_pvals[condition]
    
#     color = cmap_bias(norm_bias(reduction))
    
#     # Size based on -100 * log10(p_val) - matching RT/QP style
#     size = -100 * np.log10(p_val) if p_val > 0 else 300
#     alpha = 0.5 if p_val >= 0.05 else 1.0
    
#     ml_left = coords.get('ML_left')
#     ml_right = coords.get('ML_right')
#     ap = coords.get('AP')
    
#     if ml_left is not None and ml_right is not None and ap is not None:
#         ax.scatter(ml_left, ap, color=color, alpha=alpha, s=size, edgecolors='w')
#         ax.scatter(ml_right, ap, color=color, alpha=alpha, s=size, edgecolors='w')

# # Set axis limits
# if PLOT_AP_LIMITS is not None:
#     ax.set_ylim(bottom=PLOT_AP_LIMITS[0], top=PLOT_AP_LIMITS[1])
# if PLOT_ML_LIMITS is not None:
#     ax.set_xlim(left=PLOT_ML_LIMITS[0], right=PLOT_ML_LIMITS[1])
# elif SHOW_RIGHT_HEMISPHERE_ONLY:
#     ax.set_xlim(left=0)

# # Labels and title
# ax.set_xlabel('Mediolateral Position (mm from Bregma)', fontsize=14)
# ax.set_ylabel('Anteroposterior Position (mm from Bregma)', fontsize=14)
# ax.set_title(f'Block Bias Reduction by Stimulation Location\n({num_analyzed_sessions} sessions, {num_unique_mice} mice)', fontsize=14)
# ax.tick_params(axis='both', which='major', labelsize=12)

# # P-value legend (matching RT/QP style)
# p_values_legend = [0.001, 0.05, 0.2]
# sizes_legend = [-100 * np.log10(p) for p in p_values_legend]
# for p, size in zip(p_values_legend, sizes_legend):
#     ax.scatter([], [], s=size, label=f'p = {p}', edgecolors='w', color='black')
# ax.legend(loc='upper right', labelspacing=1.5, fontsize=10)

# # Colorbar
# sm_bias = plt.cm.ScalarMappable(cmap=cmap_bias, norm=norm_bias)
# sm_bias.set_array([])
# cbar = plt.colorbar(sm_bias, ax=ax, orientation='vertical', fraction=0.046, pad=0.04, aspect=12)
# cbar.set_label('Bias Reduction (fraction of control)', fontsize=14, labelpad=15)
# cbar.ax.tick_params(labelsize=12)

# plt.tight_layout()

# if SAVE_FIGURES:
#     plt.savefig(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_bias_reduction_heatmap.png', dpi=150, bbox_inches='tight')
#     plt.close()
# else:
#     plt.show()


# # =============================================================================
# # SAVE RESULTS
# # =============================================================================

# if SAVE_FIGURES:
#     print("\nSaving analysis results...")
    
#     # Save bias arrays for further analysis
#     np.save(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_bias_vals_LC_control.npy', bias_vals_by_condition[0])
    
#     # Save effect sizes and statistics (including QP and bias reduction data)
#     effect_size_df = pd.DataFrame([
#         {'condition': c, 
#          'effect_size': effect_sizes.get(c, np.nan),
#          'rt_mean': rt_means.get(c, np.nan),
#          'rt_pval': rt_pvals.get(c, np.nan),
#          'qp_mean': qp_means.get(c, np.nan),
#          'qp_pval': qp_pvals.get(c, np.nan),
#          'bias_reduction': bias_reduction.get(c, np.nan),
#          'bias_pval': bias_pvals.get(c, np.nan)}
#         for c in range(1, NUM_STIM_LOCATIONS + 1)
#     ])
#     effect_size_df.to_csv(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_results.csv', index=False)
    
#     print(f"Results saved to {FIGURE_SAVE_PATH}")


print("\n" + "="*60)
print("Analysis complete!")
print("="*60)
