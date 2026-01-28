#!/usr/bin/env python3
"""
Zapit Optogenetics Analysis Pipeline

Analyzes behavioral data from the IBL task with laser scanning optogenetic
inhibition (Zapit system). Mice perform a 2-AFC task with prior blocks while
receiving bilateral optogenetic stimulation at one of 52 cortical locations.

Key analyses:
- Reaction time effects by stimulation location
- Bias shift effects (block-induced choice bias)
- Lapse rate changes
- Brain heatmap visualization on Allen CCF

Usage:
    1. Configure paths and options in config.py
    2. Update session metadata in metadata_zapit.py
    3. Run this script: python zapit_analysis.py

Author: [Your name]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
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
# SESSION SELECTION
# =============================================================================

# Modify these filters to select sessions of interest
eids, trials_ranges, MouseIDs, stim_params = find_sessions_by_advanced_criteria(
    sessions,
    Stimulation_Params='zapit',
    # Add additional filters as needed:
    # Mouse_ID='SWC_NM_099',
    # Hemisphere='both',
    # Pulse_Params='motor_bilateral_mask',
)

print(f"Found {len(eids)} sessions matching criteria")


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


# -----------------------------------------------------------------------------
# Figure 1: RT Heatmap on Brain Atlas
# -----------------------------------------------------------------------------
print("  Generating RT heatmap...")

# Load Allen CCF
allenCCF_data = np.load(ALLEN_CCF_ANNOTATION)
structure_tree = pd.read_csv(ALLEN_STRUCTURE_TREE)

# Generate brain surface image
mip, edges = generate_mip_with_borders(allenCCF_data)

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Show brain surface with edges
edges_masked = np.ma.masked_where(edges == 0, edges)
ax.imshow(mip, cmap='gray', alpha=0.3, extent=[0, mip.shape[1], mip.shape[0], 0])
ax.imshow(edges_masked, cmap='binary', alpha=0.5, extent=[0, mip.shape[1], mip.shape[0], 0])

# Set up colormap for RT effect sizes
rt_effect_sizes = {c: rt_results[c]['effect_size'] for c in rt_results if c > 0 and 'effect_size' in rt_results[c]}
if rt_effect_sizes:
    min_effect = min(rt_effect_sizes.values())
    max_effect = max(rt_effect_sizes.values())
    norm = mcolors.Normalize(vmin=min_effect, vmax=max_effect)
    cmap = plt.get_cmap('coolwarm')
    
    # Plot each stim location
    for condition, coords in stim_locations.items():
        if condition == 52 or condition not in rt_effect_sizes:
            continue
        
        effect = rt_effect_sizes[condition]
        color = cmap(norm(effect))
        
        ml_left = coords.get('ML_left')
        ml_right = coords.get('ML_right')
        ap = coords.get('AP')
        
        if ml_left is not None and ml_right is not None and ap is not None:
            # Convert to CCF coordinates (approximate scaling)
            # These conversions may need adjustment based on your coordinate system
            ax.scatter(ml_left * 100 + 570, -ap * 100 + 540, color=color, s=100, edgecolors='black')
            ax.scatter(ml_right * 100 + 570, -ap * 100 + 540, color=color, s=100, edgecolors='black')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Cohen's d (RT effect size)")

ax.set_title('Reaction Time Effect by Stimulation Location')
ax.set_xlabel('ML Position')
ax.set_ylabel('AP Position')

if SAVE_FIGURES:
    plt.savefig(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_RT_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
else:
    plt.show()


# -----------------------------------------------------------------------------
# Figure 2: Bias Heatmap (P-value based)
# -----------------------------------------------------------------------------
print("  Generating bias p-value heatmap...")

# Compute p-values for block bias difference at low contrasts
p_values = {}
choices_by_condition = {cond: {'R_block': [], 'L_block': []} for cond in condition_data}

for cond, trials_list in condition_data.items():
    for trial in trials_list:
        if trial['contrast'] in BIAS_HEATMAP_CONTRASTS:
            block_type = 'R_block' if trial['probabilityLeft'] == 0.2 else 'L_block'
            choices_by_condition[cond][block_type].append(trial['choice'])

for cond, blocks in choices_by_condition.items():
    if blocks['R_block'] and blocks['L_block']:
        _, p_val = stats.ttest_ind(blocks['R_block'], blocks['L_block'])
        p_values[cond] = p_val
    else:
        p_values[cond] = np.nan

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

cmap = plt.get_cmap('coolwarm_r')
log_p_values = {c: -np.log10(p) if p > 0 else 0 for c, p in p_values.items()}

min_log_p, max_log_p = 0.7, 9.5
norm = mcolors.Normalize(vmin=min_log_p, vmax=max_log_p)

for condition, coords in stim_locations.items():
    if condition == 52:
        continue
    
    log_p = log_p_values.get(condition, 0)
    color = cmap(norm(log_p))
    
    ml_left = coords.get('ML_left')
    ml_right = coords.get('ML_right')
    ap = coords.get('AP')
    
    if ml_left is not None and ml_right is not None and ap is not None:
        ax.scatter(ml_left, ap, color=color, s=100, edgecolors='black')
        ax.scatter(ml_right, ap, color=color, s=100, edgecolors='black')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('-log10(P-value)')

ax.set_title('Block Bias Difference: -log10(P-value) by Location')
ax.set_xlabel('Mediolateral Position')
ax.set_ylabel('Anteroposterior Position')

if SAVE_FIGURES:
    plt.savefig(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_bias_pval_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
else:
    plt.show()


# -----------------------------------------------------------------------------
# Figure 3: Effect Size Heatmap
# -----------------------------------------------------------------------------
print("  Generating effect size heatmap...")

fig, ax = plt.subplots(figsize=(10, 8))

cmap = plt.get_cmap('RdBu_r')
if effect_sizes:
    es_values = list(effect_sizes.values())
    max_abs = max(abs(min(es_values)), abs(max(es_values)))
    norm = mcolors.Normalize(vmin=-max_abs, vmax=max_abs)
    
    for condition, coords in stim_locations.items():
        if condition == 52 or condition not in effect_sizes:
            continue
        
        es = effect_sizes[condition]
        color = cmap(norm(es))
        
        ml_left = coords.get('ML_left')
        ml_right = coords.get('ML_right')
        ap = coords.get('AP')
        
        if ml_left is not None and ml_right is not None and ap is not None:
            ax.scatter(ml_left, ap, color=color, s=100, edgecolors='black')
            ax.scatter(ml_right, ap, color=color, s=100, edgecolors='black')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Effect Size (normalized bias change)')

ax.set_title('Bias Shift Effect Size by Location')
ax.set_xlabel('Mediolateral Position')
ax.set_ylabel('Anteroposterior Position')

if SAVE_FIGURES:
    plt.savefig(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_effect_size_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
else:
    plt.show()


# -----------------------------------------------------------------------------
# Figure 4: Summary Statistics
# -----------------------------------------------------------------------------
print("  Generating summary statistics figure...")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Control vs Stim RT comparison (pooled)
ax1 = axes[0]
ctrl_rt = [t['reaction_times'] for t in condition_data[0] if not np.isnan(t['reaction_times'])]
all_stim_rt = []
for c in range(1, 53):
    all_stim_rt.extend([t['reaction_times'] for t in condition_data[c] if not np.isnan(t['reaction_times'])])

if ctrl_rt and all_stim_rt:
    _, p_rt = stats.mannwhitneyu(ctrl_rt, all_stim_rt)
    ax1.bar([0, 1], [np.mean(ctrl_rt), np.mean(all_stim_rt)], 
            color=['gray', 'steelblue'], tick_label=['Control', 'Stim'])
    ax1.errorbar([0, 1], [np.mean(ctrl_rt), np.mean(all_stim_rt)],
                 yerr=[stats.sem(ctrl_rt), stats.sem(all_stim_rt)], 
                 fmt='none', color='black', capsize=5)
    ax1.set_ylabel('Reaction Time (s)')
    ax1.set_title(f'RT Comparison\np = {p_rt:.3g}')

# Control vs Stim QP comparison (pooled)
ax2 = axes[1]
ctrl_qp = [t['qp_times'] for t in condition_data[0] if not np.isnan(t['qp_times'])]
all_stim_qp = []
for c in range(1, 53):
    all_stim_qp.extend([t['qp_times'] for t in condition_data[c] if not np.isnan(t['qp_times'])])

if ctrl_qp and all_stim_qp:
    _, p_qp = stats.mannwhitneyu(ctrl_qp, all_stim_qp)
    ax2.bar([0, 1], [np.mean(ctrl_qp), np.mean(all_stim_qp)],
            color=['gray', 'steelblue'], tick_label=['Control', 'Stim'])
    ax2.errorbar([0, 1], [np.mean(ctrl_qp), np.mean(all_stim_qp)],
                 yerr=[stats.sem(ctrl_qp), stats.sem(all_stim_qp)],
                 fmt='none', color='black', capsize=5)
    ax2.set_ylabel('Quiescent Period (s)')
    ax2.set_title(f'QP Comparison\np = {p_qp:.3g}')

# Bias shift summary (cycles)
ax3 = axes[2]
ctrl_bias = bias_vals_LC[0]
all_stim_bias = np.concatenate([bias_vals_LC[c] for c in range(1, 53) if len(bias_vals_LC[c]) > 0])

if len(ctrl_bias) > 0 and len(all_stim_bias) > 0:
    _, p_bias = stats.ttest_ind(ctrl_bias, all_stim_bias)
    ax3.bar([0, 1], [np.mean(ctrl_bias), np.mean(all_stim_bias)],
            color=['gray', 'steelblue'], tick_label=['Control', 'Stim'])
    ax3.errorbar([0, 1], [np.mean(ctrl_bias), np.mean(all_stim_bias)],
                 yerr=[stats.sem(ctrl_bias), stats.sem(all_stim_bias)],
                 fmt='none', color='black', capsize=5)
    ax3.set_ylabel('Bias (L-R block choice diff)')
    ax3.set_title(f'Bias Comparison\np = {p_bias:.3g}')

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_summary_stats.png', dpi=150, bbox_inches='tight')
    plt.close()
else:
    plt.show()


# =============================================================================
# SAVE RESULTS
# =============================================================================

if SAVE_FIGURES:
    print("\nSaving analysis results...")
    
    # Save bias arrays for further analysis
    np.save(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_bias_vals_LC_control.npy', bias_vals_LC[0])
    
    # Save effect sizes
    effect_size_df = pd.DataFrame([
        {'condition': c, 'effect_size': es, 
         'rt_effect': rt_results.get(c, {}).get('effect_size', np.nan),
         'rt_pval': rt_results.get(c, {}).get('p_val', np.nan),
         'bias_pval_ind': cycle_stats['ttest_ind'].get(c, np.nan)}
        for c, es in effect_sizes.items()
    ])
    effect_size_df.to_csv(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_results.csv', index=False)
    
    print(f"Results saved to {FIGURE_SAVE_PATH}")


print("\n" + "="*60)
print("Analysis complete!")
print("="*60)
