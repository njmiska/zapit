"""
Helper functions for Zapit optogenetics analysis.

This module contains reusable functions for:
- Data loading and parsing
- Trial data extraction
- Wheel analysis
- Bias calculations
- Statistical analysis
- Brain atlas visualization
"""

import re
import math
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_session_data(one, eid, flag_failed_loads=True):
    """
    Load trial and wheel data for a session from IBL database.
    
    Parameters
    ----------
    one : ONE
        ONE API instance
    eid : str
        Experiment ID
    flag_failed_loads : bool
        Whether to print warning on failed loads
        
    Returns
    -------
    trials : dict-like or None
        Trials object with all trial data
    wheel : dict-like or None
        Wheel object with position and timestamps
    """
    try:
        trials = one.load_object(eid, 'trials')
    except Exception:
        try:
            trials = one.load_object(eid, 'trials', 'alf')
        except Exception:
            if flag_failed_loads:
                print(f'Failed to load eid = {eid}')
            return None, None
    
    # Fix inconsistent trial object lengths (known IBL data issue)
    trials = fix_trials_length_inconsistency(trials)
    
    try:
        wheel = one.load_object(eid, 'wheel')
    except Exception:
        if flag_failed_loads:
            print(f'Failed to load wheel data for eid = {eid}')
        wheel = None
        
    return trials, wheel


def fix_trials_length_inconsistency(trials):
    """
    Fix inconsistent lengths in trials object by padding shorter arrays.
    
    This addresses a known issue where some trial attributes may be
    off by one element.
    
    Parameters
    ----------
    trials : dict-like
        Trials object
        
    Returns
    -------
    trials : dict-like
        Fixed trials object
    """
    lengths = [len(trials[k]) for k in trials.keys()]
    max_length = max(lengths)
    
    for k in trials.keys():
        if len(trials[k]) == max_length - 1:
            trials[k] = np.insert(trials[k], 0, 0)
            
    return trials


def load_laser_intervals(one, eid):
    """
    Load laser stimulation intervals for a session.
    
    Parameters
    ----------
    one : ONE
        ONE API instance
    eid : str
        Experiment ID
        
    Returns
    -------
    laser_intervals : ndarray
        Array of [onset, offset] times for each laser stimulation
    """
    try:
        # New format: collection is 'alf/task_00'
        laser_intervals = one.load_dataset(
            eid, '_ibl_laserStimulation.intervals', collection='alf/task_00'
        )
    except Exception:
        # Fall back to old format
        laser_intervals = one.load_dataset(eid, '_ibl_laserStimulation.intervals')
    
    return laser_intervals


def parse_zapit_log(file_path, session_start, eid=None):
    """
    Parse Zapit trial log file to extract stimulation events for a session.
    
    Parameters
    ----------
    file_path : str or Path
        Path to zapit_trials.yml file
    session_start : datetime
        Session start time
    eid : str, optional
        Experiment ID (for special case handling)
        
    Returns
    -------
    relevant_events : list
        List of event strings occurring during/after session start
    """
    # Handle known session-specific issues
    if eid == '21d33b44-f75f-4711-a2c7-0bdfe8eec386':
        session_start = datetime.strptime('2024-03-29T18:07:38.0'[:19], '%Y-%m-%dT%H:%M:%S')
    
    relevant_events = []
    
    with open(file_path, 'r') as file:
        next(file)  # Skip header line
        
        for line in file:
            # Extract timestamp (first 19 characters: YYYY-MM-DD HH:MM:SS)
            timestamp_str = line[:19]
            if len(timestamp_str) < 19:
                print('Warning: Error reading timestamp string for one line')
                continue
                
            try:
                event_timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue
            
            # Keep events that occur during or after session start
            if event_timestamp >= session_start:
                relevant_events.append(line.strip())
    
    return relevant_events


def build_stim_location_dict(laser_intervals, trials, relevant_events, trials_range, eid=None):
    """
    Build dictionary mapping trial numbers to stimulation locations.
    
    Parameters
    ----------
    laser_intervals : ndarray
        Laser stimulation intervals [onset, offset]
    trials : dict-like
        Trials object
    relevant_events : list
        Zapit log events for this session
    trials_range : range or list
        Valid trial indices
    eid : str, optional
        Experiment ID (for special case handling)
        
    Returns
    -------
    stimtrial_location_dict_all : dict
        Maps trial number -> stim location (0 = control, 1-52 = stim locations)
    """
    stimtrial_location_dict = {}
    
    for k in range(1, len(laser_intervals[:, 0]) - 2):
        # Handle known session-specific issues
        if eid == '21d33b44-f75f-4711-a2c7-0bdfe8eec386' and k < 10:
            continue
            
        # Find trial number for this laser interval
        trial_matches = np.where(laser_intervals[k, 0] == trials.intervals[:, 0])[0]
        if len(trial_matches) == 0:
            continue
        trialnum = trial_matches[0]
        
        # Extract stim location from log (characters 20-22)
        stim_location_str = relevant_events[k][20:22]
        
        # Handle special case for session with off-by-one logged events
        if eid == '5a41494f-25b9-48d4-8159-527141bd4742':
            stim_location_str = relevant_events[k-1][20:22]
            
        # Clean and convert to integer
        stim_location = int(re.sub(r'\D', '', stim_location_str))
        stimtrial_location_dict[trialnum] = stim_location
    
    # Create full dict with 0 (control) for non-stim trials
    stimtrial_location_dict_all = {k: 0 for k in trials_range}
    stimtrial_location_dict_all.update(stimtrial_location_dict)
    
    return stimtrial_location_dict_all


def load_stim_locations_coordinates(file_path):
    """
    Load stimulation location coordinates from Zapit log file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to zapit locations log file
        
    Returns
    -------
    stim_locations : dict
        Maps location number -> {'ML_left': float, 'ML_right': float, 'AP': float}
    """
    stim_locations = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    current_location = None
    for line in lines:
        line = line.strip()
        
        # Check for stimLocations header
        location_match = re.search(r'stimLocations(\d+):', line)
        if location_match:
            current_location = int(location_match.group(1))
            stim_locations[current_location] = {'ML_left': None, 'ML_right': None, 'AP': None}
            
        elif line.startswith('ML: [') and current_location is not None:
            ml_coords = re.findall(r'\[([-.\d]+), ([-.\d]+)\]', line)
            if ml_coords:
                ml_left, ml_right = map(float, ml_coords[0])
                stim_locations[current_location]['ML_left'] = ml_left
                stim_locations[current_location]['ML_right'] = ml_right
                
        elif line.startswith('AP: [') and current_location is not None:
            ap_coords = re.findall(r'\[([-.\d]+), ([-.\d]+)\]', line)
            if ap_coords:
                stim_locations[current_location]['AP'] = float(ap_coords[0][0])
    
    return stim_locations


# =============================================================================
# TRIAL DATA PROCESSING
# =============================================================================

def signed_contrast(trials):
    """
    Compute signed contrast values from trial data.
    
    Parameters
    ----------
    trials : dict-like
        Trials object with contrastLeft and contrastRight
        
    Returns
    -------
    contrast : ndarray
        Signed contrast values in percent (negative = left, positive = right)
    """
    contrast = np.nan_to_num(np.c_[trials.contrastLeft, trials.contrastRight])
    return np.diff(contrast).flatten() * 100


def compute_reaction_time(trials, trial_number):
    """
    Compute reaction time for a single trial.
    
    Parameters
    ----------
    trials : dict-like
        Trials object
    trial_number : int
        Trial index
        
    Returns
    -------
    reaction_time : float
        Reaction time in seconds
    used_trigger : bool
        Whether goCueTrigger_times was used (vs goCue_times)
    """
    try:
        rt = trials.feedback_times[trial_number] - trials.goCueTrigger_times[trial_number]
        used_trigger = True
    except (IndexError, AttributeError):
        rt = trials.feedback_times[trial_number] - trials.goCue_times[trial_number]
        used_trigger = False
    
    # Fall back if NaN
    if np.isnan(rt):
        try:
            rt = trials.feedback_times[trial_number] - trials.goCueTrigger_times[trial_number]
        except (IndexError, AttributeError):
            rt = trials.feedback_times[trial_number] - trials.goCue_times[trial_number]
    
    return rt, used_trigger


def compute_quiescent_period(trials, trial_number, used_trigger=True):
    """
    Compute quiescent period duration for a single trial.
    
    Parameters
    ----------
    trials : dict-like
        Trials object
    trial_number : int
        Trial index
    used_trigger : bool
        Whether to use goCueTrigger_times
        
    Returns
    -------
    qp_time : float
        Quiescent period duration in seconds
    """
    if used_trigger:
        return trials.goCueTrigger_times[trial_number] - trials.intervals[trial_number][0]
    else:
        return trials.goCue_times[trial_number] - trials.intervals[trial_number][0]


def create_trial_data_dict(trials, trial_number, contrast_values):
    """
    Create a dictionary of trial data for a single trial.
    
    Parameters
    ----------
    trials : dict-like
        Trials object
    trial_number : int
        Trial index
    contrast_values : ndarray
        Pre-computed signed contrast values
        
    Returns
    -------
    trial_data : dict
        Dictionary with trial information
    """
    rt, used_trigger = compute_reaction_time(trials, trial_number)
    qp = compute_quiescent_period(trials, trial_number, used_trigger)
    
    return {
        'choice': trials.choice[trial_number],
        'reaction_times': rt,
        'qp_times': qp,
        'contrast': contrast_values[trial_number],
        'feedbackType': trials.feedbackType[trial_number],
        'probabilityLeft': trials.probabilityLeft[trial_number],
    }


def get_valid_trials_range(trials_ranges_entry, num_trials):
    """
    Convert trials range specification to actual range object.
    
    Parameters
    ----------
    trials_ranges_entry : str or list
        'ALL' or list of trial indices
    num_trials : int
        Total number of trials in session
        
    Returns
    -------
    trials_range : range or list
        Valid trial indices
    """
    if trials_ranges_entry == 'ALL':
        return range(0, num_trials)
    elif trials_ranges_entry[-1] == 9998:
        # 9998 indicates "until end of session"
        return [x for x in trials_ranges_entry if x < num_trials]
    else:
        return trials_ranges_entry


# =============================================================================
# SESSION QUALITY FILTERS
# =============================================================================

def check_session_accuracy(session_data, threshold, contrast_levels=(-100, -25, 25, 100)):
    """
    Check if session meets accuracy threshold at high contrasts.
    
    Parameters
    ----------
    session_data : list
        List of trial data dictionaries
    threshold : float
        Minimum accuracy (0-1)
    contrast_levels : tuple
        Contrast levels to check
        
    Returns
    -------
    passes : bool
        Whether session meets threshold
    accuracy : float
        Computed accuracy
    """
    feedback = [t['feedbackType'] for t in session_data 
                if t['contrast'] in contrast_levels]
    
    if len(feedback) == 0:
        return False, 0.0
    
    accuracy = np.sum(np.array(feedback) == 1) / len(feedback)
    return accuracy >= threshold, accuracy


def compute_session_bias_shift(session_data):
    """
    Compute total bias shift across all contrasts for session quality check.
    
    Parameters
    ----------
    session_data : list
        List of trial data dictionaries for control trials
        
    Returns
    -------
    total_bias_shift : float
        Sum of bias shifts across all contrasts
    """
    contrasts = [-100, -25, -12.5, -6.25, 0, 6.25, 12.5, 25, 100]
    total_shift = 0
    
    for c in contrasts:
        # Get choices for this contrast in each block
        choices_Lblock = [t['choice'] for t in session_data 
                         if t['contrast'] == c and t['probabilityLeft'] == 0.8]
        choices_Rblock = [t['choice'] for t in session_data 
                         if t['contrast'] == c and t['probabilityLeft'] == 0.2]
        
        if len(choices_Lblock) > 0 and len(choices_Rblock) > 0:
            p_right_Lblock = np.sum(np.array(choices_Lblock) == 1) / len(choices_Lblock)
            p_right_Rblock = np.sum(np.array(choices_Rblock) == 1) / len(choices_Rblock)
            shift = p_right_Lblock - p_right_Rblock
            if not np.isnan(shift):
                total_shift += shift
    
    return total_shift


# =============================================================================
# WHEEL ANALYSIS
# =============================================================================

def find_nearest_wheel_index(wheel_timestamps, target_time):
    """
    Find the wheel index closest to target time.
    
    Parameters
    ----------
    wheel_timestamps : ndarray
        Wheel timestamp array
    target_time : float
        Target time to find
        
    Returns
    -------
    index : int
        Index of nearest wheel timestamp
    """
    idx = np.searchsorted(wheel_timestamps, target_time)
    
    if idx == 0:
        return idx
    elif idx == len(wheel_timestamps):
        return idx - 1
    else:
        left_diff = target_time - wheel_timestamps[idx - 1]
        right_diff = wheel_timestamps[idx] - target_time
        return idx - 1 if left_diff <= right_diff else idx


def extract_wheel_trajectory(wheel, trials, trial_number, 
                             align_to='QP', duration=10, interval=0.1):
    """
    Extract wheel movement trajectory for a single trial.
    
    Parameters
    ----------
    wheel : dict-like
        Wheel object with position and timestamps
    trials : dict-like
        Trials object
    trial_number : int
        Trial index
    align_to : str
        Alignment point: 'QP', 'goCue', 'goCue_pre', 'feedback'
    duration : float
        Duration to analyze (seconds)
    interval : float
        Time bin size (seconds)
        
    Returns
    -------
    trajectory : ndarray
        Wheel position change at each time bin
    """
    whlpos, whlt = wheel.position, wheel.timestamps
    
    # Get alignment time
    try:
        if align_to == 'goCue':
            start_time = trials.goCueTrigger_times[trial_number]
        elif align_to == 'goCue_pre':
            start_time = trials.goCueTrigger_times[trial_number] - 0.5
        elif align_to == 'QP':
            start_time = trials.intervals[trial_number][0]
        elif align_to == 'feedback':
            start_time = trials.feedback_times[trial_number] - 0.6
    except (IndexError, AttributeError):
        if align_to == 'goCue':
            start_time = trials.goCue_times[trial_number]
        elif align_to == 'goCue_pre':
            start_time = trials.goCue_times[trial_number] - 0.5
        elif align_to == 'QP':
            start_time = trials.intervals[trial_number][0]
        elif align_to == 'feedback':
            start_time = trials.feedback_times[trial_number] - 0.6
    
    wheel_start_idx = find_nearest_wheel_index(whlt, start_time)
    
    # Get event boundaries for NaN masking
    try:
        go_cue_time = trials.goCueTrigger_times[trial_number]
    except (IndexError, AttributeError):
        go_cue_time = trials.goCue_times[trial_number]
    feedback_time = trials.feedback_times[trial_number]
    
    # Compute trajectory
    num_bins = int(duration / interval)
    trajectory = np.empty(num_bins)
    trajectory[:] = np.nan
    
    for i in range(num_bins):
        t = start_time + i * interval
        wheel_end_idx = find_nearest_wheel_index(whlt, t)
        
        # Mask based on alignment type
        if align_to == 'QP' and go_cue_time < whlt[wheel_end_idx]:
            continue  # Don't use wheel movement past go cue
        elif align_to in ['goCue', 'goCue_pre'] and (feedback_time + interval) < whlt[wheel_end_idx]:
            continue  # Don't use wheel movement past feedback
        
        trajectory[i] = whlpos[wheel_end_idx] - whlpos[wheel_start_idx]
    
    return trajectory


# =============================================================================
# BIAS ANALYSIS
# =============================================================================

def calculate_choice_probability(trials_list, block_type, contrast_level):
    """
    Calculate probability of leftward choice for a given block and contrast.
    
    Parameters
    ----------
    trials_list : list
        List of trial data dictionaries
    block_type : str
        'left' (probabilityLeft=0.8) or 'right' (probabilityLeft=0.2)
    contrast_level : float
        Contrast level to filter
        
    Returns
    -------
    p_left : float or None
        Probability of leftward choice, or None if no trials
    """
    prob_left_value = 0.8 if block_type == 'left' else 0.2
    
    relevant_trials = [t for t in trials_list 
                      if t['contrast'] == contrast_level 
                      and t['probabilityLeft'] == prob_left_value]
    
    if len(relevant_trials) == 0:
        return None
    
    leftward_choices = sum(1 for t in relevant_trials if t['choice'] == -1)
    return leftward_choices / len(relevant_trials)


def compute_bias_values_by_contrast(condition_data, contrasts, num_conditions=53):
    """
    Compute bias values (L block - R block choice probability) at each contrast.
    
    Parameters
    ----------
    condition_data : dict
        Maps condition number -> list of trial data dicts
    contrasts : list
        Contrast levels to compute
    num_conditions : int
        Number of conditions (including control)
        
    Returns
    -------
    bias_values : dict
        Maps condition -> list of bias values (one per contrast)
    left_block_probs : dict
        Maps condition -> list of left block probabilities
    right_block_probs : dict
        Maps condition -> list of right block probabilities
    """
    bias_values = {c: [] for c in range(num_conditions)}
    left_block_probs = {c: [] for c in range(num_conditions)}
    right_block_probs = {c: [] for c in range(num_conditions)}
    
    for contrast in contrasts:
        # Control condition (0)
        ctrl_left = calculate_choice_probability(condition_data[0], 'left', contrast)
        ctrl_right = calculate_choice_probability(condition_data[0], 'right', contrast)
        
        if ctrl_left is not None and ctrl_right is not None:
            ctrl_bias = ctrl_left - ctrl_right
            left_block_probs[0].append(ctrl_left)
            right_block_probs[0].append(ctrl_right)
            bias_values[0].append(ctrl_bias)
        
        # Stim conditions (1-52)
        for cond in range(1, num_conditions):
            stim_left = calculate_choice_probability(condition_data[cond], 'left', contrast)
            stim_right = calculate_choice_probability(condition_data[cond], 'right', contrast)
            
            if stim_left is not None and stim_right is not None:
                stim_bias = stim_left - stim_right
                left_block_probs[cond].append(stim_left)
                right_block_probs[cond].append(stim_right)
                bias_values[cond].append(stim_bias)
            elif ctrl_left is not None:
                # Fall back to control values if no data
                left_block_probs[cond].append(ctrl_left)
                right_block_probs[cond].append(ctrl_right)
                bias_values[cond].append(ctrl_bias)
    
    return bias_values, left_block_probs, right_block_probs


def compute_bias_values_by_cycle(condition_data, trials_per_cycle=5, 
                                  low_contrast_threshold=13, num_conditions=53):
    """
    Compute bias values in cycles of N trials for low contrast trials.
    
    This provides independent samples for statistical testing.
    
    Parameters
    ----------
    condition_data : dict
        Maps condition number -> list of trial data dicts
    trials_per_cycle : int
        Number of trials per cycle
    low_contrast_threshold : float
        Maximum contrast to include (%)
    num_conditions : int
        Number of conditions
        
    Returns
    -------
    bias_vals_LC : dict
        Maps condition -> array of bias values (one per cycle)
    """
    bias_vals_LC = {c: np.array([]) for c in range(num_conditions)}
    
    for condition in range(num_conditions):
        # Filter for low contrast trials in each block
        data_Lblock = [t for t in condition_data[condition] 
                      if abs(t['contrast']) < low_contrast_threshold and t['probabilityLeft'] == 0.8]
        data_Rblock = [t for t in condition_data[condition] 
                      if abs(t['contrast']) < low_contrast_threshold and t['probabilityLeft'] == 0.2]
        
        # Determine number of complete cycles
        num_cycles = min(len(data_Lblock), len(data_Rblock)) // trials_per_cycle
        
        if num_cycles == 0:
            continue
        
        bias_vals = np.empty(num_cycles)
        bias_vals[:] = np.nan
        
        for k in range(num_cycles):
            start_idx = k * trials_per_cycle
            end_idx = (k + 1) * trials_per_cycle
            
            choices_L = [t['choice'] for t in data_Lblock[start_idx:end_idx]]
            choices_R = [t['choice'] for t in data_Rblock[start_idx:end_idx]]
            
            mean_L = np.mean(choices_L)
            mean_R = np.mean(choices_R)
            bias_vals[k] = mean_L - mean_R
        
        bias_vals_LC[condition] = bias_vals
    
    return bias_vals_LC


def compute_effect_sizes(bias_values, only_low_contrasts=False):
    """
    Compute normalized effect sizes for each condition.
    
    Effect size = -(stim_bias_sum - ctrl_bias_sum) / ctrl_bias_sum
    
    Parameters
    ----------
    bias_values : dict
        Maps condition -> list of bias values
    only_low_contrasts : bool
        Whether bias_values contains only low contrasts
        
    Returns
    -------
    effect_sizes : dict
        Maps condition -> effect size
    """
    effect_sizes = {}
    
    if only_low_contrasts:
        indices = range(5)  # Low contrasts only
    else:
        indices = range(9)  # All contrasts
    
    ctrl_sum = sum(bias_values[0][i] for i in indices if i < len(bias_values[0]))
    
    if ctrl_sum == 0:
        return effect_sizes
    
    for condition in range(1, 53):
        if not bias_values[condition]:
            continue
        
        stim_sum = sum(bias_values[condition][i] for i in indices 
                      if i < len(bias_values[condition]))
        effect_sizes[condition] = -(stim_sum - ctrl_sum) / ctrl_sum
    
    return effect_sizes


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def run_condition_comparisons(bias_values, num_conditions=53):
    """
    Run statistical comparisons between control and each stim condition.
    
    Parameters
    ----------
    bias_values : dict
        Maps condition -> list of bias values
    num_conditions : int
        Number of conditions
        
    Returns
    -------
    results : dict
        Contains 'mannwhitney', 'ttest_paired', 'kruskal', 'anova' results
    """
    results = {
        'mannwhitney': {},
        'ttest_paired': {},
    }
    
    for cond in range(1, num_conditions):
        if bias_values[0] and bias_values[cond]:
            # Mann-Whitney U test
            _, p_mw = stats.mannwhitneyu(bias_values[0], bias_values[cond], 
                                          alternative='two-sided')
            results['mannwhitney'][cond] = p_mw
            
            # Paired t-test (across contrasts)
            if len(bias_values[0]) == len(bias_values[cond]):
                _, p_paired = stats.ttest_rel(bias_values[0], bias_values[cond])
                results['ttest_paired'][cond] = p_paired
    
    # Kruskal-Wallis across all conditions
    all_groups = [bias_values[c] for c in range(num_conditions) if bias_values[c]]
    if len(all_groups) > 1:
        results['kruskal_stat'], results['kruskal_p'] = stats.kruskal(*all_groups)
        results['anova_stat'], results['anova_p'] = stats.f_oneway(*all_groups)
    
    return results


def run_cycle_comparisons(bias_vals_LC, num_conditions=53):
    """
    Run independent t-tests on cycle-based bias values.
    
    Parameters
    ----------
    bias_vals_LC : dict
        Maps condition -> array of bias values (one per cycle)
    num_conditions : int
        Number of conditions
        
    Returns
    -------
    results : dict
        Contains 'ttest_ind', 'kruskal', 'anova' results
    """
    results = {'ttest_ind': {}}
    
    for cond in range(1, num_conditions):
        if len(bias_vals_LC[cond]) > 0 and len(bias_vals_LC[0]) > 0:
            _, p_ind = stats.ttest_ind(bias_vals_LC[0], bias_vals_LC[cond])
            results['ttest_ind'][cond] = p_ind
    
    # Kruskal-Wallis and ANOVA
    all_groups = [bias_vals_LC[c] for c in range(num_conditions) 
                  if len(bias_vals_LC[c]) > 0]
    if len(all_groups) > 1:
        results['kruskal_stat'], results['kruskal_p'] = stats.kruskal(*all_groups)
        results['anova_stat'], results['anova_p'] = stats.f_oneway(*all_groups)
    
    return results


def run_rt_analysis(condition_data, num_conditions=52):
    """
    Run reaction time analysis comparing each condition to control.
    
    Parameters
    ----------
    condition_data : dict
        Maps condition number -> list of trial data dicts
    num_conditions : int
        Number of stim conditions (excluding control)
        
    Returns
    -------
    rt_results : dict
        Maps condition -> {'p_val', 'effect_size', 'mean', 'std'}
    qp_results : dict
        Maps condition -> {'p_val', 'effect_size', 'mean', 'std'}
    lapse_results : dict
        Maps condition -> {'p_val', 'lapse_rate'}
    """
    # Control condition stats
    ctrl_rt = [t['reaction_times'] for t in condition_data[0] if not np.isnan(t['reaction_times'])]
    ctrl_qp = [t['qp_times'] for t in condition_data[0] if not np.isnan(t['qp_times'])]
    ctrl_fb = [t['feedbackType'] for t in condition_data[0] if t['contrast'] in (-100, 100)]
    ctrl_lapse = ((len(ctrl_fb) - np.sum(ctrl_fb)) / 2) / len(ctrl_fb) if ctrl_fb else 0
    
    rt_results = {0: {'mean': np.mean(ctrl_rt), 'std': np.std(ctrl_rt)}}
    qp_results = {0: {'mean': np.mean(ctrl_qp), 'std': np.std(ctrl_qp)}}
    lapse_results = {0: {'p_val': np.nan, 'lapse_rate': ctrl_lapse}}
    
    for cond in range(1, num_conditions + 1):
        stim_rt = [t['reaction_times'] for t in condition_data[cond] 
                   if not np.isnan(t['reaction_times'])]
        stim_qp = [t['qp_times'] for t in condition_data[cond] 
                   if not np.isnan(t['qp_times'])]
        stim_fb = [t['feedbackType'] for t in condition_data[cond] 
                   if t['contrast'] in (-100, -25, 100, 25)]
        
        if not stim_rt or not stim_qp:
            continue
        
        stim_lapse = ((len(stim_fb) - np.sum(stim_fb)) / 2) / len(stim_fb) if stim_fb else 0
        
        # T-tests
        _, p_rt = stats.ttest_ind(stim_rt, ctrl_rt, equal_var=False)
        _, p_qp = stats.ttest_ind(stim_qp, ctrl_qp, equal_var=False)
        
        # Cohen's d effect sizes
        pooled_std_rt = np.sqrt((np.std(stim_rt)**2 + np.std(ctrl_rt)**2) / 2)
        pooled_std_qp = np.sqrt((np.std(stim_qp)**2 + np.std(ctrl_qp)**2) / 2)
        d_rt = (np.mean(stim_rt) - np.mean(ctrl_rt)) / pooled_std_rt if pooled_std_rt > 0 else 0
        d_qp = (np.mean(stim_qp) - np.mean(ctrl_qp)) / pooled_std_qp if pooled_std_qp > 0 else 0
        
        # Proportions z-test for lapse rate
        _, p_lapse = proportions_ztest([ctrl_lapse, stim_lapse], 
                                        [len(ctrl_fb), len(stim_fb)])
        
        rt_results[cond] = {
            'p_val': p_rt, 'effect_size': d_rt, 
            'mean': np.mean(stim_rt), 'std': np.std(stim_rt)
        }
        qp_results[cond] = {
            'p_val': p_qp, 'effect_size': d_qp,
            'mean': np.mean(stim_qp), 'std': np.std(stim_qp)
        }
        lapse_results[cond] = {'p_val': p_lapse, 'lapse_rate': stim_lapse}
    
    return rt_results, qp_results, lapse_results


# =============================================================================
# BRAIN ATLAS VISUALIZATION
# =============================================================================

def transform_to_ccf(x, y, z, resolution=10):
    """
    Transform stereotaxic coordinates to Allen CCF coordinates.
    
    Parameters
    ----------
    x, y, z : float
        Coordinates in micrometers
    resolution : int
        CCF resolution in micrometers per pixel
        
    Returns
    -------
    X, Y, Z : float
        Transformed coordinates in CCF space
    """
    # Bregma position for 10um resolution
    x_bregma, y_bregma, z_bregma = 540, 44, 570
    x -= x_bregma
    y -= y_bregma
    z -= z_bregma
    
    # Rotate CCF (5 degrees)
    angle_rad = 5 * (np.pi / 180)
    X = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    Y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    
    # Squeeze DV axis
    Y *= 0.9434
    
    # Scale to resolution
    X, Y, Z = X / resolution, Y / resolution, z / resolution
    
    return X, Y, Z


def generate_mip_with_borders(annotation_volume):
    """
    Generate maximum intensity projection with region borders.
    
    Parameters
    ----------
    annotation_volume : ndarray
        3D Allen CCF annotation volume (AP x DV x ML)
        
    Returns
    -------
    mip : ndarray
        2D maximum intensity projection
    edges : ndarray
        2D edge detection result for region borders
    """
    # Find dorsal surface (first non-zero label along DV axis)
    dorsal_surface_index = np.argmax(annotation_volume > 0, axis=1)
    
    ap_length, ml_length = dorsal_surface_index.shape
    dv_length = annotation_volume.shape[1]
    mip = np.zeros((ap_length, ml_length), dtype=annotation_volume.dtype)
    
    # Populate MIP
    for x in range(ml_length):
        for y in range(ap_length):
            dv_idx = dorsal_surface_index[y, x]
            if dv_idx < dv_length:
                mip[y, x] = annotation_volume[y, dv_idx, x]
    
    # Edge detection for borders
    grad_x, grad_y = np.gradient(mip)
    edges = np.sqrt(grad_x**2 + grad_y**2)
    
    return mip, edges
