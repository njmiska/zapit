"""
Configuration settings for Zapit optogenetics analysis.

This file contains all configurable parameters including:
- File paths (adjust these to your local setup)
- Analysis thresholds
- Plotting options
"""

from pathlib import Path

# =============================================================================
# FILE PATHS - Adjust these to your local setup
# =============================================================================

# Base directory for the project (update this to your local path)
BASE_DIR = Path('/Users/natemiska/python/bias_coding')

# Zapit trial log file
ZAPIT_TRIALS_LOG = BASE_DIR / 'zapit_trials.yml'

# Zapit stimulation locations log (contains AP/ML coordinates)
ZAPIT_LOCATIONS_LOG = BASE_DIR / 'zapit_log_2024_02_28__12-41.yml'

# Allen CCF atlas data (download from Allen Institute)
# See: https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
ALLEN_CCF_ANNOTATION = Path('/Users/natemiska/python/Allen/annotation_volume_10um.npy')
ALLEN_STRUCTURE_TREE = Path('/Users/natemiska/python/Allen/structure_tree_safe_2017.csv')

# Output directory for figures
FIGURE_SAVE_PATH = Path('/Users/natemiska/Desktop/other_figures/CP/')

# =============================================================================
# SESSION FILTERING THRESHOLDS
# =============================================================================

# Minimum performance at high contrasts (100%, 25%) to include session
BASELINE_PERFORMANCE_THRESHOLD = 0.79

# Minimum performance on stim trials at high contrasts
STIM_PERFORMANCE_THRESHOLD = 0.5

# Minimum number of trials required to include session
MIN_NUM_TRIALS = 300

# Minimum baseline bias shift to include session (non-Zapit sessions)
MIN_BIAS_THRESHOLD = 0.3

# Minimum baseline bias shift to include Zapit session
MIN_BIAS_THRESHOLD_ZAPIT = 1.0

# Maximum reaction time to include trial (seconds)
RT_THRESHOLD = 30

# =============================================================================
# ANALYSIS OPTIONS
# =============================================================================

# Number of stimulation locations in Zapit grid
NUM_STIM_LOCATIONS = 52

# Number of trials to average for bias assessment
TRIALS_PER_DATAPOINT = 5

# Whether to use trials immediately after stim (for lagged analysis)
USE_TRIALS_AFTER_STIM = False

# =============================================================================
# WHEEL ANALYSIS OPTIONS
# =============================================================================

# Duration of wheel movement to analyze (seconds)
WHEEL_ANALYSIS_DURATION = 10

# Time bin interval for wheel analysis (seconds)
WHEEL_TIME_INTERVAL = 0.1

# Alignment point for wheel analysis: 'QP', 'goCue', 'goCue_pre', 'feedback'
WHEEL_ALIGN_TO = 'QP'

# Whether to only include low contrast trials in wheel analysis
ONLY_LOW_CONTRASTS = False

# Threshold for defining "low" contrast (%)
LOW_CONTRAST_THRESHOLD = 13

# =============================================================================
# PLOTTING OPTIONS
# =============================================================================

# Whether to save figures to disk
SAVE_FIGURES = True

# Prefix for saved figure filenames
FIGURE_PREFIX = 'zapit'

# Flag to show failed session loads in console
FLAG_FAILED_LOADS = True

# =============================================================================
# CONTRAST LEVELS
# =============================================================================

# All contrast levels used in the task (%)
ALL_CONTRASTS = [-100.0, -25.0, -12.5, -6.25, 0.0, 6.25, 12.5, 25.0, 100.0]

# Low contrast levels for focused analysis (%)
LOW_CONTRASTS = [-12.5, -6.25, 0.0, 6.25, 12.5]

# Contrasts used for bias heatmap (typically low contrasts)
BIAS_HEATMAP_CONTRASTS = [-6.25, 0, 6.25]

# =============================================================================
# IBL API CONFIGURATION
# =============================================================================

# IBL Alyx database URL
ALYX_BASE_URL = 'https://alyx.internationalbrainlab.org'
