"""
Configuration settings for Zapit optogenetics analysis.

This file contains all configurable parameters including:
- File paths (adjust these to your local setup)
- Analysis thresholds
- Plotting options
"""

import numpy as np
from pathlib import Path

# =============================================================================
# FILE PATHS - Adjust these to your local setup
# =============================================================================

# Base directory for the project (update this to your local path)
BASE_DIR = Path('/Users/natemiska/python/zapit')

# Zapit trial log file
ZAPIT_TRIALS_LOG = BASE_DIR / 'zapit_trials.yml'

# Zapit stimulation locations log (contains AP/ML coordinates)
ZAPIT_LOCATIONS_LOG = BASE_DIR / 'zapit_log.yml'

# Allen CCF atlas data (download from Allen Institute)
# See: https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
ALLEN_CCF_ANNOTATION = Path('/Users/natemiska/python/Allen/annotation_volume_10um.npy')
ALLEN_STRUCTURE_TREE = Path('/Users/natemiska/python/Allen/structure_tree_safe_2017.csv')

# Output directory for figures
FIGURE_SAVE_PATH = Path('/Users/natemiska/Desktop/zapit_check')

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


# =============================================================================
# SESSION SELECTION CRITERIA
# =============================================================================
# These filters are passed to find_sessions_by_advanced_criteria()
# Use None to not filter on that criterion
# Use a specific value for exact match
# Use a lambda for custom filtering, e.g.: lambda x: x in ['val1', 'val2']

SESSION_FILTERS = {

    'Stimulation_Params': 'zapit',      
    # Required for Zapit analysis

    'Mouse_ID': lambda x: x in ['SWC_NM_072', 'SWC_NM_071', 'SWC_NM_057', 'SWC_NM_058', 'SWC_NM_081', 'SWC_NM_082', 'SWC_NM_085', 'SWC_NM_086', 'SWC_NM_090', 'SWC_NM_091'],                    
    # e.g., 'SWC_NM_099' or lambda x: x in [...]
    
    'Hemisphere': None,                  
    # e.g., 'both', 'left', 'right'

    'Pulse_Params': None,                
    # e.g., 'motor_bilateral_mask', '50hz'

    'Opsin': None,                       
    # e.g., 'ChR2', 'GtACR2'

    'Genetic_Line': None,                
    # e.g., 'VGAT-ChR2', 'D1-Cre'

    'Brain_Region': 'motor_bilateral',                
    # e.g., 'motor_bilateral'

    'Laser_V': None,                     
    # e.g., 2, or lambda x: x >= 1

    'Date': None,                        
    # e.g., '2024-10-24'

    'EID': None,                         
    # Specific session EID(s)
}


# =============================================================================
# PLOTTING CONFIGURATION
# =============================================================================

# Brain atlas display settings
BRAIN_BACKGROUND_COLOR = 'black'
BRAIN_BORDER_COLOR = 'white'

# Bregma position in CCF coordinates (for 10um resolution)
# Format: [AP, DV, ML] - matching original script
BREGMA_CCF = np.array([540, 0, 570])

# Scale factor (negative to handle coordinate flip)
CCF_SCALE_FACTOR = -100

# Axis limits for plotting (mm from Bregma)
# Set to None to auto-scale, or specify [min, max]
PLOT_AP_LIMITS = (-1.3, 3.8)      # AP extent (negative = posterior)
PLOT_ML_LIMITS = (0, 3.8)          # ML extent (None = show full width, or e.g., (0, 5) for right hemisphere only)

# Whether to show only right hemisphere data points
# If True, only plots ML > 0 (but still shows full brain if PLOT_ML_LIMITS is None)
SHOW_RIGHT_HEMISPHERE_ONLY = True
