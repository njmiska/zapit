# Zapit Optogenetics Analysis Pipeline

Analysis pipeline for behavioral data from the IBL task with laser scanning optogenetic inhibition (Zapit system).

## Overview

Mice perform a two-alternative forced choice (2-AFC) visual discrimination task with prior probability blocks while receiving bilateral optogenetic stimulation at one of 52 cortical locations. This pipeline analyzes the behavioral effects of stimulation at each location, with a focus on reaction time changes.

### Key Analyses
- **Reaction time effects**: How stimulation at each cortical location affects reaction times compared to control (no-stim) trials
- **Brain atlas visualization**: RT effects displayed on Allen CCF dorsal cortical projection with statistical significance indicated by marker size

### Additional Analyses (available but commented out)
- **Quiescent period effects**: How stimulation affects the time mice spend in the quiescent period before stimulus onset
- **Bias reduction effects**: How stimulation modulates block-induced choice bias at low contrasts

## Installation

### Prerequisites
1. Python 3.8+
2. IBL environment (for data access): https://docs.internationalbrainlab.org/02_installation.html

### Required packages
```bash
pip install numpy pandas matplotlib seaborn scipy
pip install ONE-api ibllib  # IBL data access
```

### Allen CCF Atlas Data

Download the annotation volume from the Allen Institute:
1. Go to: https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
2. Download `annotation_volume_10um.npy`
3. Download `structure_tree_safe_2017.csv`
4. Update paths in `config.py`

## File Structure

```
zapit_analysis/
├── zapit_analysis.py    # Main analysis script
├── zapit_helpers.py     # Helper functions module
├── config.py            # Configuration, paths, and session filters
├── metadata_zapit.py    # Session metadata database
├── zapit_log.yml        # Log of all stimulus locations to be read by main script
└── README.md            # This file
```

## Configuration

All configuration is centralized in `config.py`. Key settings include:

### File Paths
```python
BASE_DIR = Path('/your/path/to/project')
ZAPIT_TRIALS_LOG = BASE_DIR / 'zapit_trials.yml'
ZAPIT_LOCATIONS_LOG = BASE_DIR / 'zapit_log_YYYY_MM_DD__HH-MM.yml'
ALLEN_CCF_ANNOTATION = Path('/path/to/annotation_volume_10um.npy')
ALLEN_STRUCTURE_TREE = Path('/path/to/structure_tree_safe_2017.csv')
FIGURE_SAVE_PATH = Path('/path/to/output/figures')
```

### Session Selection Filters
```python
SESSION_FILTERS = {
    'Stimulation_Params': 'zapit',      # Required for Zapit analysis
    'Mouse_ID': None,                    # e.g., 'SWC_NM_099' or None for all mice
    'Brain_Region': 'motor_bilateral',   # Target region
    'Laser_V': None,                     # Specify sessions utilizing different zapit laser voltage (this is generally 2 for all sessions, though there are some at 3)
    # ... additional filters available
}
```

### Quality Control Thresholds
```python
BASELINE_PERFORMANCE_THRESHOLD = 0.79  # Min accuracy at high contrasts
STIM_PERFORMANCE_THRESHOLD = 0.5       # Min stim trial accuracy
MIN_BIAS_THRESHOLD_ZAPIT = 1.0         # Min bias shift to include session
MIN_NUM_TRIALS = 300                   # Min trials per session
RT_THRESHOLD = 30                      # Max RT to include trial (seconds)
```

### Plotting Configuration
```python
BREGMA_CCF = np.array([540, 0, 570])   # Bregma in CCF coordinates [AP, DV, ML]
CCF_SCALE_FACTOR = -100                 # Scale factor (negative handles flip)
PLOT_AP_LIMITS = (-1.2, 3.6)           # AP axis range (mm from Bregma)
PLOT_ML_LIMITS = (0, 3.7)              # ML axis range (right hemisphere)
SHOW_RIGHT_HEMISPHERE_ONLY = True      # Display only right hemisphere
```

## Usage

### Basic Usage
ensure all settings are correct in config file
```bash
conda activate iblenv
cd path/to/repo
python zapit_analysis.py
```

### Workflow
1. Edit `config.py` to set file paths and session filters
2. Run `zapit_analysis.py`
3. Outputs are saved to `FIGURE_SAVE_PATH`

## Outputs

### Figures
| File | Description |
|------|-------------|
| `{prefix}_RT_heatmap.png` | Reaction time effects on brain atlas. Color = RT deviation from control; Size = statistical significance |

### Data Tables
| File | Description |
|------|-------------|
| `{prefix}_RT_statistics.csv` | Complete RT statistics table with coordinates, means, SEMs, p-values, and trial counts for all stimulation locations |

### Additional Outputs (when additional figures enabled)
- `{prefix}_QP_heatmap.png` - Quiescent period effects
- `{prefix}_bias_reduction_heatmap.png` - Bias reduction effects
- `{prefix}_results.csv` - Summary statistics for all metrics

## Figure Interpretation

### RT Heatmap
- **Color**: Reaction time on stim trials
  - Blue = faster than control (RT decrease)
  - Red = slower than control (RT increase)
  - Colormap centered on control RT using TwoSlopeNorm
- **Circle size**: Statistical significance (larger = more significant)
  - Size = -100 × log₁₀(p-value)
  - Significance levels shown in legend
- **Opacity**: α = 1.0 if p < 0.05, α = 0.5 otherwise
- **Brain regions**: White borders show Allen CCF region boundaries

## Statistical Methods

### Reaction Time Analysis
- **Test**: Welch's independent t-test (unequal variances)
- **Comparison**: Each stim condition vs pooled control (no-stim) trials
- **Multiple conditions**: 52 stim locations tested independently
- **Reported metrics**: Mean RT, SEM, p-value, sample size

### Session Quality Control
Sessions are excluded if they fail to meet:
1. Minimum trial count threshold
2. Baseline performance threshold at high contrasts
3. Minimum bias shift threshold (ensures task engagement)

## Helper Functions

Key functions in `zapit_helpers.py`:

| Function | Description |
|----------|-------------|
| `load_session_data()` | Load trials and wheel data from IBL database |
| `parse_zapit_log()` | Parse stimulation location log file |
| `build_stim_location_dict()` | Map trial numbers to stim locations |
| `load_stim_locations_coordinates()` | Load AP/ML coordinates for each location |
| `generate_mip_with_borders()` | Create brain atlas dorsal projection with region borders |
| `check_session_accuracy()` | Validate session performance quality |
| `compute_session_bias_shift()` | Calculate bias modulation strength |

## Troubleshooting

### "Failed to load eid"
- Check network connection to IBL servers
- Verify EID exists in Alyx database
- Try: `one.search(subject='mousename')` to find valid EIDs

### "No sessions met criteria"
- Check `SESSION_FILTERS` in `config.py`
- Verify sessions in `metadata_zapit.py` have matching metadata
- Temporarily lower quality thresholds to diagnose

### Brain atlas alignment issues
- Verify `BREGMA_CCF` coordinates match your CCF version
- Check that `CCF_SCALE_FACTOR` is -100 (negative handles coordinate flip)
- Adjust `PLOT_AP_LIMITS` and `PLOT_ML_LIMITS` to frame your data

### Missing data points in figure
- Some stim locations may have insufficient trials
- Check `{prefix}_RT_statistics.csv` for trial counts per location
- Conditions with 0 trials are not plotted

## Authors

**Nate Miska**

Developed with AI pair-programming assistance (Claude, Anthropic) for code refactoring and documentation.

## Citation

If using this code, please cite:
- The International Brain Laboratory (IBL)
- Allen Institute for Brain Science (Allen CCF)

