"""
Medicaid Forecast Visualization Script
Plots historical data with ML and Stats forecast data for ATC1 and ATC2 levels

Filters to a SPECIFIC STATE (e.g., Indiana) for both historical and forecast data.
Skips plots if forecast data is not available for the selected state.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import numpy as np
from datetime import datetime

user='Lilian'
# ============================================================================
# FILE PATHS - Update these paths according to your environment
# ============================================================================
ML_FORECAST_PATH = rf"C:\Users\{user}\OneDrive - purdue.edu\VS code\Data\ATC\Graphics_aggregated\ML.xlsx"
STATS_FORECAST_PATH = rf"C:\Users\{user}\OneDrive - purdue.edu\VS code\Data\ATC\Graphics_aggregated\Stats.xlsx"
HISTORICAL_ATC1_PATH = rf"C:\Users\{user}\OneDrive - purdue.edu\VS code\Data\ATC\Graphics_aggregated\Historical_atc1.csv"
HISTORICAL_ATC2_PATH = rf"C:\Users\{user}\OneDrive - purdue.edu\VS code\Data\ATC\Graphics_aggregated\Historical_atc2.csv"

# Output paths for saved figures
OUTPUT_DIR = rf"C:\Users\{user}\OneDrive - purdue.edu\VS code\Data\ATC\Graphics_aggregated\plots\\"

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_CLASSES_TO_DISPLAY = 10  # Number of ATC classes to show per plot
FIGSIZE = (16, 10)
DPI = 150

# Date range filter (set to None to show all data)
START_DATE = "2023-01-01"  # Format: "YYYY-MM-DD" or None for no filter
END_DATE = None            # Format: "YYYY-MM-DD" or None for no filter

# Class selection method: 'Units Reimbursed' or 'Number of Prescriptions'
CLASS_SELECTION_METHOD = 'Units Reimbursed'

# ============================================================================
# STATE FILTER - SET YOUR DESIRED STATE HERE
# ============================================================================
# Filter ALL data (historical AND forecast) to this specific state.
# Only plots where forecast data exists for this state will be generated.
#
# Examples:
#   'IN'  - Indiana only
#   'IL'  - Illinois only
#   'CA'  - California only
#   None  - No filter (use whatever state each forecast has)

STATE_FILTER = 'IN'  # <-- SET YOUR STATE HERE (e.g., 'IN' for Indiana)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def extract_state(unique_id):
    """Extract state code from unique_id (e.g., 'IN_A' -> 'IN')"""
    return unique_id.split('_')[0]

def load_forecast_data(filepath, sheet_name, normalize_ids=False, state_filter=None):
    """Load forecast data from Excel file, optionally filtered by state."""
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df['ds'] = pd.to_datetime(df['ds'])
    
    if normalize_ids:
        def normalize_id(uid):
            parts = uid.split('_')
            if len(parts) >= 3 and parts[0] == parts[1]:
                return parts[0] + '_' + '_'.join(parts[2:])
            return uid
        df['unique_id'] = df['unique_id'].apply(normalize_id)
    
    # Apply state filter
    if state_filter is not None:
        df = df[df['unique_id'].apply(extract_state) == state_filter]
    
    return df

def load_historical_atc1(filepath, state_filter=None):
    """Load and prepare historical ATC1 data, filtered by state."""
    df = pd.read_csv(filepath)
    df['ds'] = pd.to_datetime(df['Year'].astype(str) + '-' + 
                              ((df['Quarter'] - 1) * 3 + 1).astype(str).str.zfill(2) + '-01')
    df['unique_id'] = df['ATC1 Class']
    
    if state_filter is not None:
        df = df[df['State'] == state_filter]
    
    return df

def load_historical_atc2(filepath, state_filter=None):
    """Load and prepare historical ATC2 data, filtered by state."""
    df = pd.read_csv(filepath)
    df['ds'] = pd.to_datetime(df['Year'].astype(str) + '-' + 
                              ((df['Quarter'] - 1) * 3 + 1).astype(str).str.zfill(2) + '-01')
    df['unique_id'] = df['State'] + '_' + df['ATC2 Class']
    
    if state_filter is not None:
        df = df[df['State'] == state_filter]
    
    return df

# ============================================================================
# CLASS SELECTION FUNCTIONS
# ============================================================================

def get_top_classes(forecast_df, historical_df, n=10, method='Units Reimbursed'):
    """Get top N unique_ids that exist in BOTH forecast and historical data."""
    forecast_ids = set(forecast_df['unique_id'].unique())
    historical_ids = set(historical_df['unique_id'].unique())
    
    common_ids = forecast_ids.intersection(historical_ids)
    
    if len(common_ids) == 0:
        return []
    
    hist_filtered = historical_df[historical_df['unique_id'].isin(common_ids)]
    totals = hist_filtered.groupby('unique_id')[method].sum().sort_values(ascending=False)
    
    return totals.head(n).index.tolist()

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_forecast_with_history(forecast_df, historical_df, metric_col, unique_ids,
                                title, ylabel, ax, start_date=None, end_date=None):
    """Plot historical data with forecast for specified unique_ids.
    
    Historical data is automatically cut off at the forecast start date
    to create a visual gap between historical and forecast lines.
    """
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_ids)))
    
    start_dt = pd.to_datetime(start_date) if start_date else None
    end_dt = pd.to_datetime(end_date) if end_date else None
    
    # Get forecast start date to cut off historical data
    forecast_start = forecast_df['ds'].min() if len(forecast_df) > 0 else None
    
    if metric_col == 'Units Reimbursed':
        scale = 1e9
        scale_label = ' (Billions)'
    elif metric_col == 'Number of Prescriptions':
        scale = 1e6
        scale_label = ' (Millions)'
    else:
        scale = 1.0
        scale_label = ''

    ylabel_scaled = f"{ylabel}{scale_label}"

    for i, uid in enumerate(unique_ids):
        color = colors[i]
        
        hist_data = historical_df[historical_df['unique_id'] == uid].copy()
        hist_data = hist_data.sort_values('ds')
        
        # Apply date filters
        if start_dt is not None:
            hist_data = hist_data[hist_data['ds'] >= start_dt]
        if end_dt is not None:
            hist_data = hist_data[hist_data['ds'] <= end_dt]
        
        # CUT OFF historical data BEFORE forecast starts (creates the gap)
        if forecast_start is not None:
            hist_data = hist_data[hist_data['ds'] < forecast_start]
        
        fcst_data = forecast_df[forecast_df['unique_id'] == uid].copy()
        fcst_data = fcst_data.sort_values('ds')
        
        if start_dt is not None:
            fcst_data = fcst_data[fcst_data['ds'] >= start_dt]
        if end_dt is not None:
            fcst_data = fcst_data[fcst_data['ds'] <= end_dt]
        
        if len(hist_data) > 0:
            ax.plot(hist_data['ds'], hist_data[metric_col] / scale, 
                   color=color, linewidth=1.5, label=f'{uid}')
        
        if len(fcst_data) > 0:
            ax.plot(fcst_data['ds'], fcst_data['forecast_point'] / scale, 
                   color=color, linewidth=2, linestyle='--')
            ax.fill_between(fcst_data['ds'], 
                          fcst_data['forecast_low_pop'] / scale, 
                          fcst_data['forecast_high_pop'] / scale,
                          color=color, alpha=0.2)
    
    if forecast_start is not None:
        ax.axvline(x=forecast_start, color='gray', linestyle=':', linewidth=1.5, 
                  label='Forecast Start')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel(ylabel_scaled, fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    def _format_quarter(x, _pos=None):
        dt = mdates.num2date(x)
        quarter = (dt.month - 1) // 3 + 1
        return f"{dt.year}Q{quarter}"

    ax.xaxis.set_major_formatter(FuncFormatter(_format_quarter))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(x, ',.3f')))

def create_single_plot(approach_name, forecast_df, historical_df, metric_col, 
                       top_classes, level, state, start_date, end_date, n_classes):
    """Create a single plot figure."""
    
    date_range_str = ""
    if start_date or end_date:
        start_str = start_date[:4] if start_date else "Start"
        end_str = end_date[:4] if end_date else "End"
        date_range_str = f" ({start_str} - {end_str})"
    
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    fig.suptitle(f'{approach_name} Forecast - State: {state}{date_range_str}', 
                 fontsize=16, fontweight='bold')
    
    metric_name = 'Units Reimbursed' if metric_col == 'Units Reimbursed' else 'Number of Prescriptions'
    plot_forecast_with_history(forecast_df, historical_df, metric_col, top_classes,
                               f'{level} Level - {metric_name} (Top {n_classes})',
                               metric_name, ax, start_date, end_date)
    plt.tight_layout()
    
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Medicaid Forecast Visualization Script")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  STATE FILTER: {STATE_FILTER}")
    print(f"  Date range: {START_DATE} to {END_DATE if END_DATE else 'End'}")
    print(f"  Class selection: {CLASS_SELECTION_METHOD}")
    print(f"  Top N classes: {NUM_CLASSES_TO_DISPLAY}")
    
    # Load historical data filtered to state
    print(f"\nLoading historical data for state: {STATE_FILTER}...")
    hist_atc1 = load_historical_atc1(HISTORICAL_ATC1_PATH, state_filter=STATE_FILTER)
    hist_atc2 = load_historical_atc2(HISTORICAL_ATC2_PATH, state_filter=STATE_FILTER)
    print(f"  Historical ATC1: {hist_atc1.shape[0]} rows, IDs: {sorted(hist_atc1['unique_id'].unique())}")
    print(f"  Historical ATC2: {hist_atc2.shape[0]} rows")
    
    # Define what to plot
    plots_config = [
        # (approach, forecast_path, normalize_ids, level, sheet_ur, sheet_nop, historical_df)
        ('ML', ML_FORECAST_PATH, False, 'ATC1', 'UR_ATC1', 'NoP_ATC1', hist_atc1),
        ('ML', ML_FORECAST_PATH, False, 'ATC2', 'UR_ATC2', 'NoP_ATC2', hist_atc2),
        ('Stats', STATS_FORECAST_PATH, True, 'ATC1', 'UR_ATC1', 'NoP_ATC1', hist_atc1),
        ('Stats', STATS_FORECAST_PATH, True, 'ATC2', 'UR_ATC2', 'NoP_ATC2', hist_atc2),
    ]
    
    plots_created = 0
    plots_skipped = 0
    
    for approach, fcst_path, normalize, level, sheet_ur, sheet_nop, hist_df in plots_config:
        print(f"\n{'-' * 50}")
        print(f"Processing: {approach} {level}")
        print(f"{'-' * 50}")
        
        # Load forecast data filtered to state
        ur_fcst = load_forecast_data(fcst_path, sheet_ur, normalize_ids=normalize, state_filter=STATE_FILTER)
        nop_fcst = load_forecast_data(fcst_path, sheet_nop, normalize_ids=normalize, state_filter=STATE_FILTER)
        
        # Check if forecast data exists for this state
        if len(ur_fcst) == 0:
            print(f"  ⚠️  SKIPPED: No {STATE_FILTER} forecast data in {approach} {level}")
            plots_skipped += 2  # Skip both UR and NoP
            continue
        
        print(f"  Forecast data found: {len(ur_fcst)} rows")
        print(f"  Forecast IDs: {sorted(ur_fcst['unique_id'].unique())[:5]}...")
        
        # Get top classes
        top_classes = get_top_classes(ur_fcst, hist_df, n=NUM_CLASSES_TO_DISPLAY, method=CLASS_SELECTION_METHOD)
        print(f"  Top classes: {top_classes}")
        
        if len(top_classes) == 0:
            print(f"  ⚠️  SKIPPED: No matching classes between forecast and historical")
            plots_skipped += 2
            continue
        
        # Create UR plot
        fig_ur = create_single_plot(approach, ur_fcst, hist_df, 'Units Reimbursed',
                                    top_classes, level, STATE_FILTER, START_DATE, END_DATE, NUM_CLASSES_TO_DISPLAY)
        out_path = f"{OUTPUT_DIR}\\{approach}_{level}_Units_Reimbursed_{STATE_FILTER}.png"
        fig_ur.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"  ✓ Saved: {out_path}")
        plots_created += 1
        
        # Create NoP plot
        fig_nop = create_single_plot(approach, nop_fcst, hist_df, 'Number of Prescriptions',
                                     top_classes, level, STATE_FILTER, START_DATE, END_DATE, NUM_CLASSES_TO_DISPLAY)
        out_path = f"{OUTPUT_DIR}\\{approach}_{level}_Number_of_Prescriptions_{STATE_FILTER}.png"
        fig_nop.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"  ✓ Saved: {out_path}")
        plots_created += 1
        
        plt.close(fig_ur)
        plt.close(fig_nop)
    
    print("\n" + "=" * 70)
    print(f"SUMMARY for state {STATE_FILTER}:")
    print(f"  Plots created: {plots_created}")
    print(f"  Plots skipped (no data): {plots_skipped}")
    print("=" * 70)
    
    if plots_skipped > 0:
        print(f"\nNote: Some plots were skipped because the forecast files")
        print(f"      don't contain data for state '{STATE_FILTER}'.")
    
    plt.show()
    print("\nDone!")