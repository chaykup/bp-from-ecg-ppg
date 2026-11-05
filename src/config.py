FS = 125    # 125 Hz sampling frequency (dataset spec)

# Heart rate - physiological range, outside this is artifact or arrhythmia
HR_MIN = 40     # bpm
HR_MAX = 180    # bpm
MIN_BEAT_DISTANCE = int(FS * 60 / HR_MAX)   # samples, beats can't be closer than HR_MAX allows (41)

# ABP filter - waveform harmonics sit below ~15 Hz at 2-3 Hz heart rate, must stay below Nyquist (62.5 Hz)
ABP_LOWPASS_HZ = 15
FILTER_ORDER = 4    # steep enough roll-off without ringing, sosfiltfilt doubles effective order

# ABP peak detection - above dicrotic notch, below smallest real pulse pressure (see tune_prominence.py)
ABP_PROMINENCE = 15     # mmHg

# Label bounds - physiological range
SBP_BOUNDS = (80, 180)  # mmHg
# DBP lower bound 40, not 60: AAMI/ISO 81060-2 requires >5% of reference DBP <= 60 mmHg, and PulseDB
# (Wang et al. 2023) argues against the >60 cutoff older studies used. 60 cut 32% of Part_1 windows
# (median DBP 63.8); none fall below 45, so 40 only catches artefacts.
DBP_BOUNDS = (40, 130)  # mmHg

# Windowing - at least MIN_BEATS beats at HR_MIN, short enough that BP is roughly constant
WINDOW_S = 8                # seconds
WINDOW = WINDOW_S * FS      # samples (1000)
MIN_BEATS = 4               # fewest beats for a median label that resists one bad peak
BEAT_COUNT_TOL = 1          # max |ECG or PPG beats - ABP beats| per window. 96% of windows differ by 0-1 (an edge beat);
                            # 2+ means missed or doubled beats, e.g. NeuroKit finds 19 R-peaks vs 50 ABP beats on Part_3_Patient_0

# Flat-signal rejection - window std below this is a disconnected lead or stuck sensor.
# Smallest stds seen on 100 Part_1 records: ECG 0.03, PPG 0.11, ABP 3.3 mmHg, so these only catch dead signals.
FLAT_STD = {"ecg": 0.01, "ppg": 0.01, "abp": 1.0}

# ABP clipping - some records' transducer saturates, e.g. Part_2_Patient_1573 sits at exactly 148.02 mmHg for 14% of samples,
# so the SBP label is the clip level. Reject a window when raw ABP stays within CLIP_EPS of its window max or min for
# CLIP_RUN+ consecutive samples. On 240 random records the longest plateau is 1-3 samples on normal windows and 11-13
# on clipped ones; real peaks change by more than 0.05 mmHg over 40 ms.
CLIP_EPS = 0.05     # mmHg
CLIP_RUN = 5        # samples (40 ms)

# ECG peak-count sanity check - SciPy find_peaks vs NeuroKit on the cleaned ECG
ECG_SCIPY_PROMINENCE_FRAC = 0.3     # prominence as a fraction of the 1st-99th pct ECG range (counts stable 0.2-0.5)
PEAK_COUNT_TOL = 0.10               # flag when counts differ by more than 10% ...
PEAK_COUNT_MIN_DIFF = 2             # ... and by more than this many beats (edge beats make short records differ by 1)

# Pulse arrival time - shorter than pre-ejection period or longer than this is a mismatched beat.
# Upper bound 0.8, not 0.6: MIMIC has an unknown ECG-PPG device delay (PulseDB: up to 500 ms), so measured
# R->foot is ~0.47 s (5-95th pct 0.27-0.55) and R->peak ~0.65 s across 94 records from all four Parts.
PAT_BOUNDS = (0.1, 0.8)     # seconds

# Model inputs (pipeline.py). The table keeps all features; these choices apply when models are trained.
# SDNN is dropped: r = 0.98 with RMSSD, and SDNN needs ~5 min of beats to mean much (8 s windows); RMSSD is the
# accepted short-term HRV measure. Other correlated pairs (amplitude/area 0.84, PAT foot/rise time -0.79) stay.
DROPPED_FEATURES = ["hrv_sdnn_ms"]
# Outliers: no rows are removed. 1.5 x IQR flags 12-19% of windows on several features because the distributions are
# skewed or bimodal (PPG gain, arrhythmias), not because the values are wrong. Fixed (not data-fitted) transforms instead:
LOG_FEATURES = ["hrv_rmssd_ms"]    # log1p: skew 3.7 -> 1.1
PIR_MAX = 5.0                      # PIR above 5 comes from a raw PPG foot near 0 (9 windows; 99.9th pct is 3.1)

# Train/test split (split.py) - grouped by record so no record's windows are on both sides
TEST_SIZE = 0.2
CV_FOLDS = 5

RANDOM_STATE = 42
