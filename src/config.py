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
DBP_BOUNDS = (60, 130)  # mmHg

# Windowing - at least MIN_BEATS beats at HR_MIN, short enough that BP is roughly constant
WINDOW_S = 8                # seconds
WINDOW = WINDOW_S * FS      # samples (1000)
MIN_BEATS = 4               # fewest beats for a median label that resists one bad peak

# ECG peak-count sanity check - SciPy find_peaks vs NeuroKit on the cleaned ECG
ECG_SCIPY_PROMINENCE_FRAC = 0.3     # prominence as a fraction of the 1st-99th pct ECG range (counts stable 0.2-0.5)
PEAK_COUNT_TOL = 0.10               # flag when counts differ by more than 10% ...
PEAK_COUNT_MIN_DIFF = 2             # ... and by more than this many beats (edge beats make short records differ by 1)

# Pulse arrival time - shorter than pre-ejection period or longer than this is a mismatched beat
PAT_BOUNDS = (0.1, 0.6)     # seconds

RANDOM_STATE = 42
