import neurokit2 as nk
from load import load_record
import config
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

# Load
record_id, ppg, abp, ecg = load_record("data/raw/Part_1.mat", 0)

# ECG
ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=config.FS)
ecg_signals_bool, ecg_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=config.FS)
ecg_R_peaks_idx = ecg_info['ECG_R_Peaks']
ecg_R_peaks_val = ecg_cleaned[ecg_R_peaks_idx]

# PPG
ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=config.FS)
ppg_signals_bool, ppg_info = nk.ppg_peaks(ppg_cleaned, sampling_rate=config.FS)
ppg_peaks_idx = ppg_info['PPG_Peaks']
ppg_peaks_val = ppg_cleaned[ppg_peaks_idx]

# Foot
ppg_foot_idx = []
for i in range(len(ppg_peaks_idx) - 1):
    start = ppg_peaks_idx[i]
    end = ppg_peaks_idx[i+1]
    local_ppg_min_idx = np.argmin(ppg_cleaned[start:end])
    idx = start + local_ppg_min_idx
    ppg_foot_idx.append(idx)
ppg_foot_val = ppg_cleaned[ppg_foot_idx]

# ABP
sos = butter(4, 15, btype='low', fs=config.FS, output='sos')
abp_cleaned = sosfiltfilt(sos, abp)
sys_peaks_idx, sys_properties = find_peaks(abp_cleaned, distance=40, prominence=0.1)
dys_troughs_idx, dys_properties = find_peaks(-abp_cleaned, distance=40, prominence=0.1)

# Overlay plot of raw vs clean ECG
plt.plot((np.arange(len(ecg))), ecg, label='Raw', color='orange', linestyle='--')
plt.plot((np.arange(len(ecg_cleaned))), ecg_cleaned, label='Cleaned', color='purple')
plt.xlim(0, 250)
plt.xlabel('Time(seconds)')
plt.title('ECG Raw vs Cleaned')
plt.legend()
plt.grid()
plt.show()

# Plot ABP
plt.plot((np.arange(len(abp_cleaned))), abp_cleaned, label='ABP', color='red')
plt.xlim(1,250)
plt.xlabel('Time(seconds)')
plt.legend()
plt.grid()
plt.show()