import neurokit2 as nk
from load import load_record
import config
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from pathlib import Path

# Load
record_id, ppg, abp, ecg = load_record(Path("data/raw/Part_1.mat"), 795)

# ECG
ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=config.FS)
ecg_signals_mask, ecg_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=config.FS)
ecg_R_peaks_idx = ecg_info['ECG_R_Peaks']

# PPG
ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=config.FS)
ppg_signals_mask, ppg_info = nk.ppg_peaks(ppg_cleaned, sampling_rate=config.FS)
ppg_peaks_idx = ppg_info['PPG_Peaks']

# Foot - foot belongs to peak_idx + 1 and first beat has no foot
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
sys_peaks_idx, sys_properties = find_peaks(abp_cleaned, distance=40, prominence=15, height=-np.inf)
dia_troughs_idx, dia_properties = find_peaks(-abp_cleaned, distance=40, prominence=15, height=-np.inf)

# Debug Plot
fig, ax = plt.subplots(3, sharex=True)

# fig.supxlabel("Time (seconds)")
ecg_t = np.arange(len(ecg_cleaned)) / config.FS
ax[0].plot(ecg_t, ecg_cleaned, label="ECG", color="blue")
ax[0].plot(ecg_t[ecg_R_peaks_idx], ecg_cleaned[ecg_R_peaks_idx], "x")
ax[0].set_ylim(-0.3,1)
ax[0].grid()
ax[0].legend()
ppg_t = np.arange(len(ppg_cleaned)) / config.FS
ax[1].plot(ppg_t, ppg_cleaned, label="PPG", color="purple")
ax[1].plot(ppg_t[ppg_peaks_idx], ppg_cleaned[ppg_peaks_idx], "x")
ax[1].plot(ppg_t[ppg_foot_idx], ppg_foot_val, "x")
ax[1].set_ylim(-1.25,2.25)
ax[1].grid()
ax[1].legend()
abp_t = np.arange(len(abp_cleaned)) / config.FS
ax[2].plot(abp_t, abp_cleaned, label="ABP", color="red")
ax[2].plot(abp_t[sys_peaks_idx], abp_cleaned[sys_peaks_idx], "x")
ax[2].plot(abp_t[dia_troughs_idx], abp_cleaned[dia_troughs_idx], "x")
ax[2].set_ylabel("mmHg")
ax[2].set_ylim(30, 150)
ax[2].grid()
ax[2].legend()
ax[2].set_xlabel("Seconds")
plt.show()

# Debug summary
print(f"\n{record_id}")
for name, idx in [("R", ecg_R_peaks_idx), ("PPG", ppg_peaks_idx), ("foot", ppg_foot_idx), ("sys", sys_peaks_idx), ("dia", dia_troughs_idx)]:
    ibi = np.diff(idx) / config.FS
    print(f"{name:5s} n={len(idx):4d} HR={60/np.median(ibi):5.1f} IBI min/max={ibi.min():.2f}/{ibi.max():.2f}s")
