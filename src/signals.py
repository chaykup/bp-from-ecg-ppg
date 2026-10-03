import neurokit2 as nk
from load import load_record
import config
import matplotlib.pyplot as plt
import numpy as np

# Load
record_id, ppg, abp, ecg = load_record("data/raw/Part_1.mat", 0)

# Process
ecg_signal, ecg_info = nk.ecg_process(ecg, sampling_rate=config.FS)
ppg_signal, ppg_info = nk.ppg_process(ppg, sampling_rate=config.FS)

# Plot
fig, ax = plt.subplots(2, 1, sharex=True)
fig.suptitle('Raw ECG vs. Cleaned ECG')

# Raw
ax[0].plot((np.arange(len(ecg_signal.ECG_Raw))), ecg_signal.ECG_Raw, label='Raw', color='yellow')
ax[0].grid()

# Cleaned
ax[1].plot((np.arange(len(ecg_signal.ECG_Clean))), ecg_signal.ECG_Clean, label='Clean', color='purple')
ax[1].grid()
ax[1].set_xlabel('Time(seconds)')
plt.show()