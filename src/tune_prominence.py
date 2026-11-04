import argparse
import config
import matplotlib.pyplot as plt
import numpy as np
from load import load_record
from scipy.signal import butter, sosfiltfilt, find_peaks
from pathlib import Path

# Histogram of every ABP peak/trough prominence to pick ABP_PROMINENCE.
# Real beats and dicrotic notches/noise should form two humps - set the threshold in the valley.
# No distance= here so prominence alone has to separate them.

parser = argparse.ArgumentParser()
parser.add_argument("--part", default="data/raw/Part_1.mat")
parser.add_argument("--limit", type=int, default=100)
args = parser.parse_args()

sos = butter(config.FILTER_ORDER, config.ABP_LOWPASS_HZ, btype='low', fs=config.FS, output='sos')

sys_prom = []
dia_prom = []
for i in range(args.limit):
    record_id, ppg, abp, ecg = load_record(Path(args.part), i)
    abp_cleaned = sosfiltfilt(sos, abp)
    _, sys_properties = find_peaks(abp_cleaned, prominence=0)
    _, dia_properties = find_peaks(-abp_cleaned, prominence=0)
    sys_prom.append(sys_properties["prominences"])
    dia_prom.append(dia_properties["prominences"])
sys_prom = np.concatenate(sys_prom)
dia_prom = np.concatenate(dia_prom)

# Summary
for name, prom in [("sys", sys_prom), ("dia", dia_prom)]:
    print(f"{name}: n={len(prom)}  5/50/95th pct={np.percentile(prom, [5, 50, 95]).round(1)} mmHg")
    for t in [5, 10, 15, 20, 25]:
        print(f"    kept at prominence >= {t:2d}: {np.sum(prom >= t)}")

# Plot
bins = np.arange(0, 100, 1)
fig, ax = plt.subplots(2, sharex=True)
ax[0].hist(sys_prom, bins=bins, color="red")
ax[0].set_title(f"Systolic peak prominence ({args.limit} records)")
ax[1].hist(dia_prom, bins=bins, color="blue")
ax[1].set_title("Diastolic trough prominence")
for a in ax:
    a.axvline(config.ABP_PROMINENCE, color="black", linestyle="--", label=f"ABP_PROMINENCE = {config.ABP_PROMINENCE}")
    a.set_yscale("log")
    a.set_ylabel("Count (log)")
    a.grid()
    a.legend()
ax[1].set_xlabel("Prominence (mmHg)")
plt.tight_layout()
plt.show()
