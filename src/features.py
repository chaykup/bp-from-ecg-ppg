import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from scipy.signal import peak_widths

import config
from load import load_record
from signals import process_record, window_record

# Feature columns in the order they appear in the table. Only ECG/PPG are used: any ABP-derived column would be leakage.
FEATURE_COLS = [
    "pat_foot_s", "pat_peak_s", "hr_bpm", "hrv_sdnn_ms", "hrv_rmssd_ms",
    "ppg_amplitude", "rise_time_s", "pulse_width_50_s", "pulse_area", "pir", "ecg_quality",
]


def _median(x) -> float:
    """Median that returns NaN instead of warning on an empty array, so a window with no valid beats gets a missing feature."""
    return float(np.median(x)) if len(x) else np.nan


def pair_beats(r_peaks: np.ndarray, ppg_foot: np.ndarray, ppg_peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair each R-peak with its PPG pulse: R -> first foot at least PAT_BOUNDS[0] later -> the peak after that foot.

    Pairing R -> next PPG peak fails on this dataset: MIMIC has an unknown ECG-PPG device delay,
    so R->peak (~0.65 s) is often longer than one beat and "next peak" lands on the previous pulse.
    Going through the foot pairs every beat correctly. Returns matched (r, foot, peak) index arrays
    with pairs whose PAT falls outside PAT_BOUNDS dropped.
    """
    lo = int(np.ceil(config.PAT_BOUNDS[0] * config.FS))     # ceil/floor so no PAT falls outside the bounds
    hi = int(np.floor(config.PAT_BOUNDS[1] * config.FS))
    j = np.searchsorted(ppg_foot, r_peaks + lo)
    ok = j < len(ppg_foot)
    r, foot = r_peaks[ok], ppg_foot[j[ok]]
    k = np.searchsorted(ppg_peaks, foot)
    ok = k < len(ppg_peaks)
    r, foot, peak = r[ok], foot[ok], ppg_peaks[k[ok]]
    ok = (foot - r <= hi) & (peak - r <= hi)
    return r[ok], foot[ok], peak[ok]


def extract_features(window: dict) -> dict:
    """Compute ECG/PPG features for one window from signals.window_record.

    Returns record_id, part, window_idx, the FEATURE_COLS, sbp and dbp. A feature with no valid beats is NaN.
    """
    fs = config.FS
    ecg, ppg, ppg_raw = window["ecg"], window["ppg"], window["ppg_raw"]
    r_peaks, ppg_peaks, ppg_foot = window["r_peaks"], window["ppg_peaks"], window["ppg_foot"]

    # Pulse arrival time (arrival, not transit: it includes the pre-ejection period)
    r, foot, peak = pair_beats(r_peaks, ppg_foot, ppg_peaks)
    pat_foot = (foot - r) / fs
    pat_peak = (peak - r) / fs

    # Heart rate and HRV
    rr = np.diff(r_peaks) / fs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # NeuroKit warns that 8 s is short for HRV; SDNN/RMSSD are still defined
        hrv = nk.hrv_time(r_peaks, sampling_rate=fs)

    # PPG morphology, beat by beat: each foot with the peak that follows it
    k = np.searchsorted(ppg_peaks, ppg_foot)
    ok = k < len(ppg_peaks)
    beat_foot, beat_peak = ppg_foot[ok], ppg_peaks[k[ok]]
    amplitude = ppg[beat_peak] - ppg[beat_foot]
    rise_time = (beat_peak - beat_foot) / fs
    # Raw PPG saturates at 0 and ~4; skip feet at 0 so PIR doesn't divide by zero
    raw_foot = ppg_raw[beat_foot]
    pir = ppg_raw[beat_peak][raw_foot > 0] / raw_foot[raw_foot > 0]
    # NeuroKit occasionally marks a peak that isn't a local max in the window (prominence 0 -> width 0); skip those
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        widths = peak_widths(ppg, ppg_peaks, rel_height=0.5)[0] / fs
    widths = widths[widths > 0]
    # Area over each foot-to-foot beat, above the starting foot
    area = [np.trapezoid(ppg[a:b + 1] - ppg[a], dx=1 / fs) for a, b in zip(ppg_foot[:-1], ppg_foot[1:])]

    quality = nk.ecg_quality(ecg, rpeaks=r_peaks, sampling_rate=fs)

    return {
        "record_id": window["record_id"],
        "part": window["record_id"].split("_Patient_")[0],
        "window_idx": window["window_idx"],
        "pat_foot_s": _median(pat_foot),
        "pat_peak_s": _median(pat_peak),
        "hr_bpm": 60 / np.median(rr),
        "hrv_sdnn_ms": float(hrv["HRV_SDNN"].iloc[0]),
        "hrv_rmssd_ms": float(hrv["HRV_RMSSD"].iloc[0]),
        "ppg_amplitude": _median(amplitude),
        "rise_time_s": _median(rise_time),
        "pulse_width_50_s": _median(widths),
        "pulse_area": _median(area),
        "pir": _median(pir),
        "ecg_quality": float(np.mean(quality)),
        "sbp": window["sbp"],
        "dbp": window["dbp"],
    }


def main():
    """Print feature medians for one long record and plot PAT vs SBP, which should trend negative within a record."""
    record_id, ppg, abp, ecg = load_record(Path("data/raw/Part_1.mat"), 100)   # long record, SBP SD 16 mmHg
    windows, _ = window_record(record_id, process_record(ppg, abp, ecg))
    rows = [extract_features(w) for w in windows]

    print(f"\n{record_id}: {len(rows)} windows")
    for col in FEATURE_COLS:
        vals = np.array([r[col] for r in rows])
        print(f"    {col:18s} median={np.nanmedian(vals):9.3f}  NaN={np.isnan(vals).sum()}")

    sbp = np.array([r["sbp"] for r in rows])
    fig, ax = plt.subplots(1, 2, sharey=True)
    for a, col in zip(ax, ["pat_foot_s", "pat_peak_s"]):
        x = np.array([r[col] for r in rows])
        corr = np.corrcoef(x, sbp)[0, 1]
        a.scatter(x, sbp, s=10)
        a.set_title(f"{col} (r = {corr:.2f})")
        a.set_xlabel("seconds")
        a.grid()
    ax[0].set_ylabel("SBP (mmHg)")
    fig.suptitle(record_id)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
