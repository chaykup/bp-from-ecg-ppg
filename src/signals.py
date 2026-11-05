import neurokit2 as nk
from load import load_record
import config
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from pathlib import Path
from collections import Counter

# ABP low-pass filter, designed once at import
ABP_SOS = butter(config.FILTER_ORDER, config.ABP_LOWPASS_HZ, btype='low', fs=config.FS, output='sos')


def process_record(ppg: np.ndarray, abp: np.ndarray, ecg: np.ndarray) -> dict:
    """Filter raw PPG, ABP and ECG and detect peaks.

    Returns a dict of cleaned signals (ecg, ppg, abp), the raw ppg_raw, and sample-index arrays:
    r_peaks (ECG R-peaks), ppg_peaks, ppg_foot, sys_peaks (ABP systolic), dia_troughs (ABP diastolic).
    """
    # ECG
    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=config.FS)
    _, ecg_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=config.FS)
    r_peaks = np.asarray(ecg_info['ECG_R_Peaks'], dtype=int)

    # PPG
    ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=config.FS)
    _, ppg_info = nk.ppg_peaks(ppg_cleaned, sampling_rate=config.FS)
    ppg_peaks = np.asarray(ppg_info['PPG_Peaks'], dtype=int)

    # Foot - foot belongs to peak_idx + 1 and first beat has no foot
    ppg_foot = []
    for i in range(len(ppg_peaks) - 1):
        start = ppg_peaks[i]
        end = ppg_peaks[i+1]
        ppg_foot.append(start + np.argmin(ppg_cleaned[start:end]))
    ppg_foot = np.asarray(ppg_foot, dtype=int)

    # ABP
    abp_cleaned = sosfiltfilt(ABP_SOS, abp)
    sys_peaks, _ = find_peaks(abp_cleaned, distance=config.MIN_BEAT_DISTANCE, prominence=config.ABP_PROMINENCE, height=-np.inf)
    dia_troughs, _ = find_peaks(-abp_cleaned, distance=config.MIN_BEAT_DISTANCE, prominence=config.ABP_PROMINENCE, height=-np.inf)

    return {
        "ecg": ecg_cleaned,
        "ppg": ppg_cleaned,
        "ppg_raw": ppg,     # raw PPG keeps its DC level, needed for PIR
        "abp": abp_cleaned,
        "abp_raw": abp,     # raw ABP for the clipping check; the low-pass filter smooths a clipped plateau
        "r_peaks": r_peaks,
        "ppg_peaks": ppg_peaks,
        "ppg_foot": ppg_foot,
        "sys_peaks": sys_peaks,
        "dia_troughs": dia_troughs,
    }


def ecg_peak_count_check(ecg_cleaned: np.ndarray, r_peaks: np.ndarray) -> dict:
    """Compare NeuroKit's R-peak count with SciPy find_peaks on the same cleaned ECG.

    A large mismatch flags a noisy record. Returns both counts, the difference, and a flagged bool.
    """
    ecg_range = np.percentile(ecg_cleaned, 99) - np.percentile(ecg_cleaned, 1)
    scipy_peaks, _ = find_peaks(ecg_cleaned, distance=config.MIN_BEAT_DISTANCE, prominence=config.ECG_SCIPY_PROMINENCE_FRAC * ecg_range)
    n_nk, n_scipy = len(r_peaks), len(scipy_peaks)
    diff = abs(n_scipy - n_nk)
    rel_diff = diff / max(n_nk, 1)
    flagged = rel_diff > config.PEAK_COUNT_TOL and diff > config.PEAK_COUNT_MIN_DIFF
    return {"n_neurokit": n_nk, "n_scipy": n_scipy, "diff": diff, "rel_diff": rel_diff, "flagged": flagged}


SIGNALS = ("ecg", "ppg", "abp")
PEAKS = ("r_peaks", "ppg_peaks", "sys_peaks", "dia_troughs")

# Rejection reasons, checked in this order. Each rejected window counts once, under its first failing reason.
REASONS = ("nan", "flat", "abp_clipped", "too_few_beats", "beat_count_mismatch", "hr_out_of_range", "sbp_out_of_bounds", "dbp_out_of_bounds")


def _peaks_in(idx: np.ndarray, start: int, end: int) -> np.ndarray:
    """Peaks detected on the whole record that fall inside [start, end)."""
    return idx[(idx >= start) & (idx < end)]


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a boolean array."""
    if not mask.any():
        return 0
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())


def is_clipped(abp_raw: np.ndarray) -> bool:
    """True when raw ABP plateaus at its max or min for config.CLIP_RUN+ samples (transducer saturation)."""
    return any(_longest_run(np.abs(abp_raw - ref) < config.CLIP_EPS) >= config.CLIP_RUN
               for ref in (abp_raw.max(), abp_raw.min()))


def window_record(record_id: str, rec: dict) -> tuple[list[dict], Counter]:
    """Cut a processed record into non-overlapping config.WINDOW windows and label each with SBP/DBP.

    rec is the dict returned by process_record. Peaks come from the whole record, so beats
    at window edges are not lost. Returns the kept windows (one dict each, with the ECG/PPG slices
    labels and window-relative ECG/PPG peak indices) and a Counter of rejection reasons plus "kept".
    """
    rows = []
    counts = Counter()
    n = len(rec["ecg"])

    for window_idx, start in enumerate(range(0, n - config.WINDOW + 1, config.WINDOW)):
        end = start + config.WINDOW
        sig = {k: rec[k][start:end] for k in SIGNALS}
        pk = {k: _peaks_in(rec[k], start, end) for k in PEAKS}

        reason = None
        if any(np.isnan(x).any() for x in sig.values()):
            reason = "nan"
        elif any(np.std(sig[k]) < config.FLAT_STD[k] for k in SIGNALS):
            reason = "flat"
        elif is_clipped(rec["abp_raw"][start:end]):
            reason = "abp_clipped"
        elif any(len(p) < config.MIN_BEATS for p in pk.values()):
            reason = "too_few_beats"
        elif any(abs(len(pk[k]) - len(pk["sys_peaks"])) > config.BEAT_COUNT_TOL for k in ("r_peaks", "ppg_peaks")):
            reason = "beat_count_mismatch"     # ECG or PPG missed/doubled beats; ABP beats are the reference
        else:
            hr = 60 / np.median(np.diff(pk["r_peaks"]) / config.FS)
            sbp = np.median(rec["abp"][pk["sys_peaks"]])
            dbp = np.median(rec["abp"][pk["dia_troughs"]])
            if not config.HR_MIN <= hr <= config.HR_MAX:
                reason = "hr_out_of_range"
            elif not config.SBP_BOUNDS[0] <= sbp <= config.SBP_BOUNDS[1]:
                reason = "sbp_out_of_bounds"
            elif not config.DBP_BOUNDS[0] <= dbp <= config.DBP_BOUNDS[1]:
                reason = "dbp_out_of_bounds"

        if reason:
            counts[reason] += 1
            continue

        counts["kept"] += 1
        rows.append({
            "record_id": record_id,
            "window_idx": window_idx,
            "start_sample": start,
            "sbp": float(sbp),
            "dbp": float(dbp),
            "ecg": sig["ecg"],
            "ppg": sig["ppg"],
            "ppg_raw": rec["ppg_raw"][start:end],
            # ECG/PPG beat positions relative to the window start, for feature extraction. No ABP peaks: ABP is only for labels.
            "r_peaks": pk["r_peaks"] - start,
            "ppg_peaks": pk["ppg_peaks"] - start,
            "ppg_foot": _peaks_in(rec["ppg_foot"], start, end) - start,
        })

    return rows, counts


def main():
    record_id, ppg, abp, ecg = load_record(Path("data/raw/Part_1.mat"), 795)
    rec = process_record(ppg, abp, ecg)
    t = np.arange(len(rec["ecg"])) / config.FS

    # Debug Plot
    fig, ax = plt.subplots(3, sharex=True)

    ax[0].plot(t, rec["ecg"], label="ECG", color="blue")
    ax[0].plot(t[rec["r_peaks"]], rec["ecg"][rec["r_peaks"]], "x")
    ax[0].set_ylim(-0.3,1)
    ax[0].grid()
    ax[0].legend()
    ax[1].plot(t, rec["ppg"], label="PPG", color="purple")
    ax[1].plot(t[rec["ppg_peaks"]], rec["ppg"][rec["ppg_peaks"]], "x")
    ax[1].plot(t[rec["ppg_foot"]], rec["ppg"][rec["ppg_foot"]], "x")
    ax[1].set_ylim(-1.25,2.25)
    ax[1].grid()
    ax[1].legend()
    ax[2].plot(t, rec["abp"], label="ABP", color="red")
    ax[2].plot(t[rec["sys_peaks"]], rec["abp"][rec["sys_peaks"]], "x")
    ax[2].plot(t[rec["dia_troughs"]], rec["abp"][rec["dia_troughs"]], "x")
    ax[2].set_ylabel("mmHg")
    ax[2].set_ylim(30, 150)
    ax[2].grid()
    ax[2].legend()
    ax[2].set_xlabel("Seconds")
    plt.show()

    # Debug summary
    print(f"\n{record_id}")
    for name, key in [("R", "r_peaks"), ("PPG", "ppg_peaks"), ("foot", "ppg_foot"), ("sys", "sys_peaks"), ("dia", "dia_troughs")]:
        idx = rec[key]
        if len(idx) < 2:
            print(f"{name:5s} n={len(idx):4d} too few beats for HR/IBI")
            continue
        ibi = np.diff(idx) / config.FS
        print(f"{name:5s} n={len(idx):4d} HR={60/np.median(ibi):5.1f} IBI min/max={ibi.min():.2f}/{ibi.max():.2f}s")

    # ECG peak-count sanity check
    check = ecg_peak_count_check(rec["ecg"], rec["r_peaks"])
    status = "FLAGGED - noisy ECG?" if check["flagged"] else "ok"
    print(f"\nECG peak count  NeuroKit={check['n_neurokit']}  SciPy={check['n_scipy']}  "
          f"diff={check['diff']} ({check['rel_diff']:.1%})  {status}")

    # Windowing and labels
    rows, counts = window_record(record_id, rec)
    print(f"\nWindows  " + "  ".join(f"{r}={counts[r]}" for r in ("kept",) + REASONS if counts[r]))
    if rows:
        sbp = [r["sbp"] for r in rows]
        dbp = [r["dbp"] for r in rows]
        print(f"SBP median={np.median(sbp):.1f} range={min(sbp):.1f}-{max(sbp):.1f}  "
              f"DBP median={np.median(dbp):.1f} range={min(dbp):.1f}-{max(dbp):.1f} mmHg")


if __name__ == "__main__":
    main()
