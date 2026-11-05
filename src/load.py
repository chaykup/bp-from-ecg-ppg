import hashlib

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_record(part: Path, i: int) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(part, "r") as file:
        part_name = part.stem
        ref = file[part_name][i, 0]
        rec = file[ref][:].T
        record_id = part_name + '_Patient_' + str(i)
        ppg, abp, ecg = rec
    return record_id, ppg.astype(np.float64), abp.astype(np.float64), ecg.astype(np.float64)

def n_records(part: Path) -> int:
    """Number of records in one Part_*.mat file."""
    with h5py.File(part, "r") as file:
        return file[part.stem].shape[0]

def find_duplicates(jobs: list[tuple[Path, int]], n_samples: int) -> dict[str, str]:
    """Find records that are copies of an earlier one, e.g. Part_1_Patient_1055-1067 == Part_3_Patient_2965-2977.

    Two records count as copies when their first n_samples of all three signals match (rounded to 4 decimals),
    so a copy cut to a different length still matches. jobs is a list of (part path, index) in file order.
    Returns {duplicate record_id: record_id of the first copy, which is kept}.
    """
    first_seen, dups = {}, {}
    by_part = {}
    for part, i in jobs:
        by_part.setdefault(part, []).append(i)
    for part, idx in by_part.items():
        with h5py.File(part, "r") as file:
            refs = file[part.stem]
            for i in idx:
                head = file[refs[i, 0]][:n_samples]     # (n_samples, 3): stored column-major, so rows are samples
                key = hashlib.sha1(np.round(head.astype(np.float64), 4).tobytes()).hexdigest()
                record_id = f"{part.stem}_Patient_{i}"
                if key in first_seen:
                    dups[record_id] = first_seen[key]
                else:
                    first_seen[key] = record_id
    return dups


def plot_waveforms(ppg: np.ndarray, ecg: np.ndarray, abp: np.ndarray):
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot((np.arange(len(ppg))/125), ppg, label='PPG', color='green')
    ax[0].grid()
    ax[0].set_xlim(0,15)
    ax[0].legend()

    ax[1].plot((np.arange(len(ecg))/125), ecg, label='ECG', color='blue')
    ax[1].grid()
    ax[1].set_xlim(0,15)
    ax[1].set_ylim(-0.25,2)
    ax[1].legend()

    ax[2].plot((np.arange(len(abp))/125), abp, label='ABP', color='red')
    ax[2].grid()
    ax[2].set_xlim(0,15)
    ax[2].set_xlabel('Time (seconds)')
    ax[2].set_ylabel('mmHg')
    ax[2].legend()

    def on_scroll(event):

        if event.inaxes is None:
            return

        cur_xmin1, cur_xmax1 = ax[0].get_xlim()
        xlim_range1 = cur_xmax1 - cur_xmin1

        cur_xmin2, cur_xmax2 = ax[1].get_xlim()
        cur_xmin3, cur_xmax3 = ax[1].get_xlim()
        scale_factor = 0.02 * xlim_range1

        if event.button == 'up':
            ax[0].set_xlim([cur_xmin1 + scale_factor, cur_xmax1 + scale_factor])
        elif event.button == 'down':
            ax[0].set_xlim([cur_xmin1 - scale_factor, cur_xmax1 - scale_factor])

        if event.button == 'up':
            ax[1].set_xlim([cur_xmin2 + scale_factor, cur_xmax2 + scale_factor])
        elif event.button == 'down':
            ax[1].set_xlim([cur_xmin2 - scale_factor, cur_xmax2 - scale_factor])

        if event.button == 'up':
            ax[2].set_xlim([cur_xmin3 + scale_factor, cur_xmax3 + scale_factor])
        elif event.button == 'down':
            ax[2].set_xlim([cur_xmin3 - scale_factor, cur_xmax3 - scale_factor])

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    plt.show()


def main():
    part = Path("data/raw/Part_1.mat")    # Set raw .mat file
    record_id, ppg, abp, ecg = load_record(part, 0) # load_record(<raw mat file>, <observation index>)
    plot_waveforms(ppg, ecg, abp)
    print(f"\nRecord ID: {record_id}")
    print(f"PPG rows: {len(ppg)}, ABP rows: {len(abp)}, ECG rows: {len(ecg)}")
    print(f"Data Types: {ppg.dtype, abp.dtype, ecg.dtype}\n")

if __name__ == "__main__":
    main()