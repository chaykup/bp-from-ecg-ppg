import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_record(part: str, i: int) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(part, "r") as file:
        part_name = list(file.keys())[1]
        ref = file[part_name][i, 0]
        rec = file[ref][:].T
        record_id = part_name + '_Patient_' + str(i)
        ppg, abp, ecg = rec
    return record_id, ppg.astype(np.float64), abp.astype(np.float64), ecg.astype(np.float64)


def main():
    part = "data/raw/Part_1.mat"
    record_id, ppg, abp, ecg = load_record(part, 0)
    print(f"\nRECORD ID: {record_id}")
    print(f"PPG rows: {len(ppg)}, ABP rows: {len(abp)}, ECG rows: {len(ecg)}")
    print(f"DATA TYPES: {ppg.dtype, abp.dtype, ecg.dtype}\n")

    # fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 15))
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
        xlim_range2 = cur_xmax2 - cur_xmin2

        cur_xmin3, cur_xmax3 = ax[1].get_xlim()
        xlim_range3 = cur_xmax3 - cur_xmin3

        scale_factor = 0.01 * xlim_range1
        scale_factor = 0.01 * xlim_range2

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

if __name__ == "__main__":
    main()