import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def convert_ts(df):
    df.index = pd.to_datetime(df.index, unit='s', utc=True)
    df.index = df.index.tz_convert('America/Toronto')
    return df

def head_count(file_path, start_time = None, total_num=31):

    df = pd.read_csv(file_path)
    df = df[df['Head Count'] != 'Class completed']

    raw_time = df['Time'].copy().to_numpy()
    raw_time = [time.split()[0].replace('.',':') for time in raw_time]
    day_info = '2025/11/24'
    raw_time = [' '.join([day_info, time]) for time in raw_time]
    raw_time = pd.to_datetime(raw_time, format='%Y/%m/%d %H:%M').tz_localize('America/Toronto')


    raw_count = df['Head Count'].copy().astype(float).to_numpy()
    true_head = raw_count.copy()
    for i in range(len(raw_count)):
        if raw_count[i] < 0.33*total_num:
            raw_count[i] = 0
        elif 0.33*total_num < raw_count[i] < 0.66*total_num:
            raw_count[i] = 1
        else:
            raw_count[i] = 2

    # df['Head Count'] = raw_count
    df = pd.DataFrame({'Time': raw_time, 'Tiers': raw_count, 'Raw Count': true_head})
    start_row = pd.DataFrame({'Time': [start_time], 'Tiers': [0], 'Raw Count': [0]})
    df = pd.concat([start_row, df])
    df.reset_index(drop=True, inplace=True)
    df.set_index('Time', inplace=True)
    df = df.resample('1s').ffill()

    time_to_start = (df.index-df.index[0]).total_seconds()
    df['Time to start'] = time_to_start
    # print(df.head(60))
    return df

def read_crossing(file_path):
    sensor_df = pd.read_csv(file_path, index_col=0)
    # Convert ms to datetime
    sensor_df = convert_ts(sensor_df)
    crossings = sensor_df.loc[sensor_df['Corrected Distance'] > 100, 'Corrected Distance'].copy()
    # print(crossings)
    crossing_ts = crossings.index
    # print(crossing_ts)
    start_ts_ = sensor_df.index[0]
    crossing_sec = [(ts - start_ts_).total_seconds() for ts in crossing_ts]
    return crossing_sec, start_ts_

def align_labels_to_windows(
    win_centers_sec: np.ndarray,
    label_times_sec: np.ndarray,
    label_tiers_per_sec: np.ndarray,
    win_sec: float = 10
) -> np.ndarray:
    """
    Majority-vote the 1 Hz labels inside each window [center - win_sec/2, center + win_sec/2).
    label_tiers_per_sec should be already in {Low, Medium, High} (strings) or mapped ints.
    Returns y_win as integer class ids (0/1/2).
    """
    half = win_sec / 2.0
    # If labels are strings, map to ints once:
    # if label_tiers_per_sec.dtype.kind in {"U", "S", "O"}:
    #     label_ids = np.array([TIER2ID.get(s, -1) for s in label_tiers_per_sec])
    # else:
    label_ids = label_tiers_per_sec.astype(int)

    y = np.full(len(win_centers_sec), fill_value=-1, dtype=int)
    for i, c in enumerate(win_centers_sec):
        a, b = c - half, c + half
        m = (label_times_sec >= a) & (label_times_sec < b)
        if not np.any(m):
            continue
        vals, counts = np.unique(label_ids[m], return_counts=True)
        dom = int(vals[np.argmax(counts)])
        y[i] = dom
    # Drop windows with no coverage (-1) later.
    return y

crossing_list, start_ts = read_crossing("room_afternoon.csv")
head_count_df = head_count("head_count.csv", start_time=start_ts)
if __name__ == "__main__":
    crossing_list, start_ts = read_crossing("room_afternoon.csv")
    print(start_ts)
    head_count_df = head_count("head_count.csv", start_time=start_ts)
    # print(head_count_df.tail(10))
    head_count_df.to_csv("head_df.csv")
    read_for_sec = pd.read_csv("compare_y_pred_y_true.csv")
    t_center = read_for_sec['time']

    raw_align = align_labels_to_windows(t_center, head_count_df['Time to start'],
                                        head_count_df['Raw Count'])
    raw_align = pd.Series(raw_align)
    raw_align.to_csv("raw_align.csv")
    print(len(raw_align))