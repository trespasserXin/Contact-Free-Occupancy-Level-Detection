import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from read_bin import read_bin

def convert_ts(df):
    df.index = pd.to_datetime(df.index, unit='s', utc=True)
    df.index = df.index.tz_convert('America/Toronto')
    return df

def ts_interpolation(df):
    # return sensor_df.resample('1ms').interpolate(method='linear', limit_area='inside')
    time = df.index.to_series()
    pos = time.groupby(time).cumcount()
    blk_size = time.groupby(time).transform('size')
    offset = pos / blk_size * pd.to_timedelta(1, unit='s')
    df = df.copy()
    df.index = df.index + offset.to_numpy()
    return df

df1 = pd.read_csv("room_afternoon.csv", index_col=0)
df2 = pd.read_csv("calib_log_v2.csv", index_col=0)
mic1 = read_bin("room317.bin")
print(mic1[0][:5])
mic2 = read_bin("room_afternoon.bin")
df_mic1 = pd.DataFrame({'raw':mic1[1]}, index=mic1[0])
df_mic2 = pd.DataFrame({'raw':mic2[1]}, index=mic2[0])

# Convert ms to datetime
df1 = convert_ts(df1)
df2 = convert_ts(df2)
df_mic1 = convert_ts(df_mic1)
df_mic2 = convert_ts(df_mic2)
# Interpolate timestamps
ts_interpolation(df_mic1)
ts_interpolation(df_mic2)

df_mic1 = df_mic1.rolling(500).mean()
df_mic2 = df_mic2.rolling(200).mean()


# for line in df1.head().loc[:, 'sensor']:
#     print(line.split(':'))

# def process_data(sensor_df):
#
#     sensor_df.rename(columns={"timestamp_ms": "timestamp"}, inplace=True)
#     # sensor_df.set_index("timestamp", inplace=True)
#     # df_acc = pd.DataFrame({"timestamp": [],"sensor type":[], "reading":[]})
#     # df_acc.set_index("timestamp", inplace=True)
#     time_array, sensor_array, reading_array = [], [], []
#     for time, line in zip(sensor_df.loc[:, "timestamp"], sensor_df.loc[:, 'sensor']):
#         splitted = line.split(':')
#         # print(type(time))
#         if splitted[0].endswith("g"):
#             time_array.append(time)
#             sensor_array.append(splitted[0])
#             reading_array.append(float(splitted[1].rstrip('g')))
#             if len(time_array) % 3 == 0 and np.mean(time_array[-3:]) != time_array[-1]:
#                 time_array[-3:] = [round(np.mean(time_array[-3:]))] * 3
#
#     df_acc = pd.DataFrame({"timestamp": time_array,"sensor type":sensor_array,
#                            "reading":reading_array})
#     df_acc["timestamp"] = pd.to_datetime(df_acc.loc[:, "timestamp"], unit='ms', utc=True)
#     df_acc["timestamp"] = df_acc["timestamp"].dt.tz_convert('America/Toronto')
#     # df_acc.drop(columns=["timestamp_ms"], inplace=True)
#     df_pivot = df_acc.pivot(index="timestamp", columns="sensor type", values="reading")
#
#     return df_pivot

# def process_data(sensor_df):
#
#     sensor_df.rename(columns={"timestamp_ms": "timestamp"}, inplace=True)
#     time_array, sensor_array, reading_array = [], [], []
#     for time, line in zip(sensor_df.loc[:, "timestamp"], sensor_df.loc[:, 'sensor']):
#         splitted = line.split(':')
#
#         # print(type(time))
#         if splitted[0].endswith("mp") or splitted[0].endswith("um"):
#             # print(splitted[0])
#             time_array.append(time)
#             sensor_array.append(splitted[0])
#             # print(splitted[0], splitted[1])
#             # read_val = splitted[1].rstrip('KOhms') if splitted[0].endswith("C") \
#             #     else splitted[1].rstrip('mm')
#             read_val = splitted[1].split()[0]
#             # reading_array.append(float(splitted[1].rstrip('KOhms')))
#             reading_array.append(float(read_val))
#
#             # if len(time_array) % 3 == 0 and np.mean(time_array[-3:]) != time_array[-1]:
#             #     time_array[-3:] = [round(np.mean(time_array[-3:]))] * 3
#
#     df_acc = pd.DataFrame({"timestamp": time_array,"sensor type":sensor_array,
#                            "reading":reading_array})
#     df_acc["timestamp"] = pd.to_datetime(df_acc.loc[:, "timestamp"], unit='ms', utc=True)
#     df_acc["timestamp"] = df_acc["timestamp"].dt.tz_convert('America/Toronto')
#     # df_acc.drop(columns=["timestamp_ms"], inplace=True)
#     df_pivot = df_acc.pivot(index="timestamp", columns="sensor type", values="reading")
#
#     return df_pivot


def process_data_new(df):

    df.rename(columns={"timestamp_ms": "timestamp"}, inplace=True)
    time_array, sensor_array, reading_array = [], [], []
    for time, line in zip(df.loc[:, "timestamp"], df.loc[:, 'sensor']):
        splitted = line.split(':')

        # print(type(time))
        if splitted[0].endswith("mp") or splitted[0].endswith("um"):
            # print(splitted[0])
            time_array.append(time)
            sensor_array.append(splitted[0])
            # print(splitted[0], splitted[1])
            # read_val = splitted[1].rstrip('KOhms') if splitted[0].endswith("C") \
            #     else splitted[1].rstrip('mm')
            read_val = splitted[1].split()[0]
            # reading_array.append(float(splitted[1].rstrip('KOhms')))
            reading_array.append(float(read_val))

            # if len(time_array) % 3 == 0 and np.mean(time_array[-3:]) != time_array[-1]:
            #     time_array[-3:] = [round(np.mean(time_array[-3:]))] * 3

    df_acc = pd.DataFrame({"timestamp": time_array,"sensor type":sensor_array,
                           "reading":reading_array})
    df_acc["timestamp"] = pd.to_datetime(df_acc.loc[:, "timestamp"], unit='ms', utc=True)
    df_acc["timestamp"] = df_acc["timestamp"].dt.tz_convert('America/Toronto')
    # df_acc.drop(columns=["timestamp_ms"], inplace=True)
    df_pivot = df_acc.pivot(index="timestamp", columns="sensor type", values="reading")

    return df_pivot

temp_series1 = df1.loc[:, 'Temp'].dropna(inplace=False)
hum_series1 = df1.loc[:, 'RH'].dropna(inplace=False)

temp_series2 = df2.loc[:, 'Temp'].dropna(inplace=False)
hum_series2 = df2.loc[:, 'RH'].dropna(inplace=False)

range_series1 = df1.loc[:, 'Corrected Distance'].dropna(inplace=False)
range_series2 = df2.loc[:, 'Corrected Distance'].dropna(inplace=False)

xg = df1.loc[:, 'Xg'].dropna(inplace=False).rolling(50).mean()
yg = df1.loc[:, 'Yg'].dropna(inplace=False).rolling(50).mean()
zg = df1.loc[:, 'Zg'].dropna(inplace=False).rolling(50).mean()

# figure(figsize=(15, 10))
# plt.subplot(2, 1, 1)
# plt.plot(temp_series1)
# plt.subplot(2, 1, 2)
# plt.scatter(hum_series1.index, hum_series1)
#
# figure(figsize=(15, 10))
# plt.subplot(2, 1, 1)
# plt.plot(temp_series2)
# plt.subplot(2, 1, 2)
# plt.scatter(hum_series2.index, hum_series2)
#
figure(figsize=(15, 15))
plt.subplot(3, 1, 1)
plt.plot(xg)
plt.title("Xg")
plt.subplot(3, 1, 2)
plt.plot(yg)
plt.title("Yg")
plt.subplot(3, 1, 3)
plt.plot(zg)
plt.title("Zg")
# plt.scatter(df1.index, range_series1)
# plt.subplot(3, 1, 3)
# plt.plot(df1.loc[:, 'Gas'])
# plt.title("Gas Sensor")
# plt.subplot(5, 1, 4)
# plt.plot(temp_series1)
# plt.title("Temperature Sensor")
# plt.subplot(5, 1, 5)
# plt.plot(hum_series1)
# plt.title("Humidity Sensor")

figure(figsize=(32, 15))
plt.subplot(3, 2, 1)
plt.plot(df_mic2.loc[:, 'raw'])
plt.title("Mic 2")
plt.subplot(3, 2, 2)
# plt.plot(df_mic2.loc[:, 'raw'])
plt.scatter(df2.index, range_series2)
plt.subplot(3, 2, 3)
plt.plot(df2.loc[:, 'Gas'])
plt.title("Gas Sensor")
plt.subplot(3, 2, 4)
plt.plot(temp_series2)
plt.title("Temperature Sensor")
plt.subplot(3, 2, 5)
plt.plot(hum_series2)
plt.title("Humidity Sensor")
plt.subplot(3, 2, 6)
plt.plot(df2.loc[:, 'Light'])
plt.title("Light Sensor")

# plt.plot(df2_acc.index, df2_acc.loc[:, ['Xg']])
# plt.subplot(3, 1, 2, sharex=plt.gca())
# plt.plot(df2_acc.index, df2_acc.loc[:, ['Yg']])
# plt.subplot(3, 1, 3, sharex=plt.gca())
# plt.plot(df2_acc.index, df2_acc.loc[:, ['Zg']])
# plt.plot(df2_acc.index, df2_acc.loc[:,'Gas'])

plt.show()





