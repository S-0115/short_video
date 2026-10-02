import os

import matplotlib.pyplot as plt
import numpy as np
from config_algorithm import VIDEO_BIT_RATE

def load_data(data_path):
    time = []
    video_id = []
    bit_rate = []
    sleep_time = []
    delay = []
    rebuf = []
    swipe = []
    buffer = []
    with open(data_path, 'rb') as f:
        for line in f:
            info = line.decode('utf-8').split(',')
            time.append(float(info[0]) / 1000.)
            if float(info[3]) != 0.:
                video_id.append(-1)
                bit_rate.append(-1)
            else:
                video_id.append(int(info[1]))
                bit_rate.append(VIDEO_BIT_RATE[int(info[2])] / 1000.)

            sleep_time.append(float(info[3]) / 1000.)
            delay.append(float(info[4]) / 1000.)
            rebuf.append(float(info[5]) / 1000.)
            swipe.append(int(info[6]))
            buffer.append(float(info[7]))

    return time, video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer


def load_trace(file_path):
    BW_ADJUST_PARA = 1
    cooked_time = []
    cooked_bw = []
    with open(file_path, 'rb') as f:
        for line in f:
            parse = line.split()
            cooked_time.append(float(parse[0]))
            cooked_bw.append(float(parse[1])*BW_ADJUST_PARA)

    return cooked_time, cooked_bw


def cut_data_by_time(time, video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer):
    split_idx = -1

    for i in range(len(time)):
        # print(time[i])
        if time[i] >= 200:
            split_idx = i
            break
    # print(split_idx)

    time = time[:split_idx]
    video_id = video_id[:split_idx]
    bit_rate = bit_rate[:split_idx]
    sleep_time = sleep_time[:split_idx]
    delay = delay[:split_idx]
    rebuf = rebuf[:split_idx]
    swipe = swipe[:split_idx]
    buffer = buffer[:split_idx]

    return time, video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer


def plot_res(time, download_video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, bw_time, bw):
    # 时间点 time

    # 视频下载索引 video_id

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(12, 6))

    # 设置坐标轴范围
    ax.set_xticks(np.arange(0, int(time[-1] + 9.9), 10).tolist())
    # ax.set_yticks(np.arange(-1, 15, 1).tolist())

    ax.set_xlim(0, 150)
    ax.set_ylim(0, 60)

    # 设置轴标签
    ax.set_xlabel('View time(s)')
    ax.set_ylabel('Bitrate(Mbps)')

    # 绘制sleep及rebuf状态区域
    # for i in range(len(time) - 1):
    #     if sleep_time[i] != 0:
    #         ax.axvspan(time[i], time[i + 1], color='pink', alpha=0.6, linewidth=0, label='sleep')
    #     elif rebuf[i] != 0:
    #         ax.axvspan(time[i], time[i + 1], color='grey', alpha=0.8, linewidth=0, label='rebuffer')
    #     else:
    #         ax.axvspan(time[i], time[i + 1], color='green', alpha=0.3, linewidth=0, label='normal play')

    # 绘制下载的比特率

    line_bitrate = ax.step(time, bit_rate, where='post', label='download bitrate', color='red')

    line_bw = ax.plot(bw_time, bw, label='throughput', color='purple')

    ax_ = ax.twinx()

    ax_.set_ylabel('buffer(s)')
    ax_.set_ylim(0, 40)
    # 绘制缓冲区变化

    line_buffer_size = ax_.plot(time, buffer, label='buffer size', color='blue')

    # 设置图例
    lines = line_bitrate + line_buffer_size + line_bw
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    # 显示图形
    plt.show()

trace_id = 4
user_id = 0
dataset = '0_subdataset'
traces_id2name = ['59', '11', '34', '60', '49', '31', '26', '25']

trace_file = '../algorithms/data/network_traces/trace_low/' + traces_id2name[trace_id]

bw_time, bw = load_trace(trace_file)


# log_dir = '../algorithms/pdas/logs/'
#
# # data_path = './log_dashlet/2000.0/0_long_mini_dataset/user_2/trace_1'
# data_path = log_dir + dataset + '/user_' + str(user_id) + '/trace_' + str(trace_id)
# time, video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer = load_data(data_path)
#
# # print(time, video_id, bit_rate, sleep_time, delay, rebuf, swipe)
# time, video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer = cut_data_by_time(time, video_id, bit_rate,
#                                                                                       sleep_time, delay, rebuf, swipe, buffer)
# # print(bit_rate)
# plot_res(time, video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, bw_time, bw)

log_dir = '../algorithms/dashlet/logs/'
data_path = log_dir + dataset + '/user_' + str(user_id) + '/trace_' + str(trace_id)
time, video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer = load_data(data_path)

# print(time, video_id, bit_rate, sleep_time, delay, rebuf, swipe)
time, video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer = cut_data_by_time(time, video_id, bit_rate,
                                                                                      sleep_time, delay, rebuf, swipe, buffer)
# print(bit_rate)
plot_res(time, video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, bw_time, bw)

# log_dir = '../algorithms/deload/logs/'
# data_path = log_dir + dataset + '/user_' + str(user_id) + '/trace_' + str(trace_id)
# time, video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer = load_data(data_path)
#
# # print(time, video_id, bit_rate, sleep_time, delay, rebuf, swipe)
# time, video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer = cut_data_by_time(time, video_id, bit_rate,
#                                                                                       sleep_time, delay, rebuf, swipe, buffer)
# # print(bit_rate)
# plot_res(time, video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, bw_time, bw)

# log_dir = '../algorithms/secbad/logs/secbad_test/logs_short_video_env/'
# data_path = log_dir + dataset + '/user_' + str(user_id) + '/trace_' + str(trace_id)
# time, video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer = load_data(data_path)
#
# # print(time, video_id, bit_rate, sleep_time, delay, rebuf, swipe)
# time, video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer = cut_data_by_time(time, video_id, bit_rate,
#                                                                                       sleep_time, delay, rebuf, swipe, buffer)
# # print(bit_rate)
# plot_res(time, video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, bw_time, bw)