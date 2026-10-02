import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from sympy.printing.pretty.pretty_symbology import line_width

plt.rcParams.update({
    "font.size": 16,          # 默认文字
    "axes.titlesize": 16,     # 标题
    "axes.labelsize": 16,     # 坐标轴标签
    "xtick.labelsize": 16,    # x 轴刻度
    "ytick.labelsize": 16,    # y 轴刻度
    "legend.fontsize": 16,    # 图例
})

from config_algorithm import VIDEO_BIT_RATE

def load_network_trace(file_path):
    BW_ADJUST_PARA = 0.95
    cooked_time = []
    cooked_bw = []
    with open(file_path, 'rb') as f:
        for line in f:
            parse = line.split()
            cooked_time.append(float(parse[0]))
            cooked_bw.append(float(parse[1])*BW_ADJUST_PARA)

    return cooked_time, cooked_bw

def load_data(data_path):
    time = []
    play_video_id = []
    download_video_id = []
    bit_rate = []
    sleep_time = []
    delay = []
    rebuf = []
    swipe = []
    buffer = []
    throughput = []
    play_chunk_bitrate = []
    play_chunk_ct = []
    view_type = []
    download_chunk_ct = []
    with open(data_path, 'rb') as f:
        for line in f:
            info = line.decode('utf-8').split(',')
            time.append(float(info[0]) / 1000.)
            play_video_id.append(int(info[1]))
            if float(info[4]) != 0.:
                download_video_id.append(-1)
                bit_rate.append(-1)
                download_chunk_ct.append(-1)
            else:
                download_video_id.append(int(info[2]))
                bit_rate.append(int(info[3]))
                download_chunk_ct.append(int(info[13]))
            sleep_time.append(float(info[4]) / 1000.)
            delay.append(float(info[5]) / 1000.)
            rebuf.append(float(info[6]) / 1000.)
            swipe.append(int(info[7]))
            buffer.append(float(info[8]))
            throughput.append(float(info[9].strip()))
            play_chunk_bitrate.append(int(info[10]))
            play_chunk_ct.append(int(info[11]))
            view_type.append(float(info[12]))

    time = np.array(time)
    play_video_id = np.array(play_video_id)
    download_video_id = np.array(download_video_id)
    bit_rate = np.array(bit_rate)
    sleep_time = np.array(sleep_time)
    delay = np.array(delay)
    rebuf = np.array(rebuf)
    swipe = np.array(swipe)
    buffer = np.array(buffer)
    throughput = np.array(throughput)
    play_chunk_bitrate = np.array(play_chunk_bitrate)
    play_chunk_ct = np.array(play_chunk_ct)
    view_type = np.array(view_type)
    download_chunk_ct = np.array(download_chunk_ct)

    return time, play_video_id, download_video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, throughput, play_chunk_bitrate, play_chunk_ct, view_type, download_chunk_ct

def cut_data_by_time(time, play_video_id, download_video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, throughput, play_chunk_bitrate, play_chunk_ct, view_type, download_chunk_ct, stime, time_len):
    left_split_idx = 0
    right_split_idx = 0

    for i in range(len(time)):
        # print(time[i])
        if time[i] >= stime:
            left_split_idx = i
            break

    for i in range(left_split_idx, len(time)):
        if time[i] >= stime + time_len:
            right_split_idx = i + 20
            break
    # print(split_idx)

    time = time[left_split_idx:right_split_idx]
    play_video_id = play_video_id[left_split_idx:right_split_idx]
    download_video_id = download_video_id[left_split_idx + 1:right_split_idx + 1]
    bit_rate = bit_rate[left_split_idx + 1:right_split_idx + 1]
    sleep_time = sleep_time[left_split_idx + 1:right_split_idx + 1]
    delay = delay[left_split_idx:right_split_idx]
    rebuf = rebuf[left_split_idx:right_split_idx]
    swipe = swipe[left_split_idx:right_split_idx]
    buffer = buffer[left_split_idx:right_split_idx]
    throughput = throughput[left_split_idx:right_split_idx]

    play_chunk_bitrate = play_chunk_bitrate[left_split_idx:right_split_idx]
    play_chunk_ct = play_chunk_ct[left_split_idx:right_split_idx]
    view_type = view_type[left_split_idx + 1:right_split_idx +1]
    download_chunk_ct = download_chunk_ct[left_split_idx + 1:right_split_idx + 1]

    return time, play_video_id, download_video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, throughput, play_chunk_bitrate, play_chunk_ct, view_type, download_chunk_ct

def plot_res(time, play_video_id, download_video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, throughput, play_chunk_bitrate, play_chunk_ct, view_type, download_chunk_ct, time_throughput, real_throughput):
    # 创建图形和轴
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [4, 4, 4]})

    # 绘制下载视频块的质量
    colors_background = [
        (239 / 255, 118 / 255, 123 / 255),
        (67 / 255, 163 / 255, 239 / 255),
    ]

    colors = [
            (26 / 255, 189 / 255, 206 / 255),
            (147 / 255, 112 / 255, 219 / 255),
            (20 / 255, 158 / 255, 17 / 255),
            (254 / 255, 140 / 255, 0 / 255),

            (0 / 255, 191 / 255, 255 / 255),
            (255 / 255, 20 / 255, 147 / 255),
              ]

    # colors = [
    #     (26 / 255, 189 / 255, 206 / 255),
    #     (147 / 255, 112 / 255, 219 / 255),
    #     (20 / 255, 158 / 255, 17 / 255),
    #     (254 / 255, 140 / 255, 0 / 255),
    #     (240 / 255, 155 / 255, 160 / 255),
    #     (0 / 255, 191 / 255, 255 / 255),
    #     (255 / 255, 20 / 255, 147 / 255),
    #     (46 / 255, 117 / 255, 182 / 255),
    #     (46 / 255, 139 / 255, 87 / 255),
    #           ]

    y_min = min(download_video_id[download_video_id >= 0])

    ax1 = axs[0]

    # 设置坐标轴范围
    ax1.set_xticks(np.arange(0, 210, 10).tolist())
    ax1.set_xlim(0, 120)

    ax1.set_ylim(0, 15)
    ax1.set_yticks(np.arange(0, 20, 5).tolist())

    # 设置轴标签
    ax1.set_xlabel('View time (s)', fontsize=16)
    ax1.set_ylabel('Video ID', fontsize=16)

    vitual_legend_item_ax1 = []

    vitual_legend_item_ax1.append(Patch(facecolor='grey', alpha=0.4, label='Download suspension'))

    # 绘制sleep及rebuf状态区域
    # 添加虚拟图例
    min_time = min(time)
    for i in range(len(time) - 1):
        if sleep_time[i] != 0:
            ax1.axvspan(time[i] - min(time), time[i + 1] - min(time), color='grey', alpha=0.4, linewidth=0,
                        label='sleep')
            continue

        # if view_type[i] == 0:
        #     ax1.axvspan(time[i] - min_time, time[i + 1] - min_time, color=colors_background[0], alpha=0.2, linewidth=0,label='short view')
        # elif view_type[i] == 1:
        #     ax1.axvspan(time[i] - min_time, time[i + 1] - min_time, color=colors_background[1], alpha=0.2, linewidth=0, label='long view')

    # 绘制网络带宽和下载的比特率折现图
    # 添加虚拟图例
    vitual_legend_item_ax1.append(Line2D([0], [0], color=colors[1], linestyle='-', linewidth=2, label='Playing video'))
    vitual_legend_item_ax1.append(Line2D([0], [0], color=colors[2], linestyle='-.', linewidth=2, label='Downloading video'))

    ax1.step(time - min(time), play_video_id - y_min + 0.1, where='post', linestyle='-', label='Play video id', color=colors[1], linewidth=2)
    ax1.step(time - min(time), download_video_id - y_min + 0.4, where='post', linestyle='--', label='Download video id',color=colors[2], linewidth=2)

    # 设置图例
    handles_all = []
    labels_all = []

    for item in vitual_legend_item_ax1:
        handles_all.append(item)
        labels_all.append(item.get_label())

    handel2label = dict(zip(labels_all, handles_all))
    # sorted_handel2label = dict(sorted(handel2label.items()))
    ax1.legend(handel2label.values(), handel2label.keys(), loc='upper center', frameon=False, fontsize=16, ncols=3, columnspacing=1.2, handlelength=1.6, handletextpad=0.5, bbox_to_anchor=(0., 1.17, 1., .102))

    ax2 = axs[1]

    # 设置坐标轴范围
    ax2.set_xticks(np.arange(0, 210, 10).tolist())
    ax2.set_xlim(0, 120)


    ax2.set_ylim(0, 5)
    ax2.set_yticks(np.arange(0, 6, 1).tolist())

    # 设置轴标签
    ax2.set_xlabel('View time (s)', fontsize=16)
    ax2.set_ylabel('Value', fontsize=16)

    vitual_legend_item_ax2 = []

    vitual_legend_item_ax2.append(Patch(facecolor='grey', alpha=0.4, label='Download suspension'))
    min_time = min(time)
    for i in range(len(time) - 1):
        if sleep_time[i] != 0:
            ax2.axvspan(time[i] - min(time), time[i + 1] - min(time), color='grey', alpha=0.4, linewidth=0,
                        label='sleep')
            continue
    # 绘制网络带宽和下载的比特率折现图
    # 添加虚拟图例

    vitual_legend_item_ax2.append(Line2D([0], [0], color=colors[3], linestyle='-', linewidth=2, label='Select segment length'))

    # ax2.step(time - min(time), play_chunk_ct - 0.1, where='post', linestyle='-', label='Select segment length of view', color=colors[3], linewidth=2)

    ax2.step(time - min(time), download_chunk_ct - 0.1, where='post', linestyle='-', label='Select segment length of download', color=colors[3], linewidth=2)

    # 设置图例
    handles_all = []
    labels_all = []

    for item in vitual_legend_item_ax2:
        handles_all.append(item)
        labels_all.append(item.get_label())

    handel2label = dict(zip(labels_all, handles_all))
    # sorted_handel2label = dict(sorted(handel2label.items()))
    ax2.legend(handel2label.values(), handel2label.keys(), loc='upper center', frameon=False, fontsize=16, ncols=3, columnspacing=1.2, handlelength=1.6, handletextpad=0.5, bbox_to_anchor=(0., 1.17, 1., .102))

    ax3 = axs[2]

    # 设置坐标轴范围
    ax3.set_xticks(np.arange(0, 210, 10).tolist())
    ax3.set_xlim(0, 120)

    ax3.set_ylim(0, 25)
    ax3.set_yticks(np.arange(0, 30, 5).tolist())

    # 设置轴标签
    ax3.set_xlabel('View time (s)', fontsize=16)
    ax3.set_ylabel('Value', fontsize=16)

    vitual_legend_item_ax3 = []

    vitual_legend_item_ax3.append(Patch(facecolor='grey', alpha=0.4, label='Download suspension'))

    # 绘制sleep及rebuf状态区域
    # 添加虚拟图例
    min_time = min(time)
    for i in range(len(time) - 1):
        if sleep_time[i] != 0:
            ax3.axvspan(time[i] - min(time), time[i + 1] - min(time), color='grey', alpha=0.4, linewidth=0,
                        label='sleep')
            continue

        # if view_type[i] == 0:
        #     ax3.axvspan(time[i] - min_time, time[i + 1] - min_time, color=colors_background[0], alpha=0.2, linewidth=0,label='short view')
        # elif view_type[i] == 1:
        #     ax3.axvspan(time[i] - min_time, time[i + 1] - min_time, color=colors_background[1], alpha=0.2, linewidth=0, label='long view')

    # 绘制网络带宽和下载的比特率折现图
    # 添加虚拟图例
    vitual_legend_item_ax3.append(Line2D([0], [0], color=colors[4], linestyle='-', linewidth=2, label='Download Bitrate (Mbps)'))
    vitual_legend_item_ax3.append(Line2D([0], [0], color=colors[5], linestyle='--', linewidth=2, label='Throughput (Mbps)'))

    real_bitrate = []
    for bitrate in bit_rate:
        if bitrate != -1:
            real_bitrate.append(VIDEO_BIT_RATE[bitrate] / 1000)
        else:
            real_bitrate.append(0)
    real_bitrate = np.array(real_bitrate)

    ax3.step(time - min(time), real_bitrate - 0.3, where='post', linestyle='-', label='Download Bitrate (Mbps)', color=colors[4], linewidth=2)
    # ax3.plot(time - min(time), throughput , linestyle='--', label='Throughput (Mbps)',color=colors[5], linewidth=2)
    ax3.plot(time_throughput - min(time_throughput), real_throughput , linestyle='--', label='Throughput (Mbps)',color=colors[5], linewidth=2)

    # 设置图例
    handles_all = []
    labels_all = []

    for item in vitual_legend_item_ax3:
        handles_all.append(item)
        labels_all.append(item.get_label())

    handel2label = dict(zip(labels_all, handles_all))
    # sorted_handel2label = dict(sorted(handel2label.items()))
    ax3.legend(handel2label.values(), handel2label.keys(), loc='upper center', frameon=False, fontsize=16, ncols=3, columnspacing=1.2, handlelength=1.6, handletextpad=0.5, bbox_to_anchor=(0., 1.17, 1., .102))


    plt.tight_layout(
        pad=0.5,  # 整体边距
        w_pad=0,  # 横向子图间距
        h_pad=0.6  # 纵向子图间距
    )
    # 显示图形

    plt.savefig('./explain_for_experiment_3_1.svg')
    plt.show()

# user_id = 0
# dataset = 'dataset'
# log_dir = '../algorithms/secbad/logs/secbad_test/logs_short_video_env/'
#
# # network_dataset = 'sampled_traces_for_test'
# network_dataset = 'trace'
# # 4 5(2986) 11 12 29 35
# network_trace_id = 8 # 36
# # ['16', '24', '37', '30', '41', '29', '3', '40', '36', '5', '17', '11', '2', '32', '10', '33', '42', '43',
# #  '4', '44', '38', '19', '34', '27', '28', '35', '12', '7', '0', '18', '14', '1', '21', '39', '22', '13',
# #  '31', '15', '6', '9', '26', '8', '23', '20', '25']
# data_path = log_dir + dataset + '/user_' + str(user_id) + '/' + network_dataset + '_' + str(network_trace_id)

data_path = '../algorithms/secbad/logs/data_for_exp3_1/' + 'trace_0'
# data_path = '../algorithms/secbad/logs/secbad_test/logs_short_video_env/dataset/user_0/trace_0'

time, play_video_id, download_video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, throughput, play_chunk_bitrate, play_chunk_ct, view_type, download_chunk_ct = load_data(data_path)

# 8: 135 1200 1400 2480 3090 4600
# 3: 1440 1915
# 18：1430 8900
stime = 147


network_path = '../algorithms/data/network_traces/' + 'sampled_traces_for_test' + '/' + str(14)
time_throughput, real_throughput = load_network_trace(network_path)

time_throughput = np.array(time_throughput)[stime: stime + 200]
interval =12
real_throughput = np.array(real_throughput)[stime - interval: stime - interval + 200]


time, play_video_id, download_video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, throughput, play_chunk_bitrate, play_chunk_ct, view_type, download_chunk_ct = cut_data_by_time(time, play_video_id, download_video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, throughput, play_chunk_bitrate, play_chunk_ct, view_type, download_chunk_ct, stime=stime, time_len=200)
# print(time)
# print( sleep_time)
plot_res(time, play_video_id, download_video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, throughput, play_chunk_bitrate, play_chunk_ct, view_type, download_chunk_ct, time_throughput, real_throughput)