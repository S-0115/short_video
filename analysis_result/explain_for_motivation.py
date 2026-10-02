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
    with open(data_path, 'rb') as f:
        for line in f:
            # print(line)
            info = line.decode('utf-8').split(',')
            time.append(float(info[0]) / 1000.)
            play_video_id.append(int(info[1]))
            if float(info[4]) != 0.:
                download_video_id.append(-1)
                bit_rate.append(-1)
            else:
                download_video_id.append(int(info[2]))
                bit_rate.append(int(info[3]))
            sleep_time.append(float(info[4]) / 1000.)
            delay.append(float(info[5]) / 1000.)
            rebuf.append(float(info[6]) / 1000.)
            swipe.append(int(info[7]))
            buffer.append(float(info[8]))
            throughput.append(float(info[9].strip()))

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

    return time, play_video_id, download_video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, throughput

def load_network_trace(file_path):
    BW_ADJUST_PARA = 1
    cooked_time = []
    cooked_bw = []
    with open(file_path, 'rb') as f:
        for line in f:
            parse = line.split()
            cooked_time.append(float(parse[0]))
            cooked_bw.append(float(parse[1])*BW_ADJUST_PARA)

    return cooked_time, cooked_bw

def cut_data_by_time(time, play_video_id, download_video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, throughput, stime, time_len):
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
    download_video_id = download_video_id[left_split_idx:right_split_idx]
    bit_rate = bit_rate[left_split_idx:right_split_idx]
    sleep_time = sleep_time[left_split_idx:right_split_idx]
    delay = delay[left_split_idx:right_split_idx]
    rebuf = rebuf[left_split_idx:right_split_idx]
    swipe = swipe[left_split_idx:right_split_idx]
    buffer = buffer[left_split_idx:right_split_idx]
    throughput = throughput[left_split_idx:right_split_idx]

    return time, play_video_id, download_video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, throughput

def plot_res(time, play_video_id, download_video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, throughput, time_throughput, real_throughput):
    # 创建图形和轴
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [4, 4]})
    ax = axs[0]

    # 设置坐标轴范围

    ax.set_xlim(0, 80)
    ax.set_xticks(np.arange(0, 85, 5).tolist())

    y_min = min(download_video_id[download_video_id >= 0])
    ax.set_ylim(0, 12)
    ax.set_yticks(np.arange(0, 12 + 2, 2).tolist())

    # 设置轴标签
    ax.set_xlabel('View time (s)', fontsize=16)
    ax.set_ylabel('Video ID', fontsize=16)

    # 图例
    vitual_legend_item_ax = []

    vitual_legend_item_ax.append(Line2D([], [], linestyle="none",marker=None,label="Bitrate: "))

    # 绘制下载视频块的质量
    # 添加虚拟图例
    # colors = ['green', 'brown', 'pink','blue','orange', 'purple']
    colors = [
        (26 / 255, 189 / 255, 206 / 255),
        (147 / 255, 112 / 255, 219 / 255),
        (20 / 255, 158 / 255, 17 / 255),
        (254 / 255, 140 / 255, 0 / 255),
        (240 / 255, 155 / 255, 160 / 255),
        (0 / 255, 191 / 255, 255 / 255),
              ]
    # colors = plt.cm.tab10(np.linspace(0.1, 1, 6))
    # colors = np.flip(colors, axis=0)
    label_bitrates = ['1Mbps', '2Mbps', '5Mbps', '9Mbps', '14Mbps', '25Mbps']
    # hatches = ['///', 'xx', '++', 'OO', '**', '..']
    # hatches = ['//', 'x', '+', 'O', '*',  '.']
    hatches = ['///', '\\\\\\', 'xx', '---', '||', '++']
    for i in range(len(label_bitrates)):
        vitual_legend_item_ax.append(Patch(facecolor=colors[i], label=label_bitrates[i], hatch=hatches[i]))

    for i in range(len(time) - 1):
        if download_video_id[i] == -1:
            continue
        x = time[i] - min(time)
        y = download_video_id[i] - y_min + 0.3
        color = colors[bit_rate[i]]
        hatch = hatches[bit_rate[i]]
        label_bitrate = label_bitrates[bit_rate[i]]
        rect = Rectangle((x, y), time[i + 1] - time[i], 0.95, facecolor=color,
                         label=label_bitrate, hatch=hatch)
        ax.add_patch(rect)

    handles_all = []
    labels_all = []
    for item in vitual_legend_item_ax:
        handles_all.append(item)
        labels_all.append(item.get_label())

    handel2label = dict(zip(labels_all, handles_all))
    # handel2label = dict(zip(labels_all, handles_all))
    legend_bitrate = ax.legend(handel2label.values(), handel2label.keys(), loc='upper center', frameon=False, fontsize=16, ncols=7, columnspacing=1.2, handlelength=1.4, handletextpad=0.5, bbox_to_anchor=(0., 1.35, 0.95, .102))

    ax.add_artist(legend_bitrate)

    vitual_legend_item_ax_1 = []
    # 绘制sleep及rebuf状态区域
    # 添加虚拟图例
    vitual_legend_item_ax_1.append(Patch(facecolor='grey', alpha=0.4, label='Download suspension'))
    for i in range(len(time) - 1):
        if sleep_time[i] != 0:
            ax.axvspan(time[i] - min(time), time[i + 1] - min(time), color='grey', alpha=0.4, linewidth=0,
                       label='sleep')
        # elif rebuf[i] != 0:
        #     ax.axvspan(time[i] - x_min, time[i + 1] - x_min, color='grey', alpha=0.8, linewidth=0, label='rebuff')
        # else:
        #     ax.axvspan(time[i] - x_min, time[i + 1] - x_min, color='green', alpha=0.3, linewidth=0, label='normal play')

    # 绘制视频播放索引
    # 添加虚拟图例
    vitual_legend_item_ax_1.append(Line2D([0], [0], color=(255 / 255, 0 / 255, 0 / 255), label='Playing video', linewidth=3))

    ax.step(time - min(time), play_video_id - y_min + 0.15, where='post', label='Playing video', color=(255 / 255, 0 / 255, 0 / 255), linewidth=3)

    # 设置图例
    handles_all_1 = []
    labels_all_1 = []

    for item in vitual_legend_item_ax_1:
        handles_all_1.append(item)
        labels_all_1.append(item.get_label())

    # 添加注释
    # ax.annotate('Download suspension', fontsize=16,
    #             xy=(37, 4), xytext=(47, 1),
    #             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', facecolor='black'))
    # ax.annotate('Playing video', fontsize=16,
    #             xy=(67, 8), xytext=(77, 3),
    #             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', facecolor='black'))

    handel2label_1 = dict(zip(labels_all_1, handles_all_1))

    # handel2label = dict(zip(labels_all, handles_all))
    ax.legend(handel2label_1.values(), handel2label_1.keys(), loc='upper center', frameon=False, fontsize=16, ncols=3, columnspacing=1.2, handlelength=1.4, handletextpad=0.5, bbox_to_anchor=(0., 1.17, 1., .102))

    ax1 = axs[1]
    vitual_legend_item_ax1 = []
    # 设置坐标轴范围
    ax1.set_xlim(0, 80)
    ax1.set_xticks(np.arange(0, 85, 5).tolist())

    ax1.set_ylim(0, 30)
    ax1.set_yticks(np.arange(0, 35, 5).tolist())

    # 设置轴标签
    ax1.set_xlabel('View time (s)', fontsize=16)
    ax1.set_ylabel('Value', fontsize=16)

    colors = [
        (255 / 255, 20 / 255, 147 / 255),
        (46 / 255, 117 / 255, 182 / 255),
        (46 / 255, 139 / 255, 87 / 255),
              ]

    # 绘制网络带宽和下载的比特率折现图
    # 添加虚拟图例
    vitual_legend_item_ax1.append(Line2D([0], [0], color=colors[0], linestyle='--', linewidth=2, label='throughput (Mbps)'))
    vitual_legend_item_ax1.append(Line2D([0], [0], color=colors[1], linestyle='-', linewidth=1.5, label='bitrate (Mbps)'))
    vitual_legend_item_ax1.append(Line2D([0], [0], color=colors[2], linestyle='-.', linewidth=2, label='buffer (s)'))

    real_bitrate = []
    for bitrate in bit_rate:
        if bitrate != -1:
            real_bitrate.append(VIDEO_BIT_RATE[bitrate] / 1000)
        else:
            real_bitrate.append(0)

    change_idx = np.where(np.diff(throughput) != 0)[0]
    x = time[change_idx] - min(time)
    y = throughput[change_idx]
    # ax1.plot(x, y, label='throughput (Mbps)', linestyle='--', linewidth=2, color=colors[0])

    ax1.plot(time - min(time), throughput, label='throughput (Mbps)', linestyle='--', linewidth=2, color=colors[0])
    # ax1.plot(time_throughput - min(time_throughput), real_throughput, label='throughput (Mbps)', linestyle='--', linewidth=2, color=colors[0])

    ax1.step(time - min(time), real_bitrate, where='post', label='bitrate (Mbps)', linestyle='-', linewidth=1.5, color=colors[1])
    ax1.plot(time - min(time), buffer, label='buffer (s)', linestyle='-.', linewidth=2, color=colors[2])

    # 绘制sleep及rebuf状态区域
    # 添加虚拟图例
    vitual_legend_item_ax1.append(Patch(facecolor='grey', alpha=0.4, label='Download suspension'))
    # vitual_legend_item_ax.append(Patch(facecolor='grey', alpha=0.8, label='rebuff'))
    # vitual_legend_item_ax.append(Patch(facecolor='green', alpha=0.3, label='normal play'))
    for i in range(len(time) - 1):
        if sleep_time[i] != 0:
            ax1.axvspan(time[i] - min(time), time[i + 1] - min(time), color='grey', alpha=0.4, linewidth=0, label='sleep')

    # 设置图例
    handles_all = []
    labels_all = []

    for item in vitual_legend_item_ax1:
        handles_all.append(item)
        labels_all.append(item.get_label())

    order = [0,1,2,3]  # 想要的顺序（索引）
    handles = [handles_all[i] for i in order]
    labels = [labels_all[i] for i in order]

    handel2label = dict(zip(labels, handles))
    # sorted_handel2label = dict(sorted(handel2label.items()))
    ax1.legend(handel2label.values(), handel2label.keys(), loc='upper center', frameon=False, fontsize=16, ncols=4, columnspacing=1.2, handlelength=1.6, handletextpad=0.5, bbox_to_anchor=(0., 1.17, 1., .102))

    plt.tight_layout(
        pad=1.6,  # 整体边距
        w_pad=0.5,  # 横向子图间距
        h_pad=0.1  # 纵向子图间距
    )
    # 显示图形

    plt.savefig('./explain_for_motivation.svg')
    plt.show()

user_id = 0
dataset = 'dataset'
log_dir = '../algorithms/dashlet/logs/'

network_dataset = 'sampled_traces_for_test'
# 4 5(2986) 11 12 29 35
# ['16', '24', '37', '30', '41', '29', '3', '40', '36', '5',
# '17', '11', '2', '32', '10', '33', '42', '43', '4', '44',
# '38', '19', '34', '27', '28', '35', '12', '7', '0', '18',
# '14', '1', '21', '39', '22', '13', '31', '15', '6', '9',
# '26', '8', '23', '20', '25']


# network trace id 35 stime 2954
network_trace_id = 35
# 27 2960
data_path = log_dir + dataset + '/user_' + str(user_id) + '/' + network_dataset + '_' + str(network_trace_id)
# data_path = log_dir + dataset + '/user_' + str(user_id) + '/' + 'trace_for_exp3_14'
time, play_video_id, download_video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, throughput = load_data(data_path)

# 2324 2950 1907 #2946
stime = 2956
time, play_video_id, download_video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, throughput = cut_data_by_time(time, play_video_id, download_video_id, bit_rate,
                                                                                      sleep_time, delay, rebuf, swipe, buffer, throughput, stime=stime, time_len=80)

network_path = '../algorithms/data/network_traces/' + network_dataset + '/' + str(network_trace_id)
time_throughput, real_throughput = load_network_trace(network_path)

time_throughput = np.array(time_throughput)[stime: stime + 100]
interval = -70
real_throughput = np.array(real_throughput)[stime - interval: stime - interval + 100]

plot_res(time, play_video_id, download_video_id, bit_rate, sleep_time, delay, rebuf, swipe, buffer, throughput, time_throughput, real_throughput)