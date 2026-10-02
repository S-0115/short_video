import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

from algorithms.PBR_ARD.config_algorithm import alpha

plt.rcParams.update({
    "font.size": 16,          # 默认文字
    "axes.titlesize": 16,     # 标题
    "axes.labelsize": 16,     # 坐标轴标签
    "xtick.labelsize": 16,    # x 轴刻度
    "ytick.labelsize": 16,    # y 轴刻度
    "legend.fontsize": 16,    # 图例
})

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

trace_id = 19
trace_file = '../algorithms/data/network_traces/sampled_traces_for_test/' + str(trace_id)

bw_time, bw = load_trace(trace_file)

t_len = 2000
for i in range(2):
    t_start = i * t_len
    fig, ax = plt.subplots(figsize=(20,4), facecolor='white')
    ax.set_facecolor('#f5f5f5')

    # ax.set_xticks(np.arange(0, t_len + 9, 50))
    ax.set_xlim(0, 2000)

    y_max = max(bw[t_start: t_start+t_len])
    # ax.set_yticks(np.arange(0, 50 + 4.9, 10))
    # ax.set_ylim(0, 50)
    # for y in np.arange(0, 50 + 4.9, 10):
    #     ax.axhline(y=y, color='grey', linestyle='-', linewidth=1.5, alpha=0.5)

    ax.set_xlabel(f'Time (s) {t_start}')
    ax.set_ylabel('Throughput (Mbps)')
    ax.plot(np.arange(t_len), bw[t_start: t_start+t_len], linewidth=1.5, color=(80 / 255, 167 / 255, 252 / 255))

    # plt.savefig('./throughput_time.svg')
    plt.tight_layout(
        pad=1.6,  # 整体边距
        w_pad=0.5,  # 横向子图间距
        h_pad=0.1  # 纵向子图间距
    )
    plt.show()

# trace_id2figs = {
#     1:[6,7],
#     2:[7],
#     8:[3],
#     9:[0],
#     12:[3,5],
#     16:[2,],
#     17:[1,7],
#     24:[1,8],
# }
#
# for trace_id in trace_id2figs:
#     print(trace_id)
#     trace_file = 'network_traces/sampled_traces_for_test/' + str(trace_id)
#
#     bw_time, bw = load_trace(trace_file)
#     t_len = 400
#     for i in trace_id2figs[trace_id]:
#         t_start = i * t_len
#         fig, ax = plt.subplots(figsize=(12,4), facecolor='white')
#         ax.set_facecolor('#f5f5f5')
#
#         ax.set_xticks(np.arange(0, t_len + 9, 50))
#         ax.set_xlim(0, 400)
#
#         y_max = max(bw[t_start: t_start+t_len])
#         ax.set_yticks(np.arange(0, 50 + 4.9, 10))
#         ax.set_ylim(0, 50)
#         for y in np.arange(0, 50 + 4.9, 10):
#             ax.axhline(y=y, color='grey', linestyle='-', linewidth=1.5, alpha=0.5)
#
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('Throughput (Mbps)')
#         ax.plot(np.arange(t_len), bw[t_start: t_start+t_len], linewidth=1.5, color=(80 / 255, 167 / 255, 252 / 255))
#
#         # plt.savefig('./throughput_time.svg')
#         plt.tight_layout(
#             pad=1.6,  # 整体边距
#             w_pad=0.5,  # 横向子图间距
#             h_pad=0.1  # 纵向子图间距
#         )
#         plt.show()
