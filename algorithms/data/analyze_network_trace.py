import os

import matplotlib.pyplot as plt
import numpy as np

from simulator import short_video_load_trace

# trace_dir = './network_traces/sampled_4G/'
# trace_dir = './network_traces/sampled_4G_train/'
# trace_dir = './network_traces/synthetic_network_trace/'
# ['24', '76', '38', '70', '12', '7', '39', '31', '9']
# trace_dir = './network_traces/sampled_traces_for_train/'
trace_dir = './network_traces/sampled_traces_for_test/'
# trace_dir = './network_traces/synthetic_network_trace/0.1/'
# trace_dir = './network_traces/sampled_trace_for_memory_runtime/'
# trace_dir = './network_traces/trace_low/'

all_cooked_time, all_cooked_bw = short_video_load_trace.load_trace(trace_dir)
trace_file_name = os.listdir(trace_dir)

print(f'trace num is {len(trace_file_name)}')

mean_traces = []
std_traces = []
cv_trace = []
max_trace = []
min_trace = []

headers = 'trace_name 均值 标准差 变异系数 高于25Mb的带宽%'
for header in headers.split(' '):
    print(f'{header:12}', end='')
print()
for i in range(len(all_cooked_bw)):
    bw = np.array(all_cooked_bw[i])#[:2000]
    mean_ = np.mean(bw)
    std_ = np.std(bw)
    cv_ = std_ / mean_
    max_ = np.max(bw)
    min_ = np.min(bw)

    mean_traces.append(mean_)
    std_traces.append(std_)
    cv_trace.append(cv_)
    max_trace.append(max_)
    min_trace.append(min_)


# info_traces = np.array([trace_file_name, mean_traces, std_traces, cv_trace,percentage_25], dtype=float).T
info_traces = np.array([trace_file_name, mean_traces, std_traces, cv_trace, max_trace, min_trace], dtype=float).T
sort_idx = np.argsort(info_traces[:, 1])
sorted_info_traces = info_traces[sort_idx,:]
print(sorted_info_traces)

    # print(f'{mean_:10.2f}', end='')
    # print(f'{std_:10.2f}', end='')
    # print(f'{cv_:10.2f}', end='')
    # print()
    # plt.plot(bw)
    # plt.title(f'{i} mean {mean_:10.2f} {std_:10.2f} {cv_:10.2f}')
    # plt.show()