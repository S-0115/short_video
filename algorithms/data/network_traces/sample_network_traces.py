import numpy as np
import os

original_network_trace_file = './NewFile-HighDensity-4G.txt'
original_network_trace = [] # kB

network_trace_num = 3

with open(original_network_trace_file, 'r') as f:
    for line in f:
        original_network_trace.append(float(line.strip())) # kB

original_network_trace = np.array(original_network_trace)

print(original_network_trace.shape)
# # seed for motivation: 11
# # seed for test2: 42    seed for test3: 66
# np.random.seed(66)
# seed_for_sample = np.random.randint(10000, size=(1000, 1))
# dataset_dir = './sampled_traces_for_test3/'
# if not os.path.exists(dataset_dir):
#     os.makedirs(dataset_dir)
#
# sampled_network_trace_num = 0
# seed_idx = 0
# while sampled_network_trace_num < network_trace_num:
#
#     np.random.seed(seed_for_sample[seed_idx])
#     begin_idx = np.random.randint(0, len(original_network_trace) - 4000)
#     end_idx = begin_idx + 4000
#
#     sampled_network_trace = original_network_trace[begin_idx:end_idx] * 8 / 1000. # Mb
#     sampled_network_trace = np.array(sampled_network_trace)
#     zero_num = np.count_nonzero(sampled_network_trace == 0)
#     if zero_num >= 20:
#         seed_idx += 1
#         print(zero_num)
#         continue
#
#     output_file = dataset_dir + str(sampled_network_trace_num)
#     with open(output_file, 'w') as f:
#         for time in range(len(sampled_network_trace)):
#             f.write(f'{time} {sampled_network_trace[time]}\n')
#     sampled_network_trace_num += 1
#     seed_idx += 1
#     print(sampled_network_trace_num)
#
#
