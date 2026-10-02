import os
import sys

import numpy as np
import torch

sys.path.append('/home/dell/Xinyu/Code/PDAS_Incendio/')
sys.path.append('/data/xinyu/pdas/')

from algorithms.count_memory_dl import measure_memory
from algorithms.incendio import run_incendio

def run():
    # dataset_dir = '../data/dataset'
    dataset_dir = '../data/dataset_2s_test'
    # dataset_dir = '../data/sub_datasets/0_subdataset'
    # dataset_dir = '../data/dataset_2s_train'
    # dataset_dir = '../data/sub_datasets/0_subdataset'

    chunklength = 2000 # ms
    # run for pdas
    epoch = 60
    # run_incendio.main('sampled_trace_for_memory_runtime', dataset_dir, str(epoch), chunklength, 'RL')
    os.system(f'python run_incendio.py '
              # f'--trace sampled_4G '
              f'--trace sampled_trace_for_memory_runtime '
              f'--dataset_dir {dataset_dir} '
              f'--epoch {epoch} '
              f'--chunklength {chunklength} '
              f'--train_type RL')


memory_usages = []
runtimes = []
for i in range(5):
    memory_usage, runtime = measure_memory(run)
    memory_usages.append(memory_usage)
    runtimes.append(runtime)
print()
mean_memory_usage = np.mean(memory_usages)
upper_memory_usage = np.max(memory_usages) - mean_memory_usage
lower_memory_usage = mean_memory_usage - np.min(memory_usages)
vary = max(upper_memory_usage, lower_memory_usage)
print(f'memory_usages: {mean_memory_usage}+-{vary}')

mean_runtime = np.mean(runtimes)
upper_runtime = np.max(runtimes) - mean_runtime
lower_runtime = mean_runtime - np.min(runtimes)
vary = max(upper_runtime, lower_runtime)
print(f'runtime: {mean_runtime}+-{vary}')
