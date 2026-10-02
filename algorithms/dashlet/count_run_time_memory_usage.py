import os
import time
import sys

import numpy as np
from typing_extensions import runtime

sys.path.append('/home/dell/Xinyu/Code/PDAS_Incendio/')
sys.path.append('/data/xinyu/pdas/')

from algorithms.count_memory_dl import measure_memory

def run_dashlet():

    chunklength = 2000  # ms
    # run for pdas
    epoch = 20
    dataset_dir = '../data/dataset'
    os.system(f'python run_dashlet.py '
              # f'--trace sampled_traces_for_motivation '
              f'--trace sampled_trace_for_memory_runtime '
              # f'--trace trace_low '
              f'--dataset_dir {dataset_dir} '
              f'--chunklength {chunklength} ')

memory_usages = []
runtimes = []
for i in range(5):
    memory_usage, runtime = measure_memory(run_dashlet)
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
