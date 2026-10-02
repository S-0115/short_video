import os
import time

dataset_dir = '../data/dataset'
# dataset_dir = '../data/dataset_2s_test'
# dataset_dir = '../data/sub_datasets/0_subdataset'
# dataset_dir = '../data/dataset_2s_train'
# dataset_dir = '../data/sub_datasets/0_subdataset'


stime = time.time()
chunklength = 2000 # ms
# run for pdas
os.system(f'python run_ard.py '
          f'--trace sampled_traces_for_test '
          # f'--trace trace_for_exp3 '
          f'--dataset_dir {dataset_dir} '
          f'--chunklength {chunklength} ')
etime = time.time()

print('time cost is :', etime - stime)
