import os
import time

dataset_dir = '../data/dataset'
# dataset_dir = '../data/dataset_2s_test'
# dataset_dir = '../data/sub_datasets/1_subdataset'
chunklength = 2000 # ms

stime = time.time()

# epoch = 120 # for rebuf penaty is 18
epoch = 120

os.system(f'python run_incendio.py '
          f'--trace sampled_traces_for_test '
          # f'--trace trace_for_exp3 '
          # f'--trace sampled_4G '
          f'--dataset_dir {dataset_dir} '
          f'--epoch {epoch} '
          f'--chunklength {chunklength} '
          f'--train_type RL')

etime = time.time()

print('time cost is :', etime - stime)