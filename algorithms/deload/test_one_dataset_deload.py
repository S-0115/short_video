import os
import time

dataset_dir = '../data/dataset'
# dataset_dir = '../data/dataset_2s_test'
# dataset_dir = '../data/dataset_2s_train'
# dataset_dir = '../data/sub_datasets/1_subdataset'
# dataset_dir = '../data/dataset'
chunklength = 2000 # ms

stime = time.time()

epoch = 50

os.system(f'python run_deload.py '
          # f'--trace trace_for_exp3 '
          f'--trace sampled_traces_for_test '
          # f'--trace sampled_4G '
          # f'--trace sampled_4G_train '
          f'--dataset_dir {dataset_dir} '
          f'--epoch {epoch} '
          f'--chunklength {chunklength} ')

etime = time.time()

print('time cost is :', etime - stime)