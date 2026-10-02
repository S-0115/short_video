import os
import time

dataset_dir = '../data/dataset'
# dataset_dir = '../data/dataset_2s_test'
# dataset_dir = '../data/sub_datasets/0_subdataset'


chunklength = 2000 # ms

stime = time.time()
# run for dashlet
os.system(f'python run_dashlet.py '
          # f'--trace sampled_traces_for_motivation '
          f'--trace sampled_traces_for_test '
          # f'--trace sampled_4G '
          # f'--trace trace_for_exp3 '
          f'--dataset_dir {dataset_dir} '
          f'--chunklength {chunklength} ')
etime = time.time()

print('time cost is :', etime - stime)
# run for fixed_preload
# os.system(f'python run.py --quickstart fixed_preload '
#           f'--trace mmgc-test '
#           f'--sub_dataset {sub_dataset} '
#           f'--video_size_dir {video_size_dir} '
#           f'--user_ret_dir {user_ret_dir} '
#           f'--sample_user_dir {sample_user_dir} '
#           f'--sample_user_prop_dir {sample_user_prop_dir}')

# run for no_save
# os.system(f'python run.py --baseline no_save '
#           f'--trace mmgc-test '
#           f'--sub_dataset {sub_dataset} '
#           f'--video_size_dir {video_size_dir} '
#           f'--user_ret_dir {user_ret_dir} '
#           f'--sample_user_dir {sample_user_dir} '
#           f'--sample_user_prop_dir {sample_user_prop_dir}')