import os
import random
import shutil

import numpy as np

dataset = './random_sampled_video/'
video_info_file = dataset + 'video_names.csv'
user_ret_dir = dataset + 'user_ret/'
user_switch_prob_dir = dataset + 'user_switch_prob/'
user_view_duration_dir = dataset + 'view_duration/'

video_name2duration = {}

with open(video_info_file, 'r') as f:
    for line in f:
        info = line.split(',')
        video_name2duration[info[1]] = info[2]

dataset_size = 100
user_num = 1
dataset_dir = './dataset/'

seed_ = 77
# 44 for test; 77 for train; 42 for dataset
np.random.seed(seed_)

random.seed(seed_)

video_names = video_name2duration.keys()

videos = random.sample(video_names, dataset_size)

random.shuffle(videos)

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

dataset_ret_dir = dataset_dir + 'user_ret/'
if not os.path.exists(dataset_ret_dir):
    os.makedirs(dataset_ret_dir)

dataset_switch_prob_dir = dataset_dir + 'user_switch_prob/'
if not os.path.exists(dataset_switch_prob_dir):
    os.makedirs(dataset_switch_prob_dir)


dataset_view_duration_dir = dataset_dir + 'view_duration/'
if not os.path.exists(dataset_view_duration_dir):
    os.makedirs(dataset_view_duration_dir)

with open(dataset_dir + 'video_names.csv', 'w') as f:
    for j in range(len(videos)):
        video_name = videos[j].split('_')[1]
        f.write(f'{j},{j}_{video_name},{video_name2duration[videos[j]]},1&2&3\n')

        source_user_ret_file = user_ret_dir + videos[j]
        des_user_ret_file = dataset_ret_dir + str(j) + '_' + video_name
        shutil.copy(source_user_ret_file, des_user_ret_file)

        source_user_switch_prob_file = user_switch_prob_dir + videos[j]
        des_user_switch_prob_file = dataset_switch_prob_dir + str(j) + '_' + video_name
        shutil.copy(source_user_switch_prob_file, des_user_switch_prob_file)

        source_view_duration_file = user_view_duration_dir + videos[j]
        des_view_duration_file = dataset_view_duration_dir + str(j) + '_' + video_name
        shutil.copy(source_view_duration_file, des_view_duration_file)


dataset_sample_user_dir = dataset_dir + 'sample_user/'
if not os.path.exists(dataset_sample_user_dir):
    os.makedirs(dataset_sample_user_dir)

view_duration_video = [None] * dataset_size

for view_duration_file_name in os.listdir(dataset_view_duration_dir):
    view_duration_file_path = os.path.join(dataset_view_duration_dir, view_duration_file_name)
    view_durations = []
    with open(view_duration_file_path, 'r') as f:
        for line in f:
            if float(line) > 0.:
                view_durations.append(float(line))
    view_duration_video[int(view_duration_file_name.split('_')[0])] = view_durations

# seed_for_sample = np.random.randint(10000, size=(1001, 1))
# for j in range(user_num):
#     np.random.seed(seed_for_sample[j])
#     seeds = np.random.randint(10000, size=(dataset_size, 2))  # reset the sample random seeds
#
#     sampled_user_j = dataset_dir + 'sample_user' + '/user_' + str(j) + '.txt'
#
#     with open(sampled_user_j, 'w') as f:
#         for video in range(dataset_size):
#
#             seed = seeds[video]
#             random.seed(seed[0])
#
#             # video_duration = float(video_name2duration[videos[video]])
#
#             view_durations = view_duration_video[video]
#             sample_playback_duration = random.sample(view_durations, 1)[0]
#
#             f.write(str(sample_playback_duration) + '\n')
#             # print(sample_playback_duration)

