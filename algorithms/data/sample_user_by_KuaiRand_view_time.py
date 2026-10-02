import os
import sys
from platform import system

import numpy as np
import random


def filter_view_duration(view_durations, begin, interval):
    filter_view_duration = []
    for view_duration in view_durations:
        if view_duration >= begin and view_duration < begin + interval:
            filter_view_duration.append(view_duration)
    return filter_view_duration

RANDOM_SEED = 42  # the random seed for user retention
np.random.seed(RANDOM_SEED)

user_num = 5
ALL_VIDEO_NUM = 100
MILLISECONDS_IN_SECOND = 1000.

dataset = './dataset/'

view_duration_dir = './view_duration/'

view_duration_video = [None] * ALL_VIDEO_NUM

for view_duration_file_name in os.listdir(view_duration_dir):
    view_duration_file_path = os.path.join(view_duration_dir, view_duration_file_name)
    view_durations = []
    with open(view_duration_file_path, 'r') as f:
        for line in f:
            view_durations.append(float(line))
    view_duration_video[int(view_duration_file_name.split('_')[0])] = view_durations

# print(view_duration_video)

interval = float(sys.argv[1])
begin = float(sys.argv[2])
print(begin, interval)

seed_for_sample = np.random.randint(10000, size=(1001, 1))
for j in range(user_num):
    np.random.seed(seed_for_sample[j])
    seeds = np.random.randint(10000, size=(ALL_VIDEO_NUM, 2))  # reset the sample random seeds

    sampled_user_j_2s = './dataset_2s/' + 'sample_user' + '/user_' + str(j) + '.txt'

    with open(sampled_user_j_2s, 'w') as f_2s:
        for video in range(ALL_VIDEO_NUM):

            seed = seeds[video]
            random.seed(seed[0])

            view_durations = view_duration_video[video]
            max_view_time = max(view_durations)
            view_durations = filter_view_duration(view_durations, begin, interval)
            if view_durations == []:
                random.seed(seed[1])
                random_generate_playback_duration = random.randint(begin, begin + interval)

                sample_playback_duration = min(max_view_time, random_generate_playback_duration)
            else:
                sample_playback_duration = random.sample(view_durations, 1)[0]

            f_2s.write(str(sample_playback_duration) + '\n')
            # print(sample_playback_duration)
