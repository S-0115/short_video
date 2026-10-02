import os

import numpy as np
from config_algorithm import VIDEO_BIT_RATE


dataset_dir = 'data/dataset/'
view_len_file = dataset_dir + 'video_names.csv'
view_time_file = dataset_dir + 'sample_user/user_0.txt'
algorithm = 'dashlet'

count_pre_chunk_num = 3
split_watch_ratio = 0.2
split_time_of_long_short_view = 12


video_view_time = []
video_len = {}

with open(view_len_file, 'r') as f:
    for line in f:
        infos = line.split(',')
        video_name = infos[1]
        video_len[video_name] = float(infos[2]) / 1000.

with open(view_time_file, 'r') as f:
    for line in f:
        video_view_time.append(float(line) / 1000.)

avg_download_bitrate_6s_long_view_video = []
avg_download_bitrate_6s_short_view_video = []

download_bitrate_video_dir = algorithm + '/log_download_bitrate/'
for video_name in video_len:
    # print(video_name, type(video_name))
    video_len_video = video_len[video_name]
    view_time = video_view_time[int(video_name.split('_')[0])]

    download_bitrate_video = []
    download_bitrate_video_file = download_bitrate_video_dir + 'download_bitrate_' + video_name
    if not os.path.exists(download_bitrate_video_file):
        continue

    with open(download_bitrate_video_file, 'r') as f:
        for line in f:
            # print(line)
            # print(np.array(eval(line)))
            download_bitrate_video.extend(np.array(eval(line)))

    r_idx = min(len(download_bitrate_video), count_pre_chunk_num)
    download_bitrate_idx = download_bitrate_video[:r_idx]
    download_bitrate = []
    for idx in download_bitrate_idx:
        download_bitrate.append(VIDEO_BIT_RATE[idx] / 1000.)

    # if view_time / video_len_video > split_watch_ratio:
    if view_time > split_time_of_long_short_view:
        avg_download_bitrate_6s_long_view_video.append(np.average(download_bitrate))
    else:
        avg_download_bitrate_6s_short_view_video.append(np.average(download_bitrate))

print(len(avg_download_bitrate_6s_long_view_video))
print(len(avg_download_bitrate_6s_short_view_video))

print(np.average(avg_download_bitrate_6s_long_view_video))
print(np.average(avg_download_bitrate_6s_short_view_video))