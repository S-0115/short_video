import os

import numpy as np

def count_view_percentage_one_dataset(dataset_path):

    video_durations = []
    video_duration_file = os.path.join(dataset_path, "video_names.csv")
    with open(video_duration_file, "r") as f:
        for line in f:
            video_durations.append(float(line.split(",")[2]))

    sample_user_dir = os.path.join(dataset_path, "sample_user")

    all_user_watch_duration = []
    all_user_watch_percentage = []
    for sample_user_file_name in os.listdir(sample_user_dir):
        sample_user_file = os.path.join(sample_user_dir, sample_user_file_name)
        one_user_watch_duration = []
        one_user_watch_percentage = []

        with open(sample_user_file, "r") as f:
            idx = 0
            for line in f:
                one_user_watch_duration.append(float(line))
                one_user_watch_percentage.append(one_user_watch_duration[-1] / video_durations[idx])
                idx += 1

        all_user_watch_duration.append(np.mean(one_user_watch_duration))
        all_user_watch_percentage.append(np.mean(one_user_watch_percentage))

    print(all_user_watch_duration)
    print(all_user_watch_percentage)
    print(np.mean(all_user_watch_duration))
    print(np.mean(all_user_watch_percentage))

count_view_percentage_one_dataset('./dataset')
# count_view_percentage_one_dataset('./dataset_2s_test')
# count_view_percentage_one_dataset('./sub_datasets/0_subdataset')
# count_view_percentage_one_dataset('./sub_datasets/1_subdataset')
# count_view_percentage_one_dataset('./sub_datasets/2_subdataset')
# count_view_percentage_one_dataset('./sub_datasets_view_percentage/0_subdataset')

# sub_datasets_dir = './sub_datasets'
# for sub_dataset_name in os.listdir(sub_datasets_dir):
#     print(sub_dataset_name)
#     sub_dataset_path = os.path.join(sub_datasets_dir, sub_dataset_name)
#     count_view_percentage_one_dataset(sub_dataset_path)