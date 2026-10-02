import numpy as np

from config_algorithm import watch_time_threhold
def load_video_dimension_data(path):
    user_watch_times = []
    with open(path, 'rb') as f:
        for line in f:
            watch_time = float(line) / 1000.
            if watch_time > watch_time_threhold:
                user_watch_times.append(watch_time)
    return user_watch_times

def compute_weibull_params(x):
    x = np.sort(x)
    # print(x)
    n = len(x)
    F_emp = (np.arange(1, n + 1) - 0.3)/(n + 0.4)

    Y = np.log(x - watch_time_threhold)
    X = np.log(-np.log(1 - F_emp))
    b=((n * np.sum(Y * X)) - (np.sum(X) * np.sum(Y))) / (n * np.sum(Y * Y) - np.sum(Y) ** 2)
    # print(Y.mean() - X.mean() / b)
    eta = np.exp(Y.mean() - X.mean() / b)  # η̂
    # # print(f'LSM 估计：β = {b:.3f}，η = {eta:.1f} 秒')

    return b, eta


if __name__ == '__main__':
    dataset_dir = '../data/dataset'
    # dataset_dir = '../data/dataset_2s_test'
    # dataset_dir = '../data/dataset_2s_train'

    # dataset_dir = '../data/sub_datasets/0_subdataset'
    # dataset_dir = '../data/sub_datasets/1_subdataset'
    # dataset_dir = '../data/sub_datasets/2_subdataset'

    # dataset_dir = '../data/sub_datasets_view_percentage/0_subdataset'
    # dataset_dir = '../data/sub_datasets_view_percentage/1_subdataset'
    # dataset_dir = '../data/sub_datasets_view_percentage/2_subdataset'

    video_names_file = dataset_dir + '/video_names.csv'

    result = []
    with open(video_names_file, 'r') as f:
        for line in f:
            infos = line.split(',')
            video_name = infos[1]
            path = dataset_dir + '/view_duration/' + video_name
            x = load_video_dimension_data(path)
            b, eta = compute_weibull_params(x)
            modified_line = f'{infos[0]},{infos[1]},{infos[2]},{b:.3f}&{eta:.3f}\n'
            result.append(modified_line)
    with open(dataset_dir + '/video_names.csv', 'w') as f:
        f.writelines(result)

    # path = '../data/dataset_2s_test/view_duration/0_3037628'
    # x = load_video_dimension_data(path)
    # print(x)
    # b, eta = compute_weibull_params(x)