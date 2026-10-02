import multiprocessing as mp

def run_mp(dataset_dir, chunklength):
    p = mp.Process(target=run_dashlet, args=(dataset_dir, chunklength))

    p.start()

def run_dashlet(dataset_dir, chunklength):
    os.system(f'python run_dashlet.py '
              # f'--trace trace_low '
              f'--trace sampled_traces_for_test '
              f'--dataset_dir {dataset_dir} '
              f'--chunklength {chunklength} ')

import os

if __name__ == '__main__':
    sub_datasets_dir = '../data/sub_datasets'
    # sub_datasets_dir = '../data/sub_datasets_view_percentage'

    sub_datasets = os.listdir(sub_datasets_dir)
    sorted_sub_datasets = [''] * len(sub_datasets)
    for sub_dataset in sub_datasets:
        idx = int(sub_dataset.split('_')[0])
        sorted_sub_datasets[idx] = sub_dataset

    for sub_dataset in sorted_sub_datasets:
        print(sub_dataset)
        dataset_dir = sub_datasets_dir + '/' +sub_dataset

        chunklength = 2000.

        # run for dashlet
        run_mp(dataset_dir, chunklength)