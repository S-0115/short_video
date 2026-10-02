import multiprocessing as mp

def run_mp(network_trace_dir, dataset_dir, chunklength):
    p = mp.Process(target=run_dashlet, args=(network_trace_dir, dataset_dir, chunklength))

    p.start()

def run_dashlet(network_trace_dir, dataset_dir, chunklength):
    os.system(f'python run_ard.py '
              f'--trace {network_trace_dir} '
              f'--dataset_dir {dataset_dir} '
              f'--chunklength {chunklength} ')

import os

if __name__ == '__main__':
    network_datasets_dir = '../data/network_traces/synthetic_network_trace'
    dataset_dir = '../data/dataset_2s_test'

    sub_dirs = os.listdir(network_datasets_dir)

    for sub_dataset in sub_dirs:
        print(sub_dataset)
        network_trace_dir = network_datasets_dir.split('/')[-1] + '/' +sub_dataset

        chunklength = 2000.

        # run for dashlet
        run_mp(network_trace_dir, dataset_dir, chunklength)