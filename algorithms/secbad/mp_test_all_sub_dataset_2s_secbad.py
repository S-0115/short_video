import argparse
import json
import time
import warnings
import numpy as np
import torch
from environments.short_video_streaming.short_video import short_video_env

import os

from config.short_video_streaming import args_short_video_streaming_non_secbad

from mixed_learner import MixedLearner
import multiprocessing as mp

def run_secbad(dataset_path, network_traces_path_test, log_dir, chunklength):
    train = False

    log_dir_all = log_dir
    config_file = log_dir_all + 'config.json'
    with open(config_file, mode='r') as f:
        config = json.load(f)

    args = argparse.Namespace(**config)

    args.results_log_dir = './logs/secbad_test'
    args.train = train

    # args.dataset_path_test = '../data/dataset_2s_test/'

    args.dataset_path_test = dataset_path
    args.network_traces_path_test = network_traces_path_test

    args.model_path = log_dir_all + 'models/'

    args.chunklength = chunklength

    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed

    for seed in seed_list:
        print('training', seed)
        args.seed = seed
        args.action_space = None

        learner = MixedLearner(args)

        learner.load_model('51000')
        with torch.no_grad():
            learner.log(time.time())

def run_mp(dataset_path, network_traces_path_test, log_dir, chunklength):
    p = mp.Process(target=run_secbad, args=(dataset_path, network_traces_path_test, log_dir, chunklength))

    p.start()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    sub_dataset_dir = '../data/sub_datasets/'
    # sub_dataset_dir = './data/shuffled_dataset_2s/'
    # sub_dataset_dir = './data/dataset_explanation/'
    sub_datasets = os.listdir(sub_dataset_dir)
    sorted_sub_datasets = [''] * len(sub_datasets)
    for sub_dataset in sub_datasets:
        idx = int(sub_dataset.split('_')[0])
        sorted_sub_datasets[idx] = sub_dataset

    network_traces_path_test = '../data/network_traces/sampled_4G/'

    log_dir = './logs/secbad_train/logs_short_video_env/secbad_12__09_29_02_57_01/'
    chunklength = 2000.
    for sub_dataset in sorted_sub_datasets:
        print(sub_dataset)
        dataset_path = os.path.join(sub_dataset_dir, sub_dataset) + '/'

        # run for dashlet
        run_mp(dataset_path, network_traces_path_test, log_dir, chunklength)