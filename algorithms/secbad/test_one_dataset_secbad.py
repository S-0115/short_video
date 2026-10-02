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


def main():
    train = False

    # log_dir_all = './logs/secbad_train/logs_short_video_env/with_latent_fixed_3'
    # log_dir_all = './logs/secbad_train/logs_short_video_env/with_latent_4s_3'
    log_dir_all = './logs/secbad_train/logs_short_video_env/with_latent_fixed_1'
    config_file = log_dir_all + '/config.json'
    with open(config_file, mode='r') as f:
        config = json.load(f)

    args = argparse.Namespace(**config)

    args.results_log_dir = './logs/secbad_test'
    args.train = train

    # args.dataset_path_test = '../data/dataset_2s_test/'
    args.dataset_path_test = '../data/dataset/'
    # args.dataset_path_test = '../data/dataset_2s_train/'
    # args.dataset_path_test = '../data/dataset_state/'
    # args.dataset_path_test = '../data/sub_datasets/1_subdataset/'

    # args.network_traces_path_test = '../data/network_traces/sampled_trace_for_memory_runtime/'
    args.network_traces_path_test = '../data/network_traces/sampled_traces_for_test/'
    # args.network_traces_path_test = '../data/network_traces/trace_for_exp3/'
    # args.network_traces_path_test = '../data/network_traces/sampled_4G_train/'

    args.model_path = log_dir_all + '/models/'

    args.chunklength = 2000.

    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed

    for seed in seed_list:
        print('training', seed)

        # args.use_best_latent_selection = False
        # args.max_input_history_length = 4

        args.seed = seed
        args.action_space = None
        args.device = 'cuda:1'

        learner = MixedLearner(args)

        # epochs = [1600, 2000, 2400]
        epochs = [600]
        for epoch in epochs:
            learner.load_model(str(epoch)) # 300 500
            with torch.no_grad():
                s_time = time.time()
                learner.test(time.time(), False, log_dir_all)
                # learner.test(time.time(), True)
                # learner.test(time.time(), False)
                e_time = time.time()
                print(f'time cost is {e_time - s_time}')
        #
        # learner.load_model(str(2000)) # 300 500
        # with torch.no_grad():
        #     learner.test(time.time())
        # learner.transform_to_onnx('1600')

    # if args has attribute visualize_model
    # if hasattr(args, 'visualize_model') and args.visualize_model:
    #     learner.visualize()
    # else:
    #     learner.train()


if __name__ == '__main__':
    main()
