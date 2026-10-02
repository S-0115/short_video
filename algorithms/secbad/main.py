import argparse
import warnings
import numpy as np
import torch
from environments.short_video_streaming.short_video import short_video_env

import os

import platform
if platform.system() == 'Linux':
    os.environ['MUJOCO_GL'] = "egl"

from config.short_video_streaming import args_short_video_streaming_non_secbad

from mixed_learner import MixedLearner


def main():
    DEBUG = False
    train = True
    CUDA = torch.cuda.is_available()

    if CUDA and not DEBUG:
        NUM_PROCESSES = 8
        EVAL_INTERVAL = 20  # 这个并不决定 test evaluation 的频率，他只是用 online 搜集到的数据算一个 training 时候的 reward
    else:
        NUM_PROCESSES = 4
        EVAL_INTERVAL = 10
        NUM_FRAMES = 5000000

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env-type', default='args_short_video_streaming_non_secbad')

    args, rest_args = parser.parse_known_args()
    env = args.env_type

    args = globals()[env].get_args(rest_args)

    # overwrite settings in config folder
    if DEBUG:
        args.exp_label = 'debug_' + args.exp_label
        args.num_frames = NUM_FRAMES
    args.num_processes = NUM_PROCESSES
    args.eval_interval = EVAL_INTERVAL
    args.train = train

    args.results_log_dir = './logs/secbad_train'

    # begin training (loop through all passed seeds)
    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed

    # args.device = 'cuda:1'
    for seed in seed_list:
        print('training', seed)
        args.seed = seed
        args.action_space = None
        learner = MixedLearner(args)

        # args.model_path = './logs/secbad_train/logs_short_video_env/secbad_12__01_15_02_09_23/models/'
        # learner.load_model('2100')

        # if args has attribute visualize_model
        # torch.autograd.set_detect_anomaly(True)
        learner.train()


if __name__ == '__main__':
    main()
