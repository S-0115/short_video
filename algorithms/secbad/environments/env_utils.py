import time

import gym
import torch

from .short_video_streaming.short_video import short_video_env as ENV
import numpy as np
import os

class VectorEnv():
    def __init__(self, encoder_input_feature, n_env, max_buffer_size, dataset_path, dataset_path_test, network_traces_path, network_traces_path_test, chunklength, train):

        self.encoder_input_feature = encoder_input_feature
        self.n_env = n_env

        self.max_buffer_size = max_buffer_size

        self.dataset_path = dataset_path
        self.dataset_path_test = dataset_path_test
        self.network_traces_path = network_traces_path
        self.network_traces_path_test = network_traces_path_test
        self.chunklength = chunklength
        self.train = train

        self._make_vector_env()
        self.test_env = ENV(encoder_input_feature=encoder_input_feature, max_buffer_size=max_buffer_size, id='test', dataset_path=self.dataset_path_test, network_traces_path=self.network_traces_path_test, chunklength=self.chunklength)

        tmp_env = ENV(encoder_input_feature=encoder_input_feature, max_buffer_size=self.max_buffer_size, network_traces_path=self.network_traces_path)
        self.get_traj_context = tmp_env.get_traj_context

        self.rew_rms = None

    def get_env_paras(self):
        return self.env_paras

    def _make_vector_env(self):
        print('Making vector env')
        # self.vec_env = gym.vector.AsyncVectorEnv([
        #     lambda: self.env_class(traj_len=self.traj_len, id=i) for i in range(self.n_env)
        # ])
        # self.vec_env = gym.vector.SyncVectorEnv([
        #     lambda: self.env_class(traj_len=self.traj_len, id=i, dataset_path=self.dataset_path,
        #                            network_traces_path=self.network_traces_path, chunklength=self.chunklength) for i in
        #     range(self.n_env)
        # ])
        # self.env_class是根据env_name找对应环境的具体实现类，然后通过gym创建多个同步的环境
        if self.train:
            self.vec_env = gym.vector.SyncVectorEnv([
                lambda i = i: ENV(encoder_input_feature=self.encoder_input_feature, max_buffer_size=self.max_buffer_size, id=i, dataset_path=self.dataset_path, network_traces_path=self.network_traces_path, chunklength=self.chunklength) for i in range(self.n_env)
            ])

        # only to get env paras
        # 获取一些环境相关的参数，例如状态空间和动作空间的维度
        tmp_env = ENV(self.encoder_input_feature, self.max_buffer_size)

        if tmp_env.action_space.__class__.__name__ == "Box":
            self.env_paras = {'dim_action': tmp_env.action_space.shape[0],
                              'action_space': tmp_env.action_space,
                              'num_feature': tmp_env.observation_space.shape[0],
                              'dim_state': tmp_env.observation_space.shape[1]}
        elif tmp_env.action_space.__class__.__name__ == "Discrete":
            self.env_paras = {'dim_action': 1,
                              'num_action': tmp_env.action_space.n,
                              'action_space': tmp_env.action_space,
                              'num_feature': tmp_env.observation_space.shape[0],
                              'dim_state': tmp_env.observation_space.shape[1]}

    def reset(self, traj_context_ls, seed=None):
        print('train env reset')
        assert len(traj_context_ls) == self.n_env
        options_dict = {'traj_context': {}}
        for i in range(self.n_env):
            options_dict['traj_context'][i] = traj_context_ls[i]
        # print(f'seed{seed}, options_dict: {options_dict}')
        init_obs = self.vec_env.reset(seed=seed, options=options_dict)
        return init_obs

    def reset_train_i(self, i):
        init_obs = self.vec_env.envs[i].reset()
        return init_obs

    def reset_test(self, traj_context, seed=None):
        print('test env reset')
        # traj_context = options['traj_context'][self.id]
        init_obs = self.test_env.reset(
            seed=seed, options={'traj_context': {self.test_env.id: traj_context}})
        return init_obs

    def get_bd_sum_test_env(self):
        return self.test_env.net_env.network.bd_sum

    def test_env_step(self, action):
        # copied from utils.helpers.env_step

        next_obs, reward, done, _,  infos = self.test_env.step(action)

        normalized_reward = None
        if self.rew_rms:
            # print(reward.shape)
            normalized_reward = (reward - self.rew_rms.mean.cpu().numpy()) / np.sqrt(self.rew_rms.var.cpu().numpy() + 1e-4)
            # normalized_reward = (reward) / np.sqrt(self.rew_rms.var.cpu().numpy() + 1e-4)
            # print(normalized_reward.shape)
            # self.rew_rms.update(torch.tensor(reward).to(self.rew_rms.device)) # 测试时，不能再对模型更新均值方差

        return next_obs, [reward, normalized_reward], done, infos

    def train_env_step(self, action):
        # print(act)
        # print(self.vec_env.step(act))
        next_obs, reward, done, _, infos = self.vec_env.step(action)

        normalized_reward = None
        if self.rew_rms:
            # print(reward.shape)
            normalized_reward = (reward - self.rew_rms.mean.cpu().numpy()) / np.sqrt(self.rew_rms.var.cpu().numpy() + 1e-4)
            # normalized_reward = (reward) / np.sqrt(self.rew_rms.var.cpu().numpy() + 1e-4)
            # print(normalized_reward.shape)
            self.rew_rms.update(torch.tensor(reward).to(self.rew_rms.device))
            # print(self.rew_rms.mean, self.rew_rms.var)
        # print(normalized_reward)
        # print(self.rew_rms.mean)
        # print(type(next_obs), type(reward), type(done), type(infos))

        return next_obs, [reward, normalized_reward], done, infos

    def train_env_i_step(self, action, i):
        # print(act)
        # print(self.vec_env.step(act))
        next_obs, reward, done, _, infos = self.vec_env.envs[i].step(action)

        return next_obs, [reward, None], done, infos

    def get_train_env_mask(self):
        mask = [env.get_mask() for env in self.vec_env.envs]
        mask = torch.as_tensor(mask, dtype=torch.bool)
        return mask

    def get_train_env_encoder_state(self):
        encoder_state = [env.get_encoder_state().astype(float)  for env in self.vec_env.envs]

        encoder_state = torch.as_tensor(np.array(encoder_state), dtype=torch.float)
        # print(encoder_state)
        return encoder_state

    def get_sleep_judge_train_env(self):
        sleep_judge = [env.get_sleep_judge() for env in self.vec_env.envs]
        sleep_judge = torch.as_tensor(sleep_judge, dtype=torch.bool)
        return sleep_judge

    def get_sleep_judge_train_env_i(self, i):
        sleep_judge = self.vec_env.envs[i].get_sleep_judge()
        sleep_judge = torch.as_tensor(sleep_judge, dtype=torch.bool)
        return sleep_judge

    def get_test_env_mask(self):
        mask = self.test_env.get_mask()
        mask = torch.as_tensor(mask, dtype=torch.bool)
        return mask

    def get_test_env_encoder_state(self):
        encoder_state = self.test_env.get_encoder_state().astype(float)
        encoder_state = torch.as_tensor(np.array(encoder_state), dtype=torch.float)
        return encoder_state

    def get_sleep_judge_test_env(self):
        sleep_judge = self.test_env.get_sleep_judge()
        sleep_judge = torch.as_tensor(sleep_judge, dtype=torch.bool)
        return sleep_judge

    def close(self):
        print('close env')
        self.vec_env.close()
        self.test_env.close()
        pass
