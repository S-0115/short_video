import math
import os
import sys

import numpy as np
import torch

# print(sys.path)
sys.path.append('/home/dell/Xinyu/Code/PDAS_Incendio/')
sys.path.append('/data/xinyu/pdas/')
sys.path.append('/mnt/mydisk/xinyu/secbad/')
from simulator import controller as env, short_video_load_trace


import gym
from gym import spaces

from config_algorithm import VIDEO_BIT_RATE
from config_algorithm import alpha, beta, gamma, theta
PAST_BW_LEN = 5

class short_video_env(gym.Env):
    def __init__(self, encoder_input_feature, max_buffer_size, id=None, dataset_path=None, network_traces_path=None, chunklength=None):
        # https://www.runoob.com/w3cnote/python-extends-init.html
        """
        """
        super().__init__()
        self.encoder_input_feature = encoder_input_feature
        self.max_buffer_size = max_buffer_size

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,5), dtype=np.float32)
        # self.action_space = spaces.Box(low=0., high=1.0, shape=(31,), dtype=np.float32)
        self.action_space = spaces.Discrete(30)

        self.dataset_path = dataset_path
        self.network_traces_path = network_traces_path

        if dataset_path and network_traces_path:
            all_cooked_time, all_cooked_bw = short_video_load_trace.load_trace(self.network_traces_path)
            self.all_cooked_time = all_cooked_time
            self.all_cooked_bw = all_cooked_bw
            user_num = os.listdir(self.dataset_path + 'sample_user/')
            user_sample_id = np.random.randint(len(user_num))
            trace_id = np.random.randint(len(all_cooked_time))

            user_swipe_trace = self.dataset_path + 'sample_user/user_' + str(user_sample_id) + '.txt'
            swipe_trace_user_i = []
            with open(user_swipe_trace, 'r') as f:
                for line in f:
                    swipe_trace_user_i.append(float(line))
            self.ALL_VIDEO_NUM = len(swipe_trace_user_i)
            self.last_rebufs = [0.] * self.ALL_VIDEO_NUM
            self.last_chunk_bitrate = [-1] * self.ALL_VIDEO_NUM

            self.chunklength = chunklength
            net_env = env.Environment(user_sample_id, all_cooked_time[trace_id], all_cooked_bw[trace_id], self.ALL_VIDEO_NUM, swipe_trace_user_i, self.dataset_path, chunklength)
            self.net_env = net_env
            retention_probs = self.calculate_retention_probabilities(self.net_env.players)
            # print(retention_probs)
            self.pre_retention_probs = retention_probs

        self.id = id  # when multiple environments are created in VectorEnvs, the id is used to distinguish them

        self.past_bandwidth = list(np.zeros(PAST_BW_LEN))
        self.past_bandwidth_ests = list(np.zeros(PAST_BW_LEN))
        self.past_errors = list(np.zeros(PAST_BW_LEN))

        self.play_video_id = 0

        self.train_mode = True

        # self.delay = 0
        self.last_info = {}
        self.observation_encoder = []

    # rewrite step function to recompute the reward function

    def step(self, action):
        # print(action)
        act = int(action[0])
        # print(act)
        selected_ct = None
        if len(action) != 1:
            selected_ct = int(action[1])
            # print(selected_ct)

        if act == 30:
            download_video_id = bit_rate = -1
            sleep_time = 200.
        else:
            download_video_id = act // 6 + self.play_video_id
            bit_rate = act % 6
            sleep_time = 0

        # print('action is ',download_video_id, bit_rate, sleep_time)
        # calculate the quality and smooth for this download step taken
        quality = 0
        smooth = 0
        # quality_rew = 0
        # smooth_rew = 0
        # waste_bytes_rew = 0
        if sleep_time == 0:
            # the last chunk id that user watched
            max_watch_chunk_id = self.net_env.user_models[
                download_video_id - self.net_env.get_start_video_id()].get_watch_chunk_cnt()
            # last downloaded chunk id
            download_chunk = self.net_env.players[download_video_id - self.net_env.get_start_video_id()].get_chunk_counter()
            if max_watch_chunk_id >= download_chunk:  # the downloaded chunk will be played
                quality = VIDEO_BIT_RATE[bit_rate]
                if download_chunk == max_watch_chunk_id:  # maintain the last_chunk_bitrate array
                    self.last_chunk_bitrate[download_video_id] = bit_rate
                    rel_id = download_video_id - self.net_env.get_start_video_id()
                    if rel_id + 1 < len(self.net_env.user_models):  # If its not the last visible video
                        if self.net_env.players[rel_id + 1].get_chunk_counter() != 0:
                            # if the next video chunk has already been downloaded before this last chunk,
                            # we include the smooth penalty here.
                            next_bitrate = self.net_env.players[rel_id + 1].get_downloaded_bitrate()[0]
                            smooth += abs(quality - VIDEO_BIT_RATE[next_bitrate])
                smooth += self.get_smooth(self.net_env, download_video_id, download_chunk, quality)
                # print("Causing smooth penalty: ", smooth, file=log_file)
            # quality_rew = self.pre_retention_probs[download_video_id - self.net_env.get_start_video_id()] * quality
            # smooth_rew = self.pre_retention_probs[download_video_id - self.net_env.get_start_video_id()] * smooth

        # print(self.net_env.get_start_video_id(), download_video_id, bit_rate, sleep_time)
        # print(self.pre_retention_probs)
        user_rets = self.net_env.user_models.copy()

        throughput = self.net_env.network.cooked_bw[self.net_env.network.mahimahi_ptr] * 0.95

        if act != 30 and selected_ct:
            self.net_env.players[download_video_id - self.net_env.get_start_video_id()].record_download_bitrate_ct(selected_ct)

        if self.net_env.play_video_id == 0 and len(self.net_env.players[0].download_chunk_bitrate) == 0:
            play_chunk_bitrate = play_chunk_ct = -1
        else:
            play_chunk_bitrate = self.net_env.players[0].get_play_chunk_bitrate()
            if self.net_env.players[0].download_bitrate_ct != []:
                # print(self.net_env.play_video_id, self.net_env.get_start_video_id())
                play_chunk_ct = self.net_env.players[0].get_play_chunk_ct()
            else:
                play_chunk_ct = -1

        # play_video_max_watch_chunk_id = self.net_env.user_models[0].get_watch_chunk_cnt()
        # if play_video_max_watch_chunk_id < 6:
        #     view_type = 0
        # else:
        #     view_type = 1

        view_type = self.pre_retention_probs


        delay, rebuf, video_size, end_of_video, \
            play_video_id, waste_bytes, rtt = self.net_env.buffer_management(download_video_id, bit_rate, sleep_time)
        # print(delay, rebuf, end_of_video)
        if play_video_id != self.play_video_id:
            user_swipe = 1
        else:
            user_swipe = 0

        buffer_size = 0
        for player in self.net_env.players:
            buffer_size += player.buffer_size / 1000.

        # buffer_size = self.net_env.players[0].get_buffer_size() / 1000.
        #     print(player.buffer_size, player.get_remain_video_num(), player.get_chunk_sum())
        # print()

        if sleep_time == 0:
            self.past_bandwidth = np.roll(self.past_bandwidth, -1)
            self.past_bandwidth[-1] = (float(video_size) / 1000000.0) / (float(delay) / 1000.0)  # MB / s
            self.last_rebufs[download_video_id] = rebuf #/ 1000.

            # self.delay = delay

        sum_wasted_bytes = waste_bytes

        # play over all videos
        if len(self.net_env.players) < 5:
            for player in self.net_env.players:
                for i in range(len(player.download_chunk_bitrate)):
                    download_bitrate = player.download_chunk_bitrate[i]
                    download_size = player.video_size[download_bitrate][i]
                    sum_wasted_bytes += download_size
            # print('time to reset')
            done = True
        else:
            done = False

        qoe = alpha * quality / 1000. - beta * rebuf / 1000. - gamma * smooth / 1000.

        reward = alpha * quality / 1000. - beta * rebuf / 1000. - gamma * smooth / 1000. # - theta * video_size * 8 / 1000000.
        # reward = alpha * quality_rew / 1000. - beta * rebuf / 1000. - gamma * smooth_rew / 1000. # - theta * video_size * 8 / 1000000.
        reward = torch.tensor(reward)

        # 2. 计算保留概率和Max Buffer阈值

        retention_probs = self.calculate_retention_probabilities(self.net_env.players)
        self.pre_retention_probs = retention_probs

        observation = self._get_obs(self.past_bandwidth, retention_probs, self.net_env.players)

        self._get_encoder_state(self.past_bandwidth, delay, self.net_env.players)

        info = {
            "reward": reward,
            "qoe": qoe,
            "quality": quality,
            "smooth": smooth,
            "rebuf": rebuf,
            "waste_bytes": waste_bytes,
            'delay': delay,
            'play_video_id': play_video_id,
            'pre_play_video_id': self.play_video_id,
            'download_video_id': download_video_id,
            'bit_rate': bit_rate,
            'sleep_time': sleep_time,
            'user_swipe': user_swipe,
            'buffer_size': buffer_size,
            'user_rets': user_rets,
            'throughput': throughput,

            'play_chunk_bitrate': play_chunk_bitrate,
            'play_chunk_ct': play_chunk_ct,
            'view_type': view_type,
        }
        # print(info)
        if done:
            self.last_info = info

        self.play_video_id = play_video_id

        return observation, reward, done, False, info

    # rewrite reset_model function to set the traj_context

    def get_smooth(self, net_env, download_video_id, chunk_id, quality):
        if download_video_id == 0 and chunk_id == 0:  # is the first chunk of all
            return 0
        if chunk_id == 0:  # needs to find the last chunk of the last video
            last_bitrate = self.last_chunk_bitrate[download_video_id - 1]
            if last_bitrate == -1:  # the neighbour chunk is not downloaded
                return 0
        else:
            last_bitrate = net_env.players[download_video_id - net_env.get_start_video_id()].get_downloaded_bitrate()[
                chunk_id - 1]
        return abs(quality - VIDEO_BIT_RATE[last_bitrate])

    def calculate_retention_probabilities(self, Players):
        # 计算每个视频块的保留概率 p_{i,m}(mc)
        retention_probs = []
        for player in Players:
            # 根据当前播放时间 mc 和用户留存率模型 H_{i,m} 计算
            mc = math.floor(player.play_timeline / self.chunklength) # 正在播放的视频块的idx
            m = player.get_chunk_counter() # 要下载的视频块的idx
            p_i_m_mc = self.calculate_retention_probability(player, mc, m)
            retention_probs.append(p_i_m_mc)
        return retention_probs

    def calculate_retention_probability(self, player, mc, m):
        if m >= player.get_chunk_sum():
            return 0.0
        # 实现保留概率的计算逻辑

        user_time, user_retent_rate = player.get_user_model()

        # 如果用户已经看完了，则留存率为0，即不需要在考虑下载该视频
        return float(user_retent_rate[m]) / float(user_retent_rate[mc])

    def _get_obs(self, past_bandwidth, retention_probs, Players):
        # 上一次下载的平均网络带宽
        # print(past_bandwidth)
        # bt = np.array(past_bandwidth) * 8
        # print(bt)
        bt = [0] * 4
        bt.append(past_bandwidth[-1] * 8) # Mbps

        # # 上一次下载的时延
        # delay = [0] * 4
        # delay.append(delay_ / 1000.) # s

        # 用户留存率，即用户从当前视频块看到正在下载的视频块的条件概率
        lj = [retention_probs[i] for i in range(len(retention_probs))]
        while len(lj) < 5:
            lj.append(1.)

        # 缓冲大小
        gj = [Players[i].get_buffer_size() / 1000. / self.max_buffer_size for i in range(len(Players))] # s
        while len(gj) < 5:
            gj.append(0.)

        # 上一次下载视频块的比特率
        qj = []
        for i in range(len(Players)):
            download_bitrate_i = Players[i].get_downloaded_bitrate()
            if len(download_bitrate_i) > 0:
                qj.append(VIDEO_BIT_RATE[download_bitrate_i[-1]] / max(VIDEO_BIT_RATE)) # Mb
            else:
                qj.append(0.)
        while len(qj) < 5:
            qj.append(0.)

        # # 上一次下载视频块导致的再缓冲时间
        # hj = self.last_rebufs[abs_cur_play_video_id: abs_cur_play_video_id + 5]  # s
        # for i in range(len(hj)):
        #     hj[i] = hj[i]
        #
        # while len(hj) < 5:
        #     hj.append(0.)
        #
        # # 上一次下载视频块的smooth
        # fj = []
        # for i in range(len(Players)):
        #     download_bitrate_i = Players[i].get_downloaded_bitrate()
        #     if len(download_bitrate_i) > 2:
        #         fj.append(abs(VIDEO_BIT_RATE[download_bitrate_i[-1]] - VIDEO_BIT_RATE[download_bitrate_i[-2]]) / 1000.) # Mb
        #     else:
        #         fj.append(0.)
        # while len(fj) < 5:
        #     fj.append(0.)

        # print(len(Players))
        # observation = np.array([bt, lj, gj, hj, qj, fj])
        # observation = np.array([bt, lj, gj, qj, fj])
        # print(observation)
        # observation = np.array([bt, delay, lj, gj, qj])
        observation = np.array([bt, lj, gj, qj])

        return observation

    def reset(self, options=None, seed=None):
        # print(f'{self.id} get seed{seed} options{options}')
        # print('env reset')
        if seed is not None:
            np.random.seed(seed)
        if self.dataset_path and self.network_traces_path:
            trace_num = len(os.listdir(self.network_traces_path))
            # print(self.network_traces_path)
            if options != None:
                # print('reset with option')
                traj_context = options['traj_context'][self.id]
                user_sample_id = traj_context[0]
                trace_id = traj_context[1]
                # print(f'setting env {user_sample_id} {trace_id}')
            else:
                # print('reset with no options')
                user_sample_id = np.random.randint(5)
                trace_id = np.random.randint(trace_num)
                # print('sampled env', user_sample_id, trace_id)
            user_swipe_trace = self.dataset_path + 'sample_user/user_' + str(user_sample_id) + '.txt'
            swipe_trace_user_i = []
            with open(user_swipe_trace, 'r') as f:
                for line in f:
                    swipe_trace_user_i.append(float(line))
            self.ALL_VIDEO_NUM = len(swipe_trace_user_i)

            self.last_rebufs = [0.] * self.ALL_VIDEO_NUM
            self.last_chunk_bitrate = [-1] * self.ALL_VIDEO_NUM

            net_env = env.Environment(user_sample_id, self.all_cooked_time[trace_id], self.all_cooked_bw[trace_id], self.ALL_VIDEO_NUM, swipe_trace_user_i, self.dataset_path, self.chunklength)
            self.net_env = net_env
            retention_probs = self.calculate_retention_probabilities(self.net_env.players)
            self.pre_retention_probs = retention_probs

        self.past_bandwidth = list(np.zeros(PAST_BW_LEN))
        self.past_bandwidth_ests = list(np.zeros(PAST_BW_LEN))
        self.past_errors = list(np.zeros(PAST_BW_LEN))

        self.play_video_id = 0

        retention_probs = self.calculate_retention_probabilities(self.net_env.players)
        observation = self._get_obs(self.past_bandwidth, retention_probs, self.net_env.players)
        self._get_encoder_state(self.past_bandwidth, 0, self.net_env.players)
        # self.delay = 0

        return observation, self.last_info

    def get_mask(self):
        mask_size = 0
        if self.action_space.__class__.__name__ == "Discrete":
            mask_size = self.action_space.n
        elif self.action_space.__class__.__name__ == "Box":
            mask_size = self.action_space.shape[0]

        # # 第一个视频的缓冲时长不小于2s
        # if self.net_env.players[0].buffer_size == 0. and self.net_env.players[0].get_remain_video_num() != 0:
        #     mask = [0.] * mask_size
        #
        #     mask[0: 6] = [1.] * 6
        #     return mask

        # 所有视频的缓冲区最大为10s
        mask = [1.] * mask_size

        for i, player in enumerate(self.net_env.players):
            if player.get_buffer_size() >= self.max_buffer_size * 1000.:
                mask[i * 6:(i + 1) * 6] = [0.0] * 6

        # 已经下载完毕的视频不需要重复下载

        for i, player in enumerate(self.net_env.players):
            if player.get_remain_video_num() == 0:
                mask[i * 6:(i + 1) * 6] = [0.0] * 6

        return mask

    def _get_encoder_state(self, past_bw, delay_, Players):
        # 网络特征
        last_bw = past_bw[-1] * 8 # Mb
        l_idx = -5
        for bw in past_bw:
            if bw == 0.:
                l_idx += 1
            else:
                break
        # mean_pask_k_bw = np.mean(past_bw[l_idx:]) * 8
        # std_pask_k_bw = np.std(past_bw[l_idx:]) * 8

        delta_bw = None
        if past_bw[-2] == 0.:
            delta_bw = 0
        else:
            delta_bw = past_bw[-1] - past_bw[-2]

        bw_feature = [last_bw, delta_bw, delay_ / 1000.]

        current_player = Players[0]
        # # 辅助特征
        # current_buffer = current_player.get_buffer_size() / 1000. # s
        #
        # assist_feature = [current_buffer, delay_ / 1000., rebuf / 1000.]

        # 用户行为特征
        play_percentage = current_player.play_timeline / current_player.video_len
        user_time, user_retent_rate = current_player.get_user_model()

        play_time_s = current_player.play_timeline / 1000.

        play_chunk_num = int(current_player.get_play_chunk())
        continue_view_probability = float(user_retent_rate[play_chunk_num])

        user_feature = [play_percentage, continue_view_probability, play_time_s]
        # observation_encoder = np.array([[last_bw], [delay_ / 1000.], [play_percentage], [continue_view_probability]])
        # observation_encoder = np.array([bw_feature, assist_feature, user_feature])
        if self.encoder_input_feature == 'all':
            observation_encoder = np.array([bw_feature, user_feature])
        elif self.encoder_input_feature == 'bw':
            observation_encoder = np.array([bw_feature])
        elif self.encoder_input_feature == 'user_behavior':
            observation_encoder = np.array([user_feature])
        else:
            assert "unavailable encoder input feature setting"

        self.observation_encoder = observation_encoder
        # return observation_encoder
        # return observation_encoder

    def get_encoder_state(self):
        return self.observation_encoder
        # return self._get_encoder_state(self.past_bandwidth, self.delay, self.net_env.players)

    def get_sleep_judge(self):
        mask_size = 0
        if self.action_space.__class__.__name__ == "Discrete":
            mask_size = self.action_space.n
        elif self.action_space.__class__.__name__ == "Box":
            mask_size = self.action_space.shape[0]

        # 所有视频的缓冲区大小为10s
        mask = [1.] * mask_size

        for i, player in enumerate(self.net_env.players):
            if player.get_buffer_size() >= self.max_buffer_size * 1000.:
                mask[i * 6:(i + 1) * 6] = [0.0] * 6

        # 已经下载完毕的视频不需要重复下载

        for i, player in enumerate(self.net_env.players):
            if player.get_remain_video_num() == 0:
                mask[i * 6:(i + 1) * 6] = [0.0] * 6

        if sum(mask) == 0:
            sleep_judge = True
        else:
            sleep_judge = False

        return sleep_judge

    def get_traj_context(self):
        """Use user swipe and network trace as context

        Args:

        Returns:
            _type_: _description_
        """
        trace_num = len(os.listdir(self.network_traces_path))
        user_sample_id = np.random.randint(5)
        trace_id = np.random.randint(trace_num)

        return [user_sample_id, trace_id]
