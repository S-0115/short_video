import itertools
import json
import math
from os import rename

import numpy as np
import sys

import torch

sys.path.append("..")

from config_algorithm import VIDEO_BIT_RATE, alpha, beta, gamma, watch_time_threhold, Bmax
# from Weibull import load_video_dimension_data, compute_weibull_params
from model.PPO import Policy

MPC_FUTURE_CHUNK_COUNT = 5
PAST_BW_LEN = 5
TAU = 200.0  # ms
PLAYER_NUM = 5
MILLISECONDS_IN_SECOND = 1000.0

USE_GPU = torch.cuda.is_available()
# USE_GPU = False
device = torch.device('cuda' if USE_GPU else 'cpu')


class Algorithm:
    def __init__(self):
        # 初始化参数
        self.sleep_time = None
        self.past_bandwidth = []
        self.past_rtt = []
        self.past_bandwidth_ests = []
        self.past_rtt_ests = []
        self.past_errors_bw = []
        self.past_errors_rtt = []
        # 记录经验池
        self.pre_download_video = None

        self.Bmax = Bmax # 最大缓冲区10s
        self.eh = 0.7
        self.el = 0.3

        self.b_u = 0.
        self.eta_u = 0.

        self.user_view_times = []

    def Initialize(self, model_path, chunklength):
        # 重置初始化
        self.past_bandwidth = list(np.zeros(PAST_BW_LEN))
        self.past_rtt = list(np.zeros(PAST_BW_LEN))
        self.past_errors_bw = list(np.zeros(PAST_BW_LEN))
        self.past_errors_rtt = list(np.zeros(PAST_BW_LEN))

        self.model = torch.load(model_path, map_location=device, weights_only=False)
        # print(self.model.high)

        self.chunklength = chunklength
        self.load_trainedOptimal()

    def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players, rtt, first_step=False):
        # print(play_video_id)
        # 1. delay: the time cost of your last operation
        # 2. rebuf: the length of rebufferment
        # 3. video_size: the size of the last downloaded chunk
        # 4. end_of_video: if the last video was ended
        # 5. play_video_id: the id of the current video
        # 6. Players: the video data of a RECOMMEND QUEUE of 5 (see specific definitions in readme)
        # 7. first_step: is this your first step?

        if first_step:  # 第一步
            self.sleep_time = 0
            self.pre_download_video = play_video_id
            download_range = torch.tensor(2)
            return play_video_id, 0, self.sleep_time, download_range

        if self.sleep_time == 0:
            self.past_bandwidth = np.roll(self.past_bandwidth, -1)
            self.past_bandwidth[-1] = (float(video_size) / 1000000.0) / (float(delay) / 1000.0)  # MB / s
            self.past_rtt = np.roll(self.past_rtt, -1)
            self.past_rtt[-1] = rtt

        start_pos = -min(5, len(self.past_bandwidth))
        while self.past_bandwidth[start_pos] == 0:
            start_pos += 1

        past_bandwidth_for_estimation = self.past_bandwidth[start_pos:]
        past_rtt_for_estimation = self.past_rtt[start_pos:]

        if self.sleep_time == 0:
            bandwidth_sum = 0
            rtt_sum = 0
            for past_val in past_bandwidth_for_estimation:
                bandwidth_sum += (1 / float(past_val))
            for past_val in past_rtt_for_estimation:
                rtt_sum += (1 / float(past_val))

            harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidth_for_estimation))
            harmonic_rtt = 1.0 / (rtt_sum / len(past_rtt_for_estimation))

            self.past_bandwidth_ests.append(harmonic_bandwidth)
            self.past_rtt_ests.append(harmonic_rtt)

            future_bandwidth = harmonic_bandwidth
            future_rtt = harmonic_rtt

        else:
            harmonic_bandwidth = self.past_bandwidth_ests[-1]
            harmonic_rtt = self.past_rtt_ests[-1]

            future_bandwidth = harmonic_bandwidth
            future_rtt = harmonic_rtt

        # 2. 选择比特率
        bitrate_players = []
        for i in range(len(Players)):
            player = Players[i]
            remaining_chunks = math.ceil(player.video_len / self.chunklength) - math.floor((player.buffer_size + player.play_timeline) / self.chunklength)  # 计算剩余的块数
            P = min(3, max(remaining_chunks, 1))  # 确保 P 不超过剩余块数
            # 计算最优的比特率选择
            last_quality = -1
            downloaded_bitrate = player.get_downloaded_bitrate()
            if len(downloaded_bitrate) > 0:
                last_quality = downloaded_bitrate[-1]
            # bit_rate = self.mpc(player.get_undownloaded_video_size(P), P, player.get_buffer_size(), last_quality, future_bandwidth)
            bit_rate = self.fastmpc(player.get_buffer_size(), last_quality, future_bandwidth * 1000. * 8)
            bitrate_players.append(bit_rate)
        # print(bitrate_players)

        # 3. 计算每个视频的需求
        demands = self.calculate_demands(Players)
        # print(demands)
        Bmax_video = 0
        buf_sizes = []
        for i in range(len(demands)):
            buf_sizes.append(Players[i].get_buffer_size())
            if Players[i].get_buffer_size() >= self.Bmax or Players[i].get_buffer_size() + Players[i].play_timeline >= Players[i].video_len - 1e-1:
            # if Players[i].get_buffer_size() >= self.Bmax:
                demands[i] = -math.inf
                Bmax_video += 1

        # print(buf_sizes, Bmax_video)
        if Bmax_video == len(Players):
            self.sleep_time = 500. # 所有视频均缓冲至最大缓冲区，睡眠
            return -1, -1, self.sleep_time, torch.tensor(0)
        # print(demands)
        selected_video = np.argmax(demands)
        # print(Bmax_video, selected_video)
        video_id = selected_video + play_video_id
        # print(selected_video)
        bitrate = bitrate_players[selected_video]
        # print(bitrate,video_id, Players[selected_video].get_buffer_size())
        self.sleep_time = 0.
        # print(retention_probs)

        # 3. 选择最佳的视频下载时长
        with torch.no_grad():
            inputs = self.get_input_data(bitrate_players, Players, selected_video, future_bandwidth, future_rtt).float()
            inputs = inputs.to(device)

            download_range = self.model.get_action(inputs)

        # print(selected_video, download_range)
        # download_range = torch.tensor(2)

        return video_id, bitrate, self.sleep_time, download_range

    def update_b_eta_user(self):
        x = np.sort(self.user_view_times)
        x = x[x > watch_time_threhold]
        n = len(x)
        if n <= 1:
            return
        # F_emp = (np.arange(1, n + 1)) / (n + 1)
        F_emp = (np.arange(1, n + 1) - 0.3) / (n + 0.4)

        Y = np.log(x - watch_time_threhold)
        X = np.log(-np.log(1 - F_emp))
        b = ((n * np.sum(Y * X)) - (np.sum(X) * np.sum(Y))) / (n * np.sum(np.square(Y)) - np.sum(Y) ** 2)
        # print(n * np.sum(np.square(Y)) - np.sum(Y) ** 2)
        eta = np.exp(Y.mean() - X.mean() / b)  # η̂

        self.b_u = b
        self.eta_u = eta
        # print(b, eta)

    def calculate_demands(self, Players):
        # 计算每个视频需求demand
        demands = []
        P_videos = []
        for i in range(len(Players)):
            player = Players[i]
            t0 = player.play_timeline / 1000.  # t0
            tau = (player.play_timeline + player.get_buffer_size()) / 1000.

            P_v = self.calculate_P_v(player, tau, t0)
            P_videos.append(P_v)
        # print(P_videos)
        for i in range(len(P_videos)):
            if i == 0:
                demands.append(P_videos[i])
            else:
                demands.append((1 - np.sum(demands)) * P_videos[i])
        # print(demands)
        return demands

    def calculate_P_v(self, player, tau, t0):
        b_v = player.beta
        eta_v = player.eta
        # print('tau t0',tau, t0)

        if tau <= watch_time_threhold:
            P_T_tau = 1.
        else:
            P_T_tau = np.exp(-((tau - watch_time_threhold) / eta_v) ** b_v)

        if t0 <= watch_time_threhold:
            P_T_t0 = 1.
        else:
            P_T_t0 = np.exp(-((t0 - watch_time_threhold) / eta_v) ** b_v)

        # print(P_T_tau, P_T_t0)

        # 如果用户已经看完了，则留存率为0，即不需要在考虑下载该视频
        return P_T_tau / P_T_t0

    def get_input_data(self, video_bitrates, Players, selected_video, est_bw, est_rtt):
        bi = [VIDEO_BIT_RATE[b] / 1000. for b in video_bitrates] # Mb
        bi = torch.tensor(bi)

        taui = [(player.get_buffer_size() + player.play_timeline) / 1000. for player in Players] # s
        taui = torch.tensor(taui)

        di = [player.video_len / 1000. for player in Players] # s
        di = torch.tensor(di)

        ti = [player.play_timeline / 1000. for player in Players] # s
        ti = torch.tensor(ti)

        hi = [self.eh * player.video_len / 1000. for player in Players] # s
        hi = torch.tensor(hi)

        li = [self.el * player.video_len / 1000. for player in Players] # s
        li = torch.tensor(li)

        bd = [est_bw * 8, est_rtt / 1000.] # Mb s
        bd = torch.tensor(bd)
        # bd.append(past_bandwidth[-1])
        # bd.append(rtt)

        idx = [selected_video]
        idx = torch.tensor(idx)
        # idx.append(selected_video)

        input = [bi, taui, di, ti, hi, li, bd, idx]
        input = torch.concat(input, dim=-1)

        return input

    def mpc(self, all_future_chunks_size, P, buffer_size, last_quality, future_bandwidth):

        CHUNK_COMBO_OPTIONS = []

        # make chunk combination options
        for combo in itertools.product(list(range(len(VIDEO_BIT_RATE))), repeat=P):
            # print(combo)
            CHUNK_COMBO_OPTIONS.append(combo)
        # print(future_bandwidth)
        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        max_reward = float('-inf')
        best_combo = ()
        start_buffer = buffer_size
        # print("start_buffer:", start_buffer)

        # start = time.time()
        for combo in CHUNK_COMBO_OPTIONS:
            # combo = full_combo[0:future_chunk_length]
            # calculate total rebuffer time for this combination (start with start_buffer and subtract
            # each download time and add 1 seconds in that order)
            curr_rebuffer_time = 0
            curr_buffer = start_buffer  # ms
            bitrate_sum = 0
            smoothness_diffs = 0
            if last_quality != -1:
                pre_quality = int(last_quality)
            else:
                pre_quality = combo[0]
            cost_sum = 0
            # print(combo)
            for position in range(0, len(combo)):
                chunk_quality = combo[position]
                download_time = MILLISECONDS_IN_SECOND * (
                        all_future_chunks_size[chunk_quality][position] / 1000000.) / (
                                    future_bandwidth)  # this is 1000. * MB/MB/s --> ms
                # print("download time:", download_time)
                if (curr_buffer < download_time):
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += self.chunklength
                bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[pre_quality])
                pre_quality = chunk_quality
            # compute reward for this combination (one reward per 5-chunk combo)

            reward = alpha * (bitrate_sum / 1000.) - gamma * (smoothness_diffs / 1000.) - beta * (
                    curr_rebuffer_time / 1000.)  # - theta * cost_sum * 8 / 1000000.

            # print(bitrate_sum, smoothness_diffs, curr_rebuffer_time, reward)

            if reward >= max_reward:
                best_combo = combo
                max_reward = reward
                send_data = best_combo[0]
                # print(bitrate_sum, smoothness_diffs, curr_rebuffer_time, reward)
        # print('max_reward', max_reward, send_data)

        bit_rate = send_data
        return bit_rate

    def load_trainedOptimal(self):
        f = open('./MPC_balanced.txt', 'r')  # training的数据

        VrateList = [200, 800, 2200, 5000, 12000, 25000]  # kbps

        BWList = range(1, 50000, 200)  # 1s的BW，Kbps

        bufferLevelList = range(0, 10000, 500)

        temp3 = {}
        for throughputIndex in range(len(BWList)):

            temp2 = {}
            for bitrateIndex in range(len(VrateList)):

                temp1 = {}
                for bufferCapacityIndex in range(len(bufferLevelList)):
                    temp1.setdefault(bufferLevelList[bufferCapacityIndex], 0)

                temp2.setdefault(VrateList[bitrateIndex], temp1)

            temp3.setdefault(BWList[throughputIndex], temp2)

        TrainedOptimal = temp3

        for line in f:
            tempList = line.split("\t")

            TrainedOptimal[int(tempList[0])][int(tempList[1])][round(float(tempList[2]), 1)] = int(tempList[3])

        f.close()

        self.TrainedOptimal = TrainedOptimal

    def fastmpc(self, buffer_size, last_quality, future_bandwidth):
        '''
        :param buffer_size: ms
        :param last_quality: idx
        :param future_bandwidth:  kb/s
        :return:
        '''
        # print(buffer_size,  last_quality, future_bandwidth)
        VrateList = [200, 800, 2200, 5000, 12000, 25000]  # kbps

        BWList = range(1, 50000, 200)  # 1s的BW，Kbps

        bufferLevelList = range(0, 10000, 500)

        if last_quality == -1:
            return 0

        preThroughput = int(future_bandwidth / 200) * 200 + 1  # transform to the times of 10
        # print(future_bandwidth)
        # print(preThroughput, VrateList[last_quality], buffer_size/1000.)
        buffer_level = int(buffer_size / 500) * 500

        VrateTmp = self.TrainedOptimal[min(preThroughput, max(BWList))][VrateList[last_quality]][min(buffer_level, max(bufferLevelList))]
        # print(VrateTmp)
        # 根据实际的参数，取出training的参数
        for bitrateIndex in range(len(VrateList)):
            if VrateTmp == VrateList[bitrateIndex]:
                return bitrateIndex