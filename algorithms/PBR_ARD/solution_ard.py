import itertools

import sys

sys.path.append("..")

# VIDEO_BIT_RATE = [750, 1200, 1850]
from config_algorithm import VIDEO_BIT_RATE, alpha, beta, gamma

MPC_FUTURE_CHUNK_COUNT = 5
PAST_BW_LEN = 5
TAU = 50.0  # ms
PLAYER_NUM = 5
MILLISECONDS_IN_SECOND = 1000.0

import numpy as np


class Algorithm:
    def __init__(self):
        # 初始化参数
        self.sleep_time = None
        self.buffer_size = 0
        self.past_bandwidth = []
        self.past_bandwidth_ests = []
        self.past_errors = []
        # 记录经验池
        self.pre_play_video_id = None
        self.last_rebufs = [0.] * 100

        self.u = 1
        self.a = 0.5

        self.Bmax = 10000. # 10s

        self.theta = 0.3 # (0, inf)0.001

        self.Z = 0.56 # (0, 0.6)0.36


    def Initialize(self, chunklength):
        # 重置初始化
        self.buffer_size = 0
        self.past_bandwidth = list(np.zeros(PAST_BW_LEN))

        self.chunklength = chunklength

    def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players, Q_set, first_step=False):
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
            self.pre_play_video_id = play_video_id
            return play_video_id, 0, self.sleep_time

        if self.sleep_time == 0:
            self.past_bandwidth = np.roll(self.past_bandwidth, -1)
            # print(video_size, delay)
            self.past_bandwidth[-1] = (float(video_size) / 1000000.0) / (float(delay) / 1000.0)  # MB / s
            # print(self.past_bandwidth)
            # 1. 更新带宽估计
            self.update_bandwidth_estimate_()
            self.update_u(Q_set)

        if self.pre_play_video_id != play_video_id:

            self.sleep_time = 0
            bitrate = self.prb()
            # print('prb')
            self.pre_play_video_id = play_video_id
            # print(play_video_id, 'prb', bitrate)
            return play_video_id, bitrate, self.sleep_time

        # 2. 遍历视频，选择最优的比特率和视频
        player = Players[0]
        remaining_chunks = player.get_remain_video_num()  # 计算剩余的块数
        # print('========',remaining_chunks)
        # breakpoint()
        if player.get_buffer_size() >= self.Bmax or remaining_chunks == 0:
            self.sleep_time = TAU
            self.pre_play_video_id = play_video_id
            return play_video_id, 0, self.sleep_time

        # 计算最优比特率决策
        self.sleep_time = 0
        bitrate = self.prb()

        # print('prb')
        # print(play_video_id, 'prb', bitrate)

        self.pre_play_video_id = play_video_id

        return play_video_id, bitrate, self.sleep_time  # 睡眠时间设为50ms

    def update_bandwidth_estimate_(self):
        # record the newest error
        curr_error = 0  # default assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if (len(self.past_bandwidth_ests) > 0) and self.past_bandwidth[-1] != 0:
            curr_error = abs(self.past_bandwidth_ests[-1] - self.past_bandwidth[-1]) / float(self.past_bandwidth[-1])
        self.past_errors.append(curr_error)
        # first get harmonic mean of last 5 bandwidths
        past_bandwidth = self.past_bandwidth[-5:]
        while past_bandwidth[0] == 0.0:
            past_bandwidth = past_bandwidth[1:]
        bandwidth_sum = 0
        for past_val in past_bandwidth:
            bandwidth_sum += (1 / float(past_val))
        harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidth))

        self.past_bandwidth_ests.append(harmonic_bandwidth)


    def prb(self):
        past_bandwidth = self.past_bandwidth[-5:]
        while past_bandwidth[0] == 0.0:
            past_bandwidth = past_bandwidth[1:] # MB
        # print(past_bandwidth)
        # print(Q_min)
        # 估计网络带宽
        est_bw = self.u * self.a * np.average(past_bandwidth) * 1000 * 8 # kb 公式7
        # print(est_bw)

        # 将网络带宽映射到最近可用比特率
        if est_bw >= VIDEO_BIT_RATE[-1]:
            bitrate = len(VIDEO_BIT_RATE) - 1
            return bitrate

        bitrate = 0
        for i in range(len(VIDEO_BIT_RATE)):
            if est_bw < VIDEO_BIT_RATE[i]:
                bitrate = i - 1
                break

        bitrate = max(0, bitrate)

        return bitrate

    def update_u(self, Q_set):
        # 根据Q值更新超参数u
        Q_min = min(Q_set)
        self.u = self.u + self.theta / self.Z * (Q_min - self.Z)