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

        # self.sleep_time = 0
        # P = min(3, remaining_chunks)  # 确保 P 不超过剩余块数
        # # 计算最优的比特率选择
        # last_quality = -1
        # downloaded_bitrate = player.get_downloaded_bitrate()
        # if len(downloaded_bitrate) > 0:
        #     last_quality = downloaded_bitrate[-1]
        # # print(player.get_undownloaded_video_size(P))
        # bitrate = self.mpc(player.get_undownloaded_video_size(P), P, player.get_buffer_size(), last_quality)


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
        est_bw = self.u * self.a * np.average(past_bandwidth) * 1000 * 8 # kb 公式7
        # print(est_bw)

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
        Q_min = min(Q_set)
        self.u = self.u + self.theta / self.Z * (Q_min - self.Z)

        # print(Q_min, self.u)
        # if Q_min < 0:
        #     breakpoint()
        # if self.u < 1:
        #     print('ssssssssssssss' * 10)


    def mpc(self, all_future_chunks_size, P, buffer_size, last_quality):

        CHUNK_COMBO_OPTIONS = []

        # make chunk combination options
        for combo in itertools.product(list(range(len(VIDEO_BIT_RATE))), repeat=P):
            # print(combo)
            CHUNK_COMBO_OPTIONS.append(combo)
        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        max_error = 0
        error_pos = -5
        if (len(self.past_errors) < 5):
            error_pos = -len(self.past_errors)
        max_error = float(max(self.past_errors[error_pos:]))
        # print(self.past_errors[error_pos:])
        future_bandwidth = self.past_bandwidth_ests[-1] / (1. + max_error)  # robustMPC here
        # print(self.past_bandwidth[-1])
        # print(self.past_bandwidth_ests[-1])
        # print(max_error)
        # print("future_bd:", future_bandwidth)

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
                        curr_rebuffer_time / 1000.) # - theta * cost_sum * 8 / 1000000.

            # print(bitrate_sum, smoothness_diffs, curr_rebuffer_time, reward)

            if reward >= max_reward:
                best_combo = combo
                max_reward = reward
                send_data = best_combo[0]
                # print(bitrate_sum, smoothness_diffs, curr_rebuffer_time, reward)
        # print('max_reward', max_reward, send_data)
        # print(best_combo, max_reward, send_data)

        bit_rate = send_data
        return bit_rate
