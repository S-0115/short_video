import itertools
import sys
import copy
import os

sys.path.append("..")

MPC_FUTURE_CHUNK_COUNT = 5
PAST_BW_LEN = 5
TAU = 500.0  # ms
PLAYER_NUM = 5
MILLISECONDS_IN_SECOND = 1000.0

from simulator.video_player import Player
import numpy as np

bitraterewards = [1143, 2064, 4449, 6086, 9193, 0]
#
penalty_weight = 700000
# buffer_threshold = bitraterewards[0] / penalty_weight
from config_algorithm import alpha, beta, gamma, theta, VIDEO_BIT_RATE

buffer_threshold = 1 / (beta * 1000)
# buffer_threshold = 1 / beta
# buffer_threshold = bitraterewards[0] / penalty_weight * 3 / 25

class Algorithm:
    def __init__(self):
        # 初始化参数
        self.sleep_time = None
        self.buffer_size = 0
        self.past_bandwidth = []
        self.past_bandwidth_ests = []
        # 权重定义
        self.probability_map = {}


    def Initialize(self, foldername, chunklength):
        # 重置初始化
        self.buffer_size = 0
        self.past_bandwidth = list(np.zeros(PAST_BW_LEN))
        self.past_bandwidth_ests = list(np.zeros(PAST_BW_LEN))
        self.probability_map = self.get_probability_map(foldername)
        self.chunklength = chunklength / 1000.

        # self.load_trainedOptimal()

    def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players, first_step=False):
        # print('play_video_id:', play_video_id)
        # 1. delay: the time cost of your last operation
        # 2. rebuf: the length of rebufferment
        # 3. video_size: the size of the last downloaded chunk
        # 4. end_of_video: if the last video was ended
        # 5. play_video_id: the id of the current video
        # 6. Players: the video data of a RECOMMEND QUEUE of 5 (see specific definitions in readme)
        # 7. first_step: is this your first step?

        if self.sleep_time == 0:
            self.past_bandwidth = np.roll(self.past_bandwidth, -1)
            self.past_bandwidth[-1] = (float(video_size) / 1000000.0) / (float(delay) / 1000.0)  # MB / s
            # 1. 更新带宽估计
            if not first_step:
                self.update_bandwidth_estimate_()

        # 2. 遍历视频，选择最优的比特率和视频
        best_video_id, best_bitrate, best_sleep_time = None, None, None

        probability_weights = self.get_probability_weights(Players)
        bitrate = self.get_bitrate(Players)
        # print(bitrate)
        if first_step:  # 第一步
            throughput = 100.
        else:
            throughput = self.past_bandwidth_ests[-1] * 1000. * 8 # kb / s
        buffer_plan = self.dash_sv(Players, probability_weights, bitrate, throughput)

        # print(buffer_plan)
        for i in range(len(buffer_plan)):
            if buffer_plan[i] != -2:
                best_video_id = play_video_id + i
                best_bitrate = buffer_plan[i]
                best_sleep_time = self.sleep_time = 0.
                break

        # print(best_video_id, best_bitrate, best_sleep_time, self.sleep_time)

        if best_video_id is not None:
            # player = Players[best_video_id - play_video_id]
            # # print(player.get_buffer_size())
            # if player.get_buffer_size() >= 8000.:
            #     # breakpoint()
            #     self.sleep_time = self.chunklength * 1000. - Players[0].play_timeline % (self.chunklength * 1000.)
            #     return -1, -1, self.sleep_time
            # else:
            return best_video_id, best_bitrate, best_sleep_time
        else:
            self.sleep_time = self.chunklength * 1000.# - Players[0].play_timeline % (self.chunklength * 1000.)
            # self.sleep_time = 100.
            return play_video_id, 0, self.sleep_time  # 睡眠时间设为50ms

    def update_bandwidth_estimate_(self):
        # first get harmonic mean of last 5 bandwidths
        past_bandwidth = self.past_bandwidth[-5:]
        while past_bandwidth[0] == 0.0:
            past_bandwidth = past_bandwidth[1:]
        bandwidth_sum = 0
        for past_val in past_bandwidth:
            bandwidth_sum += (1 / float(past_val))
        harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidth))

        self.past_bandwidth_ests.append(harmonic_bandwidth)

    def get_probability_map(self, foldername):
        probability_map_f = {}

        filenames = os.listdir(foldername)

        for filename in filenames:
            # key = filename.strip(".txt")
            key = filename.strip()
            data = np.loadtxt(foldername + filename)

            probability_map_f[key] = data

        return probability_map_f

    def get_probability_weights(self, Players):
        '''
        该方法旨在获取视频列表中所有视频的用户聚合的滑动概率分布
        '''

        probability_weights = []
        for i in range(len(Players)):
            vid = Players[i].video_name
            probability_weights.append(self.probability_map[vid])

        return probability_weights

    def get_bitrate(self, Players):
        '''
        本方法旨在获取推荐列表中各个视频共有多少个比特率，各个比特率下块的大小又是多少
        '''
        bitrate_list = []

        for eidx in range(len(Players)):
            player = Players[eidx]
            bitrate_list.append([])
            for i in range(len(player.video_size[0])): # chunk位置
                bitrate_list[-1].append([])
                for j in range(len(player.video_size)): # 比特率
                    bitrate_list[eidx][i].append(player.video_size[j][i] * 8 / 1000.0)  # in kb

        return bitrate_list

    def parse_buffer_status(self, Players):
        '''
        这个方法的目的在于提取推荐列表里的视频的要buffer的视频块的序号,视频的时长,上一次的比特率以及对于当前正在观看的视频的观看进度
        '''
        buffer_length = []
        video_duration = []

        last_quality = []

        for i in range(len(Players)):
            player = Players[i]

            idx = player.video_chunk_counter
            buffer_length.append(idx)

            video_duration.append(player.video_len / 1000.)
            if len(player.download_chunk_bitrate) == 0:
                last_quality.append(-1)
            else:
                last_quality.append(player.download_chunk_bitrate[-1])

        # total buffered video in seconds - not played video in the buffer
        current_cursor = min(buffer_length[0] * self.chunklength, video_duration[0]) - Players[0].buffer_size / 1000.

        return buffer_length, video_duration, last_quality, current_cursor

    def dash_sv(self, players, probability_weights, bitrate_profile, estimate_throughput):
        '''
        这个方法负责根据推荐列表中的视频的相关信息，滑动概率分布以及预估的网络带宽找出最优的决策序列
        实质上是dashlet论文里核心代码的实现
        '''

        ret = [-2, -2, -2, -2, -2]
        # print(bitrate_profile)
        # 推荐列表里的视频的要buffer的视频块的序号,视频的时长,上一次的比特率以及对于当前正在观看的视频的观看进度
        buffer_length, video_duration, last_quality, current_cursor = self.parse_buffer_status(players)
        # print(buffer_length, video_duration, last_quality, current_cursor)
        look_forward_time = 25
        danger_zone_time = 5
        # print(video_duration)
        total_lengths = [int((vduration - 0.00000001) / self.chunklength) + 1 for vduration in
                         video_duration]  # 通过video_duration计算推荐列表里的每个视频的chunk总数
        # print('total chunk num is ', total_lengths)

        cursor_idx = int(current_cursor / self.chunklength) + 1

        # 获得当前正在播放的视频的播放进度的int表示
        current_playback_ts = int(current_cursor)
        # print(f'current play timeline is ', current_playback_ts)

        # 获得推荐列表里视频的滑动概率分布，对于当前正在播放的视频，只统计还没有播放的视频的滑动概率分布
        update_weights = copy.deepcopy(probability_weights)

        update_weights[0] = update_weights[0][current_playback_ts:] / np.sum(update_weights[0][current_playback_ts:])

        nvideos = 5

        # 接下来的部分主要是计算从当前视频的观看的位置，一直看到第i个视频的第j个视频块的概率，即在第i个视频第j个视频块滑动的概率，结果会被记录到total_distribution中
        head_distribution = [np.array([1.0]) for i in range(nvideos)]

        # 这里计算由上一个视频的任意时刻滑动到下一个视频任意时刻的滑动概率
        for i in range(1, nvideos):
            # print(i, head_distribution[i - 1].shape)
            # print(update_weights[i -1].shape)
            head_distribution[i] = np.convolve(head_distribution[i - 1], update_weights[i - 1]) # 离散线性卷积，计算由上一个视频的任意时刻滑动到下一个视频任意时刻的滑动概率
        # print(head_distribution)

        total_distribution = {}

        danger_zone_dict = {}

        candidate_high = copy.deepcopy(buffer_length)

        for i in range(nvideos):
            for j in range(buffer_length[i], total_lengths[i]):
                # 这里会计算用户直接从i, j往后面接着看的概率,要更新滑动概率分布,被i,j之前的块,其发生滑动的概率会变成0,因为在这样的情况下,他们已经看过或下载完了
                if i == 0:
                    shift_distance = j * int(self.chunklength) - int(current_cursor)
                else:
                    shift_distance = j * int(self.chunklength)
                shift_array = np.array([0.0 for ai in range(shift_distance)])

                # 这里相当与计算从第i,j个块后，第i个视频的用户滑动概率分布，其中head distribution是由当前播放视频滑动到第i，j个视频块的滑动概率分布，update weights是第i，j个块后的滑动概率分布
                total_distribution[(i, j)] = np.concatenate((shift_array, head_distribution[i])) * np.sum(update_weights[i][shift_distance:])
                # 然后，还会计算那些看到这些视频块的预计的penalty，然后会把那些penalty超过阈值的序号记录下来
                # penalty有两种，this_penalty和danger_penalty，实际就是论文里所说的预期rebuffer,他们之间的区别在于horizon不同，this_penalty是对于长期的min(未来25s的，视频剩余时间)，danger_penalty是对于短期的min(未来5s的，视频剩余时间)。
                # 对于较远的未来的预估的this_penalty的块，会将他们记为该视频高风险的块，如果该块的近期的预估的danger_penalty也较高，同时会将其记录到danger_zone_dict中
                this_penalty = 0
                danger_penalty = 0
                # 计算penalty如论文公式11，（look_forward_time - tidx） =》 Trebuf， total_distribution[(i, j)][tidx] =》 f
                for tidx in range(min(look_forward_time, len(total_distribution[(i, j)]))):
                    this_penalty += (look_forward_time - tidx) * total_distribution[(i, j)][tidx]

                for tidx in range(min(danger_zone_time, len(total_distribution[(i, j)]))):
                    danger_penalty += (danger_zone_time - tidx) * total_distribution[(i, j)][tidx]

                if this_penalty > buffer_threshold:
                    # print(this_penalty)
                    candidate_high[i] = j + 1

                if danger_penalty > buffer_threshold:
                    danger_zone_dict[(i, j)] = 1

                # print((i, j, this_penalty))
        # 计算远期未来再缓冲风险较高的块的总数
        candidate_num = 0
        for i in range(nvideos):
            candidate_num += (candidate_high[i] - buffer_length[i])

        # 如果没有高风险块的话，就可以直接返回ret了，表示这个视频不需要继续下载以降低再缓冲的风险
        if candidate_num == 0:
            return ret

        # 如果有高风险的块的话，那么接下来要做的就是决定高风险块的比特率了
        # 首先，就是要估计可以接受的最大比特率target_bitrate
        target_bitrate = look_forward_time * estimate_throughput / candidate_num

        # 然后，选择预期再缓冲风险最大的视频块
        max_penalty = 0
        max_buffer_i = 0
        # print(total_distribution.keys())
        for i in range(nvideos):
            j = buffer_length[i]

            # 下面这段代码是说，虽然第i个视频的第j个块是高风险的，但是它属于是正在播放或已经播放的视频块，我们不需要考虑他的buffer问题，因为它不在total_distribution中
            if (i, j) not in total_distribution.keys():
                continue

            # 接下来,就计算这个视频会导致的this_penalty,和之前不同的是,观测的未来长度不一样了,在这里,高风险块越多,就预测更短的未来.
            look_forward_local = max(int(look_forward_time * 2 / candidate_num) + 1, 8)
            # print(int(look_forward_time * 2 / candidate_num) + 1, 10)
            this_penalty = 0
            for tidx in range(min(look_forward_local, len(total_distribution[(i, j)]))):
                this_penalty += (look_forward_local - tidx) * total_distribution[(i, j)][tidx]

            if max_penalty < this_penalty:
                max_penalty = this_penalty
                max_buffer_i = i

        # 这样就得到了最需要进行处理的那个视频在推荐列表中的序号及其要buffer的视频块序号
        max_buffer_j = buffer_length[max_buffer_i]

        # 如果这个块在短的未来来看,也是很危险的,就降低其可接受的比特率上限target_bitrate
        if (max_buffer_i, max_buffer_j) in danger_zone_dict.keys():
            target_bitrate /= 2

        # 下面就开始对该视频块做比特率决策
        # 如果对这个视频来说最大的penalty实际上也很小的话,就不做预取了
        # print(max_penalty)
        if max_penalty < 0.00001:
            return ret

        # fastMPC for dashlet bitrate adaption
        player = players[max_buffer_i]

        P = min(player.get_remain_video_num(), 5)
        bitrate_choice = self.mpc(player.get_undownloaded_video_size(P), P, estimate_throughput / 1000 / 8, player.get_buffer_size(), last_quality[max_buffer_i])

        # bitrate_choice = self.fastmpc(estimate_throughput, player.get_buffer_size(), last_quality[max_buffer_i])

        ret[max_buffer_i] = bitrate_choice
        # print(ret)
        return ret

        # # 这里就是一个比较了,我们只需要在该视频提供的可供选择的比特率里,选一个与target_bitrate最为接近的且小于它的比特率就可以了
        # bitrate_choice = 0
        # # print(bitrate_profile)
        # # print(max_buffer_i, max_buffer_j)
        # # print(len(bitrate_profile[max_buffer_i]))
        # for i in range(1, len(bitrate_profile[max_buffer_i][max_buffer_j])):
        #     # chunk_duration = min(self.chunklength, video_duration[max_buffer_i] - max_buffer_j * self.chunklength)
        #
        #     # print("=====================")
        #     # print(bitrate_profile[max_buffer_i][max_buffer_j][i] / self.chunklength)
        #     # print(target_bitrate)
        #
        #     if bitrate_profile[max_buffer_i][max_buffer_j][i] < target_bitrate:
        #         # print(bitrate_profile[max_buffer_i][max_buffer_j][i])
        #         bitrate_choice = i
        #
        # # 这里还考虑了平滑的事,为了防止前后两次决策的比特率差距过大
        # # # Take care of smoothness, reduce the bitrate to align with the formal chunk
        # # if last_quality[max_buffer_i] != -1:
        # #     if bitrate_choice != (len(bitrate_profile[max_buffer_i][max_buffer_j]) - 1):
        # #         bitrate_choice = last_quality[max_buffer_i]
        #
        # # if bitrate_choice != (len(bitrate_profile[max_buffer_i][max_buffer_j]) - 1):
        # #     print("change")
        #
        # ret[max_buffer_i] = bitrate_choice
        #
        # return ret

    def mpc(self, all_future_chunks_size, P, future_bandwidth, buffer_size, last_quality):
        CHUNK_COMBO_OPTIONS = []

        # make chunk combination options
        for combo in itertools.product(list(range(len(VIDEO_BIT_RATE))), repeat=P):
            # print(combo)
            CHUNK_COMBO_OPTIONS.append(combo)
        # future bandwidth prediction

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
                pre_quality = 0
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

    def fastmpc(self, future_bandwidth, buffer_size, last_quality):
        '''
        :param buffer_size: ms
        :param last_quality: idx
        :param future_bandwidth:  kb/s
        :return:
        '''
        # print(future_bandwidth, buffer_size, last_quality)
        VrateList = [200, 800, 2200, 5000, 12000, 25000]  # kbps

        BWList = range(1, 50000, 200)  # 1s的BW，Kbps

        bufferLevelList = range(0, 10000, 500)

        if last_quality == -1:
            return 0

        preThroughput = int(future_bandwidth / 200) * 200 + 1  # transform to the times of 10
        # print(future_bandwidth)
        buffer_level = int(buffer_size / 500) * 500

        # print(preThroughput, VrateList[last_quality], buffer_level)
        VrateTmp = self.TrainedOptimal[min(preThroughput, max(BWList))][VrateList[last_quality]][min(buffer_level, max(bufferLevelList))]
        # print(VrateTmp)
        # 根据实际的参数，取出training的参数
        for bitrateIndex in range(len(VrateList)):
            if VrateTmp == VrateList[bitrateIndex]:
                return bitrateIndex

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