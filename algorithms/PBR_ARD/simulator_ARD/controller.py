# input: download_video_id, bitrate, sleep_time
# output: info needed by schedule algorithm
# buffer: ms

import numpy as np
import math

from .video_player import Player
from .user_module import Retention
from .network_module import Network

NEW = 0
DEL = 1

# VIDEO_BIT_RATE = [750, 1200, 1850]  # Kbps
from config_algorithm import VIDEO_BIT_RATE
PLAYER_NUM = 5

class Environment:
    def __init__(self, user_sample_id, all_cooked_time, all_cooked_bw, video_num, seeds, dataset_dir, chunklength):
        self.video_size_dir = dataset_dir + '/short_video_size/'
        self.user_ret_dir = dataset_dir + '/user_ret/'
        self.chunklength = chunklength

        self.players = []
        self.seeds = seeds
        self.user_sample_id = user_sample_id
        global USER_FILE

        self.user_models = []  # Record the user action(Retention class) for the current video, update synchronized with players
        self.video_num = video_num
        self.video_cnt = 0
        self.play_video_id = 0
        self.network = Network(all_cooked_time, all_cooked_bw)
        self.timeline = 0.0
        # for ratio
        self.total_watched_len = 0.0
        self.total_downloaded_len = 0.0

        # print(self.watch_ratio)

        # self.download_permit = set()
        for p in range(PLAYER_NUM):
            # self.download_permit.add(p)
            self.players.append(Player(p, self.video_size_dir, self.user_ret_dir, self.chunklength))
            user_time, user_retent_rate = self.players[-1].get_user_model()
            self.user_models.append(Retention(user_time, user_retent_rate, seeds[self.video_cnt], self.chunklength))
            self.total_watched_len += self.user_models[-1].get_ret_duration()  # sum the total watch duration
            self.video_cnt += 1
        
        self.start_video_id = 0
        self.end_video_id = PLAYER_NUM - 1

        self.FPS = 25
        self.Kp = 15
        self.Kr = 1
        self.IF_PF = 5

        self.D = 1 / self.FPS

        self.Q_set = []
        self.past_bw_download_frame = []

    def player_op(self, operation):
        if operation == NEW:
            # print('--------------ADD--------------')
            if self.video_cnt >= self.video_num:  # If exceed video cnt, no add
                return
            self.players.append(Player(self.video_cnt, self.video_size_dir, self.user_ret_dir, self.chunklength))
            self.end_video_id += 1
            # print("adding: ", self.video_num)
            user_time, user_retent_rate = self.players[-1].get_user_model()
            self.user_models.append(Retention(user_time, user_retent_rate, self.seeds[self.video_cnt], self.chunklength))
            self.video_cnt += 1
            self.total_watched_len += self.user_models[-1].get_ret_duration()  # sum the total watch duration
        else:
            # print('--------------DEL--------------')
            self.players.remove(self.players[0])
            self.user_models.remove(self.user_models[0])
    
    def get_start_video_id(self):
        return self.start_video_id

    def get_wasted_time_ratio(self):
        return self.total_downloaded_len / self.total_watched_len

    def play_videos(self, action_time):  # play for action_time from the start of current players queue
        # print("\n\nPlaying Video ", self.start_video_id)
        wasted_bw = 0
        buffer = 0
        user_swipe = False

        # Continues to play if all the following conditions are satisfied:
        # 1) there's still action_time len
        # 2) the last video hasn't caused rebuf
        # 3) the video queue is not empty (will break inside the loop if its already empty)
        while buffer >= 0 and action_time > 0:
            # print("time_left:", action_time)
            # the timeline of the current video before this play step
            timeline_before_play = self.players[0].play_timeline
            # print("timeline_before_play: ", timeline_before_play)
            # the remain time length of the current video
            video_remain_time = self.user_models[0].get_ret_duration() - timeline_before_play
            # print("video_remain_time: ", video_remain_time)
            # the maximum play time of the current video
            max_play_time = min(action_time, video_remain_time)
            # print("max_play_time: ", video_remain_time)
            # timeline_after_play is the actual time when the play action ended( <=max_play_tm + before_play )
            timeline_after_play, buffer = self.players[0].video_play(max_play_time)
            # print("timeline_after_play: ", timeline_after_play)
            # the actual time length of this play action
            actual_play_time = timeline_after_play - timeline_before_play
            # print("actual_play_time: ", timeline_after_play)
            # consume the action_time
            # print("time_left:", action_time, timeline_before_play, actual_play_time, video_remain_time, max_play_time)
            action_time -= actual_play_time

            if action_time < 1e-3:
                action_time = 0

            # if the current video has ended
            # print(actual_play_time, video_remain_time)
            if video_remain_time - actual_play_time < 1e-3:
                # Output: the downloaded time length, the total time length, the watch duration
                # print("\nUser stopped watching Video ", self.start_video_id, "( ", self.players[0].get_video_len(), " ms ) :")
                # print("User watched for ", self.user_models[0].get_ret_duration(), " ms, you downloaded ", self.players[0].get_chunk_counter()*self.chunklength, " sec.")

                # use watch duration as an arg for the calculation of wasted_bandwidth of this current video
                wasted_bw += self.players[0].bandwidth_waste(self.user_models[0])

                # Forward the queue head to the next video
                self.player_op(DEL)
                self.start_video_id += 1
                self.player_op(NEW)
                self.play_video_id += 1
                user_swipe = True

            if self.play_video_id >= self.video_num:
                # if it has come to the end of the list
                # print("played out!")
                break

        if buffer < 0:  # action ends because a video stuck(needs rebuffer)
            buffer = (-1) * action_time  # rebuf time is the remain action time(cause the player will stuck for this time too)
        return buffer, wasted_bw, user_swipe
              
    def buffer_management(self, download_video_id, bitrate, sleep_time, first_step):
        if first_step:
            self.Q_set = []
        buffer = 0
        rebuf = 0
        end_of_video = False
        delay = 0
        video_size = 0
        wasted_bytes = 0
        if first_step:
            self.past_bw_download_frame = []

        play_interrupt = False

        user_switch_video = False

        if sleep_time > 0:
            sleep_time = int(sleep_time)
            delay = sleep_time
            rtt = 0.
            buffer, wasted, user_swipe = self.play_videos(sleep_time)
            self.network.network_sleep(sleep_time)
            # Return the end flag for the current playing video
            if self.play_video_id == self.video_num:  # if user leaves
                end_of_video = True
            else:
                end_of_video = (self.players[self.play_video_id-self.start_video_id].get_remain_video_num() == 0)
        else:
            wasted = 0
            video_size = self.players[download_video_id-self.start_video_id].get_video_size(bitrate)

            # print(bitrate, video_size)
            self.players[download_video_id - self.start_video_id].record_download_bitrate(bitrate)

            total_frames_num = int(self.FPS * self.chunklength / 1000.)
            frame_size = 1 / total_frames_num * video_size

            # print(total_frames_num, I_frame_size, P_frame_size)

            player = self.players[0]

            w = None
            if len(player.get_downloaded_bitrate()) == player.get_chunk_sum():
                w = 0

            if first_step:
                total_frames_num -= (self.Kp - 1)

            # 按帧进行下载
            i = 0
            while i < total_frames_num:
                if user_switch_video:
                    left_frame_num = total_frames_num - i
                    video_size_ = left_frame_num * frame_size

                    delay_, rtt_ = self.network.network_simu(video_size_)
                    delay_ = delay_ - rtt_
                    # print(f'time cost {delay_}')
                    delay += delay_
                    #
                    buffer_, wasted_, user_swipe = self.play_videos(delay_)
                    #
                    # if buffer_ < 0:
                    #     rebuf += abs(buffer_)
                    wasted += wasted_
                    #
                    # rebuf += delay_
                    # wasted += wasted_
                    break

                if i == 0:
                    if first_step:
                        for j in range(self.Kp):

                            video_size_ = frame_size

                            delay_, rtt_ = self.network.network_simu(video_size_)
                            if j == 0:
                                rtt = rtt_
                                delay_ = delay_
                                self.past_bw_download_frame.append(video_size_ / (delay_ - rtt_))  # B/ms
                            else:
                                delay_ = delay_ - rtt_
                                self.past_bw_download_frame.append(video_size_ / delay_)  # B/ms

                            delay += delay_
                            buffer_, wasted_, user_swipe = self.play_videos(delay_)
                            if user_swipe:
                                user_switch_video = True
                            if buffer_ < 0:
                                rebuf += abs(buffer_)
                            wasted += wasted_

                        self.Q_set.append(self.Kp * self.D)
                        download_ = self.Kp * self.D * 1000.  # ms
                        player.video_download(download_)

                    else:
                        video_size_ = frame_size
                        delay_, rtt_ = self.network.network_simu(video_size_)
                        rtt = rtt_
                        delay += delay_
                        buffer_, wasted_, user_swipe = self.play_videos(delay_)
                        if user_swipe:
                            user_switch_video = True

                        if buffer_ < 0:
                            play_interrupt = True
                            self.ard(frame_size, total_frames_num - i, True, w)
                            if self.Kr == 1:
                                play_interrupt = False
                                self.Q_set.append(self.Kr * self.D)
                        else:
                            self.Q_set.append(self.Q_set[-1] + self.D - delay_ / 1000.)

                        self.past_bw_download_frame.append(video_size_ / (delay_ - rtt_))  # B/ms
                        wasted += wasted_
                        if not play_interrupt:
                            download_ = self.D * 1000.  # ms
                            player.video_download(download_)

                else:
                    video_size_ = frame_size

                    delay_, rtt_ = self.network.network_simu(video_size_)
                    delay_ = delay_ - rtt_
                    delay += delay_
                    buffer_, wasted_, user_swipe = self.play_videos(delay_)
                    if user_swipe:
                        user_switch_video = True

                    if buffer_ < 0:
                        rebuf += abs(buffer_)
                        play_interrupt = True
                        self.ard(frame_size, total_frames_num - i, False, w)
                        if self.Kr == 1:
                            play_interrupt = False
                            self.Q_set.append(self.Kr * self.D)
                    else:
                        self.Q_set.append(self.Q_set[-1] + self.D - delay_ / 1000.)

                    self.past_bw_download_frame.append(video_size_ / delay_)  # B/ms
                    wasted += wasted_
                    if not play_interrupt:
                        download_ = self.D * 1000.
                        player.video_download(download_)
                # print(i, delay_, buffer_, user_switch_video)

                if play_interrupt:
                    # print('ard')
                    for j in range(self.Kr -1):
                        video_size_ = frame_size
                        delay_, rtt_ = self.network.network_simu(video_size_)
                        delay_ = delay_ - rtt_
                        delay += delay_
                        buffer_, wasted_, user_swipe = self.play_videos(delay_)
                        if user_swipe:
                            user_switch_video = True
                        if buffer_ < 0:
                            rebuf += abs(buffer_)

                        self.past_bw_download_frame.append(video_size_ / delay_)  # B/ms
                        wasted += wasted_


                    self.Q_set.append(self.Kr * self.D)
                    # print(self.Kr)
                    # print(self.Q_set[-1])

                    download_ = self.Kr * self.D * 1000. # ms
                    player.video_download(download_)
                    play_interrupt = False

                    i += self.Kr - 1


                i += 1

            player.video_download_complete()


            delay = math.floor(delay)

        # Sum up the bandwidth wastage
        wasted_bytes += wasted
        if buffer < 0:
            rebuf = abs(buffer)

        return delay, rebuf, video_size, end_of_video, self.play_video_id, wasted_bytes, rtt, self.Q_set

    def ard(self, A, y, occur_I_frame, w):

        if y == 1:
            self.Kr = 1
            return
        lambda_ = 0.8
        delta2 = 15

        t0 = 0
        k0 = 1

        past_bw = self.past_bw_download_frame[-delta2:]

        Kr_1 = []
        Kr_2 = []
        start_pos = -min(len(past_bw), delta2)

        est_bw = lambda_ / delta2 * np.sum(past_bw[start_pos:]) # kB/s B/ms
        # if est_bw == 0:
        #     print(past_bw)
        for kr in range(1, y + 1):
            if occur_I_frame:
                t1 = (kr * A) / est_bw # ms
            else:
                t1 = kr * A / est_bw
            pk = t1 + kr * self.D * 1000. # ms
            dk = t0 + t1
            if dk <= pk:
                Kr_1.append(kr)

            t2 = (y - k0 + 1) * self.D * 1000. + t1
            u = (t2 - t0) * est_bw
            v = y * A

            if w is None:
                w = y * A
            if (u - v) >= w:
                Kr_2.append(kr)
        # print(past_bw, Kr_1, Kr_2, y)
        if Kr_1 == [] or Kr_2 == []:
            # print('no candidate')
            self.Kr = max(1, y)
            return
        Kr_1_inf = min(Kr_1)
        Kr_2_inf = min(Kr_2)

        Kr = max(Kr_1_inf, Kr_2_inf)
        self.Kr = Kr
        # print(self.Kr)
