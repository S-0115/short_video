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
            # print("max_play_time: ", max_play_time)
            # timeline_after_play is the actual time when the play action ended( <=max_play_tm + before_play )
            timeline_after_play, buffer = self.players[0].video_play(max_play_time)
            # print("timeline_after_play: ", timeline_after_play)
            # the actual time length of this play action
            actual_play_time = timeline_after_play - timeline_before_play
            # print("actual_play_time: ", timeline_after_play)
            # consume the action_time
            action_time -= actual_play_time
            if action_time < 1e-2:
                action_time = 0

            # if the current video has ended
            # print(actual_play_time, video_remain_time)
            if  video_remain_time - actual_play_time < 1e-1:

                # Forward the queue head to the next video
                wasted_bw += self.players[0].bandwidth_waste(self.user_models[0])
                self.player_op(DEL)
                self.start_video_id += 1
                self.player_op(NEW)
                self.play_video_id += 1

            if self.play_video_id >= self.video_num:
                # if it has come to the end of the list
                # print("played out!")
                break
        if buffer < 0:  # action ends because a video stuck(needs rebuffer)
            buffer = (-1) * action_time  # rebuf time is the remain action time(cause the player will stuck for this time too)
        return buffer, wasted_bw
              
    def buffer_management(self, download_video_id, bitrate, download_range, sleep_time):
        buffer = 0
        rebuf = 0
        end_of_video = False
        delay = 0
        video_size = 0
        wasted_bytes = 0

        wasted_after_download = 0

        if sleep_time > 0:
            sleep_time = int(sleep_time)
            delay = sleep_time
            rtt = 0.
            buffer, wasted = self.play_videos(sleep_time)
            self.network.network_sleep(sleep_time)
            # Return the end flag for the current playing video
            if self.play_video_id == self.video_num:  # if user leaves
                end_of_video = True
            else:
                end_of_video = (self.players[self.play_video_id-self.start_video_id].get_remain_video_num() == 0)
        else:
            video_size = self.players[0].get_video_size(bitrate, download_range)
            # print(bitrate, download_range, video_size)

            self.players[download_video_id - self.start_video_id].record_download_bitrate(bitrate, download_range)
            delay, rtt = self.network.network_simu(video_size)
            delay = math.floor(delay) # ms
            rtt = math.floor(rtt)
            # print(delay, rtt)
            # print("the actual download delay is:", delay)
            # print("\n\n")
            # play_timeline, buffer = self.players[self.play_video_id - self.start_video_id].video_play(delay)
            wasted_after_download = self.players[download_video_id - self.play_video_id].bandwidth_waste(
                self.user_models[download_video_id - self.play_video_id])

            buffer, wasted = self.play_videos(delay)

            self.total_downloaded_len += download_range * 1000.  # sum up the total downloaded time
            if download_video_id < self.start_video_id:
                # If the video has already been ended, we only accumulate the wastage
                # print("Extra chunk downloaded for Video ", download_video_id,
                #       " which the user already finished watching.\n")

                # wasted += video_size  # Since its already fluently played, the download must be redundant
                end_of_video = True
            else:
                if self.play_video_id == self.video_num:  # if user leaves
                    end_of_video = True
                else:
                    player = self.players[download_video_id - self.start_video_id]
                    # print(player.download_chunk_length)
                    # print(player.video_len, player.play_timeline, player.buffer_size, download_range)
                    download_ = min(player.video_len - player.play_timeline - player.buffer_size, download_range * 1000.)
                    # print(player.video_len, player.play_timeline, player.buffer_size, download_range)
                    end_of_video = self.players[download_video_id-self.start_video_id].video_download(download_)

        # Sum up the bandwidth wastage
        wasted_bytes += wasted
        if buffer < 0:
            rebuf = abs(buffer)

        return delay, rebuf, video_size, end_of_video, self.play_video_id, wasted_bytes, rtt, wasted_after_download

    def buffer_management_second_download(self, download_video_id, bitrate_1, download_range_1, bitrate_2, download_range_2,  sleep_time):
        buffer = 0
        rebuf = 0
        end_of_video = False
        delay = 0
        video_size = 0
        wasted_bytes = 0

        wasted_after_download = 0

        if sleep_time > 0:
            sleep_time = int(sleep_time)
            delay = sleep_time
            rtt = 0.
            buffer, wasted = self.play_videos(sleep_time)
            self.network.network_sleep(sleep_time)
            # Return the end flag for the current playing video
            if self.play_video_id == self.video_num:  # if user leaves
                end_of_video = True
            else:
                end_of_video = (self.players[self.play_video_id-self.start_video_id].get_remain_video_num() == 0)
        else:
            video_size = self.players[0].get_video_size(bitrate_1, download_range_1) + self.players[0].get_video_size(bitrate_2, download_range_2)
            # print(bitrate, download_range, video_size)

            self.players[download_video_id - self.start_video_id].record_download_bitrate(bitrate_1, download_range_1)
            self.players[download_video_id - self.start_video_id].record_download_bitrate(bitrate_2, download_range_2)

            delay, rtt = self.network.network_simu(video_size)
            delay = math.floor(delay) # ms
            rtt = math.floor(rtt)
            # print(delay, rtt)
            # print("the actual download delay is:", delay)
            # print("\n\n")
            # play_timeline, buffer = self.players[self.play_video_id - self.start_video_id].video_play(delay)
            wasted_after_download = self.players[download_video_id - self.play_video_id].bandwidth_waste(
                self.user_models[download_video_id - self.play_video_id])

            buffer, wasted = self.play_videos(delay)

            self.total_downloaded_len += (download_range_1 + download_range_2) * 1000.  # sum up the total downloaded time
            if download_video_id < self.start_video_id:
                # If the video has already been ended, we only accumulate the wastage
                # print("Extra chunk downloaded for Video ", download_video_id,
                #       " which the user already finished watching.\n")

                # wasted += video_size  # Since its already fluently played, the download must be redundant
                end_of_video = True
            else:
                if self.play_video_id == self.video_num:  # if user leaves
                    end_of_video = True
                else:
                    player = self.players[download_video_id - self.start_video_id]
                    # print(player.download_chunk_length)
                    # print(player.video_len, player.play_timeline, player.buffer_size, download_range)
                    download_ = min(player.video_len - player.play_timeline - player.buffer_size, (download_range_1 + download_range_2) * 1000.)
                    # print(player.video_len, player.play_timeline, player.buffer_size, download_range)
                    end_of_video = self.players[download_video_id-self.start_video_id].video_download(download_)

        # Sum up the bandwidth wastage
        wasted_bytes += wasted
        if buffer < 0:
            rebuf = abs(buffer)

        return delay, rebuf, video_size, end_of_video, self.play_video_id, wasted_bytes, rtt, wasted_after_download
