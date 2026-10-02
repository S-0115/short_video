# multi-video play
import numpy as np
import math
import os

from config_algorithm import VIDEO_BIT_RATE
BITRATE_LEVELS = len(VIDEO_BIT_RATE)
MILLISECONDS_IN_SECOND = 1000.0
VIDEO_SIZE_SCALE = 1.0  # chunk size

class Player:
    # initialize each new video and player buffer
    def __init__(self, video_num, video_size_file, user_ret, chunklength):
        self.video_size_file = video_size_file
        self.user_ret = user_ret
        self.chunklength = chunklength
        # initialize each new video
        self.video_size = {}  # in bytes
        # videos = []
        # for root, dirs, files in os.walk(self.video_size_file):
        #     videos = dirs
        #     videos = self.sort_video(videos)
        #     break

        # num and len of the video, all chunks are counted instead of -1 chunk
        idx = 0
        with open(self.user_ret + '/../video_names.csv') as f:
            for line in f:
                if idx == video_num:
                    video_info = line.split(',')
                    self.video_len = float(video_info[2])
                    self.video_name = video_info[1]
                    weibull = video_info[3].split('&')
                    self.beta = float(weibull[0])
                    self.eta = float(weibull[1])
                idx += 1
        # print(video_num)
        # print(self.video_name)
        # print(self.video_len)
        # self.video_len = self.chunk_num * self.chunklength  # ms
        # print(self.video_len)
        self.chunk_num = math.ceil(self.video_len / self.chunklength)
        # print(self.chunk_num)
        # print(self.chunklength)

        for bitrate in range(len(VIDEO_BIT_RATE)):
            self.video_size[bitrate] = []
            # file_name = self.video_size_file + self.video_name + '/video_size_' + str(bitrate)
            # with open(file_name) as f:
            #     for line in f:
            #         self.video_size[bitrate].append(int(line.split()[0])/VIDEO_SIZE_SCALE)
            for i in range(self.chunk_num):
                self.video_size[bitrate].append(VIDEO_BIT_RATE[bitrate] * self.chunklength / 1000. * 1000. / 8) # B
            if self.video_len % self.chunklength != 0:
                self.video_size[bitrate][-1] = VIDEO_BIT_RATE[bitrate] * (self.video_len % self.chunklength) / 1000. * 1000. / 8  # B
            # print(self.video_size[bitrate][-1])
        # print(self.video_size)
        
        # download chunk counter
        self.video_chunk_counter = 0
        self.video_chunk_remain = self.chunk_num - self.video_chunk_counter
        
        self.download_chunk_bitrate = []
        self.download_bitrate_ct = []
        
        # play chunk counter
        self.video_play_counter = 0
        
        # play timeline of this video
        self.play_timeline = 0.0
        
        # initialize the buffer
        self.buffer_size = 0  # ms
        
        # initialize preload size
        self.preload_size = 0 # B

        # initialize corresponding user watch time
        self.user_time = []  # user_model which players have access to!!
        self.user_retent_rate = []

        with open(self.user_ret + self.video_name) as file:
            for line in file:
                self.user_time.append(float(line.split()[0]) * MILLISECONDS_IN_SECOND)
                self.user_retent_rate.append(line.split()[1])

    def get_user_model(self):
        return self.user_time, self.user_retent_rate
    
    def get_video_len(self):
        return self.video_len

    def get_video_size(self, quality):
        try:
            video_chunk_size = self.video_size[quality][self.video_chunk_counter]
        except IndexError:
            raise Exception("You're downloading chunk ["+str(self.video_chunk_counter)+"] is out of range. "\
                            + "\n   % Hint: The valid chunk id is from 0 to " + str(self.chunk_num-1) + " %")
        # self.preload_size += video_chunk_size
        return video_chunk_size

    def get_video_quality(self, chunk_id):
        if chunk_id >= len(self.download_chunk_bitrate) or chunk_id < 0:
            return -1  # means no video downloaded
        return self.download_chunk_bitrate[chunk_id]

    # get size of all preloaded chunks
    def get_preload_size(self):
        return self.preload_size
    
    def get_undownloaded_video_size(self, P):  # the undownloaded video size
        chunk_playing = self.get_chunk_counter()
        future_videosize = []
        for i in range(BITRATE_LEVELS):
            size_in_level = []
            for k in range(P):
                size_in_level.append(self.video_size[i][int(chunk_playing+k)])
            future_videosize.append(size_in_level)
        return future_videosize

    def get_future_video_size(self, P):  # the unplayed video size
        interval = 1
        chunk_playing = self.get_play_chunk()
        if chunk_playing % 1 == 0:  # Check whether it is an integer
            interval = 0

        future_videosize = []
        for i in range(BITRATE_LEVELS):
            size_in_level = []
            for k in range(P):
                size_in_level.append(self.video_size[i][int(chunk_playing + interval + k)])
            future_videosize.append(size_in_level)
        return future_videosize

    def get_play_chunk(self):
        return self.play_timeline / self.chunklength
    
    def get_remain_video_num(self):
        self.video_chunk_remain = self.chunk_num - self.video_chunk_counter
        return self.video_chunk_remain
    
    def get_chunk_sum(self):
        return self.chunk_num

    def get_chunk_counter(self):
        return self.video_chunk_counter

    def get_buffer_size(self):
        return self.buffer_size
    
    def record_download_bitrate(self, bit_rate):
        self.download_chunk_bitrate.append(bit_rate)
        self.preload_size += self.video_size[bit_rate][self.video_chunk_counter]

    def record_download_bitrate_ct(self, selected_ct):
        self.download_bitrate_ct.append(selected_ct)
    
    def get_downloaded_bitrate(self):
        return self.download_chunk_bitrate
    
    def bandwidth_waste(self, user_ret):
        download_len = len(self.download_chunk_bitrate)
        waste_start_chunk = math.ceil(user_ret.get_ret_duration() / self.chunklength)
        sum_waste_each_video = 0
        for i in range(waste_start_chunk, download_len):
            download_bitrate = self.download_chunk_bitrate[i]
            download_size = self.video_size[download_bitrate][i]
            sum_waste_each_video += download_size

        log_dir = './log_download_bitrate'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(os.path.join(log_dir, 'download_bitrate_' + str(self.video_name)), 'w') as f:
            f.write(self.download_chunk_bitrate.__str__())

        return sum_waste_each_video
            
    # download the video, buffer increase.
    def video_download(self, download_len):  # ms
        self.buffer_size += download_len
        self.video_chunk_counter += 1
        end_of_video = False
        if self.video_chunk_counter >= self.chunk_num:
            end_of_video = True
        return end_of_video

    # play the video, buffer decrease. Return the remaining buffer, negative number means rebuf
    def video_play(self, play_time):  # ms
        buffer = self.buffer_size - play_time
        self.play_timeline += np.minimum(self.buffer_size, play_time)   # rebuffering time is not included in timeline
        self.buffer_size = np.maximum(self.buffer_size - play_time, 0.0)
        # print(self.buffer_size, play_time, "play_timeline", self.play_timeline)
        return self.play_timeline, buffer

    def get_play_chunk_bitrate(self):
        video_chunk_idx = math.floor(self.play_timeline / self.chunklength)
        # print(self.play_timeline, video_chunk_idx)
        # print(self.download_chunk_bitrate)
        video_chunk_idx = min(video_chunk_idx, len(self.download_chunk_bitrate) - 1)
        if video_chunk_idx == -1:
            return -1
        playing_chunk_bitrate = self.download_chunk_bitrate[video_chunk_idx]

        return playing_chunk_bitrate

    def get_play_chunk_ct(self):
        video_chunk_idx = math.floor(self.play_timeline / self.chunklength)
        # print(video_chunk_idx)
        video_chunk_idx = min(video_chunk_idx, len(self.download_chunk_bitrate) - 1)
        if video_chunk_idx == -1:
            return -1
        playing_chunk_ct = self.download_bitrate_ct[video_chunk_idx]

        return playing_chunk_ct

    def sort_video(self, video):
        sorted_video = [''] * len(video)
        for i in range(len(video)):
            video_name = video[i]
            video_index = int(video_name.split('_')[0])
            sorted_video[video_index] = video[i]

        return sorted_video




