import math
import sys, os

import torch

sys.path.append('./simulator/')
sys.path.append('/home/dell/Xinyu/Code/PDAS_Incendio/')
sys.path.append('/data/xinyu/pdas/')
import argparse
import random
import numpy as np
from simulator_deload import controller as env, short_video_load_trace

parser = argparse.ArgumentParser()
parser.add_argument('--trace', type=str, help='The network trace you are testing (fixed, high, low, medium, middle)')

parser.add_argument('--dataset_dir', type=str, help='Is testing quickstart')
parser.add_argument('--chunklength', type=float, help='')
parser.add_argument('--train_type', type=str, help='')

parser.add_argument('--epoch', type=str, help='indicate BM BA model')

args = parser.parse_args()

# QoE 参数
from config_algorithm import VIDEO_BIT_RATE
from config_algorithm import alpha, beta, gamma, theta

# 计算smooth惩罚
def get_smooth(net_env, download_video_id, downloaded_time, quality):
    if download_video_id == 0 and downloaded_time == 0:  # is the first chunk of all
        return 0
    if downloaded_time == 0:  # needs to find the last chunk of the last video
        last_bitrate = last_chunk_bitrate[download_video_id - 1]
        if last_bitrate == -1:  # the neighbour chunk is not downloaded
            return 0
    else:
        last_bitrate = net_env.players[download_video_id - net_env.get_start_video_id()].get_downloaded_bitrate()[-1]
    return abs(quality - VIDEO_BIT_RATE[last_bitrate])

def test(trace_id, user_sample_id, model_path, swipe_trace_user_i, network=None):

    # 初始化算法
    from solution_deload import Algorithm
    solution = Algorithm()
    solution.Initialize(model_path, args.chunklength)

    # 初始化环境
    net_env = env.Environment(user_sample_id, all_cooked_time[trace_id], all_cooked_bw[trace_id], ALL_VIDEO_NUM, swipe_trace_user_i, args.dataset_dir, args.chunklength)
    if network is not None:
        # print(network.mahimahi_ptr, '=================')
        net_env.network = network

    # 算法决策  第一次
    first_step = True
    download_video_id, bitrate, sleep_time, download_range = solution.run(0, 0, 0, False, 0, net_env.players, 0, first_step)  # take the first step

    # 相关性能指标统计 初始化
    # 带宽浪费
    sum_wasted_bytes = 0

    # QoE及其组成
    QoE = 0
    quality_all = 0
    smooth_all = 0
    rebuffer_all = 0

    # 带宽使用
    bandwidth_usage = 0  # record total bandwidth usage
    sleep_all = 0

    real_time = 0
    pre_play_video_id = 0

    # 观看时长
    view_length = 0
    download_length_all = 0
    play_video_id = 0

    # 分视频统计 Qoe相关指标
    performance_view_time = {0:np.zeros(5), 1:np.zeros(5)}
    qoe_video = [0] * ALL_VIDEO_NUM
    quality_video = [0] * ALL_VIDEO_NUM
    smooth_video = [0] * ALL_VIDEO_NUM
    rebuffer_video = [0] * ALL_VIDEO_NUM
    bw_wastage_video = [0] * ALL_VIDEO_NUM

    count_view_chunk_view_time = [0] * 2
    count_view_chunk_video = [0] * ALL_VIDEO_NUM
    count_video_num_view_time = [0] * 2

    while True:
        # 计算本次下载的quality和smooth
        quality = 0
        smooth = 0
        sleep_all += sleep_time
        download_twice = False
        if sleep_time == 0:
            if download_video_id >= ALL_VIDEO_NUM - 4:
                download_length_all += download_range.item() * 1000.
            else:
                player = net_env.players[download_video_id - play_video_id]
                download_length = player.get_buffer_size() + player.play_timeline
                last_chunk_time_left = min(player.video_len, math.ceil(download_length / args.chunklength) * args.chunklength) - download_length

                video_left = (player.video_len - download_length) / 1000. # s
                if download_range.item() > video_left:
                    # 最多下载完整个视频
                    download_range = torch.tensor(video_left)

                if last_chunk_time_left > 1e-3 and last_chunk_time_left + 1e-3 < 2000.:
                    if last_chunk_time_left < download_range.item() * 1000.:
                        download_range_1 = torch.tensor(last_chunk_time_left / 1000.)
                        bitrate_1 = player.get_downloaded_bitrate()[-1]

                        download_range_2 = download_range - download_range_1
                        bitrate_2 = bitrate

                        download_twice = True
                        # print('download twice')
                        # breakpoint()
                    else:
                        bitrate = player.get_downloaded_bitrate()[-1]
                        # print('download once')

                # the last chunk id that user watched
                max_watch_time = net_env.user_models[download_video_id - net_env.get_start_video_id()].get_ret_duration() # ms
                # last downloaded chunk id
                downloaded_time = (net_env.players[download_video_id - net_env.get_start_video_id()].play_timeline
                                   + net_env.players[download_video_id - net_env.get_start_video_id()].buffer_size) # ms

                download_chunk_id = math.floor(downloaded_time / args.chunklength)

                # print(download_video_id, bitrate, max_watch_time, downloaded_time)
                if max_watch_time >= downloaded_time:  # the downloaded chunk will be played
                    if download_twice:
                        if max_watch_time < downloaded_time + download_range_1.item() * 1000. + 1e-3:
                            quality = VIDEO_BIT_RATE[bitrate_1]
                        else:
                            quality = VIDEO_BIT_RATE[bitrate_2]
                    else:
                        quality = VIDEO_BIT_RATE[bitrate]

                    if downloaded_time + download_range.item() * 1000. + 1e-3 >= max_watch_time:  # maintain the last_chunk_bitrate array
                        last_chunk_bitrate[download_video_id] = bitrate
                        rel_id = download_video_id - net_env.get_start_video_id()
                        if rel_id + 1 < len(net_env.user_models):  # If its not the last visible video
                            if net_env.players[rel_id + 1].get_buffer_size() != 0:
                                # if the next video chunk has already been downloaded before this last chunk,
                                # we include the smooth penalty here.
                                next_bitrate = net_env.players[rel_id + 1].get_downloaded_bitrate()[0]
                                smooth += abs(quality - VIDEO_BIT_RATE[next_bitrate])
                    smooth += get_smooth(net_env, download_video_id, downloaded_time, quality)


                    if downloaded_time + download_range.item() * 1000. >= max_watch_time:
                        total_watch_chunk_num_minus_1 = math.floor(max_watch_time / args.chunklength)
                        t = total_watch_chunk_num_minus_1 * args.chunklength - downloaded_time + args.chunklength
                        if download_twice:
                            quality = VIDEO_BIT_RATE[bitrate_1] * download_range_1.item() * 1000. / args.chunklength + VIDEO_BIT_RATE[bitrate_2] * max(0, t - download_range_1.item() * 1000.) / args.chunklength
                            # print(f'twice download a {quality} {bitrate_1} {download_range_1.item()} {bitrate_2} {download_range_2.item()}')
                        else:
                            quality = VIDEO_BIT_RATE[bitrate] * t / args.chunklength

                        view_length += t
                    else:
                        if download_twice:
                            quality = VIDEO_BIT_RATE[bitrate_1] * download_range_1.item() * 1000. / args.chunklength + \
                                      VIDEO_BIT_RATE[bitrate_2] * download_range_2.item() * 1000. / args.chunklength
                            # print(f'twice download b {quality} {bitrate_1} {download_range_1.item()} {bitrate_2} {download_range_2.item()}')
                        else:
                            quality = VIDEO_BIT_RATE[bitrate] * download_range.item() * 1000. / args.chunklength
                        view_length += download_range.item() * 1000.
                # print(quality, smooth, view_length)
                download_length_all += download_range.item() * 1000.

        user_view_time = net_env.user_models[0].get_ret_duration() / 1000.

        user_rets = net_env.user_models.copy()

        if not download_twice or sleep_time != 0:
            delay, rebuf, video_size, end_of_video, \
            play_video_id, waste_bytes, rtt, _ = net_env.buffer_management(download_video_id, bitrate, download_range.item(), sleep_time)
            # print(play_video_id, download_video_id, bitrate, download_range.item(), sleep_time, rebuf)
        else:

            delay, rebuf, video_size, end_of_video, \
                play_video_id, waste_bytes, rtt, _ = net_env.buffer_management_second_download(download_video_id, bitrate_1,
                                                                               download_range_1.item(), bitrate_2, download_range_2.item(), sleep_time)

            # print(play_video_id, download_video_id, bitrate_1, download_range_1.item(), bitrate_2, download_range_2.item(), sleep_time, rebuf)
        # print(download_twice)
        # print(download_video_id, bitrate, download_range.item(), sleep_time)
        # breakpoint()

        if pre_play_video_id == play_video_id:
            user_swipe = 0
        else:
            user_swipe = 1
            solution.user_view_times.append(user_view_time)
            # if len(solution.user_view_times) > 5:
            #     solution.update_b_eta_user()

        buffer_size = 0
        for player in net_env.players:
            buffer_size += player.buffer_size / 1000.

        throughput = net_env.network.cooked_bw[net_env.network.mahimahi_ptr] * 0.95

        # log本次下载的相关信息
        print(f'{real_time},{pre_play_video_id},{download_video_id},{bitrate},{sleep_time},{delay},{rebuf},{user_swipe},{buffer_size},{throughput}', file=log_file)

        if sleep_time != 0:
            real_time += int(sleep_time)
        else:
            real_time += int(delay)

        # 更新带宽使用
        bandwidth_usage += video_size

        # 更新带宽浪费
        sum_wasted_bytes += waste_bytes  # Sum up the bandwidth wastage

        # 更新qoe
        one_step_QoE = alpha * quality / 1000. - beta * rebuf / 1000. - gamma * smooth / 1000.

        # 更新对于视频的qoe数据
        qoe_video[download_video_id] += one_step_QoE
        quality_video[download_video_id] += quality / 1000.
        smooth_video[download_video_id] += smooth / 1000.
        rebuffer_video[download_video_id] += rebuf
        bw_wastage_video[pre_play_video_id] += waste_bytes

        # 更新对应观看时长的数据
        if user_swipe == 1:
            for idx in range(play_video_id - pre_play_video_id):
                count_view_chunk_video[pre_play_video_id + idx] += math.ceil(user_rets[idx].get_ret_duration() / args.chunklength)
                # view_chunk_num = view_length / args.chunklength
                # print(idx, view_chunk_num, sum(count_view_chunk_video), play_video_id- pre_play_video_id)
                # print(count_view_chunk_video)

                category_view_time = min(1, int(user_rets[idx].get_ret_duration() / 1000. / 12))
                performance_view_time[category_view_time][0] += qoe_video[pre_play_video_id + idx]
                performance_view_time[category_view_time][1] += quality_video[pre_play_video_id + idx]
                performance_view_time[category_view_time][2] += smooth_video[pre_play_video_id + idx]
                performance_view_time[category_view_time][3] += rebuffer_video[pre_play_video_id + idx]
                performance_view_time[category_view_time][4] += bw_wastage_video[pre_play_video_id + idx]

                count_view_chunk_view_time[category_view_time] += count_view_chunk_video[pre_play_video_id + idx]

                count_video_num_view_time[category_view_time] += 1

                # print(
                #     f'{count_view_chunk_video[pre_play_video_id + idx]},{len(user_rets[idx].user_time)},{qoe_video[pre_play_video_id + idx]}',
                #     file=log_file)

        pre_play_video_id = play_video_id

        # 更新qoe
        QoE += one_step_QoE

        quality_all += quality / 1000.
        smooth_all += smooth / 1000.
        rebuffer_all += rebuf / 1000.

        # 用户退出
        if len(net_env.players) < 5:
            for player in net_env.players:
                for i in range(len(player.download_chunk_bitrate)):
                    download_bitrate = player.download_chunk_bitrate[i]
                    download_len = player.download_chunk_length[i]
                    download_size = VIDEO_BIT_RATE[download_bitrate] * download_len
                    sum_wasted_bytes += download_size
            break

        # Apply the participant's algorithm to decide the args for the next step
        download_video_id, bitrate, sleep_time, download_range = solution.run(delay, rebuf, video_size, end_of_video, play_video_id, net_env.players, rtt, False)
        # print(play_video_id, download_video_id, bitrate, sleep_time, download_range.item())

    for catefory_ in performance_view_time:
        performance_view_time[catefory_][0] /= count_view_chunk_view_time[catefory_]
        performance_view_time[catefory_][1] /= count_view_chunk_view_time[catefory_]
        performance_view_time[catefory_][2] /= count_view_chunk_view_time[catefory_]
        performance_view_time[catefory_][3] /= count_view_chunk_view_time[catefory_]

    # print(count_view_chunk_view_time)
    # print(performance_view_time)
    # print(count_video_num_view_time)

    # Score
    S = QoE - theta * bandwidth_usage * 8 / 1000000.

    view_chunk_num = view_length / args.chunklength
    download_chunk_num = download_length_all / args.chunklength

    print("Your score is: ", S)
    # QoE
    print("Your QoE is: ", QoE / view_chunk_num)
    # wasted_bytes
    # print("Your sum of wasted bytes is: ", sum_wasted_bytes)
    # print("Your sum of used bytes is: ", bandwidth_usage)

    print("Quality_all: ", quality_all / view_chunk_num)
    print("Smooth_all: ", smooth_all / (view_chunk_num - 1))
    print("Rebuffer_all: ", rebuffer_all / view_chunk_num * 1000.)
    print("bw_wastage_all: ", sum_wasted_bytes / 1000000.)

    return (np.array([QoE / view_chunk_num, quality_all / view_chunk_num, smooth_all / (view_chunk_num - 1), rebuffer_all / view_chunk_num * 1000., sum_wasted_bytes / 1000000.]),
            performance_view_time[0], performance_view_time[1])

def test_all_traces(trace, user_sample_id, model_path, f_performance):

    # log文件路径
    LOG_DIR = 'logs/' + str(os.path.basename(args.dataset_dir)) + '/user_' + str(user_sample_id) + '/'
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    avg_0 = np.zeros(5)
    avg_1 = np.zeros(5)
    avg_all = np.zeros(5)

    # 网络轨迹
    cooked_trace_folder = '../data/network_traces/' + trace + '/'
    global all_cooked_time, all_cooked_bw, ALL_VIDEO_NUM, last_chunk_bitrate
    all_cooked_time, all_cooked_bw = short_video_load_trace.load_trace(cooked_trace_folder)

    # 用户行为
    user_swipe_dir = args.dataset_dir + '/sample_user/'
    user_swipe_trace = user_swipe_dir + '/user_' + str(user_sample_id) + '.txt'
    swipe_trace_user_i = []
    with open(user_swipe_trace, 'r') as f:
        for line in f:
            swipe_trace_user_i.append(float(line))
    ALL_VIDEO_NUM = len(swipe_trace_user_i)
    # record the last chunk(which will be played) of each video to aid the calculation of smoothness
    last_chunk_bitrate = [-1] * ALL_VIDEO_NUM
    # print(seeds)
    global log_file
    cooked_files = os.listdir(cooked_trace_folder)
    for i in range(len(all_cooked_time)):
        LOG_FILE = LOG_DIR + str(args.trace) + '_' + str(cooked_files[i])
        log_file = open(LOG_FILE, 'w')

        print(f'------------user {user_sample_id} trace {i} --------------')
        # avg += test(i, user_sample_id, swipe_trace_user_i)
        _all, _0, _1 = test(i, user_sample_id, model_path, swipe_trace_user_i)
        avg_all += _all
        avg_0 += _0
        avg_1 += _1
        print('---------------------------------------\n\n')

        f_performance.write(f'{i},{_all[0]},{_all[1]},{_all[2]},{_all[3]},{_all[4]}\n')
        f_performance.flush()

    avg_all /= len(all_cooked_time)
    avg_0 /= len(all_cooked_time)
    avg_1 /= len(all_cooked_time)

    print("\n\nYour average indexes under [", trace, "] network is: ")
    print("QoE: ", avg_all[0])
    print("Quality_all: ", avg_all[1])
    print("Smooth_all: ", avg_all[2])
    print("Rebuffer_all: ", avg_all[3])

    return avg_all, avg_0, avg_1

def test_user_samples(trace, sample_cnt, model_path):  # test 50 user sample
    '''

    :param trace: 训练所需的数据集
    :param sample_cnt: 要采样的用户的个数
    :param probability_map_dir: 滑动概率分布
    :return:
    '''

    seed_for_sample = np.random.randint(10000, size=(1001, 1))
    avgs_all = np.zeros(5)
    avgs_0 = np.zeros(5)
    avgs_1 = np.zeros(5)

    performance_under_sub_dataset = './deload_' + str(args.chunklength/ 1000) + trace.split('/')[0]
    f = open(performance_under_sub_dataset, 'a')

    for j in range(sample_cnt):
        print('------------sample user ', j, '--------------')
        # global seeds
        # np.random.seed(seed_for_sample[j])
        # seeds = np.random.randint(10000, size=(ALL_VIDEO_NUM, 2))  # reset the sample random seeds
        _all, _0, _1 = test_all_traces(trace, j, model_path, f)
        avgs_all += _all
        avgs_0 += _0
        avgs_1 += _1

    avgs_all /= sample_cnt
    avgs_0 /= sample_cnt
    avgs_1 /= sample_cnt

    print("\nQoE: ", avgs_all[0])
    print("Quality_all: ", avgs_all[1])
    print("Smooth_all: ", avgs_all[2])
    print("Rebuffer_time: ", avgs_all[3])

    f.write(
        f'{args.epoch},{os.path.basename(args.dataset_dir)},{args.trace},{avgs_all[0]},{avgs_all[1]},{avgs_all[2]},{avgs_all[3]},{avgs_all[4]}\n')
    f.write(
        f'{args.epoch},{os.path.basename(args.dataset_dir)},{args.trace},{avgs_0[0]},{avgs_0[1]},{avgs_0[2]},{avgs_0[3]},{avgs_0[4]}\n')
    f.write(
        f'{args.epoch},{os.path.basename(args.dataset_dir)},{args.trace},{avgs_1[0]},{avgs_1[1]},{avgs_1[2]},{avgs_1[3]},{avgs_1[4]}\n')

    return avgs_all, avgs_0, avgs_1

if __name__ == '__main__':

    nn_model_save_path = './models/'
    epoch = args.epoch

    MODEL_SAVE_PATH = nn_model_save_path + 'policy_' + epoch + '.pt'

    avgs = test_user_samples(args.trace, 1, MODEL_SAVE_PATH)