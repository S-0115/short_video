import math
import sys, os
sys.path.append('/home/dell/Xinyu/Code/PDAS_Incendio/')
sys.path.append('/data/xinyu/pdas/')
import argparse
import numpy as np
from simulator import controller as env, short_video_load_trace

parser = argparse.ArgumentParser()
parser.add_argument('--trace', type=str, help='The network trace you are testing (fixed, high, low, medium, middle)')

parser.add_argument('--dataset_dir', type=str, help='')

parser.add_argument('--chunklength', type=float, help='ms')

args = parser.parse_args()

# QoE arguments

from config_algorithm import VIDEO_BIT_RATE
from config_algorithm import alpha, beta, gamma, theta

# calculate the smooth penalty for an action to download:
# chunk:[chunk_id] of the video:[download_video_id] with bitrate:[quality]
def get_smooth(net_env, download_video_id, chunk_id, quality):
    if download_video_id == 0 and chunk_id == 0:  # is the first chunk of all
        return 0
    if chunk_id == 0:  # needs to find the last chunk of the last video
        last_bitrate = last_chunk_bitrate[download_video_id - 1]
        if last_bitrate == -1:  # the neighbour chunk is not downloaded
            return 0
    else:
        last_bitrate = net_env.players[download_video_id - net_env.get_start_video_id()].get_downloaded_bitrate()[chunk_id - 1]
    return abs(quality - VIDEO_BIT_RATE[last_bitrate])

def test(trace_id, user_sample_id, swipe_trace_user_i, network=None):

    # print('------------trace ', trace_id, '--------------', file=log_file)

    from solution_pdas import Algorithm

    # start the test

    solution = Algorithm()
    solution.Initialize(args.chunklength)

    # all_cooked_time, all_cooked_bw = short_video_load_trace.load_trace(trace_path)
    net_env = env.Environment(user_sample_id, all_cooked_time[trace_id], all_cooked_bw[trace_id], ALL_VIDEO_NUM, swipe_trace_user_i, args.dataset_dir, args.chunklength)
    if network is not None:
        # print(network.mahimahi_ptr, '=================')
        net_env.network = network
    # Decision variables
    download_video_id, bit_rate, sleep_time = solution.run(0, 0, 0, False, 0, net_env.players, True)  # take the first step

    # sum of wasted bytes for a user
    QoE = 0

    quality_all = 0
    smooth_all = 0
    rebuffer_all = 0

    sum_wasted_bytes = 0

    bandwidth_usage = 0  # record total bandwidth usage

    real_time = 0
    pre_play_video_id = 0

    view_chunk_num = 0
    download_chunk_num = 0

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
        # calculate the quality and smooth for this download step taken
        quality = 0
        smooth = 0

        if sleep_time == 0:
            if download_video_id >= ALL_VIDEO_NUM - 4:
                download_chunk_num += 1
            else:
                # the last chunk id that user watched
                max_watch_chunk_id = net_env.user_models[
                    download_video_id - net_env.get_start_video_id()].get_watch_chunk_cnt()
                # last downloaded chunk id
                download_chunk = net_env.players[download_video_id - net_env.get_start_video_id()].get_chunk_counter()
                if max_watch_chunk_id >= download_chunk:  # the downloaded chunk will be played
                    quality = VIDEO_BIT_RATE[bit_rate]
                    if download_chunk == max_watch_chunk_id:  # maintain the last_chunk_bitrate array
                        last_chunk_bitrate[download_video_id] = bit_rate
                        rel_id = download_video_id - net_env.get_start_video_id()
                        if rel_id + 1 < len(net_env.user_models):  # If its not the last visible video
                            if net_env.players[rel_id + 1].get_chunk_counter() != 0:
                                # if the next video chunk has already been downloaded before this last chunk,
                                # we include the smooth penalty here.
                                next_bitrate = net_env.players[rel_id + 1].get_downloaded_bitrate()[0]
                                smooth += abs(quality - VIDEO_BIT_RATE[next_bitrate])
                    smooth += get_smooth(net_env, download_video_id, download_chunk, quality)
                    view_chunk_num += 1
                download_chunk_num += 1

        user_rets = net_env.user_models.copy()

        buffer_size = 0
        for player in net_env.players:
            buffer_size += player.buffer_size / 1000.

        throughput = net_env.network.cooked_bw[net_env.network.mahimahi_ptr] * 0.95

        delay, rebuf, video_size, end_of_video, \
        play_video_id, waste_bytes, rtt = net_env.buffer_management(download_video_id, bit_rate, sleep_time)
        # print(delay, rebuf, video_size, end_of_video, play_video_id, waste_bytes)
        if pre_play_video_id == play_video_id:
            user_swipe = 0
        else:
            user_swipe = 1

        print(f'{real_time},{pre_play_video_id},{download_video_id},{bit_rate},{sleep_time},{delay},{rebuf},{user_swipe},{buffer_size},{throughput}', file=log_file)

        if sleep_time != 0:
            real_time += int(sleep_time)
        else:
            real_time += int(delay)

        # Update bandwidth usage
        bandwidth_usage += video_size

        # Update bandwidth wastage
        sum_wasted_bytes += waste_bytes  # Sum up the bandwidth wastage

        # Update QoE:

        one_step_QoE = alpha * quality / 1000. - beta * rebuf / 1000. - gamma * smooth / 1000.

        qoe_video[download_video_id] += one_step_QoE
        quality_video[download_video_id] += quality / 1000.
        smooth_video[download_video_id] += smooth / 1000.
        rebuffer_video[download_video_id] += rebuf
        bw_wastage_video[pre_play_video_id] += waste_bytes

        if user_swipe == 1:
            for idx in range(play_video_id - pre_play_video_id):
                count_view_chunk_video[pre_play_video_id + idx] += math.ceil(user_rets[idx].get_ret_duration() / args.chunklength)

                category_view_time = min(1, int(user_rets[idx].get_ret_duration() / 1000. / 12))
                performance_view_time[category_view_time][0] += qoe_video[pre_play_video_id + idx]
                performance_view_time[category_view_time][1] += quality_video[pre_play_video_id + idx]
                performance_view_time[category_view_time][2] += smooth_video[pre_play_video_id + idx]
                performance_view_time[category_view_time][3] += rebuffer_video[pre_play_video_id + idx]
                performance_view_time[category_view_time][4] += bw_wastage_video[pre_play_video_id + idx]

                count_view_chunk_view_time[category_view_time] += count_view_chunk_video[pre_play_video_id + idx]

                count_video_num_view_time[category_view_time] += 1

                print(
                    f'{count_view_chunk_video[pre_play_video_id + idx]},{len(user_rets[idx].user_time)},{qoe_video[pre_play_video_id + idx]}',
                    file=log_file)

        pre_play_video_id = play_video_id

        QoE += one_step_QoE

        quality_all += quality / 1000.
        smooth_all += smooth / 1000.
        rebuffer_all += rebuf / 1000.

        # play over all videos
        if len(net_env.players) < 5:
            for player in net_env.players:
                for i in range(len(player.download_chunk_bitrate)):
                    download_bitrate = player.download_chunk_bitrate[i]
                    download_size = player.video_size[download_bitrate][i]
                    sum_wasted_bytes += download_size
            break

        # Apply the participant's algorithm to decide the args for the next step
        download_video_id, bit_rate, sleep_time = solution.run(delay, rebuf, video_size, end_of_video, play_video_id, net_env.players, False)
        # print(download_video_id, bit_rate, sleep_time)

    for catefory_ in performance_view_time:
        performance_view_time[catefory_][0] /= count_view_chunk_view_time[catefory_]
        performance_view_time[catefory_][1] /= count_view_chunk_view_time[catefory_]
        performance_view_time[catefory_][2] /= count_view_chunk_view_time[catefory_]
        performance_view_time[catefory_][3] /= count_view_chunk_view_time[catefory_]
        # performance_view_time[catefory_][4] /= count_view_chunk_view_time[catefory_]

    # print(count_view_chunk_view_time)
    # print(performance_view_time)
    # print(count_video_num_view_time)

    # Score
    S = QoE - theta * bandwidth_usage * 8 / 1000000.

    # print(view_chunk_num)
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

    # end the test
    # print('------------trace ', trace_id, '--------------\n\n', file=log_file)
    return (S, np.array([QoE / view_chunk_num, quality_all / view_chunk_num, smooth_all / (view_chunk_num - 1), rebuffer_all / view_chunk_num * 1000., sum_wasted_bytes / 1000000.]),
            performance_view_time[0], performance_view_time[1])

def test_all_traces(trace, user_sample_id, f_performance):

    LOG_DIR = 'logs/' + str(os.path.basename(args.dataset_dir)) + '/user_' + str(user_sample_id) + '/'
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    S_all = 0
    avg_0 = np.zeros(5)
    avg_1 = np.zeros(5)
    avg_all = np.zeros(5)
    cooked_trace_folder = '../data/network_traces/' + trace + '/'
    global all_cooked_time, all_cooked_bw, ALL_VIDEO_NUM, last_chunk_bitrate
    all_cooked_time, all_cooked_bw = short_video_load_trace.load_trace(cooked_trace_folder)

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
        S, _all, _0, _1 = test(i, user_sample_id, swipe_trace_user_i)
        S_all += S
        avg_all += _all
        avg_0 += _0
        avg_1 += _1
        print('---------------------------------------\n\n')

        f_performance.write(f'{i},{_all[0]},{_all[1]},{_all[2]},{_all[3]},{_all[4]}\n')
        f_performance.flush()

    S_all /= len(all_cooked_time)
    avg_all /= len(all_cooked_time)
    avg_0 /= len(all_cooked_time)
    avg_1 /= len(all_cooked_time)

    print("\n\nYour average indexes under [", trace, "] network is: ")
    print("Score: ", S_all)
    print("QoE: ", avg_all[0])
    print("Quality_all: ", avg_all[1])
    print("Smooth_all: ", avg_all[2])
    print("Rebuffer_all: ", avg_all[3])

    return avg_all, avg_0, avg_1

def test_user_samples(trace, sample_cnt):  # test 50 user sample
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

    performance_under_sub_dataset = './pdas_' + str(args.chunklength/ 1000) + trace.split('/')[0]
    f = open(performance_under_sub_dataset, 'a')

    for j in range(sample_cnt):
        print('------------sample user ', j, '--------------')
        # global seeds
        # np.random.seed(seed_for_sample[j])
        # seeds = np.random.randint(10000, size=(ALL_VIDEO_NUM, 2))  # reset the sample random seeds
        _all, _0, _1= test_all_traces(trace, j, f)
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
        f'{os.path.basename(args.dataset_dir)},{args.trace},{avgs_all[0]},{avgs_all[1]},{avgs_all[2]},{avgs_all[3]},{avgs_all[4]}\n')
    f.write(
        f'{os.path.basename(args.dataset_dir)},{args.trace},{avgs_0[0]},{avgs_0[1]},{avgs_0[2]},{avgs_0[3]},{avgs_0[4]}\n')
    f.write(
        f'{os.path.basename(args.dataset_dir)},{args.trace},{avgs_1[0]},{avgs_1[1]},{avgs_1[2]},{avgs_1[3]},{avgs_1[4]}\n')

    return avgs_all, avgs_0, avgs_1

if __name__ == '__main__':
    # assert args.trace in ["mixed", "high", "low", "medium"]
        test_user_samples(args.trace, 1)
