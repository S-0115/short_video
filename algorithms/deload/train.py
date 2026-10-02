import itertools
import math
import sys, os

import torch

from exp_buffer import ReplayBuffer
from simulator_deload import controller as env, short_video_load_trace
from model.PPO import Policy

import argparse
import random
import numpy as np
import multiprocess as mp

parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO")
parser.add_argument("--N", type=int, default=int(8), help=" number of agent")
parser.add_argument("--max_train_steps", type=int, default=int(2000), help=" Maximum number of training steps")
parser.add_argument("--episode_limit", type=int, default=100, help="Maximum number of steps per episode")
parser.add_argument("--evaluate_freq", type=float, default=50, help="Evaluate the policy every 'evaluate_freq' steps")

parser.add_argument("--batch_size", type=int, default=4, help="Batch size (the number of episodes)")
parser.add_argument("--mini_batch_size", type=int, default=2, help="Minibatch size (the number of episodes)")
parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
parser.add_argument("--epsilon", type=float, default=0.2, help="Discount factor")
parser.add_argument("--gamma", type=float, default=1., help="Discount factor")
parser.add_argument("--K_epochs", type=int, default=2, help="GAE parameter")


parser.add_argument('--trace', type=str, default='sampled_4G_train', help='The network trace used for train')

parser.add_argument('--dataset_dir', type=str, default='../data/dataset_2s_train/', help='Is testing quickstart')
parser.add_argument('--chunklength', type=float, default=2000., help='')


args = parser.parse_args()

# QoE 参数
from config_algorithm import VIDEO_BIT_RATE
from config_algorithm import alpha, beta, gamma, theta, watch_time_threhold, Bmax

ALL_VIDEO_NUM = 100
all_cooked_time = []
all_cooked_bw = []

# ppo 训练参数
NUM_AGENTS = args.N
MODEL_SAVE_INTERVAL = args.evaluate_freq
TRAIN_SEQ_LEN = args.episode_limit
TRAIN_TRACES = '../data/network_traces/' + args.trace + '/'
DEFAULT_ID = 0
DEFAULT_BITRATE = 0
DEFAULT_SLEEP = 0
PAST_BW_LEN = 5
TAU = 500.

b_IN_B = 8
b_IN_kb = 1000

start_epoch = 1

USE_GPU = torch.cuda.is_available()
device = torch.device('cuda' if USE_GPU else 'cpu')

batch_size = args.batch_size

# 算法参数，观看时长上下界
eh = 0.7
el = 0.3

def test_model(epoch):
    with torch.no_grad():
        dataset_dir = '../data/dataset_2s_test'
        chunklength = 2000.

        os.system(f'python run_deload.py '
                  f'--trace sampled_4G '
                  f'--dataset_dir {dataset_dir} '
                  f'--epoch {epoch} '
                  f'--chunklength {chunklength} ')

def run_with_timeout(func, timeout, epoch):
    p = mp.Process(target=func, args=(epoch,))
    p.start()
    # p.join(timeout)
    #
    # if p.is_alive():
    #     p.terminate()
    #     p.join()
    #     print('=' * 20)
    #     print('time out')
    #     print('=' * 20)

def cul_reward(at, bt, wt, btt, qt):
    # 计算本次下载决策的奖励
    '''
    :param at: s downloadrange
    :param bt: VIDEO_BIT_RATE[bt] kbps selected video bitrate
    :param wt: B bandwidth wastage caused by current download action
    :param btt: ms rebuff caused by current download action
    :param qt: MB average bandwidth of previous download action
    :return: reward
    '''

    wt = wt * 8 / 1000. # kb
    wt = min(wt , 1200) # 带宽浪费上限截断为1.2Mb = 1200kb
    reward = at * VIDEO_BIT_RATE[bt] / 1000. - 0.01 * wt - 1.85 * btt / 1000. * qt * 8
    # print(reward)
    # print(at * VIDEO_BIT_RATE[bt] / 1000.)
    # print(0.01 * wt)
    # print(1.85 * btt / 1000. * qt * 8)
    # print()

    return reward

def update_b_eta_user(x):
    # 根据用户观看行为更新weibull分布

    # print(x)
    x = np.sort(x)
    x = x[x > watch_time_threhold]
    n = len(x)
    if n <= 1:
        return 0, 0

    # F_emp = (np.arange(1, n + 1)) / (n + 1)
    F_emp = (np.arange(1, n + 1) - 0.3)/(n + 0.4)

    Y = np.log(x - watch_time_threhold)
    X = np.log(-np.log(1 - F_emp))
    b=((n * np.sum(Y * X)) - (np.sum(X) * np.sum(Y))) / (n * np.sum(np.square(Y)) - np.sum(Y) ** 2)
    # print(n * np.sum(np.square(Y)) - np.sum(Y) ** 2)
    eta = np.exp(Y.mean() - X.mean() / b)  # η̂

    return b, eta


def calculate_demands(Players):
    # 计算每个视频需求demand
    demands = []
    P_videos = []
    for i in range(len(Players)):
        player = Players[i]
        t0 = player.play_timeline / 1000.  # t0
        tau = (player.play_timeline + player.get_buffer_size()) / 1000.

        P_v = calculate_P_v(player, tau, t0)
        P_videos.append(P_v)
    # print(P_videos)
    for i in range(len(P_videos)):
        if i == 0:
            demands.append(P_videos[i])
        else:
            demands.append((1 - np.sum(demands)) * P_videos[i])
    # print(demands)
    return demands


def calculate_P_v(player, tau, t0):
    b_v = player.beta
    eta_v = player.eta

    if tau <= watch_time_threhold:
        P_T_tau = 1.
    else:
        P_T_tau = np.exp(-((tau - watch_time_threhold) / eta_v) ** b_v)

    if t0 <= watch_time_threhold:
        P_T_t0 = 1.
    else:
        P_T_t0 = np.exp(-((t0 - watch_time_threhold) / eta_v) ** b_v)

    # 如果用户已经看完了，则留存率为0，即不需要在考虑下载该视频
    return P_T_tau / P_T_t0

def get_input_data(video_bitrates, Players, selected_video, est_bw, est_rtt):
    # 获取算法的输入状态

    bi = [VIDEO_BIT_RATE[b] / 1000. for b in video_bitrates] # Mb
    while len(bi) < 5:
        bi.append(0.)
    bi = torch.tensor(bi)

    taui = [(player.get_buffer_size() + player.play_timeline) / 1000. for player in Players] # s
    while len(taui) < 5:
        taui.append(0.)
    taui = torch.tensor(taui)

    di = [player.video_len / 1000. for player in Players] # s
    while len(di) < 5:
        di.append(0.)
    di = torch.tensor(di)

    ti = [player.play_timeline / 1000. for player in Players] # s
    while len(ti) < 5:
        ti.append(0.)
    ti = torch.tensor(ti)

    hi = [eh * player.video_len / 1000. for player in Players] # s
    while len(hi) < 5:
        hi.append(0.)
    hi = torch.tensor(hi)

    li = [el * player.video_len / 1000. for player in Players] # s
    while len(li) < 5:
        li.append(0.)
    li = torch.tensor(li)

    bd = [est_bw * 8, est_rtt / 1000.] # Mb, s
    bd = torch.tensor(bd)

    idx = [selected_video]
    idx = torch.tensor(idx)
    # idx.append(selected_video)

    input = [bi, taui, di, ti, hi, li, bd, idx]
    input = torch.concat(input, dim=-1)

    return input

def store_work_agent_data(replay_buffer, s_batchs, raw_a_batchs, r_batchs, v_batchs, log_prob_batchs, dones):
    # 将子agent执行的轨迹存储到经验缓冲区

    s_batchs = torch.tensor(np.array(s_batchs)).permute(1, 0, 2).tolist()
    raw_a_batchs = torch.tensor(np.array(raw_a_batchs)).permute(1, 0).tolist()
    r_batchs = torch.tensor(np.array(r_batchs)).permute(1, 0).tolist()
    v_batchs = torch.tensor(np.array(v_batchs)).permute(1, 0).tolist()
    log_prob_batchs = torch.tensor(np.array(log_prob_batchs)).permute(1, 0).tolist()
    dones = torch.tensor(np.array(dones)).permute(1, 0).tolist()

    # print(mask_batchs.shape, torch.tensor(np.array(s_bm_batchs)).shape)
    # print(len(s_bm_batchs), len(r_batchs), len(v_batchs),len(dones))
    for i in range(len(s_batchs)):
        replay_buffer.store_transition(i, s_batchs[i],
                                       v_batchs[i],
                                       raw_a_batchs[i],
                                       log_prob_batchs[i],
                                       r_batchs[i],
                                       dones[i],)
    # replay_buffer.store_last_value(len(v_batchs) - 1, v_batchs[-1])

    print('data restore, ', replay_buffer.episode_num)

def central_agent(net_params_queues, exp_queues, args):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    layers = [128, 64]
    policy = Policy(layers, 33, args)

    policy = policy.to(device)

    # 获取模型参数
    net_params = policy.state_dict()

    for i in range(NUM_AGENTS):
        net_params_queues[i].put(net_params)

    replay_buffer = ReplayBuffer(NUM_AGENTS, TRAIN_SEQ_LEN, batch_size)
    replay_buffer.reset_buffer()

    epoch = start_epoch
    while True:
        s_batchs = []
        raw_a_batchs = []
        r_batchs = []
        v_batchs = []
        log_prob_batchs = []
        dones = []

        for i in range(NUM_AGENTS):
            # 获取子agent轨迹
            s_batch, raw_a_batch, r_batch, v_batch, log_prob_batch, done = exp_queues[i].get()
            # print(mask)
            s_batchs.append(s_batch)
            raw_a_batchs.append(raw_a_batch)
            r_batchs.append(r_batch)
            v_batchs.append(v_batch)
            log_prob_batchs.append(log_prob_batch)
            dones.append(done)

        # 将轨迹存储到经验缓冲区
        store_work_agent_data(replay_buffer, s_batchs, raw_a_batchs,
                              r_batchs, v_batchs, log_prob_batchs, dones)

        if replay_buffer.episode_num == batch_size:

            print('=' * 20, 'training of epoch ', epoch, '=' * 20)
            # 抽样，更新
            policy.train_ppo(epoch, replay_buffer)
            # 更新完毕，清楚缓冲
            replay_buffer.reset_buffer()

            if epoch % MODEL_SAVE_INTERVAL == 0:
                print("---------epoch %d--------" % epoch)
                # Save the neural net parameters to disk.

                nn_model = f'./models/policy_{epoch}.pt'

                torch.save(policy, nn_model)

                run_with_timeout(test_model, 600, epoch)

            # 分发模型参数
            net_params = policy.state_dict()

            for i in range(NUM_AGENTS):
                net_params_queues[i].put(net_params)

            epoch += 1
        if epoch >= args.max_train_steps + 1:
            sys.exit(0)


def work_agent(idx_agent, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue, args):
    TrainedOptimal = load_trainedOptimal()
    with torch.no_grad():
        # Initial the a3c
        layers = [128, 64]
        policy = Policy(layers, 33, args)

        net_params = net_params_queue.get()

        policy.load_state_dict(net_params)

        policy = policy.to(device)

        # 初始化 state, action, reward batch
        s_batch = []
        raw_a_batch = []
        r_batch = []
        v_batch = []
        log_prob_batch = []
        done = []

        pre_download_video = None
        last_chunk_bitrate = [-1] * 100
        past_bandwidth = list(np.zeros(PAST_BW_LEN))
        past_rtt = list(np.zeros(PAST_BW_LEN))
        past_bandwidth_ests =[]
        past_rtt_ests = []

        # 随机选用户和网络轨迹
        user_sample_id = random.randint(0, 4)
        network_trace_idx = random.randint(0, len(all_cooked_time) - 1)

        user_swipe_dir = args.dataset_dir + '/sample_user/'
        user_swipe_trace = user_swipe_dir + '/user_' + str(user_sample_id) + '.txt'
        seeds = []

        with open(user_swipe_trace, 'r') as f:
            for line in f:
                seeds.append(float(line))

        # 初始化环境
        net_env = env.Environment(user_sample_id, all_cooked_time[network_trace_idx], all_cooked_bw[network_trace_idx], ALL_VIDEO_NUM,
                                  seeds, args.dataset_dir, args.chunklength)

        send_data_count = 0

        play_video_id = 0
        pre_play_video_id = 0
        rtt = 0
        first_step = True

        user_view_times = []
        while True:
            if len(net_env.players) < 5:
                # 用户退出，重新初始化环境以及相关参数
                user_sample_id = random.randint(0, 4)
                user_swipe_dir = args.dataset_dir + '/sample_user/'
                user_swipe_trace = user_swipe_dir + '/user_' + str(user_sample_id) + '.txt'
                seeds = []

                with open(user_swipe_trace, 'r') as f:
                    for line in f:
                        seeds.append(float(line))

                # network_trace_idx = (network_trace_idx + 1) % len(all_cooked_time)
                network_trace_idx = random.randint(0, len(all_cooked_time) - 1)
                # Initial the environment
                net_env = env.Environment(user_sample_id, all_cooked_time[network_trace_idx],
                                          all_cooked_bw[network_trace_idx], ALL_VIDEO_NUM,
                                          seeds, args.dataset_dir, args.chunklength)
                play_video_id = 0
                pre_play_video_id = 0

                past_bandwidth = list(np.zeros(PAST_BW_LEN))
                past_rtt = list(np.zeros(PAST_BW_LEN))
                past_bandwidth_ests = []
                past_rtt_ests = []
                last_chunk_bitrate = [-1] * 100
                first_step = True

                user_view_times = []

            bitrate_players = []
            download_twice = False
            if first_step:
                # 第一次决策
                download_video_id = 0
                bitrate = 0
                sleep_time = 0
                download_range = torch.tensor(2)
            else:
                # 非第一次
                # 估计网络吞吐量和rtt
                start_pos = -min(5, len(past_bandwidth))
                while past_bandwidth[start_pos] == 0:
                    start_pos += 1

                past_bandwidth_for_estimation = past_bandwidth[start_pos:]
                past_rtt_for_estimation = past_rtt[start_pos:]

                if sleep_time == 0:
                    # print(past_bandwidth_for_estimation)
                    bandwidth_sum = 0
                    rtt_sum = 0
                    for past_val in past_bandwidth_for_estimation:
                        bandwidth_sum += (1 / float(past_val))
                    for past_val in past_rtt_for_estimation:
                        rtt_sum += (1 / float(past_val))

                    harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidth_for_estimation))
                    harmonic_rtt = 1.0 / (rtt_sum / len(past_rtt_for_estimation))

                    past_bandwidth_ests.append(harmonic_bandwidth)

                    past_rtt_ests.append(harmonic_rtt)

                    future_bandwidth = harmonic_bandwidth
                    future_rtt = harmonic_rtt

                else:
                    harmonic_bandwidth = past_bandwidth_ests[-1]
                    harmonic_rtt = past_rtt_ests[-1]

                    future_bandwidth = harmonic_bandwidth
                    future_rtt = harmonic_rtt

                for player in net_env.players:
                    remaining_chunks = math.ceil(player.video_len / args.chunklength) - math.floor((player.buffer_size + player.play_timeline) / args.chunklength)  # 计算剩余的块数
                    # 计算最优的比特率选择
                    last_quality = -1
                    downloaded_bitrate = player.get_downloaded_bitrate()
                    if len(downloaded_bitrate) > 0:
                        last_quality = downloaded_bitrate[-1]

                    # P = min(3, max(remaining_chunks, 1))  # 确保 P 不超过剩余块数
                    # bitrate = mpc(player.get_undownloaded_video_size(P), P, player.get_buffer_size(), last_quality, future_bandwidth)
                    bitrate = fastmpc(TrainedOptimal, player.get_buffer_size(), last_quality, future_bandwidth * 1000. * 8)
                    # print(player.get_buffer_size(), last_quality, future_bandwidth * 1000. * 8, bitrate)
                    bitrate_players.append(bitrate)

                # print(bitrate_players)
                # 3. 计算每个视频的需求
                demands = calculate_demands(net_env.players)
                Bmax_video = 0
                for i in range(len(demands)):
                    if net_env.players[i].get_buffer_size() >= Bmax or net_env.players[i].get_buffer_size() + net_env.players[i].play_timeline >= net_env.players[i].video_len - 1e-1:
                        demands[i] = -math.inf
                        Bmax_video += 1
                # print(idx_agent, demands)

                # 根据需求选择下载的视频
                if Bmax_video == len(net_env.players):
                    sleep_time = 500.  # 所有视频均缓冲至最大缓冲区，睡眠500ms
                else:
                    sleep_time = 0.

                    selected_video = np.argmax(demands)
                    download_video_id = selected_video + play_video_id
                    bitrate = bitrate_players[selected_video]
                    state = get_input_data(bitrate_players, net_env.players, selected_video, future_bandwidth, future_rtt).float()
                    # state = state.to('cpu')
                    state = state.to(device)

                    s_batch.append(state.tolist())

                    # Decide the actions for the next step
                    value, (raw_action, download_range), log_prob = policy.act(state)

                    v_batch.append(value.item())
                    raw_a_batch.append(raw_action.item())
                    log_prob_batch.append(log_prob.item())

                    player = net_env.players[selected_video]
                    download_length = player.get_buffer_size() + player.play_timeline
                    last_chunk_time_left = min(player.video_len, math.ceil(download_length / args.chunklength) * args.chunklength) - download_length

                    video_left = (player.video_len - download_length) / 1000.  # s

                    # 判断是否需要分两次下载
                    if download_range.item() > video_left:
                        download_range = torch.tensor(video_left)

                    if last_chunk_time_left > 1e-3 and last_chunk_time_left + 1e-3 < 2000.:
                        if last_chunk_time_left < download_range.item() * 1000.:
                            download_range_1 = torch.tensor(last_chunk_time_left / 1000.)
                            bitrate_1 = player.get_downloaded_bitrate()[-1]

                            download_range_2 = download_range - download_range_1

                            bitrate_2 = bitrate

                            download_twice = True
                        else:
                            bitrate = player.get_downloaded_bitrate()[-1]


            # print(download_video_id, bit_rate, sleep_time)
            user_view_time = net_env.user_models[0].get_ret_duration() / 1000. #s
            if not download_twice or sleep_time != 0:
                delay, rebuf, video_size, end_of_video, \
                play_video_id, waste_bytes, rtt, waste_bytes_after_download = net_env.buffer_management(download_video_id, bitrate, download_range.item(), sleep_time)
                # print('once', waste_bytes)
                # print(play_video_id)
                # print(download_video_id, bitrate, download_range.item(),sleep_time)
            else:
                delay, rebuf, video_size, end_of_video, \
                play_video_id, waste_bytes, rtt, waste_bytes_after_download = net_env.buffer_management_second_download(download_video_id, bitrate_1,
                                                                               download_range_1.item(), bitrate_2, download_range_2.item(), sleep_time)
                # print(download_video_id, bitrate_1,download_range_1.item(), bitrate_2, download_range_2.item(),sleep_time)

            # print(idx_agent, net_env.players[0].play_timeline, net_env.players[0].buffer_size, net_env.players[0].video_len)
            if play_video_id != pre_play_video_id:
                user_view_times.append(user_view_time)
                # if len(user_view_times) > 5:
                #     b_u, eta_u = update_b_eta_user(user_view_times)

            pre_play_video_id = play_video_id

            # 更新吞吐量和rtt记录
            if sleep_time == 0.:
                past_bandwidth = np.roll(past_bandwidth, -1)
                past_bandwidth[-1] = (float(video_size) / 1000000.0) / (float(delay) / 1000.0)  # MB / s
                # print(delay, rtt)
                past_rtt = np.roll(past_rtt, -1)
                past_rtt[-1] = rtt
                # print(rtt)

            # 计算上一步的reward，由于预取由启发式方法完成，所以仅收集模型进行决策的数据，即下载决策的相关数据
            if sleep_time == 0. and not first_step:
                if download_twice:
                    at = (download_range_1 + download_range_2).item()
                else:
                    at = download_range.item()
                reward = cul_reward(at, bitrate, waste_bytes_after_download, rebuf, past_bandwidth[-1])
                r_batch.append(reward)

                if len(net_env.players) < 5:
                    done.append(1)
                else:
                    done.append(0)

            first_step = False

            if len(net_env.players) < 5 and len(done) > 0:
                done[-1] = 1
            # print(idx_agent, len(r_batch))
            if len(r_batch) >= TRAIN_SEQ_LEN : # or len(net_env.players) == 0: # 可以在这里加判定，没有视频就加上dones=1，把数据补全到一个batch

                exp_queue.put([s_batch[:],  # ignore the first chuck
                               raw_a_batch[:],
                               r_batch[:],  # control over it
                               v_batch[:],
                               log_prob_batch[:],
                               done[:]
                              ])

                del s_batch[:]
                del raw_a_batch[:]
                del r_batch[:]
                del v_batch[:]
                del log_prob_batch[:]
                del done[:]

                send_data_count += 1
                if send_data_count == batch_size:
                    # synchronize the network parameters from the coordinator
                    net_params = net_params_queue.get()
                    policy.load_state_dict(net_params)
                    send_data_count = 0

def mpc(all_future_chunks_size, P, buffer_size, last_quality, future_bandwidth):
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
        # print("combo:", combo)
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
            download_time = 1000. * (
                    all_future_chunks_size[chunk_quality][position] / 1000000.) / (
                                future_bandwidth)  # this is 1000. * MB/MB/s --> ms
            # print("download time:", download_time)
            if (curr_buffer < download_time):
                curr_rebuffer_time += (download_time - curr_buffer)
                curr_buffer = 0
            else:
                curr_buffer -= download_time
            curr_buffer += args.chunklength
            bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
            smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[pre_quality])
            pre_quality = chunk_quality
        # compute reward for this combination (one reward per 5-chunk combo)

        reward = alpha * (bitrate_sum / 1000.) - gamma * (smoothness_diffs / 1000.) - beta * (
                    curr_rebuffer_time / 1000.)
        # print(reward)
        # print(bitrate_sum, smoothness_diffs, curr_rebuffer_time, reward)

        if reward >= max_reward:
            best_combo = combo
            max_reward = reward
            send_data = best_combo[0]
            # print(bitrate_sum, smoothness_diffs, curr_rebuffer_time, reward)
    # print(best_combo, max_reward)
    # print('max_reward', max_reward, send_data)

    bit_rate = send_data
    return bit_rate

def fastmpc(TrainedOptimal, buffer_size, last_quality, future_bandwidth):
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
    VrateTmp = TrainedOptimal[min(preThroughput, max(BWList))][VrateList[last_quality]][min(buffer_level, max(bufferLevelList))]
    # print(VrateTmp)
    # 根据实际的参数，取出training的参数
    for bitrateIndex in range(len(VrateList)):
        if VrateTmp == VrateList[bitrateIndex]:
            return bitrateIndex

def load_trainedOptimal():
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
    return TrainedOptimal

def main(args):
    # np.random.seed(RANDOM_SEED)
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues, args))
    coordinator.start()

    all_cooked_time, all_cooked_bw = short_video_load_trace.load_trace(TRAIN_TRACES)
    work_agents = []
    for i in range(NUM_AGENTS):
        work_agents.append(mp.Process(target=work_agent,
                                      args=(i, all_cooked_time, all_cooked_bw, net_params_queues[i], exp_queues[i], args)))
    for i in range(NUM_AGENTS):
        work_agents[i].start()
    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main(args)