import sys, os
import time

import torch
from torch.distributions import Categorical

from Incendio_model import BM_Actor, BA_Actor, Critic, Trainer_RL
from replay_buffer import ReplayBuffer
from run_incendio import test_user_samples
from simulator import controller as env, short_video_load_trace

sys.path.append('./simulator/')
import argparse
import random
import numpy as np
import math
import multiprocess as mp

parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO")
parser.add_argument("--N", type=int, default=int(8), help=" number of agent")
parser.add_argument("--max_train_steps", type=int, default=int(1001), help=" Maximum number of training steps")
parser.add_argument("--episode_limit", type=int, default=100, help="Maximum number of steps per episode")
parser.add_argument("--evaluate_freq", type=float, default=10, help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--critic_pretrain_epoch", type=float, default=20, help="train critic before RL train start")
parser.add_argument("--norm_rew", type=bool, default=True, help="norm_rew")

parser.add_argument("--batch_size", type=int, default=8, help="Batch size (the number of episodes)")
parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
parser.add_argument("--K_epochs", type=int, default=1, help="GAE parameter")
parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
parser.add_argument("--use_value_clip", type=float, default=True, help="Whether to use value clip.")
parser.add_argument("--entropy_coef", type=float, default=0.005, help="Trick 5: policy entropy")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-8")
parser.add_argument("--use_lr_decay", type=float, default=False, help="")


parser.add_argument('--trace', type=str, default='sampled_4G_train', help='The network trace you are testing (fixed, high, low, medium, middle)')

parser.add_argument('--dataset_dir', type=str, default='../data/dataset_2s_train/', help='Is testing quickstart')
parser.add_argument('--chunklength', type=float, default=2000., help='')


args = parser.parse_args()

# QoE arguments
from config_algorithm import VIDEO_BIT_RATE
from config_algorithm import alpha, beta, gamma, theta

nn_model_save_path= './model_' + str(beta) + '/RL/'

ALL_VIDEO_NUM = 100
all_cooked_time = []
all_cooked_bw = []

# For training a3c
NUM_AGENTS = args.N
MODEL_SAVE_INTERVAL = args.evaluate_freq
TRAIN_SEQ_LEN = args.episode_limit
TRAIN_TRACES = '../data/network_traces/' + args.trace + '/'
DEFAULT_ID = 0
DEFAULT_BITRATE = 0
DEFAULT_SLEEP = 0
PAST_BW_LEN = 5
TAU = 200.

b_IN_B = 8
b_IN_kb = 1000

NN_MODEL_BM = './model_' + str(beta) + '/IL/' + 'bm_agent/bm_actor_epoch_100.pth'
NN_MODEL_BA = './model_' + str(beta) + '/IL/' + 'ba_agent/ba_actor_epoch_100.pth'
NN_MODEL_CRITIC = None

start_epoch = 1

USE_GPU = torch.cuda.is_available()
batch_size = args.batch_size

def test_model(epoch):
    with torch.no_grad():
        dataset_dir = '../data/dataset_2s_test'
        chunklength = 2000.

        os.system(f'python run_incendio.py '
                  f'--trace sampled_4G '
                  f'--dataset_dir {dataset_dir} '
                  f'--epoch {epoch} '
                  f'--chunklength {chunklength} '
                  f'--train_type RL')

def run_with_timeout(func, timeout, epoch, BM_MODEL_SAVE_PATH, BA_MODEL_SAVE_PATH):
    p = mp.Process(target=func, args=(epoch,))
    p.start()
    p.join()

    # if p.is_alive():
    #     p.terminate()
    #     p.join()
    #     print('=' * 20)
    #     print('time out')
    #     print('=' * 20)

def cul_reward(quality, rebuffer, smooth, bandwidth_usage, sleep_time):
    # 计算本次下载决策的奖励
    if sleep_time == 0 :
        reward = alpha * quality / 1000. - beta * rebuffer / 1000. - gamma * smooth / 1000. - theta * bandwidth_usage * 8. / 1000000.
    else:
        reward = - beta * rebuffer / 1000.
    return reward

def calculate_retention_probabilities(Players):
    # 计算每个视频块的保留概率 p_{i,m}(mc)
    retention_probs = []
    for player in Players:
        # 根据当前播放时间 mc 和用户留存率模型 H_{i,m} 计算
        mc = math.floor(player.play_timeline / args.chunklength)
        m = player.get_chunk_counter()
        p_i_m_mc = calculate_retention_probability(player, mc, m)
        retention_probs.append(p_i_m_mc)
    return retention_probs

def calculate_retention_probability(player, mc, m):
    if m >= player.get_chunk_sum():
        return 0.0
    # 实现保留概率的计算逻辑

    user_time, user_retent_rate = player.get_user_model()

    # 如果用户已经看完了，则留存率为0，即不需要在考虑下载该视频
    return float(user_retent_rate[m]) / float(user_retent_rate[mc])


def get_input_data(past_bandwidth, retention_probs, last_rebufs, Players, abs_cur_play_video_id):
    # 获取算法的输入状态
    bt = [bd * 8. for bd in past_bandwidth] # Mb/s

    lj = [retention_probs[i] for i in range(len(retention_probs))]
    while len(lj) < 5:
        lj.append(0.)

    gj = [Players[i].get_buffer_size() / 1000. for i in range(len(Players))] # norm 5s
    while len(gj) < 5:
        gj.append(0.)

    uj = []
    for i in range(len(Players)):
        if Players[i].get_remain_video_num() > 0:
            uj.append(np.average(Players[i].get_future_video_size(1)) * 8. / 1000000.) # Mb
        else:
            uj.append(0.)
    while len(uj) < 5:
        uj.append(0.)

    hj = last_rebufs[abs_cur_play_video_id: abs_cur_play_video_id + 5]
    while len(hj) < 5:
        hj.append(0.)

    qj = []
    for i in range(len(Players)):
        download_bitrate_i = Players[i].get_downloaded_bitrate()
        if len(download_bitrate_i) > 0:
            qj.append(VIDEO_BIT_RATE[download_bitrate_i[-1]] / max(VIDEO_BIT_RATE))
        else:
            qj.append(0.)
    while len(qj) < 5:
        qj.append(0.)

    fj = []
    for i in range(len(Players)):
        download_bitrate_i = Players[i].get_downloaded_bitrate()
        if len(download_bitrate_i) > 2:
            fj.append(abs(VIDEO_BIT_RATE[download_bitrate_i[-1]] - VIDEO_BIT_RATE[download_bitrate_i[-2]]) / max(VIDEO_BIT_RATE))
        else:
            fj.append(0.)
    while len(fj) < 5:
        fj.append(0.)

    return [bt, lj, gj, uj, hj, qj, fj]

def store_work_agent_data(replay_buffer, s_bm_batchs, s_ba_batchs, bts_btachs, a_bm_batchs, a_ba_batchs, r_batchs, v_batchs, bm_log_prob_batchs, ba_log_prob_batchs, dones_bm, dones_ba, bm_mask_batchs, ba_mask_batchs):
    # 将子agent执行的轨迹存储到经验缓冲区
    s_bm_batchs = torch.tensor(np.array(s_bm_batchs)).permute(1, 0, 2, 3).tolist() # 10,32,4,5
    s_ba_batchs = torch.tensor(np.array(s_ba_batchs)).permute(1, 0, 2, 3).tolist()
    bts_btachs = torch.tensor(np.array(bts_btachs)).permute(1, 0, 2, 3).tolist()
    # print(bts_btachs)
    a_bm_batchs = torch.tensor(np.array(a_bm_batchs)).permute(1, 0).tolist()
    a_ba_batchs = torch.tensor(np.array(a_ba_batchs)).permute(1, 0).tolist()
    r_batchs = torch.tensor(np.array(r_batchs)).permute(1, 0).tolist()
    v_batchs = torch.tensor(np.array(v_batchs)).permute(1, 0).tolist()
    bm_log_prob_batchs = torch.tensor(np.array(bm_log_prob_batchs)).permute(1, 0).tolist()
    ba_log_prob_batchs = torch.tensor(np.array(ba_log_prob_batchs)).permute(1, 0).tolist()
    dones_bm = torch.tensor(np.array(dones_bm)).permute(1, 0).tolist()
    dones_ba = torch.tensor(np.array(dones_ba)).permute(1, 0).tolist()
    bm_mask_batchs = torch.tensor(np.array(bm_mask_batchs)).permute(1, 0, 2, 3).tolist()
    ba_mask_batchs = torch.tensor(np.array(ba_mask_batchs)).permute(1, 0).tolist()

    # print(mask_batchs.shape, torch.tensor(np.array(s_bm_batchs)).shape)
    # print(len(s_bm_batchs), len(r_batchs), len(v_batchs),len(dones))
    for i in range(len(s_bm_batchs)):
        replay_buffer.store_transition(i, s_bm_batchs[i], s_ba_batchs[i], bts_btachs[i],
                                       v_batchs[i],
                                       a_bm_batchs[i], a_ba_batchs[i],
                                       bm_log_prob_batchs[i], ba_log_prob_batchs[i],
                                       r_batchs[i],
                                       dones_bm[i],
                                       dones_ba[i],
                                       bm_mask_batchs[i],
                                       ba_mask_batchs[i])
    replay_buffer.store_last_value(len(v_batchs) - 1, v_batchs[-1])

    # print('data restore, ', replay_buffer.episode_num)

def central_agent(net_params_queues, exp_queues, args):
    set_seed()
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    # 初始化模型
    bm_actor = BM_Actor()
    ba_actor = BA_Actor()

    critic = Critic()

    trainer_rl = Trainer_RL(args.lr, bm_actor, ba_actor, critic)

    if NN_MODEL_BM is not None and NN_MODEL_BA is not None:
        trainer_rl.load_model(NN_MODEL_BM, NN_MODEL_BA)
        print('BM BA AGENT MODEL LOAD')

    if NN_MODEL_CRITIC is not None:
        trainer_rl.load_critic(NN_MODEL_CRITIC)
        print('Critic MODEL LOAD')

    trainer_rl.Initial(args)

    if USE_GPU:
        trainer_rl.bm_actor = trainer_rl.bm_actor.cuda()
        trainer_rl.ba_actor = trainer_rl.ba_actor.cuda()
        trainer_rl.critic = trainer_rl.critic.cuda()

    # 获取模型参数并分发给子agent
    bm_actor_net_params, ba_actor_net_params, critic_net_params = trainer_rl.get_network_params()
    for i in range(NUM_AGENTS):
        net_params_queues[i].put([bm_actor_net_params, ba_actor_net_params, critic_net_params])

    replay_buffer = ReplayBuffer(NUM_AGENTS, TRAIN_SEQ_LEN, batch_size, args.norm_rew)
    replay_buffer.reset_buffer()

    epoch = start_epoch

    pre_reward_mean = -math.inf
    reward_mean = 0
    score_decay_count = 0

    while True:
        s_bm_batchs = []
        s_ba_batchs = []
        bts_btachs = []
        a_bm_batchs = []
        a_ba_batchs = []
        r_batchs = []
        v_batchs = []
        bm_log_prob_batchs = []
        ba_log_prob_batchs = []
        dones_bm = []
        dones_ba = []
        bm_mask_batch = []
        ba_mask_batch = []

        # 获取子agent轨迹
        for i in range(NUM_AGENTS):
            s_bm_batch, s_ba_batch, bts_btach, a_bm_batch, a_ba_batch, r_batch, v_batch, bm_log_prob_batch, ba_log_prob_batch, done_bm, done_ba, bm_mask, ba_mask = exp_queues[i].get()
            # print(mask)
            s_bm_batchs.append(s_bm_batch)
            s_ba_batchs.append(s_ba_batch)
            bts_btachs.append(bts_btach)
            a_bm_batchs.append(a_bm_batch)
            a_ba_batchs.append(a_ba_batch)
            r_batchs.append(r_batch)
            v_batchs.append(v_batch)
            bm_log_prob_batchs.append(bm_log_prob_batch)
            ba_log_prob_batchs.append(ba_log_prob_batch)
            dones_bm.append(done_bm)
            dones_ba.append(done_ba)
            bm_mask_batch.append(bm_mask)
            ba_mask_batch.append(ba_mask)

            reward_mean += np.mean(r_batch)

        # 将轨迹存储到经验缓冲区
        store_work_agent_data(replay_buffer, s_bm_batchs, s_ba_batchs, bts_btachs, a_bm_batchs,
                              a_ba_batchs, r_batchs, v_batchs, bm_log_prob_batchs, ba_log_prob_batchs, dones_bm, dones_ba,
                              bm_mask_batch, ba_mask_batch)
        if replay_buffer.episode_num == batch_size:

            entropy_decay = False
            if reward_mean < pre_reward_mean:
                score_decay_count += 1
                if score_decay_count == 100:
                    score_decay_count = 0
                    entropy_decay = True
            pre_reward_mean = reward_mean
            reward_mean = 0

            print('=' * 20, 'training of epoch ', epoch, '=' * 20)
            # 抽样，更新
            # torch.autograd.set_detect_anomaly(True)
            trainer_rl.train(epoch, replay_buffer)
            # 更新完毕，清楚缓冲
            replay_buffer.reset_buffer()
            if epoch % MODEL_SAVE_INTERVAL == 0 and epoch > args.critic_pretrain_epoch:
                print("---------epoch %d--------" % epoch)
                # Save the neural net parameters to disk.
                BM_MODEL_SAVE_PATH = nn_model_save_path + 'bm_agent/bm_actor_epoch_' + str(epoch) + '.pth'
                BA_MODEL_SAVE_PATH = nn_model_save_path + 'ba_agent/ba_actor_epoch_' + str(epoch) + '.pth'

                CRITIC_SAVE_PATH = nn_model_save_path + 'critic/critic_epoch_' + str(epoch) + '.pth'

                trainer_rl.save_model(BM_MODEL_SAVE_PATH, BA_MODEL_SAVE_PATH)

                trainer_rl.save_critic(CRITIC_SAVE_PATH)

                run_with_timeout(test_model, 600, epoch, BM_MODEL_SAVE_PATH, BA_MODEL_SAVE_PATH)

                if entropy_decay:
                    trainer_rl.entropy_decay()

            # 分发模型参数
            bm_actor_net_params, ba_actor_net_params, critic_net_params = trainer_rl.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put([bm_actor_net_params, ba_actor_net_params, critic_net_params])

            epoch += 1

        if epoch == args.max_train_steps:
            sys.exit(0)


def work_agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue, args):
    set_seed()
    with torch.no_grad():
        bm_actor = BM_Actor()
        ba_actor = BA_Actor()
        critic = Critic()

        trainer = Trainer_RL(args.lr, bm_actor, ba_actor, critic)
        trainer.Initial(args)

        bm_actor_net_params, ba_actor_net_params, critic_net_params = net_params_queue.get()

        trainer.set_network_params(bm_actor_net_params, ba_actor_net_params, critic_net_params)

        if USE_GPU:
            trainer.bm_actor = trainer.bm_actor.cuda()
            trainer.ba_actor = trainer.ba_actor.cuda()
            trainer.critic = trainer.critic.cuda()

        # 初始化 state, action, reward batch
        s_bm_batch = []
        s_ba_batch = []
        bts_batch = []
        a_bm_batch = []
        a_ba_batch = []
        bm_log_prob_batch = []
        ba_log_prob_batch = []
        r_batch = []
        v_batch = []
        dones_batch_bm = []
        dones_batch_ba = []
        bm_mask_batch = []
        ba_mask_batch = []

        pre_download_video = None
        last_rebufs = [0.] * 100
        last_chunk_bitrate = [-1] * 100
        past_bandwidth = list(np.zeros(PAST_BW_LEN))

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
        while True:
            if len(net_env.players) < 5:
                # 用户退出，重新初始化环境以及相关参数
                # print('env changed')
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

                past_bandwidth = list(np.zeros(PAST_BW_LEN))
                last_rebufs = [0.] * 100
                last_chunk_bitrate = [-1] * 100

            user_rets = calculate_retention_probabilities(net_env.players)

            state_actor = get_input_data(past_bandwidth, user_rets, last_rebufs, net_env.players, play_video_id)

            inputs = torch.tensor(state_actor).reshape(1, 7, 5).float()

            bts = []
            for input in inputs:
                bts.append(list(input[0]))
            bts = torch.tensor(bts).reshape(len(bts), 5, 1)

            states_bm = inputs[:, 0:4, :].reshape(inputs.shape[0], 1, 4, 5)
            states_ba = inputs
            bts = bts
            # print(states_bm.shape, states_ba.shape, bts.shape)

            s_bm_batch.append(states_bm.tolist()[0][0])
            # print(states_bm.shape,states_ba.shape, bts.shape)
            # print(bts.tolist())
            # print(s_bm_batch)
            s_ba_batch.append(states_ba.tolist()[0])
            # print(s_ba_batch)
            bts_batch.append(bts.tolist()[0])
            # print(bts.shape)

            if USE_GPU:
                states_bm, states_ba, bts = states_bm.cuda(), states_ba.cuda(), bts.cuda()

            # Decide the actions for the next step
            pi_video, bm_hidden_state = trainer.bm_actor(states_bm, bts)

            # print(pi_video)
            # print(torch.argmax(pi_video))
            # 4. 决策输出，如果没有合适的块可供下载，则返回睡眠时间
            sleep_time = 0.

            # IAM 无效动作掩码
            mask = torch.ones_like(pi_video)
            for i in range(5):
                if net_env.players[i].get_remain_video_num() == 0:
                    mask[0][i] = 0.
            if net_env.players[0].buffer_size == 0. and net_env.players[0].get_remain_video_num() != 0:
                mask[0][0] = 1.
                mask[0][1] = mask[0][2] = mask[0][3] = mask[0][4] = mask[0][5] = 0.
            # print(mask)
            bm_mask_batch.append(mask.tolist())
            pi_video.masked_fill_(mask == 0., -float('inf'))
            # print(pi_video)
            dist_pi = Categorical(logits=pi_video)

            a_bm = dist_pi.sample()

            if a_bm.item() == 5:
                a_bm_logprob = dist_pi.log_prob(a_bm)
                sleep_time = TAU
            else:
                a_bm_logprob = dist_pi.log_prob(a_bm)
                download_video_id = a_bm.item() + play_video_id

            a_bm_batch.append(a_bm.item())
            # print(pi_video)
            # print(pi_video.shape)
            bm_log_prob_batch.append(a_bm_logprob.item())

            # bm_hidden_state = bm_actor.last_hidden_output
            # print(a_bm, bm_log_prob_batch)

            pi_bitrate, ba_hidden_state = trainer.ba_actor(states_ba, bts)

            dist = Categorical(logits=pi_bitrate)
            bit_rate = dist.sample()
            a_ba_logprob = dist.log_prob(bit_rate)

            bit_rate = bit_rate.item()

            # print(pi_bitrate.shape)
            # ba_log_prob_batch.append(a_ba_logprob.item())
            # print(bit_rate, ba_log_prob_batch)

            if a_bm.item() != 5:
                a_ba_batch.append(bit_rate)
                ba_log_prob_batch.append(a_ba_logprob.item())
                ba_mask_batch.append(1.0)
            else:
                bit_rate = 0
                a_ba_batch.append(5)
                ba_log_prob_batch.append(0.)
                ba_mask_batch.append(0.)

            # ba_hidden_state = ba_actor.last_hidden_output

            # print(bm_hidden_state.shape, ba_hidden_state.shape)
            critic_input = torch.concat([bm_hidden_state, ba_hidden_state], dim=1)
            # print(critic_input.shape)

            value = trainer.critic(critic_input)

            v_batch.append(value.item())

            # 计算上一步的reward
            quality = 0
            smooth = 0
            if sleep_time == 0:
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
                    smooth += get_smooth(net_env, last_chunk_bitrate, download_video_id, download_chunk, quality)
                quality = quality * user_rets[a_bm.item()]
                smooth = smooth * user_rets[a_bm.item()]
            # print(download_video_id, bit_rate, sleep_time)

            # 环境模型下载过程
            delay, rebuf, video_size, end_of_video, \
            play_video_id, waste_bytes, rtt = net_env.buffer_management(download_video_id, bit_rate, sleep_time)

            if sleep_time == 0:
                last_rebufs[download_video_id] = rebuf / 1000.

                past_bandwidth = np.roll(past_bandwidth, -1)
                past_bandwidth[-1] = (float(video_size) / 1000000.0) / (float(delay) / 1000.0)  # MB / s

            bitrate_usage = video_size

            reward = cul_reward(quality, rebuf, smooth, bitrate_usage, sleep_time)

            r_batch.append(reward)

            if len(net_env.players) < 5:
                dones_batch_bm.append(1)
                # dones_batch_ba.append(1)
            else:
                dones_batch_bm.append(0)
                # dones_batch_ba.append(0)

            if sleep_time == 0 and len(net_env.players) >= 5:
                dones_batch_ba.append(0)
            else:
                dones_batch_ba.append(1)

            if len(r_batch) >= TRAIN_SEQ_LEN : # or len(net_env.players) == 0: # 可以在这里加判定，没有视频就加上dones=1，把数据补全到一个batch

                user_rets = calculate_retention_probabilities(net_env.players)
                state_actor = get_input_data(past_bandwidth, user_rets, last_rebufs, net_env.players, play_video_id)
                # print(state_actor)
                inputs = torch.tensor(state_actor).reshape(1, 7, 5).float()
                bts = []
                for input in inputs:
                    bts.append(list(input[0]))
                bts = torch.tensor(bts).reshape(len(bts), 5, 1)

                states_bm = inputs[:, 0:4, :].reshape(inputs.shape[0], 1, 4, 5)
                states_ba = inputs
                bts = bts

                if USE_GPU:
                    states_bm, states_ba, bts = states_bm.cuda(), states_ba.cuda(), bts.cuda()

                # Decide the actions for the next step
                _, bm_hidden_state = trainer.bm_actor(states_bm, bts)
                _, ba_hidden_state = trainer.ba_actor(states_ba, bts)

                # print(bm_hidden_state.shape, ba_hidden_state.shape)
                critic_input = torch.concat([bm_hidden_state, ba_hidden_state], dim=1)
                # print(critic_input.shape)

                value = trainer.critic(critic_input)

                v_batch.append(value.item())

                exp_queue.put([s_bm_batch[:],  # ignore the first chuck
                               s_ba_batch[:],
                               bts_batch[:],
                               a_bm_batch[:],  # since we don't have the
                               a_ba_batch[:],
                               r_batch[:],  # control over it
                               v_batch[:],
                               bm_log_prob_batch[:],
                               ba_log_prob_batch[:],
                               dones_batch_bm[:],
                               dones_batch_ba[:],
                               bm_mask_batch[:],
                               ba_mask_batch[:]])

                del s_bm_batch[:]
                del s_ba_batch[:]
                del bts_batch[:]
                del a_bm_batch[:]
                del a_ba_batch[:]
                del r_batch[:]
                del v_batch[:]
                del bm_log_prob_batch[:]
                del ba_log_prob_batch[:]
                del dones_batch_bm[:]
                del dones_batch_ba[:]
                del bm_mask_batch[:]
                del ba_mask_batch[:]

                send_data_count += 1
                if send_data_count == batch_size:
                    # synchronize the network parameters from the coordinator
                    bm_actor_net_params, ba_actor_net_params, critic_net_params = net_params_queue.get()
                    trainer.set_network_params(bm_actor_net_params, ba_actor_net_params, critic_net_params)
                    send_data_count = 0


def get_smooth(net_env, last_chunk_bitrate, download_video_id, chunk_id, quality):
    if download_video_id == 0 and chunk_id == 0:  # is the first chunk of all
        return 0
    if chunk_id == 0:  # needs to find the last chunk of the last video
        last_bitrate = last_chunk_bitrate[download_video_id - 1]
        if last_bitrate == -1:  # the neighbour chunk is not downloaded
            return 0
    else:
        last_bitrate = net_env.players[download_video_id - net_env.get_start_video_id()].get_downloaded_bitrate()[chunk_id - 1]
    return abs(quality - VIDEO_BIT_RATE[last_bitrate])

def set_seed():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

def main(args):

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues, args))
    coordinator.start()

    all_cooked_time, all_cooked_bw = short_video_load_trace.load_trace(TRAIN_TRACES)
    work_agents = []
    for i in range(NUM_AGENTS):
        work_agents.append(mp.Process(target=work_agent,
                                      args=(i % 5, all_cooked_time, all_cooked_bw, net_params_queues[i], exp_queues[i], args)))
    for i in range(NUM_AGENTS):
        work_agents[i].start()
    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main(args)