import math
from distutils.command.bdist import bdist

from scipy.linalg import bandwidth

from algorithms.secbad.algorithms.a2c import A2C
from algorithms.secbad.algorithms.ppo import PPO
from algorithms.secbad.models.policy import Policy

from algorithms.secbad.algorithms.dqn import DQN
from algorithms.secbad.models.policy_dqn import DQNPolicy
from algorithms.secbad.algorithms.sac import SAC

from environments.env_utils import VectorEnv
import os
import time

from utils.hidden_recoder import HiddenRecoder
# import non_envs

import gym
import numpy as np
import torch

from algorithms.adaptive_online_storage import AdaptiveOnlineStorage

from utils import helpers as utl
from utils.tb_logger import TBLogger
from vae import VaribadVAE

from scipy.stats import norm


class MixedLearner:
    """
    Meta-Learner class with the main training loop for variBAD.
    """

    def __init__(self, args):

        self.args = args
        self.device = args.device
        utl.seed(self.args.seed)

        # 计算总更新次数
        self.num_updates = int(
            args.num_frames) // args.policy_num_steps // args.num_processes
        self.frames = 0
        self.iter_idx = 0

        # 一个封装了tensorboard的summarwriter的类，用于记录训练过程中的相关数据
        if self.args.train:
            self.logger = TBLogger(self.args, self.args.exp_label)
            self.logs_detail_performance = open(os.path.join(self.logger.full_output_folder, 'logs.txt'), 'w')
        else:
            self.logger = None
            self.full_output_folder = './logs/secbad_test/logs_short_video_env/'
            if not os.path.exists(self.full_output_folder):
                os.makedirs(self.full_output_folder)
            self.logs_detail_performance = open(self.full_output_folder + '/logs.txt', 'a')

        # 初始化向量化环境，参数包括环境名、并行的环境数量、traj_len和是否归一化reward
        self.envs = VectorEnv(encoder_input_feature=self.args.encoder_input_feature,
                              n_env=self.args.num_processes, max_buffer_size=self.args.max_buffer_size,
                              dataset_path=self.args.dataset_path, dataset_path_test=self.args.dataset_path_test,
                              network_traces_path=self.args.network_traces_path,
                              network_traces_path_test=self.args.network_traces_path_test,
                              chunklength=self.args.chunklength, train=self.args.train)

        env_paras = self.envs.get_env_paras()
        # print(env_paras)
        self.args.num_feature = env_paras['num_feature']
        self.args.action_dim = env_paras['dim_action']
        self.args.num_action = env_paras['num_action']
        self.args.state_dim = env_paras['dim_state']
        self.args.action_space = env_paras['action_space']

        # VAE是指encoder-decoder架构，同时还有计算loss的方法
        self.vae = VaribadVAE(
            self.args, self.logger, lambda: self.iter_idx, self.num_updates)
        # replay buffer，存储历史轨迹以训练PPO
        self.policy_storage = self.initialise_policy_storage()
        # 这里初始化PPO的网络
        self.policy = self.initialise_policy()

        if self.args.norm_rew_for_policy:
            self.envs.rew_rms = utl.RunningMeanStd(self.device, shape=(1,))

    def train(self):
        """ Main Meta-Training loop """
        start_time = time.time()

        # 这里获得了envs中每个长度为traj_len的env的上下文，以ant_dir为例，这里的上下文就是目标的方向
        # 从ant_dir的get_traj_context的代码来看，这里的获得的上下文是随机生成的固定的值
        if self.args.seed is not None:
            np.random.seed(self.args.seed)
        seed = np.random.randint(1000, size=self.envs.n_env).tolist()
        traj_context = [self.envs.get_traj_context() for i in range(self.envs.n_env)]
        # print(traj_context)
        # 调用envs.reset会根据随机种子初始化环境，获得多个初始状态不同的环境
        prev_state = self.envs.reset(seed=seed, traj_context_ls=traj_context)[0]
        prev_state_encoder = self.envs.get_train_env_encoder_state()
        # print(prev_state)
        # insert initial observation / embeddings to rollout storage
        self.policy_storage.prev_state[0].copy_(torch.tensor(prev_state))

        with torch.no_grad():
            print('log once before trainging')
            # self.log(start_time)

        hidden_recs = [HiddenRecoder(self.device, self.vae.encoder, self.args.max_input_history_length) for _ in range(self.envs.n_env)]
        for hidden_rec in hidden_recs:
            hidden_rec.encoder_init(0)

        reward_decoder = self.vae.reward_decoder

        step_idxs = [0 for _ in range(self.envs.n_env)]

        download_suspend = self.envs.get_sleep_judge_train_env()

        latent_mean, latent_logvar, hidden_state = self.vae.encoder.prior(self.envs.n_env)
        latent_mean = latent_mean.squeeze()
        latent_logvar = latent_logvar.squeeze()
        # print(latent_mean.shape, latent_logvar.shape, hidden_state.shape)
        # print(self.vae.encoder.state_encoder.fc1s)
        for self.iter_idx in range(self.num_updates):
            print('=' * 10, self.iter_idx, '/', self.num_updates, '=' * 10)

            # First, re-compute the hidden states given the current rollouts (since the VAE might've changed)
            # 通过RNN结构的encoder,提取隐藏的特征向量，然后存到policy_storage里

            # add this initial hidden state to the policy storage
            # make sure we emptied buffers
            if self.args.pass_latent_to_policy:

                assert len(self.policy_storage.latent_mean) == 0
                self.policy_storage.latent_mean.append(latent_mean.clone())
                self.policy_storage.latent_logvar.append(latent_logvar.clone())

            # rollout policies for a few steps
            # 和 env 交互，用得到的 trajectory 来算 encoding，并且把 policy_num_steps 这么多步的 encoding, action, 各种信息都存到 policy_storage 里面
            # XXX truncate 之后还需要用下面的 for loop 来算 encoding 么？
            # self.args.policy_num_steps 这个数 还是得大一点，起码在这么多次step内，env的task要变一次
            # 主要是这个循环得到的 encoding 后面 updating 的时候是怎么用的
            # 接下来for循环的代码主要就是为了收集经验
            curr_step = 0
            for step in range(self.args.policy_num_steps):
                # print(step)
                # print(prev_state)
                # sample actions from policy
                # 采样动作，这里deterministic=False，就指是根据概率随机sample的动作
                mask = self.envs.get_train_env_mask().to(self.device)
                # print(mask)
                # print(torch.all(mask == True, dim=-1).sum())

                with torch.no_grad():
                    # value is the return predicted by the policy, action is the action suggested by the policy
                    # print( latent_mean.shape, latent_logvar.shape, hidden_state.shape)
                    value, action = utl.select_action(
                        args=self.args,
                        policy=self.policy,
                        state=torch.tensor(prev_state).float().to(self.device),
                        mask=mask,
                        deterministic=False,
                        latent_mean=latent_mean,
                        latent_logvar=latent_logvar,
                    )
                # print(action, action.shape)
                # take step in the environment
                # 执行随机sample的动作，获得reward和接下来的state
                # print(self.envs.train_env_step(action, self.args))
                next_state, (rew_raw, rew_normalized), done, infos = self.envs.train_env_step(action)
                # next_state_encoder = self.envs.get_train_env_encoder_state()
                # print(rew_normalized.min(), rew_normalized.max())

                # reward prediction of vae
                if self.args.norm_rew_for_policy and self.args.pass_norm_rew_to_vae:
                    rew_vae = rew_normalized
                else:
                    rew_vae = rew_raw

                download_suspend = self.envs.get_sleep_judge_train_env()

                for i in range(self.envs.n_env):
                    while download_suspend[i]:
                        action_i = torch.as_tensor([30])

                        next_state_i, (_, _), done_i, _ = self.envs.train_env_i_step(action_i, i)

                        next_state[i] = next_state_i
                        done[i] = done_i
                        if done_i:
                            # print(next_state_i.shape)
                            next_state_i = self.envs.reset_train_i(i)[0]
                            # print(next_state_i.shape)
                            next_state[i] = next_state_i

                        download_suspend[i] = self.envs.get_sleep_judge_train_env_i(i)

                next_state_encoder = self.envs.get_train_env_encoder_state()
                # 将训练和推理变为一致
                r_t = None
                if self.args.pass_latent_to_policy:
                    latent_mean, latent_logvar = [], []
                    r_t = []
                    for i in range(self.envs.n_env):

                        hidden_rec = hidden_recs[i]

                        step_idxs[i] += 1
                        step_idx = step_idxs[i]
                        # print(action, next_state, rew_raw)
                        next_state_encoder_i = next_state_encoder[i].unsqueeze(0)
                        next_state_policy_i = torch.tensor(next_state[i]).unsqueeze(0)
                        # print(state.shape, rew.shape, act.shape)
                        # print(next_state_encoder_i.shape)
                        curr_latent_mean, curr_latent_logvar, best_unchange_length, pred_rew_ct = self.inference(
                            hidden_rec, next_state_encoder_i, next_state_policy_i, step_idx, reward_decoder)

                        latent_mean.append(curr_latent_mean)
                        latent_logvar.append(curr_latent_logvar)
                        r_t.append(torch.tensor(best_unchange_length))
                    latent_mean = torch.stack(latent_mean).squeeze()
                    latent_logvar = torch.stack(latent_logvar).squeeze()
                    r_t = torch.stack(r_t).unsqueeze(-1)
                    # print(r_t.shape)

                # print(done)
                # 从infos里提取r_t的值
                done = torch.from_numpy(np.array(done, dtype=int)).to(
                    self.device).float().view((-1, 1))

                if self.args.pass_latent_to_policy:
                    for i in range(self.envs.n_env):
                        done_flag = done[i]
                        if done_flag != 0: # 表示环境reset
                            step_idxs[i] = 0
                            hidden_recs[i] = HiddenRecoder(self.device, self.vae.encoder, self.args.max_input_history_length)
                            curr_latent_mean, curr_latent_logvar = hidden_recs[i].encoder_init(0)
                            latent_mean[i] = curr_latent_mean
                            latent_logvar[i] = curr_latent_logvar

                # # create mask for episode ends
                masks_done = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done]).to(self.device)

                # before resetting, update the embedding and add to vae buffer
                # 收集经验存储到了rollout_storage、vae_buffer(用于计算loss)和policy_storage
                # print(prev_state.shape)
                # print(prev_state)
                if self.args.pass_latent_to_policy:
                    self.vae.rollout_storage.insert(prev_state_encoder.clone(),
                                                    action.detach().clone(),
                                                    next_state_encoder.clone(),
                                                    torch.tensor(prev_state).clone(),
                                                    torch.tensor(next_state).clone(),
                                                    torch.tensor(rew_vae).clone().unsqueeze(-1),
                                                    done.clone(),
                                                    mask,
                                                    r_ts=r_t.clone() if self.args.learner_type == 'secbad' else None)

                curr_step += 1
                # add the obs before reset to the policy storage
                self.policy_storage.next_state[step] = torch.tensor(next_state)

                self.policy_storage.insert(
                    state=torch.tensor(next_state),
                    actions=action,
                    rewards_raw=torch.tensor(rew_raw).unsqueeze(-1),
                    # 6-17 use rew_raw as rewards_normalised for now
                    rewards_normalised=torch.tensor(rew_normalized).unsqueeze(
                        -1) if self.args.norm_rew_for_policy else None,
                    value_preds=value,
                    mask=mask,
                    masks=masks_done,
                    r_t=r_t if self.args.learner_type == 'secbad' else None,
                    latent_mean=latent_mean,
                    latent_logvar=latent_logvar,
                )

                prev_state = next_state
                prev_state_encoder = next_state_encoder

                self.frames += self.args.num_processes

            # 用收集到的经验，更新模型的参数
            # --- UPDATE ---
            # self.args.precollect_len 是为了在 self.vae.rollout_storage 里面记录 self.args.precollect_len 这么多个 trajectory，
            # 搜集了足够多的 traj 后，再进行 update
            if self.frames >= self.args.precollect_len:
                print('args updating')
                # 这里用的 prev_state, belief, latent_mean, latent_logvar 都是上一个 loop 的结果
                self.update(state=torch.tensor(prev_state).float().to(self.device),
                            latent_mean=latent_mean,
                            latent_logvar=latent_logvar)

                # log
                with torch.no_grad():
                    self.log(start_time)

            # clean up after update
            # 把 policy_storage 的 latent_samples, latent_mean, latent_logvar 都清空
            # 但是 actions, prev_state, rewards_raw, value_preds 还留着，size 仍为 self.args.policy_num_steps，只不过如果 policy_storage 满了，在 insert 的时候是会从头覆盖原有的记录的
            self.policy_storage.after_update()

        self.logs_detail_performance.close()
        self.envs.close()

    def initialise_policy_storage(self):
        # AdaptiveOnlineStorage 记录的是 num_processes 这么多个 trajectory 从开始到结尾的 state latent belief 这些
        return AdaptiveOnlineStorage(args=self.args,
                                     num_steps=self.args.policy_num_steps,
                                     num_processes=self.args.num_processes,
                                     num_feature=self.args.num_feature,
                                     state_dim=self.args.state_dim,
                                     latent_dim=self.args.latent_dim,
                                     action_space=self.args.action_space,
                                     hidden_size=self.args.encoder_gru_hidden_size,
                                     normalise_rewards=self.args.norm_rew_for_policy,
                                     )

    def initialise_policy(self):
        if self.args.policy in ['a2c', 'ppo']:
            # initialise policy network
            policy_net = Policy(
                args=self.args,
                pass_state_to_policy=self.args.pass_state_to_policy,
                pass_latent_to_policy=self.args.pass_latent_to_policy,
                num_feature=self.args.num_feature,
                dim_state=self.args.state_dim,
                dim_latent=self.args.latent_dim * 2,
                hidden_layers=self.args.policy_layers,
                activation_function=self.args.policy_activation_function,
                action_space=self.envs.get_env_paras()['action_space'],
            ).to(self.device)

            # initialise policy trainer
            if self.args.policy == 'a2c':
                policy = A2C(
                    self.args,
                    policy_net,
                    self.args.policy_value_loss_coef,
                    self.args.policy_entropy_coef,
                    policy_optimiser=self.args.policy_optimiser,
                    policy_anneal_lr=self.args.policy_anneal_lr,
                    train_steps=self.num_updates,
                    optimiser_vae=self.vae.optimiser_vae,
                    lr=self.args.lr_policy,
                    eps=self.args.policy_eps,
                )
            elif self.args.policy == 'ppo':
                policy = PPO(
                    self.args,
                    policy_net,
                    self.args.policy_value_loss_coef,
                    self.args.policy_entropy_coef,
                    policy_optimiser=self.args.policy_optimiser,
                    policy_anneal_lr=self.args.policy_anneal_lr,
                    train_steps=self.num_updates,
                    lr=self.args.lr_policy,
                    eps=self.args.policy_eps,
                    ppo_epoch=self.args.ppo_num_epochs,
                    num_mini_batch=self.args.ppo_num_minibatch,
                    use_huber_loss=self.args.ppo_use_huberloss,
                    use_clipped_value_loss=self.args.ppo_use_clipped_value_loss,
                    clip_param=self.args.ppo_clip_param,
                    optimiser_vae=self.vae.optimiser_vae,
                )
        elif self.args.policy == 'dqn':
            q_net = DQNPolicy(
                # input
                args=self.args,
                pass_state_to_policy = self.args.pass_state_to_policy,
                pass_latent_to_policy = self.args.pass_latent_to_policy,
                num_feature = self.args.num_feature,
                dim_state = self.args.state_dim,
                dim_latent = self.args.latent_dim * 2,
                hidden_layers = self.args.policy_layers,
                activation_function = self.args.policy_activation_function,
                action_space = self.envs.get_env_paras()['action_space'],
            ).to(self.device)
            target_q_net = DQNPolicy(
                # input
                args=self.args,
                pass_state_to_policy = self.args.pass_state_to_policy,
                pass_latent_to_policy = self.args.pass_latent_to_policy,
                num_feature = self.args.num_feature,
                dim_state = self.args.state_dim,
                dim_latent = self.args.latent_dim * 2,
                hidden_layers = self.args.policy_layers,
                activation_function = self.args.policy_activation_function,
                action_space = self.envs.get_env_paras()['action_space'],
            ).to(self.device)
            policy = DQN(
                args=self.args,
                q_net=q_net,
                target_q_net=target_q_net,
                gamma=self.args.dqn_gamma,
                lr=self.args.lr_policy,
                target_update_interval=self.args.target_update_interval,
                eps=self.args.policy_eps,
                get_iter_idx=lambda: self.iter_idx,
                train_steps=self.num_updates,
                policy_anneal_lr=self.args.policy_anneal_lr,
            )
        elif self.args.policy == 'sac':
            policy = SAC(
                args=self.args,
                dim_state=self.args.state_dim,
                num_feature=self.args.num_feature,
                dim_latent=self.args.latent_dim * 2,
                hidden_layers = self.args.policy_layers,
                activation_function = self.args.policy_activation_function,
                dim_action=self.envs.get_env_paras()['action_space'].n,
                actor_lr=self.args.actor_lr_sac,
                critic_lr=self.args.critic_lr_sac,
                alpha_lr=self.args.alpha_lr_sac,
                target_entropy=self.args.target_entropy_sac,
                tau=self.args.tau_sac,
                gamma=self.args.gamma_sac,
            )
        else:
            raise NotImplementedError

        return policy

    # def get_value(self, state, belief, latent_mean, latent_logvar):
    def get_value(self, state, latent_mean, latent_logvar):
        # get_latent_for_policy按照 args 里 agent 能看到哪些东西 把 latent_mean, latent_logvar 拼起来
        latent = utl.get_latent_for_policy(
            self.args, latent_mean=latent_mean, latent_logvar=latent_logvar)
        # 然后就是用 policy 的 V-network 算 value
        return self.policy.actor_critic.get_value(state=state, latent=latent).detach()

    def update(self, state, latent_mean, latent_logvar):
        # update policy (if we are not pre-training, have enough data in the vae buffer, and are not at iteration 0)
        if self.args.policy in ['a2c', 'ppo']:
            # bootstrap next value prediction
            with torch.no_grad():
                next_value = self.get_value(state=state,
                                            latent_mean=latent_mean,
                                            latent_logvar=latent_logvar)
            # compute returns for current rollouts
            # 按照gae的方法算td reward，把计算得到的结果保存为self.policy_storage.returns
            self.policy_storage.compute_returns(next_value, self.args.policy_use_gae, self.args.policy_gamma,
                                                self.args.policy_tau)
            # update agent (this will also call the VAE update!)
            # 参数更新，policy和encoder-decoder
            self.policy.update(
                policy_storage=self.policy_storage)

        elif self.args.policy == 'dqn':
            self.policy.update(
                policy_storage=self.policy_storage,
                rollout_storage=self.vae.rollout_storage,
                encoder=self.vae.encoder,
            )
        elif self.args.policy == 'sac':
            self.policy.update(
                policy_storage=self.policy_storage,
                rollout_storage=self.vae.rollout_storage,
                encoder=self.vae.encoder,
            )

        if self.args.pass_latent_to_policy:
            for _ in range(self.args.num_vae_updates):
                self.vae.compute_vae_loss(update=True)
            if self.vae.anneal_lr:
                self.vae.lr_scheduler.step()


    def log(self, start_time):
        # --- save models ---
        if (self.iter_idx) % self.args.save_interval == 0:
            save_path = os.path.join(self.logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            idx_labels = ['']
            if self.args.save_intermediate_models:
                idx_labels.append(int(self.iter_idx))

            for idx_label in idx_labels:
                if self.args.policy in ['ppo', 'a2c']:
                    torch.save(self.policy.actor_critic, os.path.join(
                        save_path, f"policy{idx_label}.pt"))

                    if self.policy.actor_critic.pass_state_to_policy and self.policy.actor_critic.norm_state:
                        utl.save_obj(self.policy.actor_critic.state_rms,
                                     save_path, f"env_state_rms{idx_label}")
                    if self.policy.actor_critic.pass_latent_to_policy and self.policy.actor_critic.norm_latent:
                        utl.save_obj(self.policy.actor_critic.latent_rms,
                                     save_path, f"env_latent_rms{idx_label}")
                elif self.args.policy == 'dqn':
                    torch.save(self.policy.q_net, os.path.join(
                        save_path, f"q_net{idx_label}.pt"))
                    torch.save(self.policy.target_q_net, os.path.join(
                        save_path, f"target_q_net{idx_label}.pt"))

                    if self.policy.q_net.pass_state_to_policy and self.policy.q_net.norm_state:
                        utl.save_obj(self.policy.q_net.state_rms,
                                     save_path, f"env_state_rms{idx_label}")
                    if self.policy.q_net.pass_latent_to_policy and self.policy.q_net.norm_latent:
                        utl.save_obj(self.policy.q_net.latent_rms,
                                     save_path, f"env_latent_rms{idx_label}")
                elif self.args.policy == 'sac':
                    torch.save(self.policy.actor, os.path.join(
                        save_path, f"actor{idx_label}.pt"))
                    torch.save(self.policy.target_critic_1, os.path.join(
                        save_path, f"target_critic1_{idx_label}.pt"))
                    torch.save(self.policy.target_critic_2, os.path.join(
                        save_path, f"target_critic2_{idx_label}.pt"))

                torch.save(self.vae.encoder, os.path.join(
                    save_path, f"encoder{idx_label}.pt"))
                if self.vae.reward_decoder is not None:
                    torch.save(self.vae.reward_decoder, os.path.join(
                        save_path, f"reward_decoder{idx_label}.pt"))

                if self.args.norm_rew_for_policy:
                    utl.save_obj(self.envs.rew_rms, save_path, f"env_rew_rms{idx_label}")
        # --- visualise behaviour of policy ---

        if (self.iter_idx) % self.args.eval_interval == 0:
            args = self.args

            # log the return avg/std across tasks (=processes)
            qoe_users = []
            quality_users = []
            smooth_users = []
            rebuf_users = []
            bd_wastage_users = []

            pred_rew_mae_users = []
            # bd_sum_users = []

            all_detail_performance_folder = self.logger.full_output_folder + '/detailed_log/'

            all_detail_performance_folder_epoch = all_detail_performance_folder + str(self.iter_idx) + '/'
            if not os.path.exists(all_detail_performance_folder_epoch):
                os.makedirs(all_detail_performance_folder_epoch)


            for user in range(1):
                all_detail_performance = open(all_detail_performance_folder_epoch + 'log_' + str(user) + '.txt', 'w')

                print(f"========== user {user} ==========")
                qoe_traces = []
                quality_traces = []
                smooth_traces = []
                rebuf_traces = []
                bd_wastage_traces = []

                pred_rew_mae_traces = []
                # bd_sum_traces = []

                for trace in range(len(os.listdir(self.envs.test_env.network_traces_path))):

                    selected_history_length = [0] * (self.args.max_input_history_length + 1)

                    print(f'========== user {user} trace {trace} ==========')
                    vis_context = [user, trace]

                    qoe_all = []
                    quality_all = []
                    smooth_all = []
                    rebuf_all = []
                    bd_wastage_all = []

                    pred_rew_mae_all = []
                    # bd_sum = None

                    state = self.envs.reset_test(vis_context)[0]
                    state = torch.tensor(np.array([state])).float().to(self.device)
                    encoder_state = self.envs.get_test_env_encoder_state()

                    hidden_rec = HiddenRecoder(self.device, self.vae.encoder, self.args.max_input_history_length)

                    curr_latent_mean, curr_latent_logvar= hidden_rec.encoder_init(0)

                    step_idx = 0
                    real_time = 0

                    pred_rew_ct = None
                    while True:
                        # print('begin')
                        step_idx += 1
                        # print(f'step_idx: {step_idx}, {time.time() - start_time}')
                        # print(curr_latent_mean.shape, curr_latent_logvar.shape, curr_latent_logvar.shape)

                        # 获取隐含特征
                        latent = utl.get_latent_for_policy(args,
                                                           latent_mean=curr_latent_mean,
                                                           latent_logvar=curr_latent_logvar)
                        # print(state)

                        # 获取掩码
                        mask = self.envs.get_test_env_mask()
                        if mask is not None:
                            mask = mask.to(self.device)
                        # print(f'get latent')
                        # print(state, latent, mask)

                        # 进行决策
                        if self.args.policy in ['ppo', 'a2c']:
                            _, action = self.policy.act(
                                state=state.float(), latent=latent, mask=mask, deterministic=True)
                        elif self.args.policy == 'dqn':
                            action = self.policy.act(
                                state=state.float(), latent=latent, mask=mask, deterministic=True)
                        elif self.args.policy == 'sac':
                            action = self.policy.act(
                                state=state.float(), latent=latent, mask=mask, deterministic=True)

                        next_state, (rew_raw, rew_normalized), done, infos = self.envs.test_env_step(action.cpu().numpy())
                        next_state = torch.tensor(np.array([next_state])).float().to(self.device)
                        # next_state_encoder = self.envs.get_test_env_encoder_state()

                        # print(step_idx, action)
                        # print(state, rew, rew_normalised, done, infos)
                        # print(f'time for step')
                        if self.args.norm_rew_for_policy and self.args.pass_norm_rew_to_vae:
                            rew_vae = rew_normalized.item()
                        else:
                            rew_vae = rew_raw.item()

                        qoe_all.append(infos["qoe"])
                        quality_all.append(infos["quality"] / 1000.)
                        smooth_all.append(infos["smooth"] / 1000.)
                        rebuf_all.append(infos["rebuf"] / 1000.)
                        bd_wastage_all.append(infos["waste_bytes"])

                        if done:
                            # bd_sum = self.envs.get_bd_sum_test_env()
                            break

                        if pred_rew_ct:
                            pred_rew_mae_all.append(abs((rew_vae - pred_rew_ct)))

                        # print(pred_rew_mre_all)

                        download_suspend = self.envs.get_sleep_judge_test_env()

                        while download_suspend:
                            action = torch.as_tensor([30])

                            next_state, (_, _), done, infos = self.envs.test_env_step(action)

                            next_state = torch.tensor(np.array([next_state])).float().to(self.device)

                            qoe_all.append(infos["qoe"])
                            quality_all.append(infos["quality"] / 1000.)
                            smooth_all.append(infos["smooth"] / 1000.)
                            rebuf_all.append(infos["rebuf"] / 1000.)
                            bd_wastage_all.append(infos["waste_bytes"])

                            download_suspend = self.envs.get_sleep_judge_test_env()

                        if self.args.pass_latent_to_policy:

                            # update embedding
                            reward_decoder = self.vae.reward_decoder
                            next_state_encoder = self.envs.get_test_env_encoder_state().unsqueeze(0)
                            next_state_policy = next_state
                            # print(next_state_encoder.shape)
                            curr_latent_mean, curr_latent_logvar, best_unchange_length, pred_rew_ct = self.inference(
                                hidden_rec, next_state_encoder, next_state_policy, step_idx, reward_decoder)

                            selected_history_length[best_unchange_length - 1] += 1

                        if done:
                            # bd_sum = self.envs.get_bd_sum_test_env()
                            break

                        state = next_state

                    # print(rewards)
                    view_chunk_num = len(quality_all) - quality_all.count(0)

                    qoe = np.sum(qoe_all) / view_chunk_num
                    quality = np.sum(quality_all) / view_chunk_num
                    smooth = np.sum(smooth_all) / (view_chunk_num - 1)
                    rebuf = np.sum(rebuf_all) / view_chunk_num

                    bandwidth_wastage = np.sum(bd_wastage_all)

                    if len(pred_rew_mae_all) > 0:
                        mae = np.average(pred_rew_mae_all)
                    else:
                        mae = -1

                    qoe_traces.append(qoe)
                    quality_traces.append(quality)
                    smooth_traces.append(smooth)
                    rebuf_traces.append(rebuf)
                    bd_wastage_traces.append(bandwidth_wastage)

                    pred_rew_mae_traces.append(mae)
                    # bd_sum_traces.append(bd_sum)

                    print(f"Time Cost {int((time.time() - start_time))}, "
                          f"\n QoE: {qoe}"
                          f"\n Quality: {quality}"
                          f"\n Smooth: {smooth}"
                          f"\n Rebuf: {rebuf}"
                          f"\n bandwidth_wastage: {bandwidth_wastage}"
                          # f"\n bd_sum: {bd_sum}"
                          f"\n prediction error: {mae}\n"
                          )
                    print(selected_history_length)
                    all_detail_performance.write(
                    f'{trace},{qoe},{quality},{smooth},{rebuf},{bandwidth_wastage},{mae}\n')
                    all_detail_performance.write(selected_history_length.__str__() + '\n')
                    all_detail_performance.flush()

                avg_qoe_traces = np.average(qoe_traces)
                avg_quality_traces = np.average(quality_traces)
                avg_smooth_traces = np.average(smooth_traces)
                avg_rebuf_traces = np.average(rebuf_traces)

                avg_bd_wastage_traces = np.average(bd_wastage_traces)

                avg_pred_rew_mae_traces = np.average(pred_rew_mae_traces)
                # avg_bd_sum_traces = np.average(bd_sum_traces)

                qoe_users.append(avg_qoe_traces)
                quality_users.append(avg_quality_traces)
                smooth_users.append(avg_smooth_traces)
                rebuf_users.append(avg_rebuf_traces)
                bd_wastage_users.append(avg_bd_wastage_traces)

                pred_rew_mae_users.append(avg_pred_rew_mae_traces)
                # bd_sum_users.append(avg_bd_sum_traces)

                print(f"average performance of user {user} under all traces"
                      f"\n QoE: {avg_qoe_traces}"
                      f"\n Quality: {avg_quality_traces}"
                      f"\n Smooth: {avg_smooth_traces}"
                      f"\n Rebuf: {avg_rebuf_traces}"
                      f"\n bandwidth_wastage: {avg_bd_wastage_traces}"
                      # f"\n bd_sum: {avg_bd_sum_traces}"
                      f"\n prediction error: {avg_pred_rew_mae_traces}\n"
                      )

                all_detail_performance.write(
                    f'{user},{avg_qoe_traces},{avg_quality_traces},{avg_smooth_traces},{avg_rebuf_traces},{avg_bd_wastage_traces},{avg_pred_rew_mae_traces}\n')

            avg_qoe_user = np.average(qoe_users)
            avg_quality_user = np.average(quality_users)
            avg_smooth_user = np.average(smooth_users)
            avg_rebuf_user = np.average(rebuf_users)

            avg_bd_wastage_user = np.average(bd_wastage_users)

            avg_pred_rew_mae_user = np.average(pred_rew_mae_users)
            # avg_bd_sum_user = np.average(bd_sum_users)

            print(f"average performance of all user"
                  f"\n QoE: {avg_qoe_user}"
                  f"\n Quality: {avg_quality_user}"
                  f"\n Smooth: {avg_smooth_user}"
                  f"\n Rebuf: {avg_rebuf_user}"
                  f"\n bandwidth_wastage: {avg_bd_wastage_user}",
                  # f"\n bd_sum: {avg_bd_sum_user}"
                  f"\n prediction error: {avg_pred_rew_mae_user}\n"
                  )

            self.logger.add(
                'QoE', avg_qoe_user, self.iter_idx)
            self.logger.add(
                'Quality', avg_quality_user, self.iter_idx)
            self.logger.add(
                'Rebuf', avg_rebuf_user, self.iter_idx)
            self.logger.add(
                'Smooth', avg_smooth_user, self.iter_idx)
            self.logger.add(
                'bandwidth_wastage', avg_bd_wastage_user, self.iter_idx)

            self.logger.add(
                'mae', avg_pred_rew_mae_user, self.iter_idx)
            # self.logger.add(
            #     'bd_sum', avg_bd_sum_user, self.iter_idx)

            dataset_name = args.dataset_path_test.split('/')[-2]
            self.logs_detail_performance.write(
                f'{dataset_name},{self.iter_idx},{avg_qoe_user},{avg_quality_user},{avg_smooth_user},{avg_rebuf_user},{avg_bd_wastage_user},{avg_pred_rew_mae_user}\n')
            self.logs_detail_performance.flush()

    def test(self, start_time, log_detail=False, log_dir=None):
        # print(self.vae.encoder.state_encoder.fc1s)
        # breakpoint()
        log_dir_test = self.full_output_folder + '/' + self.args.dataset_path_test.split('/')[-2] + '/'

        if not os.path.exists(log_dir_test):
            os.makedirs(log_dir_test)

        args = self.args

        # log the return avg/std across tasks (=processes)
        qoe_users = []
        quality_users = []
        smooth_users = []
        rebuf_users = []
        bd_wastage_users = []

        pred_rew_mae_users = []
        # bd_sum_users = []

        all_detail_performance_folder = './logs/secbad_test/logs_short_video_env/detailed_log/'

        # all_detail_performance_folder_epoch = all_detail_performance_folder + str(self.iter_idx + 1) + '/'
        if log_dir:
            all_detail_performance_folder_epoch = all_detail_performance_folder + log_dir.split('/')[-1] + '/'
        else:
            all_detail_performance_folder_epoch = all_detail_performance_folder + str(self.iter_idx + 1) + '/'

        if not os.path.exists(all_detail_performance_folder_epoch):
            os.makedirs(all_detail_performance_folder_epoch)
        self.logs_detail_performance = open(all_detail_performance_folder_epoch + '/logs.txt', 'a')

        performance_view_time_all = {0: np.zeros(5), 1: np.zeros(5)}
        selected_history_len_view_time_all = {
            0: np.zeros(self.args.max_input_history_length + 1),
            1: np.zeros(self.args.max_input_history_length + 1),
        }
        user_num = 1
        for user in range(user_num):
            all_detail_performance = open(all_detail_performance_folder_epoch + 'log_' + self.args.encoder_type + '_' +
                                          str(self.args.max_input_history_length + 1) + '_' + self.epoch, 'w')

            log_dir_user = log_dir_test + 'user_' + str(user) + '/'
            if not os.path.exists(log_dir_user):
                os.makedirs(log_dir_user)

            print(f"========== user {user} ==========")
            qoe_traces = []
            quality_traces = []
            smooth_traces = []
            rebuf_traces = []
            bd_wastage_traces = []

            pred_rew_mae_traces = []
            # bd_sum_traces = []

            performance_view_time_user = {0: np.zeros(5), 1: np.zeros(5)}
            selected_history_len_view_time_user = {
                0: np.zeros(self.args.max_input_history_length + 1),
                1: np.zeros(self.args.max_input_history_length + 1),
            }

            trace_num = len(os.listdir(self.envs.test_env.network_traces_path))

            for trace in range(trace_num):
                quality_play_chunk_video = [[] for i in range(self.envs.test_env.ALL_VIDEO_NUM)]
                time_play_chunk_video = []

                selected_history_len = [0] * (self.args.max_input_history_length + 1)
                best_unchange_length = 1

                performance_view_time = {0: np.zeros(5), 1: np.zeros(5)}
                selected_history_len_view_time = {
                    0: np.zeros(self.args.max_input_history_length + 1),
                    1: np.zeros(self.args.max_input_history_length + 1),
                }
                qoe_video = [0] * self.envs.test_env.ALL_VIDEO_NUM
                quality_video = [0] * self.envs.test_env.ALL_VIDEO_NUM
                smooth_video = [0] * self.envs.test_env.ALL_VIDEO_NUM
                rebuffer_video = [0] * self.envs.test_env.ALL_VIDEO_NUM
                bw_wastage_video = [0] * self.envs.test_env.ALL_VIDEO_NUM

                selected_history_len_video = [np.zeros(self.args.max_input_history_length + 1) for i in range(self.envs.test_env.ALL_VIDEO_NUM)]

                count_view_chunk_view_time = [0] * 3
                count_view_chunk_video = [0] * self.envs.test_env.ALL_VIDEO_NUM
                count_video_num_view_time = [0] * 3

                log_file_trace = log_dir_user + 'trace_' + str(trace)

                if log_detail:
                    f = open(log_file_trace, 'w')
                    f_ = open(log_file_trace + '_', 'w')

                print(f'========== user {user} trace {trace} ==========')
                vis_context = [user, trace]

                qoe_all = []
                quality_all = []
                smooth_all = []
                rebuf_all = []
                bd_wastage_all = []

                pred_rew_mae_all = []
                # bd_sum = None

                state = self.envs.reset_test(vis_context)[0]
                state = torch.tensor(np.array([state])).float().to(self.device)
                state_encoder = self.envs.get_test_env_encoder_state()

                hidden_rec = HiddenRecoder(self.device, self.vae.encoder, self.args.max_input_history_length)

                curr_latent_mean, curr_latent_logvar= hidden_rec.encoder_init(0)

                step_idx = 0
                real_time = 0

                pred_rew_ct = None

                pre_latent = None
                while True:
                    # print('begin')
                    step_idx += 1
                    # print(f'step_idx: {step_idx}, {time.time() - start_time}')
                    # print(curr_latent_mean.shape, curr_latent_logvar.shape, curr_latent_logvar.shape)

                    latent = utl.get_latent_for_policy(args,
                                                       latent_mean=curr_latent_mean,
                                                       latent_logvar=curr_latent_logvar)
                    latent_distance = -1
                    if pre_latent is not None:
                        latent_distance = calcu_dis(latent, pre_latent)
                        # print(latent, pre_latent)
                        # print(latent_distance)
                        # breakpoint()
                    pre_latent = latent
                    # print(state)
                    mask = self.envs.get_test_env_mask()
                    if mask is not None:
                        mask = mask.to(self.device)
                    # print(f'get latent')
                    # print(state, latent, mask)
                    if self.args.policy in ['ppo', 'a2c']:
                        _, action = self.policy.act(
                            state=state.float(), latent=latent, mask=mask, deterministic=True)
                    elif self.args.policy == 'dqn':
                        action = self.policy.act(
                            state=state.float(), latent=latent, mask=mask, deterministic=True)
                    elif self.args.policy == 'sac':
                        action = self.policy.act(
                            state=state.float(), latent=latent, mask=mask, deterministic=True)

                    action = action.cpu().numpy()
                    # print(action)
                    if self.args.policy == 'dqn':
                        action = np.concatenate([[action], np.array([[[best_unchange_length]]])])
                    else:
                        action = np.concatenate([action, np.array([[[best_unchange_length]]])])
                    # print(action)
                    next_state, (rew_raw, rew_normalized), done, infos = self.envs.test_env_step(action)
                    next_state = torch.tensor(np.array([next_state])).float().to(self.device)
                    # next_state_encoder = self.envs.get_test_env_encoder_state()

                    if self.args.norm_rew_for_policy and self.args.pass_norm_rew_to_vae:
                        rew_vae = rew_normalized.item()
                    else:
                        rew_vae = rew_raw.item()
                    # print(step_idx, action)
                    # print(state, rew_raw, rew_normalised, done, infos)
                    # print(f'time for step')
                    delay = infos['delay']

                    real_time += delay

                    play_video_id = infos['play_video_id']
                    pre_play_video_id = infos['pre_play_video_id']
                    download_video_id = infos['download_video_id']

                    one_step_qoe = infos['qoe']
                    bit_rate = infos['bit_rate']
                    quality = infos['quality']
                    smooth = infos['smooth']
                    rebuf = infos['rebuf']

                    waste_bytes = infos['waste_bytes']

                    sleep_time = infos['sleep_time']

                    user_swipe = infos['user_swipe']
                    buffer_size = infos['buffer_size']
                    user_rets = infos['user_rets']

                    throughput = infos['throughput']

                    play_chunk_bitrate = infos['play_chunk_bitrate']
                    play_chunk_ct = infos['play_chunk_ct']
                    view_type = infos['view_type']

                    if log_detail:
                        f.write(
                            f'{real_time},{play_video_id},{download_video_id},{bit_rate},{sleep_time},'
                            f'{delay},{rebuf},{user_swipe},{buffer_size},{throughput},{play_chunk_bitrate},'
                            f'{play_chunk_ct},{view_type[0]},{best_unchange_length},{latent_distance}\n')

                        f_.write(f'{real_time}#{latent.tolist()[0]}\n')

                    qoe_all.append(one_step_qoe)
                    quality_all.append(quality / 1000.)
                    smooth_all.append(smooth / 1000.)
                    rebuf_all.append(rebuf / 1000.)
                    bd_wastage_all.append(waste_bytes)

                    qoe_video[download_video_id] += one_step_qoe
                    quality_video[download_video_id] += quality / 1000.
                    smooth_video[download_video_id] += smooth / 1000.
                    rebuffer_video[download_video_id] += rebuf
                    bw_wastage_video[pre_play_video_id] += waste_bytes

                    if pred_rew_ct:
                        selected_history_len_video[download_video_id][best_unchange_length - 1] += 1
                        # selected_history_len_video[download_video_id][best_unchange_length - 1] += abs((rew_vae - pred_rew_ct))

                    if quality != 0:
                        quality_play_chunk_video[download_video_id].append(quality / 1000.)

                    if user_swipe == 1:
                        for idx in range(play_video_id - pre_play_video_id):
                            count_view_chunk_video[pre_play_video_id + idx] += math.ceil(
                                user_rets[idx].get_ret_duration() / args.chunklength)

                            category_view_time = min(1, int(user_rets[idx].get_ret_duration() / 1000. / 12))
                            performance_view_time[category_view_time][0] += qoe_video[pre_play_video_id + idx]
                            performance_view_time[category_view_time][1] += quality_video[pre_play_video_id + idx]
                            performance_view_time[category_view_time][2] += smooth_video[pre_play_video_id + idx]
                            performance_view_time[category_view_time][3] += rebuffer_video[pre_play_video_id + idx]
                            performance_view_time[category_view_time][4] += bw_wastage_video[pre_play_video_id + idx]

                            selected_history_len_view_time[category_view_time] += selected_history_len_video[pre_play_video_id + idx]

                            count_view_chunk_view_time[category_view_time] += count_view_chunk_video[
                                pre_play_video_id + idx]

                            count_video_num_view_time[category_view_time] += 1

                    if done:
                        # bd_sum = self.envs.get_bd_sum_test_env()
                        break

                    if pred_rew_ct and rew_vae != 0:
                        pred_rew_mae_all.append(abs((rew_vae - pred_rew_ct)))
                        # print(rew_vae, pred_rew_ct)

                    download_suspend = self.envs.get_sleep_judge_test_env()

                    while download_suspend:
                        action = torch.as_tensor([30])

                        next_state, (_, _), done, infos = self.envs.test_env_step(action)

                        next_state = torch.tensor(np.array([next_state])).float().to(self.device)

                        delay = infos['delay']

                        real_time += delay

                        play_video_id = infos['play_video_id']
                        pre_play_video_id = infos['pre_play_video_id']
                        download_video_id = infos['download_video_id']

                        one_step_qoe = infos['qoe']
                        bit_rate = infos['bit_rate']
                        quality = infos['quality']
                        smooth = infos['smooth']
                        rebuf = infos['rebuf']

                        waste_bytes = infos['waste_bytes']

                        sleep_time = infos['sleep_time']

                        user_swipe = infos['user_swipe']
                        buffer_size = infos['buffer_size']
                        user_rets = infos['user_rets']

                        throughput = infos['throughput']

                        play_chunk_bitrate = infos['play_chunk_bitrate']
                        play_chunk_ct = infos['play_chunk_ct']
                        view_type = infos['view_type']

                        if log_detail:
                            f.write(
                                f'{real_time},{play_video_id},{download_video_id},{bit_rate},{sleep_time},'
                                f'{delay},{rebuf},{user_swipe},{buffer_size},{throughput},{play_chunk_bitrate},'
                                f'{play_chunk_ct},{view_type[0]},{best_unchange_length},{-1}\n')

                        qoe_all.append(one_step_qoe)
                        quality_all.append(quality / 1000.)
                        smooth_all.append(smooth / 1000.)
                        rebuf_all.append(rebuf / 1000.)
                        bd_wastage_all.append(waste_bytes)

                        qoe_video[download_video_id] += one_step_qoe
                        quality_video[download_video_id] += quality / 1000.
                        smooth_video[download_video_id] += smooth / 1000.
                        rebuffer_video[download_video_id] += rebuf
                        bw_wastage_video[pre_play_video_id] += waste_bytes

                        if user_swipe == 1:
                            for idx in range(play_video_id - pre_play_video_id):
                                count_view_chunk_video[pre_play_video_id + idx] += math.ceil(
                                    user_rets[idx].get_ret_duration() / args.chunklength)

                                category_view_time = min(1, int(user_rets[idx].get_ret_duration() / 1000. / 12))
                                performance_view_time[category_view_time][0] += qoe_video[pre_play_video_id + idx]
                                performance_view_time[category_view_time][1] += quality_video[pre_play_video_id + idx]
                                performance_view_time[category_view_time][2] += smooth_video[pre_play_video_id + idx]
                                performance_view_time[category_view_time][3] += rebuffer_video[pre_play_video_id + idx]
                                performance_view_time[category_view_time][4] += bw_wastage_video[pre_play_video_id + idx]

                                selected_history_len_view_time[category_view_time] += selected_history_len_video[
                                    pre_play_video_id + idx]

                                count_view_chunk_view_time[category_view_time] += count_view_chunk_video[
                                    pre_play_video_id + idx]

                                count_video_num_view_time[category_view_time] += 1

                        download_suspend = self.envs.get_sleep_judge_test_env()
                    # print(pred_rew_mre_all)

                    if self.vae.encoder is not None and self.args.pass_latent_to_policy:
                        # update embedding
                        reward_decoder = self.vae.reward_decoder
                        next_state_encoder = self.envs.get_test_env_encoder_state().unsqueeze(0)
                        next_state_policy = next_state
                        # print(next_state_encoder)

                        curr_latent_mean, curr_latent_logvar, best_unchange_length, pred_rew_ct = self.inference(
                            hidden_rec, next_state_encoder, next_state_policy, step_idx, reward_decoder)
                        selected_history_len[best_unchange_length - 1] += 1

                    if done:
                        # bd_sum = self.envs.get_bd_sum_test_env()
                        break

                    state = next_state

                for catefory_ in performance_view_time:
                    performance_view_time[catefory_][0] /= count_view_chunk_view_time[catefory_]
                    performance_view_time[catefory_][1] /= count_view_chunk_view_time[catefory_]
                    performance_view_time[catefory_][2] /= count_view_chunk_view_time[catefory_]
                    performance_view_time[catefory_][3] /= count_view_chunk_view_time[catefory_]

                performance_view_time_user[0] += performance_view_time[0]
                performance_view_time_user[1] += performance_view_time[1]

                selected_history_len_view_time_user[0] += selected_history_len_view_time[0]
                selected_history_len_view_time_user[1] += selected_history_len_view_time[1]

                # print(rewards)
                view_chunk_num = len(quality_all) - quality_all.count(0)
                # print(len(quality_all), quality_all.count(0))
                # print(count_view_chunk_view_time)

                qoe = np.sum(qoe_all) / view_chunk_num
                quality = np.sum(quality_all) / view_chunk_num
                smooth = np.sum(smooth_all) / (view_chunk_num - 1)
                rebuf = np.sum(rebuf_all) / view_chunk_num
                bandwidth_wastage = np.sum(bd_wastage_all)

                if len(pred_rew_mae_all) > 0:
                    mae = np.average(pred_rew_mae_all)
                else:
                    mae = -1

                qoe_traces.append(qoe)
                quality_traces.append(quality)
                smooth_traces.append(smooth)
                rebuf_traces.append(rebuf)
                bd_wastage_traces.append(bandwidth_wastage)

                pred_rew_mae_traces.append(mae)
                # bd_sum_traces.append(bd_sum)

                print(f"Time Cost {int((time.time() - start_time))}, "
                      f"\n QoE: {qoe}"
                      f"\n Quality: {quality}"
                      f"\n Smooth: {smooth}"
                      f"\n Rebuf: {rebuf}"
                      f"\n bandwidth_wastage: {bandwidth_wastage}"
                      # f"\n bd_sum: {bd_sum}"
                      f"\n prediction error: {mae}\n"
                      )
                print(selected_history_len)
                # print(selected_history_len_video)
                print(performance_view_time)
                print(selected_history_len_view_time)

                print(count_view_chunk_view_time)
                # print(count_video_num_view_time)
                # print(play_chunk_quality)
                all_detail_performance.write(f'{performance_view_time}\n')
                all_detail_performance.write(f'{selected_history_len_view_time}\n')
                all_detail_performance.write(
                f'{trace},{qoe},{quality},{smooth},{rebuf},{bandwidth_wastage},{mae}\n')
                all_detail_performance.write(f'{selected_history_len}\n\n')

                all_detail_performance.flush()
            avg_qoe_traces = np.average(qoe_traces)
            avg_quality_traces = np.average(quality_traces)
            avg_smooth_traces = np.average(smooth_traces)
            avg_rebuf_traces = np.average(rebuf_traces)

            avg_bd_wastage_traces = np.average(bd_wastage_traces)

            avg_pred_rew_mae_traces = np.average(pred_rew_mae_traces)
            # avg_bd_sum_traces = np.average(bd_sum_traces)

            qoe_users.append(avg_qoe_traces)
            quality_users.append(avg_quality_traces)
            smooth_users.append(avg_smooth_traces)
            rebuf_users.append(avg_rebuf_traces)
            bd_wastage_users.append(avg_bd_wastage_traces)

            pred_rew_mae_users.append(avg_pred_rew_mae_traces)
            # bd_sum_users.append(avg_bd_sum_traces)

            print(f"average performance of user {user} under all traces"
                  f"\n QoE: {avg_qoe_traces}"
                  f"\n Quality: {avg_quality_traces}"
                  f"\n Smooth: {avg_smooth_traces}"
                  f"\n Rebuf: {avg_rebuf_traces}"
                  f"\n bandwidth_wastage: {avg_bd_wastage_traces}"
                  # f"\n bd_sum: {avg_bd_sum_traces}"
                  f"\n prediction error: {avg_pred_rew_mae_traces}\n"
                  )

            # all_detail_performance.write(
            #     f'{user},{avg_qoe_traces},{avg_quality_traces},{avg_smooth_traces},{avg_rebuf_traces},{avg_bd_usage_traces / 1000000.},{avg_bd_wastage_traces / 1000000.},{avg_bd_sum_traces / 1000000.}\n')
            all_detail_performance.write(
                f'{user},{avg_qoe_traces},{avg_quality_traces},{avg_smooth_traces},{avg_rebuf_traces},{avg_bd_wastage_traces},{avg_pred_rew_mae_traces}\n')

            performance_view_time_all[0] += performance_view_time_user[0] / trace_num
            performance_view_time_all[1] += performance_view_time_user[1] / trace_num

            selected_history_len_view_time_all[0] += selected_history_len_view_time_user[0]
            selected_history_len_view_time_all[1] += selected_history_len_view_time_user[1]

        avg_qoe_user = np.average(qoe_users)
        avg_quality_user = np.average(quality_users)
        avg_smooth_user = np.average(smooth_users)
        avg_rebuf_user = np.average(rebuf_users)
        avg_bd_usage_user = np.average(bd_wastage_users)

        avg_pred_rew_mae_user = np.average(pred_rew_mae_users)
        # avg_bd_sum_user = np.average(bd_sum_users)

        print(f"average performance of all user"
              f"\n QoE: {avg_qoe_user}"
              f"\n Quality: {avg_quality_user}"
              f"\n Smooth: {avg_smooth_user}"
              f"\n Rebuf: {avg_rebuf_user}"
              f"\n bandwidth_usage: {avg_bd_usage_user}"
              f"\n prediction error: {avg_pred_rew_mae_user}\n"
              )

        dataset_name = args.dataset_path_test.split('/')[-2]

        performance_view_time_all[0] = performance_view_time_all[0] / user_num
        performance_view_time_all[1] = performance_view_time_all[1] / user_num

        self.logs_detail_performance.write(
            f'{dataset_name},{performance_view_time_all[0][0]},{performance_view_time_all[0][1]},{performance_view_time_all[0][2]},{performance_view_time_all[0][3]},{performance_view_time_all[0][4]}\n')

        self.logs_detail_performance.write(
            f'{selected_history_len_view_time_all[0]}\n')

        self.logs_detail_performance.write(
            f'{dataset_name},{performance_view_time_all[1][0]},{performance_view_time_all[1][1]},{performance_view_time_all[1][2]},{performance_view_time_all[1][3]},{performance_view_time_all[1][4]}\n')

        self.logs_detail_performance.write(
            f'{selected_history_len_view_time_all[1]}\n')

        self.logs_detail_performance.write(
            f'{dataset_name},{avg_qoe_user},{avg_quality_user},{avg_smooth_user},{avg_rebuf_user},{avg_bd_usage_user},{avg_pred_rew_mae_user}\n\n')

        self.logs_detail_performance.flush()


    def load_model(self, epoch=''):
        self.epoch = epoch

        if self.args.policy in ['a2c', 'ppo']:
            self.policy.actor_critic = torch.load(os.path.join(self.args.model_path, 'policy' + epoch + '.pt'),
                                                  map_location=self.device, weights_only=False)
            self.policy.device = self.device
            self.policy.actor_critic.device = self.device

        elif self.args.policy in ['dqn']:
            self.policy.q_net = torch.load(os.path.join(self.args.model_path, 'q_net' + epoch + '.pt'),
                                                  map_location=self.device, weights_only=False)
            self.policy.target_q_net = torch.load(os.path.join(self.args.model_path, 'target_q_net' + epoch + '.pt'),
                                                  map_location=self.device, weights_only=False)
            self.policy.device = self.device
        elif self.args.policy in ['sac']:
            self.policy.actor = torch.load(os.path.join(self.args.model_path, 'actor' + epoch + '.pt'),
                                           map_location=self.device, weights_only=False)
            self.policy.target_critic_1 = torch.load(os.path.join(self.args.model_path, 'target_critic1_' + epoch + '.pt'),
                                           map_location=self.device, weights_only=False)
            self.policy.target_critic2_ = torch.load(os.path.join(self.args.model_path, 'target_critic2_' + epoch + '.pt'),
                                           map_location=self.device, weights_only=False)

        self.vae.encoder = torch.load(os.path.join(self.args.model_path, 'encoder' + epoch + '.pt'),
                                      map_location=self.device, weights_only=False)
        self.vae.encoder.device = self.device
        if self.vae.reward_decoder is not None:
            self.vae.reward_decoder = torch.load(os.path.join(self.args.model_path, 'reward_decoder' + epoch + '.pt'),
                                                 map_location=self.device, weights_only=False)
            self.vae.reward_decoder.device = self.device

        # print(self.policy.actor_critic.state_rms.mean)
        if self.args.norm_rew_for_policy:
            self.envs.rew_rms = utl.load_obj(self.args.model_path, 'env_rew_rms' + epoch)
            print(self.envs.rew_rms.mean, self.envs.rew_rms.var)

    def inference(self, hidden_rec, state_encoder, state_policy, step_idx, reward_decoder):

        hidden_rec.encoder_step(state_encoder)

        # 消融
        if not self.args.use_best_latent_selection:
            reset_after = max(step_idx - self.args.max_input_history_length, 0)
            latent_mean, latent_logvar = hidden_rec.get_record(
                reset_after=reset_after, up_to=step_idx, label='latent')
            latent = torch.cat((latent_mean, latent_logvar), dim=-1)
            pred_rew = reward_decoder(
                latent, state_policy.to(self.device))

            return latent_mean, latent_logvar, self.args.max_input_history_length + 1, pred_rew.item()

        # 预测决策奖励
        pred_rews = []
        for reset_after in range(max(step_idx - self.args.max_input_history_length, 0), step_idx + 1):
            latent_mean, latent_logvar = hidden_rec.get_record(
                reset_after=reset_after, up_to=step_idx, label='latent')

            latent = torch.cat((latent_mean, latent_logvar), dim=-1)
            second_term = 1.

            if reward_decoder is not None:
                pred_rew = reward_decoder(
                    latent, state_policy.to(self.device))
                pred_rews.append(pred_rew.mean(dim=-1).item())

        # 根据预测奖励选择隐含状态
        pred_rew_ct = max(pred_rews)
        best_unchange_length = len(pred_rews) - pred_rews.index(pred_rew_ct)
        best_reset_after = step_idx + 1 - best_unchange_length

        curr_latent_mean, curr_latent_logvar = hidden_rec.get_record(
            reset_after=best_reset_after, up_to=step_idx, label='latent')

        return curr_latent_mean.clone(), curr_latent_logvar.clone(), best_unchange_length, pred_rew_ct

def calcu_dis(a, b):
    dis = (abs(a-b) ** 2).sum()
    return dis