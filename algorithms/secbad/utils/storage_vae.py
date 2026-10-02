import numpy as np
import torch

class RolloutStorageVAE(object):
    def __init__(self,
                 num_processes,
                 max_trajectory_len,
                 zero_pad,
                 max_num_rollouts,
                 num_feature_policy,
                 state_dim_policy,
                 num_feature_encoder,
                 state_dim_encoder,
                 action_dim,
                 num_action,
                 vae_buffer_add_thresh):
        """
        Store everything that is needed for the VAE update
        :param num_processes:
        """
        self.action_dim = action_dim
        # print(action_dim)

        # prob of adding new trajectories
        self.vae_buffer_add_thresh = vae_buffer_add_thresh
        # maximum buffer len (number of trajectories)
        self.max_buffer_size = max_num_rollouts
        self.insert_idx = 0  # at which index we're currently inserting new data
        self.buffer_len = 0  # how much of the buffer has been filled

        # how long a trajectory can be at max (horizon)
        self.max_traj_len = max_trajectory_len
        # whether to zero-pad to maximum length (zero's at the end!)
        self.zero_pad = zero_pad

        # buffers for completed rollouts (stored on CPU)
        if self.max_buffer_size > 0:
            # 要存储 max_buffer_size 这么多个 完整的 （长度为 max_traj_len） trajectory
            self.prev_state = torch.zeros(
                (self.max_traj_len, self.max_buffer_size, num_feature_encoder, state_dim_encoder))
            self.next_state = torch.zeros(
                (self.max_traj_len, self.max_buffer_size, num_feature_encoder, state_dim_encoder))

            self.prev_state_policy = torch.zeros(
                (self.max_traj_len, self.max_buffer_size, num_feature_policy, state_dim_policy))
            self.next_state_policy = torch.zeros(
                (self.max_traj_len, self.max_buffer_size, num_feature_policy, state_dim_policy))

            self.actions = torch.zeros(
                (self.max_traj_len, self.max_buffer_size, action_dim))
            self.mask = torch.ones(
                (self.max_traj_len, self.max_buffer_size, num_action))
            self.rewards = torch.zeros(
                (self.max_traj_len, self.max_buffer_size, 1))
            self.r_ts = torch.zeros(
                (self.max_traj_len, self.max_buffer_size, 1))
            self.r_ts = torch.zeros(
                (self.max_traj_len, self.max_buffer_size, 1))
            self.done = torch.zeros(
                (self.max_traj_len, self.max_buffer_size, 1))

        # storage for each running process (stored on GPU)
        self.num_processes = num_processes
        # count environment steps so we know where to insert
        self.curr_timestep = torch.zeros((num_processes)).long()

        self.running_prev_state = torch.zeros((self.max_traj_len, num_processes, num_feature_encoder, state_dim_encoder))  # for each episode will have obs 0...N-1
        self.running_next_state = torch.zeros((self.max_traj_len, num_processes, num_feature_encoder, state_dim_encoder))  # for each episode will have obs 1...N

        self.running_prev_state_policy = torch.zeros((self.max_traj_len, num_processes, num_feature_policy,state_dim_policy))  # for each episode will have obs 0...N-1
        self.running_next_state_policy = torch.zeros((self.max_traj_len, num_processes, num_feature_policy,state_dim_policy))  # for each episode will have obs 1...N

        self.running_rewards = torch.zeros(
            (self.max_traj_len, num_processes, 1))
        self.running_r_ts = torch.zeros(
            (self.max_traj_len, num_processes, 1))
        self.running_actions = torch.zeros(
            (self.max_traj_len, num_processes, action_dim))
        self.running_mask = torch.zeros(
            (self.max_traj_len, num_processes, num_action))

        self.running_done = torch.zeros(
            (self.max_traj_len, num_processes, 1))

    def get_running_batch(self):
        """
        Returns the batch of data from the current running environments
        (zero-padded to maximal trajectory length since different processes can have different trajectory lengths)
        :return:
        """
        return self.running_prev_state, self.running_next_state, self.running_actions, self.running_rewards, self.curr_timestep

    def insert(self, prev_state, actions, next_state, prev_state_policy, next_state_policy, rewards, done, mask, r_ts=None):
        '''

        :param mask:
        :param prev_state: s_t
        :param actions: a_t
        :param next_state: s_t+1
        :param rewards: r_t
        :param done:
        :param r_ts:
        :return:
        '''
        # add to temporary buffer
        # print(mask.shape)
        for i in range(self.num_processes):

            self.running_prev_state[self.curr_timestep[i],
                                    i] = prev_state[i]
            self.running_next_state[self.curr_timestep[i],
                                    i] = next_state[i]

            self.running_prev_state_policy[self.curr_timestep[i],
                                    i] = prev_state_policy[i]
            self.running_next_state_policy[self.curr_timestep[i],
                                    i] = next_state_policy[i]

            self.running_rewards[self.curr_timestep[i], i] = rewards[i]
            if r_ts is not None:
                self.running_r_ts[self.curr_timestep[i], i] = r_ts[i]
            self.running_actions[self.curr_timestep[i], i] = actions[i]
            self.running_mask[self.curr_timestep[i], i] = mask[i]
            # print(sum(mask[i])==6)
            # if sum(mask[i]) == 0:
            #     print(i)

            self.running_done[self.curr_timestep[i], i] = done[i]

            self.curr_timestep[i] += 1

            # if we are at the end of a task, dump the data into the larger buffer
            if self.curr_timestep[i] == self.max_traj_len:
                # add to permanent (up to max_buffer_len) buffer
                if self.max_buffer_size > 0:
                    if self.vae_buffer_add_thresh >= np.random.uniform(0, 1):
                        # check where to insert data
                        if self.insert_idx + 1 > self.max_buffer_size:
                            # keep track of how much we filled the buffer (for sampling from it)
                            self.buffer_len = self.insert_idx
                            # this will keep some entries at the end of the buffer without overwriting them,
                            # but the buffer is large enough to make this negligible
                            self.insert_idx = 0
                        else:
                            self.buffer_len = max(
                                self.buffer_len, self.insert_idx)
                        # add; note: num trajectories are along dim=1,
                        # trajectory length along dim=0, to match pytorch RNN interface
                        self.prev_state[:, self.insert_idx] = self.running_prev_state[:, i].to(
                            'cpu')
                        self.next_state[:, self.insert_idx] = self.running_next_state[:, i].to(
                            'cpu')

                        self.prev_state_policy[:, self.insert_idx] = self.running_prev_state_policy[:, i].to(
                            'cpu')
                        self.next_state_policy[:, self.insert_idx] = self.running_next_state_policy[:, i].to(
                            'cpu')

                        self.actions[:, self.insert_idx] = self.running_actions[:, i].to(
                            'cpu')
                        self.mask[:, self.insert_idx] = self.running_mask[:, i].to(
                            'cpu')

                        self.rewards[:, self.insert_idx] = self.running_rewards[:, i].to(
                            'cpu')
                        if r_ts is not None:
                            self.r_ts[:, self.insert_idx] = self.running_r_ts[:, i].to(
                                'cpu')

                        self.done[:, self.insert_idx] = self.running_done[:, i].to(
                            'cpu'
                        )
                        self.insert_idx += 1

                    # empty running buffer
                    self.running_prev_state[:, i] *= 0
                    self.running_next_state[:, i] *= 0
                    self.running_rewards[:, i] *= 0
                    if r_ts is not None:
                        self.running_r_ts[:, i] *= 0
                    self.running_actions[:, i] *= 0
                    self.running_mask[:, i] *= 0
                    self.running_done[:, i] *= 0
                    self.curr_timestep[i] = 0

    def ready_for_update(self):
        return len(self) > 0

    def __len__(self):
        return self.buffer_len

    def get_batch(self, batchsize=5, replace=False):
        # TODO: check if we can get rid of num_enc_len and num_rollouts (call it batchsize instead)

        batchsize = min(self.buffer_len, batchsize)
        # print(batchsize, self.buffer_len)
        # select the indices for the processes from which we pick
        rollout_indices = np.random.choice(
            range(self.buffer_len - 1), batchsize, replace=replace)
        # trajectory length of the individual rollouts we picked
        # print(rollout_indices)

        # select the rollouts we want
        prev_obs = self.prev_state[:, rollout_indices, :]
        next_obs = self.next_state[:, rollout_indices, :]

        prev_obs_policy = self.prev_state_policy[:, rollout_indices, :]
        next_obs_policy = self.next_state_policy[:, rollout_indices, :]

        actions = self.actions[:, rollout_indices, :]
        rewards = self.rewards[:, rollout_indices, :]
        r_ts = self.r_ts[:, rollout_indices, :]
        # print(next_obs[-1])

        n_rewards = self.rewards[:, rollout_indices, :]
        # print(n_rewards)
        n_rewards = torch.roll(n_rewards, -1, 0)
        n_rewards[-1] = self.rewards[0, rollout_indices + 1, :]
        # print(n_rewards)

        return prev_obs, next_obs, prev_obs_policy, next_obs_policy, n_rewards, actions, \
            rewards, r_ts

    def get_batch_dqn(self, batchsize=5, replace=False):
        # TODO: check if we can get rid of num_enc_len and num_rollouts (call it batchsize instead)

        batchsize = min(self.buffer_len, batchsize)
        # print(batchsize, self.buffer_len)
        # select the indices for the processes from which we pick
        rollout_indices = np.random.choice(
            range(self.buffer_len - 1), batchsize, replace=replace)
        # trajectory length of the individual rollouts we picked
        # print(rollout_indices)

        # select the rollouts we want
        prev_obs = self.prev_state[:, rollout_indices, :]
        next_obs = self.next_state[:, rollout_indices, :]
        actions = self.actions[:, rollout_indices, :]
        rewards = self.rewards[:, rollout_indices, :]

        prev_obs_policy = self.prev_state_policy[:, rollout_indices, :]
        next_obs_policy = self.next_state_policy[:, rollout_indices, :]


        mask = self.mask[:, rollout_indices, :]
        # print(torch.all(mask == 0, dim=-1).sum())
        # print(next_obs[-1])
        n_mask = self.mask[:, rollout_indices, :]
        n_mask = torch.roll(n_mask, -1, 0)
        n_mask[-1] = self.mask[0, rollout_indices + 1, :]
        # print(torch.all(n_mask == 0, dim=-1).sum())
        # print(n_mask[-1])

        n_rewards = self.rewards[:, rollout_indices, :]
        # print(n_rewards)
        n_rewards = torch.roll(n_rewards, -1, 0)
        n_rewards[-1] = self.rewards[0, rollout_indices + 1, :]
        # print(n_rewards)
        r_ts = self.r_ts[:, rollout_indices, :]

        dones = self.done[:, rollout_indices, :]

        return prev_obs, next_obs, prev_obs_policy, next_obs_policy, n_rewards, actions, mask, n_mask, \
            rewards, r_ts, dones

