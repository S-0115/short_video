"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

Used for on-policy rollout storages.
"""
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from algorithms.secbad.utils import helpers as utl


class AdaptiveOnlineStorage(object):
    def __init__(self,
                 args, num_steps, num_processes,
                 num_feature,
                 state_dim,
                 action_space,
                 hidden_size, latent_dim, normalise_rewards, **kwargs):

        self.args = args
        self.device = args.device

        self.num_feature = num_feature
        self.state_dim = state_dim

        # how many steps to do per update (= size of online buffer)
        self.num_steps = num_steps
        self.num_processes = num_processes  # number of parallel processes
        self.step = 0  # keep track of current environment step

        # normalisation of the rewards
        self.normalise_rewards = normalise_rewards

        # inputs to the policy
        # this will include s_0 when state was reset (hence num_steps+1)
        self.prev_state = torch.zeros(num_steps + 1, num_processes, num_feature, state_dim)
        # next_state will include s_N when state was reset, skipping s_0
        # (only used if we need to re-compute embeddings after backpropagating RL loss through encoder)
        self.next_state = torch.zeros(num_steps, num_processes, num_feature, state_dim)
        if self.args.pass_latent_to_policy:
            # latent variables (of VAE)
            self.latent_dim = latent_dim
            self.latent_mean = []
            self.latent_logvar = []
            # hidden states of RNN (necessary if we want to re-compute embeddings)
            self.hidden_size = hidden_size
        else:
            self.latent_mean = None
            self.latent_logvar = None

        # rewards and end of episodes
        self.rewards_raw = torch.zeros(num_steps, num_processes, 1)
        self.rewards_normalised = torch.zeros(num_steps, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.mask = torch.ones(num_steps, num_processes, action_space.n)
        self.r_t = torch.zeros(
            num_steps + 1, num_processes, 1)  # TODO +1 or not?

        # actions
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()

        # values and returns
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)

        self.to_device()

    def to_device(self):
        if self.args.pass_state_to_policy:
            self.prev_state = self.prev_state.to(self.device)
        if self.args.pass_latent_to_policy:
            self.latent_mean = [t.to(self.device) for t in self.latent_mean]
            self.latent_logvar = [t.to(self.device) for t in self.latent_logvar]
        self.next_state = self.next_state.to(self.device)
        self.rewards_raw = self.rewards_raw.to(self.device)
        self.rewards_normalised = self.rewards_normalised.to(self.device)
        self.masks = self.masks.to(self.device)
        self.mask = self.mask.to(self.device)
        self.value_preds = self.value_preds.to(self.device)
        self.returns = self.returns.to(self.device)
        self.actions = self.actions.to(self.device)

    def insert(self,
               state,
               actions,
               rewards_raw,
               rewards_normalised,
               value_preds,
               mask,
               masks,
               r_t,
               latent_mean=None,
               latent_logvar=None,
               **kwargs,
               ):
        self.prev_state[self.step + 1].copy_(state)
        if self.args.pass_latent_to_policy:
            self.latent_mean.append(latent_mean.detach().clone())
            self.latent_logvar.append(latent_logvar.detach().clone())
        self.actions[self.step] = actions.detach().clone()
        self.rewards_raw[self.step].copy_(rewards_raw)
        if rewards_normalised != None:
            self.rewards_normalised[self.step].copy_(rewards_normalised)

        if value_preds != None:
            if isinstance(value_preds, list):
                self.value_preds[self.step].copy_(value_preds[0].detach())
            else:
                self.value_preds[self.step].copy_(value_preds.detach())
        self.masks[self.step + 1].copy_(masks)
        self.mask[self.step] = mask.detach().clone()
        if r_t is not None:
            self.r_t[self.step + 1].copy_(r_t)
        self.step = (self.step + 1) % self.num_steps

    def before_update(self, policy):
        latent = utl.get_latent_for_policy(self.args,
                                           latent_mean=torch.stack(
                                               self.latent_mean[:-1]) if self.latent_mean is not None else None,
                                           latent_logvar=torch.stack(
                                               self.latent_logvar[:-1]) if self.latent_mean is not None else None)
        # given policy, 看到 storage 里的 prev_state 后会得到一个 policy 认为的 action 分布，
        # action_log_probs 算的是 storage 里的 actions 在 policy 认为的 action 分布上的 log_prob

        _, action_log_probs, _ = policy.evaluate_actions(self.prev_state[:-1],
                                                         latent,
                                                         self.mask,
                                                         #  self.beliefs[:-
                                                         #               1] if self.beliefs is not None else None,
                                                         #  self.tasks[:-
                                                         #             1] if self.tasks is not None else None,
                                                         self.actions)
        self.action_log_probs = action_log_probs.detach()

    def after_update(self):
        self.prev_state[0].copy_(self.prev_state[-1])
        if self.args.pass_latent_to_policy:
            self.latent_mean = []
            self.latent_logvar = []
        self.masks[0].copy_(self.masks[-1])
        self.r_t[0].copy_(self.r_t[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):

        if self.normalise_rewards:
            rewards = self.rewards_normalised.clone()
        else:
            rewards = self.rewards_raw.clone()

        self._compute_returns(next_value=next_value, rewards=rewards, value_preds=self.value_preds,
                              returns=self.returns,
                              gamma=gamma, tau=tau, use_gae=use_gae)

    def _compute_returns(self, next_value, rewards, value_preds, returns, gamma, tau, use_gae):
        # next_value: reward for next step predicted by the network
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        if use_gae:
            value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(rewards.size(0))):
                delta = rewards[step] + gamma * value_preds[step +
                                                            1] * self.masks[step + 1] - value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                returns[step] = gae + value_preds[step]
        else:
            returns[-1] = 0
            for step in reversed(range(rewards.size(0))):
                returns[step] = returns[step + 1] * gamma * \
                                self.masks[step + 1] + rewards[step]

    def num_transitions(self):
        return len(self.prev_state) * self.num_processes

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards_raw.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        # print(f'num_steps{num_steps}, num_processes{num_processes}')
        # print(f'batch_size: {batch_size}, mini_batch_size: {mini_batch_size}')
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:

            if self.args.pass_state_to_policy:
                state_batch = self.prev_state[:-
                1].reshape(-1, *self.prev_state.size()[2:])[indices]
            else:
                state_batch = None
            if self.args.pass_latent_to_policy:
                latent_mean_batch = torch.cat(self.latent_mean[:-1])[indices]
                latent_logvar_batch = torch.cat(
                    self.latent_logvar[:-1])[indices]
            else:
                latent_mean_batch = latent_logvar_batch = None

            actions_batch = self.actions.reshape(-1,
                                                 self.actions.size(-1))[indices]

            value_preds_batch = self.value_preds[:-1].reshape(-1, 1)[indices]
            return_batch = self.returns[:-1].reshape(-1, 1)[indices]

            old_action_log_probs_batch = self.action_log_probs.reshape(-1, 1)[
                indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.reshape(-1, 1)[indices]

            mask_batch = self.mask.reshape(-1, self.mask.size(-1))[indices]

            yield state_batch, \
                actions_batch, \
                latent_mean_batch, latent_logvar_batch, \
                value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ, \
                mask_batch
