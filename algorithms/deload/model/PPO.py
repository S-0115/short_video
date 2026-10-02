"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import numpy as np
import torch
import torch.nn as nn

from torch.distributions import Normal
from torch.utils.data import BatchSampler, SequentialSampler
USE_GPU = torch.cuda.is_available()
# USE_GPU = False
device = torch.device('cuda' if USE_GPU else 'cpu')

class Policy(nn.Module):
    def __init__(self, layers, input_dim, args):
        super(Policy, self).__init__()

        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.episode_limit = args.episode_limit
        self.K_epochs = args.K_epochs
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size

        self.high = 3.
        self.low = 0.1

        self.activation_function = nn.ReLU()
        # initialise actor and critic
        layers = [int(h) for h in layers]
        self.actor_layers = nn.ModuleList()
        self.critic_layers = nn.ModuleList()
        for i in range(len(layers)):
            fc = nn.Linear(input_dim, layers[i])
            self.actor_layers.append(fc)
            fc = nn.Linear(input_dim, layers[i])
            self.critic_layers.append(fc)
            input_dim = layers[i]
        self.critic_linear = nn.Linear(layers[-1], 1)

        self.critic_linear.weight.data.mul_(0.01)
        self.critic_linear.bias.data.zero_()

        # output distributions of the policy
        self.dist = FixedNormal(layers[-1]).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

    def forward_actor(self, inputs):
        h = inputs
        for i in range(len(self.actor_layers)):
            h = self.actor_layers[i](h)
            h = self.activation_function(h)
        return h

    def forward_critic(self, inputs):
        h = inputs
        for i in range(len(self.critic_layers)):
            h = self.critic_layers[i](h)
            h = self.activation_function(h)
        return h

    def forward(self, state):

        # forward through critic/actor part
        hidden_critic = self.forward_critic(state)
        hidden_actor = self.forward_actor(state)

        return self.critic_linear(hidden_critic), hidden_actor

    def act(self, state):
        """
        Returns the (raw) actions and their value.
        """
        value, actor_features = self.forward(state)
        # print(actor_features)

        self.distribution = self.dist(actor_features)
        # print('dist', dist.logits)

        raw_action = self.distribution.rsample()

        action = torch.sigmoid(raw_action) * (self.high - self.low) + self.low
        action = torch.round(action, decimals=1)

        # print(action)

        # action_log_prob = dist.log_prob(action)
        action_log_prob = self.distribution.log_prob(raw_action)

        return value, (raw_action, action), action_log_prob

    def get_value(self, state):
        hidden_critic = self.forward_critic(state)

        return self.critic_linear(hidden_critic)

    def get_action(self, state):

        hidden_actor = self.forward_actor(state)
        dist = self.dist(hidden_actor)
        # print('dist', dist.logits)
        # raw = dist.sample()
        raw = dist.mean

        action = torch.sigmoid(raw) * (self.high - self.low) + self.low
        action = torch.round(action, decimals=1)

        return action

    def evaluate_actions(self, state, raw_action):
        # print(state.shape, action.shape)
        value, actor_features = self.forward(state)

        # print(value.shape)
        dist = self.dist(actor_features)

        log_prob = dist.log_prob(raw_action.unsqueeze(-1))  # [B, N, 1]

        return value, log_prob

    def train_ppo(self, epoch, replay_buffer):
        batch = replay_buffer.get_training_data()  # get training data

        with torch.no_grad():  # adv and td_target have no gradient

            B, T, N = batch['r_n'].shape
            return_batch = torch.zeros_like(batch['r_n'], dtype=torch.float32)
            g = torch.zeros(B, N, dtype=torch.float32, device=batch['r_n'].device)
            # print(batch['r_n'].shape, returns_batch.shape, g.shape, batch['done_n'].shape)
            # 逆序计算
            for t in reversed(range(T)):
                # 如果 done=1，则 g 重置为 0
                g = batch['r_n'][:, t] + self.gamma * g * (1 - batch['done_n'][:, t])
                # print(batch['r_n'][:, t])
                # print(g)
                return_batch[:, t] = g

            adv = return_batch - batch['v_n']

            # print(adv.shape, return_batch.shape)
            # print(adv.mean().item(), adv.std().item())
            # print(adv.mean(), adv.std())
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            # print(adv)

            if USE_GPU:
                adv = adv.to(device)
                return_batch = return_batch.to(device)

        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                """
                    get probs_now and values_now
                    probs_now.shape=(mini_batch_size, episode_limit, N, action_dim_bm)
                    values_now.shape=(mini_batch_size, episode_limit, N)
                """
                states_batch = batch['obs_n'][index].detach()

                a_n_batch = batch['raw_a_n'][index].detach()
                # print(probs_bm_now.shape)
                # print(a_bm_n_batch.shape)

                values_now, a_logprob_n_now = self.evaluate_actions(states_batch, a_n_batch)
                # a/b=exp(log(a)-log(b))
                # print(a_bm_logprob_n_now)

                a_logprob_n_batch = batch['a_logprob_n'][index].detach()
                # print(a_logprob_n_now.shape, a_logprob_n_batch.shape)
                # print(dist_entropy.shape)
                ratios = torch.exp(
                    a_logprob_n_now.squeeze(-1) - a_logprob_n_batch)  # ratios_bm.shape=(mini_batch_size, episode_limit, N)

                surr1 = ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
                action_loss = -torch.min(surr1, surr2) * adv[index]

                actor_loss = action_loss.mean()

                print(f'actor loss: {actor_loss}')
                # print(surr1_bm)

                values_old = batch["v_n"][index].detach()
                # print(values_old.shape, values_now.shape, return_batch[index].shape)
                values_error_clip = torch.clamp(values_now.squeeze(-1), values_old - self.epsilon, values_old + self.epsilon) - \
                                    return_batch[index]
                values_error_original = values_now.squeeze(-1) - return_batch[index]
                critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2).mean() * 0.5

                # critic_loss = torch.clamp(critic_loss, max=1.0).mean()
                # print(return_batch[index].unsqueeze(-1).shape)

                # critic_loss = (values_now - return_batch[index].unsqueeze(-1)) ** 2
                # print(f'value now {values_now.shape} return_batch {return_batch.shape}')

                self.optimizer.zero_grad()

                loss = actor_loss.mean() + critic_loss.mean()

                loss.backward()
                print(f'critic loss : {critic_loss.mean()}')

                # for name, params in self.named_parameters():
                #     print(f'{name} : {params}')

                # torch.nn.utils.clip_grad_norm_(list(self.parameters()),1.)

                self.optimizer.step()

class FixedNormal(nn.Module):
    def __init__(self, num_inputs):
        super(FixedNormal, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(num_inputs, 1),
            nn.Tanh()
        )

        self.std = nn.Sequential(
            nn.Linear(num_inputs, 1),
            nn.Softplus()
        )

    def forward(self, x):
        mu = self.mu(x)
        # print(mu.shape)
        std = self.std(x) + 1e-6

        return Normal(mu, std)
