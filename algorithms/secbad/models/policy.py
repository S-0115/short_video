"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import numpy as np
import torch
import torch.nn as nn

from algorithms.secbad.utils import helpers as utl
try:
    from torch.distributions import TanhTransform, TransformedDistribution, Normal, Categorical


    class TanhNormal(TransformedDistribution):
        def __init__(self, base_distribution, transforms, validate_args=None):
            super().__init__(base_distribution, transforms, validate_args=None)

    @property
    def mean(self):
        x = self.base_dist.mean
        for transform in self.transforms:
            x = transform(x)
        return x

except ImportError:
    print('You are probably running MuJoCo 131, so PyTorch Transforms cannot be used. '
          'Do not set norm_actions_pre_sampling, this will break.')
    pass

class Policy(nn.Module):
    def __init__(self,
                 args,
                 # input
                 pass_state_to_policy,
                 pass_latent_to_policy,
                 num_feature,
                 dim_state,
                 dim_latent,
                 # hidden
                 hidden_layers,
                 activation_function,  # tanh, relu, leaky-relu
                 # output
                 action_space,
                 ):
        """
        The policy can get any of these as input:
        - state (given by environment)
        - task (in the (belief) oracle setting)
        - latent variable (from VAE)
        """
        super(Policy, self).__init__()

        self.args = args
        self.device = args.device

        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        elif activation_function == 'leaky-relu':
            self.activation_function = nn.LeakyReLU()
        else:
            raise ValueError

        self.pass_state_to_policy = pass_state_to_policy
        self.pass_latent_to_policy = pass_latent_to_policy
        # set normalisation parameters for the inputs
        # (will be updated from outside using the RL batches)
        self.norm_state = self.args.norm_state_for_policy and (
            dim_state is not None)
        if self.pass_state_to_policy and self.norm_state:
            self.state_rms = utl.RunningMeanStd(self.device, shape=(num_feature, dim_state))
        self.norm_latent = self.args.norm_latent_for_policy and (
            dim_latent is not None)
        if self.pass_latent_to_policy and self.norm_latent:
            self.latent_rms = utl.RunningMeanStd(self.device, shape=(dim_latent,))

        curr_input_dim = dim_state * int(self.pass_state_to_policy) + \
            dim_latent * int(self.pass_latent_to_policy)

        # initialise encoders for separate inputs
        self.use_state_encoder = self.args.state_embedding_dim is not None
        if self.pass_state_to_policy and self.use_state_encoder:
            self.state_encoder_actor = utl.FeatureExtractor_State(
                self.args.num_feature, dim_state, self.args.state_embedding_dim, self.activation_function)
            self.state_encoder_critic = utl.FeatureExtractor_State(
                self.args.num_feature, dim_state, self.args.state_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - dim_state + \
                self.args.state_embedding_dim * self.args.num_feature
        self.use_latent_encoder = self.args.latent_embedding_dim is not None
        if self.pass_latent_to_policy and self.use_latent_encoder:
            self.latent_encoder_actor = utl.FeatureExtractor(
                dim_latent, self.args.latent_embedding_dim, self.activation_function)
            self.latent_encoder_critic = utl.FeatureExtractor(
                dim_latent, self.args.latent_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - dim_latent + \
                self.args.latent_embedding_dim

        # initialise actor and critic
        hidden_layers = [int(h) for h in hidden_layers]
        self.actor_layers = nn.ModuleList()
        self.critic_layers = nn.ModuleList()
        for i in range(len(hidden_layers)):
            fc = nn.Linear(curr_input_dim, hidden_layers[i])
            self.actor_layers.append(fc)
            fc = nn.Linear(curr_input_dim, hidden_layers[i])
            self.critic_layers.append(fc)
            curr_input_dim = hidden_layers[i]
        self.critic_linear = nn.Linear(hidden_layers[-1], 1)

        # self.critic_linear.weight.data.mul_(0.01)
        # self.critic_linear.bias.data.zero_()

        # output distributions of the policy
        num_outputs = action_space.n
        self.dist = FixedCategorical(hidden_layers[-1], num_outputs)

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

    def forward_act(self, state, latent):
        if state is not None:
            state = state.clone()
        if latent is not None:
            latent = latent.clone()

        # print(state.device, latent.device)
        # print('>=' * 10)
        # print(state.shape, latent.shape)
        # print(state.shape)
        # print(state)
        if self.pass_state_to_policy:
            if self.norm_state:
                mean_state = self.state_rms.mean.detach()
                var_state = self.state_rms.var.detach()
                state = (state - mean_state) / torch.sqrt(var_state + 1e-4)
            if self.use_state_encoder:
                state_actor = self.state_encoder_actor(state)
                state_critic = self.state_encoder_critic(state)
                # assert torch.isfinite(state).all(), "state encoder 导致的"
        else:
            state_actor = state_critic = torch.zeros(0, ).to(self.device)

        if self.pass_latent_to_policy:
            if self.norm_latent:
                mean_latent = self.latent_rms.mean.detach()
                var_latent  = self.latent_rms.var.detach()
                latent = (latent - mean_latent) / torch.sqrt(var_latent + 1e-4)
            if self.use_latent_encoder:
                latent_actor = self.latent_encoder_actor(latent)
                latent_critic = self.latent_encoder_critic(latent)
        else:
            latent_actor = latent_critic = torch.zeros(0, ).to(self.device)

        # concatenate inputs
        # print(state.shape)
        # print(latent.shape)
        # if self.pass_latent_to_policy:
        actor_inputs = torch.cat((state_actor, latent_actor), dim=-1)
        critic_inputs = torch.cat((state_critic, latent_critic), dim=-1)
        # forward through critic/actor part
        hidden_actor = self.forward_actor(actor_inputs)
        hidden_critic = self.forward_critic(critic_inputs)

        return self.critic_linear(hidden_critic), hidden_actor

    def forward(self, state, latent, mask):
        if state is not None:
            state = state.clone()
        if latent is not None:
            latent = latent.clone()

        # print(state.device, latent.device)
        # print('>=' * 10)
        # print(state.shape, latent.shape)
        # print(state.shape)
        # print(state)
        if self.pass_state_to_policy:
            if self.norm_state:
                mean_state = self.state_rms.mean.detach()
                var_state = self.state_rms.var.detach()
                state = (state - mean_state) / torch.sqrt(var_state + 1e-4)
            if self.use_state_encoder:
                state_actor = self.state_encoder_actor(state)
        else:
            state_actor = torch.zeros(0, ).to(self.device)

        if self.pass_latent_to_policy:
            if self.norm_latent:
                mean_latent = self.latent_rms.mean.detach()
                var_latent  = self.latent_rms.var.detach()
                latent = (latent - mean_latent) / torch.sqrt(var_latent + 1e-4)
            if self.use_latent_encoder:
                latent_actor = self.latent_encoder_actor(latent)
        else:
            latent_actor = torch.zeros(0, ).to(self.device)

        # concatenate inputs
        # print(state.shape)
        # print(latent.shape)
        # if self.pass_latent_to_policy:
        actor_inputs = torch.cat((state_actor, latent_actor), dim=-1)
        # forward through critic/actor part
        hidden_actor = self.forward_actor(actor_inputs)

        if not self.args.use_invalid_action_mask:
            mask = None

        dist = self.dist(hidden_actor, mask)

        action = dist.logits.argmax(dim=-1, keepdim=True).unsqueeze(-1)

        return action


    def act(self, state, latent, mask, deterministic=False):
        """
        Returns the (raw) actions and their value.
        """
        # print(state.shape, latent.shape, mask.shape)
        value, actor_features = self.forward_act(
            state=state, latent=latent)
        # print(actor_features)
        if not self.args.use_invalid_action_mask:
            mask = None

        dist = self.dist(actor_features, mask)
        # print('dist', dist.logits)

        if deterministic:
            action = dist.logits.argmax(dim=-1, keepdim=True).unsqueeze(-1)
        else:
            action = dist.sample().unsqueeze(-1)
        # print(action)
        # print(action_log_probs)
        # print(action_log_probs.shape)
        # print(dist.logits.shape)
        # print(action.shape)
        # breakpoint()
        return value, action

    def get_value(self, state, latent):
        value, _ = self.forward_act(state, latent)
        return value

    def update_rms(self, args, policy_storage):
        """ Update normalisation parameters for inputs with current data """
        with torch.no_grad():
            if self.pass_state_to_policy and self.norm_state:
                self.state_rms.update(policy_storage.prev_state[:-1])
            if self.pass_latent_to_policy and self.norm_latent:
                latent = utl.get_latent_for_policy(args,
                                                   torch.cat(
                                                       policy_storage.latent_mean[:-1]),
                                                   torch.cat(
                                                       policy_storage.latent_logvar[:-1])
                                                   )
                latent = latent.clone()
                # print(f'latent shape is {latent.shape}')
                self.latent_rms.update(latent)

    def evaluate_actions(self, state, latent, mask, action):
        # print('mask', mask.shape)
        value, actor_features = self.forward_act(state, latent)

        if not self.args.use_invalid_action_mask:
            mask = None

        dist = self.dist(actor_features, mask)

        # print('dist', dist.logits.shape)
        # print(action.shape)
        # action = action.unsqueeze(-1)
        action_log_probs = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
        # print(action_log_probs.shape)
        # print(action_log_probs)
        dist_entropy = dist.entropy().mean()
        # print("entropy requires_grad:", dist_entropy.requires_grad)

        return value, action_log_probs, dist_entropy


def init(module, weight_init, bias_init, gain=1.0):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


class FixedCategorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(FixedCategorical, self).__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x, mask):
        x = self.linear(x)
        # print(x)
        if mask is not None:
            x = x.masked_fill(mask == 0, -float('inf'))
        # print(x)
        return Categorical(logits=x)
