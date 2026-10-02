import math

import torch
import torch.nn as nn

from algorithms.secbad.utils import helpers as utl


class DQNPolicy(nn.Module):
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
                 activation_function,
                 # output
                 action_space):
        super().__init__()

        self.args = args
        self.device = args.device
        self.num_actions = action_space.n

        # -------- activation --------
        if activation_function == 'tanh':
            self.activation = nn.Tanh()
        elif activation_function == 'relu':
            self.activation = nn.ReLU()
        elif activation_function == 'leaky-relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError

        self.pass_state_to_policy = pass_state_to_policy
        self.pass_latent_to_policy = pass_latent_to_policy

        # -------- RMS --------
        self.norm_state = args.norm_state_for_policy and dim_state is not None
        if self.pass_state_to_policy and self.norm_state:
            self.state_rms = utl.RunningMeanStd(
                self.device, shape=(num_feature, dim_state))

        self.norm_latent = args.norm_latent_for_policy and dim_latent is not None
        if self.pass_latent_to_policy and self.norm_latent:
            self.latent_rms = utl.RunningMeanStd(
                self.device, shape=(dim_latent,))

        # -------- input dim --------
        curr_input_dim = (
            dim_state * int(self.pass_state_to_policy)
            + dim_latent * int(self.pass_latent_to_policy)
        )

        # -------- encoders --------
        self.use_state_encoder = args.state_embedding_dim is not None
        if self.pass_state_to_policy and self.use_state_encoder:
            self.state_encoder = utl.FeatureExtractor_State(
                num_feature,
                dim_state,
                args.state_embedding_dim,
                self.activation
            )
            curr_input_dim = (
                curr_input_dim - dim_state
                + args.state_embedding_dim * num_feature
            )

        self.use_latent_encoder = args.latent_embedding_dim is not None
        if self.pass_latent_to_policy and self.use_latent_encoder:
            self.latent_encoder = utl.FeatureExtractor(
                dim_latent,
                args.latent_embedding_dim,
                self.activation
            )
            curr_input_dim = (
                curr_input_dim - dim_latent
                + args.latent_embedding_dim
            )

        # -------- Q network --------
        hidden_layers = [int(h) for h in hidden_layers]
        self.q_layers = nn.ModuleList()

        for h in hidden_layers:
            self.q_layers.append(nn.Linear(curr_input_dim, h))
            curr_input_dim = h

        self.q_head = nn.Linear(curr_input_dim, self.num_actions)

    # ======================================================
    # forward: NEVER mask here
    # ======================================================
    def forward(self, state, latent):
        # print(state.shape, latent.shape)

        state = state.clone()
        if latent is not None:
            latent = latent.clone()

        # state
        if self.pass_state_to_policy:
            if self.norm_state:
                state = (state - self.state_rms.mean.detach()) / \
                        torch.sqrt(self.state_rms.var.detach() + 1e-4)
            if self.use_state_encoder:
                state = self.state_encoder(state)
        else:
            state = torch.zeros(0, device=self.device)

        # latent
        if self.pass_latent_to_policy:
            if self.norm_latent:
                latent = (latent - self.latent_rms.mean.detach()) / \
                         torch.sqrt(self.latent_rms.var.detach() + 1e-4)
            if self.use_latent_encoder:
                latent = self.latent_encoder(latent)
        else:
            latent = torch.zeros(0, device=self.device)

        x = torch.cat((state, latent), dim=-1)

        for layer in self.q_layers:
            x = self.activation(layer(x))

        return self.q_head(x)

    # ======================================================
    # ε-greedy action (mask 必须在这里)
    # ======================================================
    def act(self, state, latent, mask, epsilon=0.0):
        if torch.rand(1).item() < epsilon:
            actions = []
            for m in mask:
                valid = torch.where(m == 1)[0]
                idx = torch.randint(len(valid), (1,))
                actions.append(valid[idx])
            return torch.stack(actions)

        with torch.no_grad():
            q = self.forward(state, latent)
            q = q.masked_fill(mask == 0, -math.inf)
            return q.argmax(dim=-1, keepdim=True)

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
