import torch
import torch.nn as nn
from torch.nn import functional as F

from algorithms.secbad.utils import helpers as utl

class RewardDecoder(nn.Module):
    def __init__(self,
                 args,
                 layers,
                 latent_dim,
                 latent_embed_dim,
                 state_dim,
                 state_embed_dim,
                 pred_type='deterministic',
                 input_state=True,
                 input_latent=True,
                 ):
        super(RewardDecoder, self).__init__()

        self.args = args

        self.pred_type = pred_type
        self.input_state = input_state
        self.input_latent = input_latent
        self.device = self.args.device

        activation_function = args.policy_activation_function

        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        elif activation_function == 'leaky-relu':
            self.activation_function = nn.LeakyReLU()
        else:
            raise ValueError

        if self.input_state:
            self.state_encoder = utl.FeatureExtractor_State(
                self.args.num_feature, state_dim, state_embed_dim, self.activation_function)
        else:
            self.state_encoder = None

        if self.input_latent:
            self.latent_encoder = utl.FeatureExtractor(
                latent_dim, latent_embed_dim, self.activation_function)
        else:
            self.latent_encoder = None

        curr_input_dim = 0
        if self.input_state:
            curr_input_dim += state_embed_dim * self.args.num_feature

        if self.input_latent:
            curr_input_dim += latent_embed_dim
            # curr_input_dim += latent_dim

        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        if pred_type == 'gaussian':
            self.fc_out = nn.Linear(curr_input_dim, 2)
        else:
            self.fc_out = nn.Linear(curr_input_dim, 1)
        self.fc_out.weight.data.mul_(0.01)
        self.fc_out.bias.data.zero_()

    # def forward(self, latent_state, next_state, prev_state=None, actions=None):
    def forward(self, latent_state, state):
        # print(actions.shape)
        # print(state.device, latent_state.device)
        # print(latent_state.shape, state.shape)
        if self.input_state:
            # print(state.shape)
            hns = self.state_encoder(state)
        else:
            hns = torch.zeros(0, ).to(self.device)

        if self.input_latent:
            hnl = self.latent_encoder(latent_state)
            # hnl = latent_state
        else:
            hnl = torch.zeros(0, ).to(self.device)
        # print(hns.device, hnl.device)
        # print(hns.shape, hnl.shape)
        h = torch.cat((hns, hnl), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        out = self.fc_out(h)
        if self.pred_type == 'gaussian':
            mu, logvar = out.chunk(2, dim=-1)
            return mu, logvar
        else:
            return out
