import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from algorithms.secbad.utils import helpers as utl


class RNNEncoder(nn.Module):
    def __init__(self,
                 args,
                 # network size
                 layers_before_gru=(),
                 hidden_size=64,  # hidden_size 是 RNN pass through 的隐藏层的大小，latent_dim 是 RNN 每个 step 的 output，latent 应该是会做为 PPO policy & value 的输入
                 layers_after_gru=(),
                 latent_dim=32,
                 state_dim=2,
                 state_embed_dim=10,
                 ):
        super(RNNEncoder, self).__init__()

        activation_function = args.policy_activation_function

        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        elif activation_function == 'leaky-relu':
            self.activation_function = nn.LeakyReLU()
        else:
            raise ValueError

        self.args = args
        self.device = args.device

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

        # embed action, state, reward
        curr_input_dim = 0
        if args.encoder_input_feature != 'all':
            input_feature_num_encoder = 1
        else:
            input_feature_num_encoder = 2
        self.state_encoder = utl.FeatureExtractor_Encoder(
            input_feature_num_encoder, 3, state_embed_dim, self.activation_function)
        curr_input_dim += input_feature_num_encoder * state_embed_dim

        # fully connected layers before the recurrent cell
        self.fc_before_gru = nn.ModuleList([])
        for i in range(len(layers_before_gru)):
            self.fc_before_gru.append(
                nn.Linear(curr_input_dim, layers_before_gru[i]))
            curr_input_dim = layers_before_gru[i]

        # recurrent unit
        # TODO: TEST RNN vs GRU vs LSTM
        self.encoder_type = args.encoder_type
        if self.encoder_type == "rnn":
            self.model = nn.RNN(
                                input_size=curr_input_dim,
                                hidden_size=hidden_size,
                                num_layers=1,
                                )
        elif self.encoder_type == "gru":
            self.model = nn.GRU(
                                input_size=curr_input_dim,
                                hidden_size=hidden_size,
                                num_layers=1,
                                )
        elif self.encoder_type == "lstm":
            self.model = nn.LSTM(
                                input_size=curr_input_dim,
                                hidden_size=hidden_size,
                                num_layers=1,
                                )
        elif self.encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=curr_input_dim,
                nhead=2,
                dim_feedforward=hidden_size,
                batch_first=False
            )

            self.model = nn.TransformerEncoder(
                encoder_layer,
                num_layers=2
            )
            self.post_model = nn.Linear(curr_input_dim, 64)

        # 对rnn,gru,lstm做参数初始化   transformer的参数类型太复杂且初始参数已经由pytorch优化过，在这里不重复进行初始化
        if self.encoder_type in ["rnn", "gru", "lstm"]:
            for name, param in self.model.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

        # fully connected layers after the recurrent cell
        curr_input_dim = hidden_size
        self.fc_after_gru = nn.ModuleList([])
        for i in range(len(layers_after_gru)):
            self.fc_after_gru.append(
                nn.Linear(curr_input_dim, layers_after_gru[i]))
            curr_input_dim = layers_after_gru[i]

        # output layer
        self.fc_mu = nn.Linear(curr_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(curr_input_dim, latent_dim)

    def prior(self, batch_size):

        # TODO: add option to incorporate the initial state

        # we start out with a hidden state of zero
        if self.encoder_type == "rnn" or self.encoder_type == "gru":
            hidden_state = torch.zeros(
                (1, batch_size, self.hidden_size), requires_grad=True).to(self.device)
        elif self.encoder_type == "lstm":
            h_n = torch.zeros(
                (1, batch_size, self.hidden_size), requires_grad=True).to(self.device)
            c_n = torch.zeros(
                (1, batch_size, self.hidden_size), requires_grad=True).to(self.device)
            hidden_state = (h_n, c_n)
        elif self.encoder_type == "transformer":
            hidden_state = []

        h = torch.zeros(
            (1, batch_size, self.hidden_size), requires_grad=True).to(self.device)
        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_gru)):
            # print(i, h.device)
            h = F.relu(self.fc_after_gru[i](h))

        # outputs
        # print(next(self.fc_mu.parameters()).device, h.device)
        latent_mean = self.fc_mu(h)
        latent_logvar = self.fc_logvar(h)

        return latent_mean, latent_logvar, hidden_state

    def encode_sequence(self, h, hidden_state, r_t):
        '''
        :param h:
        :param hidden_state:
        :param r_t: r_t == None时，为收集轨迹。 r_t ！= None时，为vae更新参数。
        :return:
        '''
        if self.encoder_type == "rnn" or self.encoder_type == "gru":
            if r_t is None:
                # print('r_t None')
                output, _ = self.model(h, hidden_state)
                hidden = output.clone()
            else:
                # print('r_t not None')
                batch_size = r_t.shape[1]
                output = []
                for batch_id in range(batch_size):
                    output_curr_batch = []
                    r_t_curr_batch = r_t[:, batch_id, :]
                    for i in range(1, len(r_t_curr_batch) + 1):
                        r_t_ = int(r_t_curr_batch[i - 1].item())

                        left_idx = max(0, i - r_t_)
                        if left_idx == i:
                            left_idx -= 1
                        # print(left_idx, i, batch_id)
                        curr_input = h[left_idx:i, batch_id, :]
                        output_curr_h, _ = self.model(curr_input)
                        # print(output_curr_h.shape)
                        # print(output_curr_h[-1].shape)
                        output_curr_batch.append(output_curr_h[-1].unsqueeze(0).unsqueeze(0))
                    output_curr_batch = torch.cat(output_curr_batch, dim=0)
                    # print(output_curr_batch.shape)
                    output.append(output_curr_batch)
                output = torch.cat(output, dim=1)
                hidden = output.clone()
        elif self.encoder_type == "lstm":
            if r_t is None:
                # print('r_t None')
                output, (h_n, c_n) = self.model(h, hidden_state)
                hidden = (h_n, c_n)
            else:
                # print('r_t not None')
                batch_size = r_t.shape[1]
                output = []
                for batch_id in range(batch_size):
                    output_curr_batch = []
                    r_t_curr_batch = r_t[:, batch_id, :]
                    for i in range(1, len(r_t_curr_batch) + 1):
                        r_t_ = int(r_t_curr_batch[i - 1].item())

                        left_idx = max(0, i - r_t_)
                        if left_idx == i:
                            left_idx -= 1
                        # print(left_idx, i, batch_id)
                        curr_input = h[left_idx:i, batch_id, :]
                        output_curr_h, _ = self.model(curr_input)
                        # print(output_curr_h.shape)
                        # print(output_curr_h[-1].shape)
                        output_curr_batch.append(output_curr_h[-1].unsqueeze(0).unsqueeze(0))
                    output_curr_batch = torch.cat(output_curr_batch, dim=0)
                    # print(output_curr_batch.shape)
                    output.append(output_curr_batch)
                output = torch.cat(output, dim=1)
                hidden = output.clone()
        elif self.encoder_type == "transformer":

            if r_t is None:
                # print('r_t None')
                # causal mask（强烈建议）
                seq_len = h.shape[0]
                mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=h.device),
                    diagonal=1
                ).bool()

                output_model = self.model(h, mask=mask)
                output = self.post_model(output_model)[-1].unsqueeze(0)
                hidden = None
            else:
                # print('r_t not None')
                batch_size = r_t.shape[1]
                output = []
                for batch_id in range(batch_size):
                    output_curr_batch = []
                    r_t_curr_batch = r_t[:, batch_id, :]
                    for i in range(1, len(r_t_curr_batch) + 1):
                        r_t_ = int(r_t_curr_batch[i - 1].item())

                        left_idx = max(0, i - r_t_)
                        if left_idx == i:
                            left_idx -= 1
                        # print(left_idx, i, batch_id)
                        curr_input = h[left_idx:i, batch_id, :]
                        output_model_curr_h = self.model(curr_input)[-1].unsqueeze(0)
                        output_curr_h = self.post_model(output_model_curr_h)
                        # print(output_curr_h.shape)
                        # print(output_curr_h[-1].shape)
                        output_curr_batch.append(output_curr_h[-1].unsqueeze(0).unsqueeze(0))
                    output_curr_batch = torch.cat(output_curr_batch, dim=0)
                    # print(output_curr_batch.shape)
                    output.append(output_curr_batch)
                output = torch.cat(output, dim=1)
                hidden = None

        return output, hidden

    def forward(self, states, hidden_state, r_t=None):
    # def forward(self, states, h_n, c_n, r_t=None):
        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        For one-step predictions, sequence_len=1 and hidden_state!=None.
        For feeding in entire trajectories, sequence_len>1 and hidden_state=None.
        In the latter case, we return embeddings of length sequence_len+1 since they include the prior.
        """
        # shape should be: sequence_len x batch_size x hidden_size
        # print(states.shape, hidden_state.shape)
        # print(r_t)
        # print(actions.shape, states.shape, rewards.shape)
        states = states.reshape((-1, *states.shape[-3:]))

        # print(actions.shape, states.shape, rewards.shape)

        if hidden_state is not None:
            # if the sequence_len is one, this will add a dimension at dim 0 (otherwise will be the same)
            # print(type(hidden_state))
            if type(hidden_state) is tuple:
                # print(hidden_state[0].shape)
                hidden_state = tuple([hidden_state_.reshape((-1, *hidden_state_.shape[-2:])) for hidden_state_ in hidden_state])
                # print(hidden_state[0].shape)
            elif type(hidden_state) is list:
                # 针对transformer的处理
                hidden_state.append(states)
                states = torch.cat(hidden_state, dim=0)
            else:
                hidden_state = hidden_state.reshape((-1, *hidden_state.shape[-2:]))

        # hidden_state = tuple([h_n, c_n])

        # extract features for states, actions, rewards
        # 这个是丢给GRU的前一步，就是简单的MLP
        # print(actions.shape, states.shape, rewards.shape)
        h = self.state_encoder(states)

        # forward through fully connected layers before GRU
        for i in range(len(self.fc_before_gru)):
            h = F.relu(self.fc_before_gru[i](h))

        gru_h, output = self.encode_sequence(h, hidden_state, r_t)
        # print(gru_h.shape)

        if output is None:
            output = hidden_state

        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_gru)):
            gru_h = F.relu(self.fc_after_gru[i](gru_h))

        # outputs

        latent_mean = self.fc_mu(gru_h)
        latent_logvar = self.fc_logvar(gru_h)

        if latent_mean.shape[0] == 1:
            latent_mean, latent_logvar = latent_mean[0], latent_logvar[0]

        return latent_mean, latent_logvar, output
