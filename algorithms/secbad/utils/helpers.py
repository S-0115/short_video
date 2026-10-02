import os
import pickle
# import pickle5 as pickle
import random
import warnings
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def seed(seed):
    print('Seeding random, torch, numpy.')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

def select_action(args,
                  policy,
                  deterministic,
                  state=None,
                  mask=None,
                  latent_mean=None, latent_logvar=None
                  ):
    """ Select action using the policy. """
    latent = get_latent_for_policy(args=args, latent_mean=latent_mean, latent_logvar=latent_logvar)
    action = policy.act(state=state, latent=latent, mask=mask,
                        deterministic=deterministic)
    # print(action)
    if isinstance(action, list) or isinstance(action, tuple):
        value, action = action
    else:
        value = None

    action = action.to(args.device)

    return value, action


def get_latent_for_policy(args, latent_mean=None, latent_logvar=None):

    if (latent_mean is None) and (latent_logvar is None):
        return None

    latent = torch.cat((latent_mean, latent_logvar), dim=-1)

    return latent

class FeatureExtractor_State(nn.Module):
    """ Used for extrating features for states """

    def __init__(self, input_feature_num, input_size, output_size, activation_function):
        super(FeatureExtractor_State, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        self.fc1s = nn.ModuleList([])
        # self.fc1s.append(nn.Linear(1, output_size))
        self.fc1s.append(nn.Linear(1, output_size))
        for i in range(input_feature_num - 1):
            self.fc1s.append(nn.Linear(input_size, output_size))

    def forward(self, inputs):
        inputs = inputs.float()
        # print(inputs.shape)

        outputs = []

        for i in range(len(self.fc1s)):
            if i == 0:
                output = self.activation_function(self.fc1s[i](inputs[..., i, -1:]))
            else:
                output = self.activation_function(self.fc1s[i](inputs[..., i, :]))
            outputs.append(output)

        merge_net = torch.cat(outputs, dim=-1)
        # print(merge_net.shape)

        return merge_net

# 只提取用户行为一个特征
class FeatureExtractor_Encoder(nn.Module):
    """ Used for extrating features for states """

    def __init__(self, input_feature_num, input_size, output_size, activation_function):
        super(FeatureExtractor_Encoder, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        self.fc1s = nn.ModuleList([])
        for i in range(input_feature_num):
            self.fc1s.append(nn.Linear(input_size, output_size))

    def forward(self, inputs):
        inputs = inputs.float()
        # print(inputs.shape)
        outputs = []
        for i in range(len(self.fc1s)):
            output = self.activation_function(self.fc1s[i](inputs[..., i, :]))
            outputs.append(output)
        merge_net = torch.cat(outputs, dim=-1)

        return merge_net

class FeatureExtractor(nn.Module):
    """ Used for extrating features for actions/rewards """

    def __init__(self, input_size, output_size, activation_function):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, inputs):
        inputs = inputs.float()
        output = self.activation_function(self.fc(inputs))
        return output

def save_obj(obj, folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'rb') as f:
        return pickle.load(f)


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # PyTorch version.
    def __init__(self, device, epsilon=1e-4, shape=()):
        self.device = device
        self.mean = torch.zeros(shape).float().to(device)
        self.var = torch.ones(shape).float().to(device)
        self.count = epsilon
        self.shape = shape

    def update(self, x):
        # print(f'x shape is {x.shape}')
        shape = (-1,) + self.shape
        # print(shape)
        x = x.view(shape)
        # print(f'view x shape is {x.shape}')
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
        # print(f'update is {self.mean} {self.var} {self.count}')

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))
