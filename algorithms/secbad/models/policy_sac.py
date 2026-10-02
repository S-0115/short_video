import torch
import torch.nn as nn

from algorithms.secbad.utils import helpers as utl


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)  # 第一层全连接层，输入状态维度
        self.fc2 = nn.Linear(256, 256)  # 第二层全连接层
        self.mean = nn.Linear(256, action_dim)  # 输出动作均值
        self.log_std = nn.Linear(256, action_dim)  # 输出动作的对数标准差
        self.max_action = max_action  # 动作的最大值，用于缩放

    def forward(self, state):
        x = torch.relu(self.fc1(state))  # 激活第一层
        x = torch.relu(self.fc2(x))  # 激活第二层
        mean = self.mean(x)  # 计算动作均值
        log_std = self.log_std(x).clamp(-20, 2)  # 将对数标准差限制在合理范围内
        std = torch.exp(log_std)  # 通过对数标准差计算标准差
        return mean, std  # 返回均值和标准差

    def sample(self, state):
        mean, std = self.forward(state)  # 获取动作分布的均值和标准差
        normal = torch.distributions.Normal(mean, std)  # 正态分布
        x_t = normal.rsample()  # 使用重参数化技巧采样
        y_t = torch.tanh(x_t)  # 使用 Tanh 将动作限制在 [-1, 1]
        action = y_t * self.max_action  # 缩放动作到最大值范围
        log_prob = normal.log_prob(x_t)  # 计算动作的对数概率
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)  # Tanh 的修正项
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # 对每个维度求和


# Q 网络（价值函数，用于评估状态-动作对的价值）
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)  # 输入包括状态和动作
        self.fc2 = nn.Linear(256, 256)  # 第二层全连接层
        self.fc3 = nn.Linear(256, 1)  # 输出单一 Q 值

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # 将状态和动作连接起来作为输入
        x = torch.relu(self.fc1(x))  # 激活第一层
        x = torch.relu(self.fc2(x))  # 激活第二层
        x = self.fc3(x)  # 输出 Q 值
        return x  # 返回 Q 值



class SACPolicy(nn.Module):
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

        # initialise actor and critic


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
            q = q.masked_fill(mask == 0, -1e9)
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
