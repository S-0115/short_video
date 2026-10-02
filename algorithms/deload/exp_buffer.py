import numpy as np
import torch

USE_GPU = torch.cuda.is_available()
# USE_GPU = False
device = torch.device('cuda' if USE_GPU else 'cpu')

class ReplayBuffer:
    def __init__(self, N, episode_limit, batch_size):
        self.N = N
        self.episode_limit = episode_limit
        self.batch_size = batch_size
        self.episode_num = 0
        self.buffer = None
        self.reset_buffer()

        self.mean = 0.
        self.var = 1.
        self.count = 0
        # create a buffer (dictionary)

        self.episode_count = 0

    def reset_buffer(self):
        self.buffer = {'obs_n': np.empty([self.batch_size, self.episode_limit, self.N, 33]),
                       'v_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'a_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'raw_a_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'a_logprob_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'r_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'done_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       }
        self.episode_num = 0

    def store_transition(self, episode_step, obs_n, v_n, raw_a_n, a_logprob_n, r_n, done_n):
        # print(episode_step)
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['raw_a_n'][self.episode_num][episode_step] = raw_a_n
        self.buffer['a_logprob_n'][self.episode_num][episode_step] = a_logprob_n
        # self.buffer['r_n'][self.episode_num][episode_step] = r_n
        self.buffer['done_n'][self.episode_num][episode_step] = done_n

        batch_count = len(r_n)
        total_count = self.count + batch_count
        delta = np.mean(r_n) - self.mean
        self.mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = np.var(r_n) * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

        norm_r_n = (r_n - self.mean) / (np.sqrt(self.var) + 1e-8)
        # print(norm_r_n)
        self.buffer['r_n'][self.episode_num][episode_step] = norm_r_n

        # self.buffer['r_n'][self.episode_num][episode_step] = r_n

        self.episode_count += 1
        # print(self.episode_count)
        if self.episode_count == self.episode_limit:
            self.episode_num += 1
            self.episode_count = 0

    # def store_last_value(self, episode_step, v_n):
    #     self.buffer['v_n'][self.episode_num][episode_step] = v_n
    #     self.episode_num += 1

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32).to(device)
        return batch