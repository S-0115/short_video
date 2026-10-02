import gym
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F

from models.decoder import RewardDecoder
from models.encoder import RNNEncoder
from utils.storage_vae import RolloutStorageVAE

class VaribadVAE:
    """
    VAE of VariBAD:
    - has an encoder and decoder
    - can compute the ELBO loss
    - can update the VAE (encoder+decoder)
    """

    def __init__(self, args, logger, get_iter_idx, train_steps):

        self.args = args
        self.device = args.device

        self.logger = logger
        self.get_iter_idx = get_iter_idx

        # initialise the encoder
        self.encoder = self.initialise_encoder()

        # initialise the decoders (returns None for unused decoders)
        self.reward_decoder = self.initialise_decoder()

        # initialise rollout storage for the VAE update
        # (this differs from the data that the on-policy RL algorithm uses)
        self.rollout_storage = RolloutStorageVAE(num_processes=self.args.num_processes,
                                                 max_trajectory_len=self.args.traj_len,
                                                 zero_pad=True,
                                                 max_num_rollouts=self.args.size_vae_buffer,

                                                 num_feature_policy=self.args.num_feature,
                                                 state_dim_policy=self.args.state_dim,

                                                 num_feature_encoder=2,
                                                 state_dim_encoder=3,

                                                 action_dim=self.args.action_dim,
                                                 num_action=self.args.num_action,
                                                 # probability of adding a new trajectory to buffer
                                                 vae_buffer_add_thresh=self.args.vae_buffer_add_thresh,
                                                 )

        # initalise optimiser for the encoder and decoders
        decoder_params = []
        if self.args.decode_reward:
            decoder_params.extend(self.reward_decoder.parameters())
        self.optimiser_vae = torch.optim.Adam(
            [*self.encoder.parameters(), *decoder_params], lr=self.args.lr_vae)

        self.anneal_lr = self.args.vae_anneal_lr
        if self.anneal_lr:
            def lam(f): return 1 - f / train_steps
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimiser_vae, lr_lambda=lam)

    def initialise_encoder(self):
        """ Initialises and returns an RNN encoder """
        encoder = RNNEncoder(
            args=self.args,
            layers_before_gru=self.args.encoder_layers_before_gru,
            hidden_size=self.args.encoder_gru_hidden_size,
            layers_after_gru=self.args.encoder_layers_after_gru,
            latent_dim=self.args.latent_dim,
            state_dim=self.args.state_dim,
            state_embed_dim=self.args.encoder_state_embedding_dim,
        ).to(self.device)
        return encoder

    def initialise_decoder(self):
        """ Initialises and returns the (state/reward/task) decoder as specified in self.args """

        latent_dim = self.args.latent_dim * 2

        reward_decoder = RewardDecoder(
            args=self.args,
            layers=self.args.reward_decoder_layers,
            latent_dim=latent_dim,
            latent_embed_dim=self.args.latent_embedding_dim,
            state_dim=self.args.state_dim,
            state_embed_dim=self.args.state_embedding_dim,
            pred_type=self.args.rew_pred_type,
            input_state=self.args.input_state,
            input_latent=self.args.input_latent,
        ).to(self.device)

        return reward_decoder

    # def compute_rew_reconstruction_loss(self, latent, prev_obs, next_obs, action, reward, return_predictions=False):
    def compute_rew_reconstruction_loss(self, latent, prev_obs, n_rewards):
        """ 计算reward预测损失，MSE"""

        if self.args.rew_pred_type == 'gaussian':
            mu, logvar = self.reward_decoder(latent, prev_obs)
            var = logvar.exp()
            loss = 0.5 * (logvar + (n_rewards - mu).pow(2) / var).mean(dim=-1)
        else:
            rew_pred = self.reward_decoder(latent, prev_obs)
            loss = 0.5 * (n_rewards - rew_pred).pow(2).mean(dim=-1)
        return loss

    def compute_loss(self, latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, vae_prev_obs_policy, vae_next_obs_policy, vae_n_rewards, vae_actions,
                     vae_rewards, r_t=None):
        # 获取隐含特征
        latent = torch.cat((latent_mean, latent_logvar), dim=-1)
        # 计算损失
        rew_reconstruction_loss = self.compute_rew_reconstruction_loss(latent, vae_next_obs_policy, vae_n_rewards)
        # print(rew_reconstruction_loss.shape)
        rew_reconstruction_loss = rew_reconstruction_loss.mean()

        return rew_reconstruction_loss

    def compute_vae_loss(self, update=False):
        """ Returns the VAE loss """

        if not self.rollout_storage.ready_for_update():
            print('vae not ready')
            return 0

        # 抽取轨迹，重新计算隐含特征和loss
        vae_prev_obs, vae_next_obs, vae_prev_obs_policy, vae_next_obs_policy, vae_n_rewards, vae_actions, vae_rewards, r_t = self.rollout_storage.get_batch(batchsize=self.args.vae_batch_num_trajs)
        latent_mean, latent_logvar, _ = self.encoder(
                                                        states=vae_next_obs.to(self.device),
                                                        hidden_state=None,
                                                        r_t=r_t.to(self.device)
                                                        )

        loss = self.compute_loss(latent_mean, latent_logvar,
                                 vae_prev_obs.to(self.device), vae_next_obs.to(self.device),
                                 vae_prev_obs_policy.to(self.device), vae_next_obs_policy.to(self.device),
                                 vae_n_rewards.to(self.device), vae_actions.to(self.device),
                                 vae_rewards.to(self.device), r_t=r_t.to(self.device)).mean()
        print(f'loss: {loss}')

        if update:
            self.optimiser_vae.zero_grad()
            loss.backward()

            # update
            self.optimiser_vae.step()

        self.log(loss)

        return loss

    def log(self, loss):

        iter_idx = self.get_iter_idx()
        if iter_idx % self.args.log_interval == 0:
            self.logger.add('vae_losses__sum',loss, iter_idx)