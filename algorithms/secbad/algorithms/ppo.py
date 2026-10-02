import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algorithms.secbad.utils import helpers as utl


class PPO:
    def __init__(self,
                 args,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 policy_optimiser,
                 policy_anneal_lr,
                 train_steps,
                 optimiser_vae=None,
                 lr=None,
                 clip_param=0.2,
                 ppo_epoch=5,
                 num_mini_batch=5,
                 eps=None,
                 use_huber_loss=True,
                 use_clipped_value_loss=True,
                 ):
        self.args = args

        # the model
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_huber_loss = use_huber_loss

        self.train_steps = train_steps

        # optimiser
        if policy_optimiser == 'adam':
            self.optimiser = optim.Adam(
                self.actor_critic.parameters(), lr=lr, eps=eps)
        elif policy_optimiser == 'rmsprop':
            self.optimiser = optim.RMSprop(
                self.actor_critic.parameters(), lr=lr, eps=eps, alpha=0.99)
        self.optimiser_vae = optimiser_vae

        self.policy_anneal_lr = policy_anneal_lr
        if policy_anneal_lr:
            def lam(f): return 1 - f / train_steps
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimiser, lr_lambda=lam)

    def update(self,
               policy_storage=None
               ):

        # -- get action values --
        advantages = policy_storage.returns[:-
                                            1] - policy_storage.value_preds[:-1]
        # print(advantages.mean(), advantages.std())
        # print(policy_storage.returns.shape)
        # print(policy_storage.rewards_normalised.shape)
        # print(policy_storage.returns.squeeze()[:,0])
        # print(policy_storage.rewards_normalised.squeeze()[:,0])
        # breakpoint()
        if self.args.norm_adv_for_policy:
            advantages = (advantages - advantages.mean()) / \
                (advantages.std() + 1e-8)

        # update the normalisation parameters of policy inputs before updating
        self.actor_critic.update_rms(
            args=self.args, policy_storage=policy_storage)
        # call this to make sure that the action_log_probs are computed
        # (needs to be done right here because of some caching thing when normalising actions)

        policy_storage.before_update(self.actor_critic)

        for e in range(self.ppo_epoch):

            data_generator = policy_storage.feed_forward_generator(
                advantages, self.num_mini_batch)
            sample_idx = 0
            for sample in data_generator:
                sample_idx = sample_idx + 1
                state_batch, actions_batch, latent_mean_batch, latent_logvar_batch, value_preds_batch, \
                    return_batch, old_action_log_probs_batch, adv_targ, mask_batch = sample
                # print('shape is ',state_batch.shape, actions_batch.shape, mask_batch.shape)

                if state_batch is not None:
                    state_batch = state_batch.detach()
                if latent_mean_batch is not None:
                    latent_mean_batch = latent_mean_batch.detach()
                    latent_logvar_batch = latent_logvar_batch.detach()

                if mask_batch is not None:
                    mask_batch = mask_batch.detach()
                if actions_batch is not None:
                    actions_batch = actions_batch.detach()

                latent_batch = utl.get_latent_for_policy(args=self.args,
                                                         latent_mean=latent_mean_batch,
                                                         latent_logvar=latent_logvar_batch
                                                         )
                if latent_batch is not None:
                    latent_batch = latent_batch.detach()
                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = \
                    self.actor_critic.evaluate_actions(state=state_batch, latent=latent_batch, mask=mask_batch,
                                                       action=actions_batch)

                # action_log_probs = torch.nan_to_num(action_log_probs, nan=-1e8, posinf=-1e8, neginf=-1e8)
                # old_action_log_probs_batch = torch.nan_to_num(old_action_log_probs_batch, nan=-1e8, posinf=-1e8, neginf=-1e8)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                # print(f"Ratio: {ratio.mean().item():.4f} ± {ratio.std().item():.4f}") # ratio 应该在1左右波动
                # print(ratio)

                # print(adv_targ)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_huber_loss and self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = F.smooth_l1_loss(
                        values, return_batch, reduction='none')
                    value_losses_clipped = F.smooth_l1_loss(
                        value_pred_clipped, return_batch, reduction='none')
                    value_loss = 0.5 * \
                        torch.max(value_losses, value_losses_clipped).mean()
                elif self.use_huber_loss:
                    value_loss = F.smooth_l1_loss(values, return_batch)
                elif self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * \
                        torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # print(value_loss.shape, action_loss.shape, dist_entropy.shape)
                # ------------------------------------------------------------------
                # 计算 Loss
                # ------------------------------------------------------------------
                loss = value_loss * self.value_loss_coef + action_loss - self.entropy_coef * dist_entropy

                self.optimiser.zero_grad()

                loss.backward()

                # clip gradients
                # nn.utils.clip_grad_norm_(
                #     self.actor_critic.parameters(), self.args.policy_max_grad_norm)

                self.optimiser.step()
                # compute gradients (will attach to all networks involved in this computation)
                # print(f'critic loss:{value_loss}, actor_loss:{actor_loss} action loss:{action_loss} entropy:{dist_entropy}')

                print(f'loss:{loss}, value_loss:{value_loss} action loss:{action_loss} entropy:{dist_entropy}')



        if self.policy_anneal_lr:
            self.lr_scheduler.step()

        # self.entropy_coef = max(0.05, self.entropy_coef - 0.00005)

    def act(self, state, latent, mask, deterministic=False):
        # return self.actor_critic.act(state=state, latent=latent, belief=belief, task=task, deterministic=deterministic)
        # print(state)
        # print(latent)
        # print(mask)
        return self.actor_critic.act(state=state, latent=latent, mask=mask, deterministic=deterministic)
