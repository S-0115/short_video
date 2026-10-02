"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import torch.nn as nn
import torch.optim as optim

from algorithms.secbad.utils import helpers as utl


class A2C:
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
                 eps=None,
                 ):
        self.args = args

        # the model
        self.actor_critic = actor_critic

        # coefficients for mixing the value and entropy loss
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        # optimiser
        if policy_optimiser == 'adam':
            self.optimiser = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        elif policy_optimiser == 'rmsprop':
            self.optimiser = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=0.99)
        self.optimiser_vae = optimiser_vae

        self.lr_scheduler = None
        self.policy_anneal_lr = policy_anneal_lr
        if policy_anneal_lr:
            lam = lambda f: 1 - f / train_steps
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda=lam)

    def update(self,
               policy_storage
               ):

        # get action values
        advantages = policy_storage.returns[:-1] - policy_storage.value_preds[:-1]

        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)

        # update the normalisation parameters of policy inputs before updating
        self.actor_critic.update_rms(args=self.args, policy_storage=policy_storage)

        # call this to make sure that the action_log_probs are computed
        # (needs to be done right here because of some caching thing when normalising actions)
        policy_storage.before_update(self.actor_critic)


        for i in range(4):
            data_generator = policy_storage.feed_forward_generator(advantages, 1)
            for sample in data_generator:

                state_batch, actions_batch, latent_mean_batch, latent_logvar_batch, value_preds_batch, \
                    return_batch, old_action_log_probs_batch, adv_targ, mask_batch = sample

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

                values, action_log_probs, dist_entropy = \
                    self.actor_critic.evaluate_actions(state=state_batch, latent=latent_batch, mask=mask_batch,
                                                       action=actions_batch)

                # --  UPDATE --

                # zero out the gradients
                self.optimiser.zero_grad()
                # compute policy loss and backprop
                value_loss = 0.5 * (return_batch - values).pow(2).mean()
                action_loss = -(adv_targ * action_log_probs).mean()

                # (loss = value loss + action loss + entropy loss, weighted)
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

                print(f'loss:{loss}, value_loss:{value_loss} action loss:{action_loss} entropy:{dist_entropy}')
                # compute gradients (will attach to all networks involved in this computation)
                loss.backward()
                # nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.policy_max_grad_norm)

                # update
                self.optimiser.step()

        if self.policy_anneal_lr:
            self.lr_scheduler.step()

    def act(self, state, latent, mask, deterministic=False):
        # return self.actor_critic.act(state=state, latent=latent, belief=belief, task=task, deterministic=deterministic)
        return self.actor_critic.act(state=state, latent=latent, mask=mask, deterministic=deterministic)
