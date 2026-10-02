import os.path
from modulefinder import Module

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SequentialSampler

ACTION_EPS = 1e-4
ENTROPY_EPS = 1e-6
GAMMA = 0.99
USE_GPU = torch.cuda.is_available()
device = "cuda" if USE_GPU else "cpu"


class BM_Actor(nn.Module):
	def __init__(self):
		super(BM_Actor, self).__init__()
		# Actor network
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=5, padding=2)
		self.conv3 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=5, padding=2)

		self.gru = nn.GRU(input_size=1, hidden_size=64, batch_first=True)

		self.fc1 = nn.Linear(144, 64)
		self.fc2 = nn.Linear(64, 6)

	def forward(self, states, bts):
		# print(states.shape)
		output_conv1 = F.leaky_relu(self.conv1(states))
		# print(output_conv1.shape)
		output_conv2 = F.leaky_relu(self.conv2(output_conv1))
		# print(output_conv2.shape)
		output_conv3 = F.leaky_relu(self.conv3(output_conv2))
		# print(output_conv3.shape)

		out, hidden = self.gru(bts)
		# print(bts.shape)
		# print(out)
		# print(out[:,-1,:].shape)
		out_gru = F.leaky_relu(out[:, -1, :])
		# print(out_gru.shape)

		flatten_conv3 = torch.flatten(output_conv3, start_dim=1)
		# print(flatten_conv3.shape)

		merge_input = torch.concat([flatten_conv3, out_gru], dim=1)
		# print(merge_input.shape)

		last_hidden = self.fc1(merge_input)
		output_fc1 = F.leaky_relu(last_hidden).clone()

		pi_video = self.fc2(output_fc1)
		# print(pi_video.shape)
		# print(pi_video)
		return pi_video, output_fc1


class BA_Actor(nn.Module):
	def __init__(self):
		super(BA_Actor, self).__init__()
		# Actor network

		self.fc1s = nn.ModuleList()
		for i in range(7):
			self.fc1s.append(nn.Linear(1, 64))

		self.fc2s = nn.ModuleList()
		for i in range(7):
			self.fc2s.append(nn.Linear(64, 16))

		self.fc3s = nn.ModuleList()
		for i in range(7):
			self.fc3s.append(nn.Linear(16, 4))

		self.gru = nn.GRU(input_size=1, hidden_size=64, batch_first=True)

		self.fc4 = nn.Linear(204, 64)
		self.fc5 = nn.Linear(64, 6)

	def forward(self, states, bts):
		# print(states.shape)
		states = states.unsqueeze(-1)
		# print(states.shape)
		output_fc1 = []
		for i, fc in enumerate(self.fc1s):
			input_fc = states[..., i, :, :]
			output_fc1.append(F.leaky_relu(fc(input_fc)))
		# print(output_fc1)
		# print(output_fc1.shape)

		output_fc2 = []
		for i, fc in enumerate(self.fc2s):
			output_fc2.append(F.leaky_relu(fc(output_fc1[i])))
		# print(output_fc2.shape)

		output_fc3 = []
		for i, fc in enumerate(self.fc3s):
			output_fc3.append(F.leaky_relu(fc(output_fc2[i])))
		# print(output_fc3.shape)
		# print(output_fc3[0].shape)
		output_fc = torch.stack(output_fc3, dim=1)
		# print(output_fc.shape)
		output_fc = torch.flatten(output_fc, start_dim=1)
		# print(output_fc.shape)

		out, hidden = self.gru(bts)
		out_gru = F.leaky_relu(out[:, -1, :])
		# print(out_gru.shape)

		merge_input = torch.concat([output_fc, out_gru], dim=1)
		# print(merge_input.shape)

		last_hidden = self.fc4(merge_input)
		output_fc4 = F.leaky_relu(last_hidden).clone()

		pi_bitrate = self.fc5(output_fc4)

		return pi_bitrate, output_fc4


class Critic(nn.Module):
	def __init__(self):
		super(Critic, self).__init__()
		# Actor network
		self.fc = nn.Linear(128, 1)
		self.fc.weight.data.mul_(0.01)
		self.fc.bias.data.zero_()

	def forward(self, inputs):
		value = self.fc(inputs)

		return value


class Trainer_RL:
	def __init__(self, learning_rate, bm_actor, ba_actor, critic):

		super(Trainer_RL, self).__init__()
		self.lr = learning_rate

		self.bm_actor = bm_actor
		self.ba_actor = ba_actor
		self.critic = critic

	def Initial(self, args):
		self.N = args.N
		self.episode_limit = args.episode_limit

		self.batch_size = args.batch_size
		self.mini_batch_size = args.mini_batch_size
		self.max_train_steps = args.max_train_steps
		self.lr = args.lr
		self.gamma = args.gamma
		self.lamda = args.lamda
		self.epsilon = args.epsilon
		self.K_epochs = args.K_epochs
		self.entropy_coef = args.entropy_coef
		self.set_adam_eps = args.set_adam_eps
		self.use_grad_clip = args.use_grad_clip
		self.use_adv_norm = args.use_adv_norm
		self.use_value_clip = args.use_value_clip
		self.use_lr_decay = args.use_lr_decay

		self.critic_pretrain_epoch = args.critic_pretrain_epoch

		if args.set_adam_eps:
			self.ac_optimizer = torch.optim.Adam(list(self.bm_actor.parameters()) + list(self.ba_actor.parameters()),
												 lr=self.lr,
												 eps=1e-8)
			self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
													 lr=self.lr,
													 eps=1e-8)
		else:
			self.ac_optimizer = torch.optim.Adam(list(self.bm_actor.parameters()) + list(self.ba_actor.parameters()),
												 lr=self.lr)
			self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
													 lr=self.lr)

	def entropy_decay(self):
		"""
		Decay entropy.
		"""
		self.entropy_coef = max(self.entropy_coef - 0.01, 0.01)

	def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
		lr_now = self.lr * (1 - total_steps / self.max_train_steps)
		for p in self.ac_optimizer.param_groups:
			p['lr'] = lr_now

	def get_network_params(self):
		return self.bm_actor.state_dict(), self.ba_actor.state_dict(), self.critic.state_dict()

	def set_network_params(self, input_network_params_BM, input_network_params_BA, input_network_params_CRITIC):
		self.bm_actor.load_state_dict(input_network_params_BM)
		self.ba_actor.load_state_dict(input_network_params_BA)
		self.critic.load_state_dict(input_network_params_CRITIC)

	def get_inputs(self, batch):
		bm_actor_inputs = batch['obs_bm_n'].reshape(self.batch_size, batch['obs_bm_n'].shape[1], self.N, 1, 4,
													5).tolist()
		ba_actor_inputs = batch['obs_ba_n'].reshape(self.batch_size, batch['obs_ba_n'].shape[1], self.N, 7, 5).tolist()
		bts = batch['obs_bts_n'].reshape(self.batch_size, batch['obs_bts_n'].shape[1], self.N, 5, 1).tolist()
		bm_mask = batch['bm_mask'].tolist()

		return bm_actor_inputs, ba_actor_inputs, bts, bm_mask

	def train(self, epoch, replay_buffer):
		batch = replay_buffer.get_training_data()  # get training data
		# Calculate the advantage using GAE
		adv = []
		gae = 0

		with torch.no_grad():  # adv and td_target have no gradient
			deltas_bm = batch['r_n'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_bm_n']) - batch['v_n'][:,
																									 :-1]  # deltas_bm.shape=(batch_size,episode_limit,N)
			# print(deltas_bm.shape, batch['done_bm_n'].shape)
			for t in reversed(range(self.episode_limit)):
				gae = deltas_bm[:, t] + self.gamma * self.lamda * gae * (1 - batch['done_bm_n'][:, t])
				adv.insert(0, gae)
			adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,episode_limit,N)
			v_target = adv + batch['v_n'][:, :-1]  # v_target.shape(batch_size,episode_limit,N)
			# print(adv.shape, v_target.shape)
			# print(adv.mean().item(), adv.std().item())
			if self.use_adv_norm:  # Trick 1: advantage normalization
				adv = (adv - adv.mean()) / (adv.std() + 1e-8)
				# print(adv.mean(), adv.std())
			if USE_GPU:
				adv = adv.cuda()
				v_target = v_target.cuda()

		"""
			Get actor_inputs and critic_inputs
			actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
			critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
		"""
		bm_actor_inputs, ba_actor_inputs, bts, bm_masks = self.get_inputs(batch)

		# Optimize policy for K epochs:
		for _ in range(self.K_epochs):
			for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
				"""
					get probs_now and values_now
					probs_now.shape=(mini_batch_size, episode_limit, N, action_dim_bm)
					values_now.shape=(mini_batch_size, episode_limit, N)
				"""
				bm_states_batch = torch.tensor(bm_actor_inputs)[index]
				ba_states_batch = torch.tensor(ba_actor_inputs)[index]
				bts_batch = torch.tensor(bts)[index]
				bm_masks_batch = torch.tensor(bm_masks)[index]
				ba_masks_batch = batch['ba_mask'][index].detach()
				if USE_GPU:
					ba_masks_batch = ba_masks_batch.cuda()

				# cr_states_batch = torch.tensor(critic_inputs)[index]

				probs_bm_now = []
				probs_ba_now = []
				values_now = []
				# print(len(index))
				for i in range(len(index)):
					bm_states, ba_states, bts_state, masks, masks_ba = bm_states_batch[i], ba_states_batch[i], \
					bts_batch[i], bm_masks_batch[i], ba_masks_batch[i]
					probs_bm = []
					probs_ba = []
					values = []
					for bm_state, ba_state, bt_state, bm_mask, ba_mask in zip(bm_states, ba_states, bts_state, masks,
																			  masks_ba):
						if USE_GPU:
							bm_state = bm_state.cuda()
							ba_state = ba_state.cuda()
							bt_state = bt_state.cuda()
							# cr_state = cr_state.cuda()
							bm_mask = bm_mask.cuda()
						bm_mask = bm_mask.squeeze(dim=1)

						prob_bm, bm_hidden = self.bm_actor(bm_state, bt_state)
						# print(prob_bm.shape)
						# print(bm_mask.shape)
						prob_bm.masked_fill_(bm_mask == 0, -float('Inf'))

						# bm_hidden = self.bm_actor.last_hidden_output.clone().detach()

						prob_ba, ba_hidden = self.ba_actor(ba_state, bt_state)

						# ba_hidden = self.ba_actor.last_hidden_output.clone().detach()
						# print(bm_hidden.shape, ba_hidden.shape)
						cr_input = torch.concat([bm_hidden.detach(), ba_hidden.detach()], dim=-1)
						# print(cr_input.shape)
						value = self.critic(cr_input)
						# print(prob_bm.shape, prob_ba.shape, value.shape)

						probs_bm.append(prob_bm)
						probs_ba.append(prob_ba)
						values.append(value)

					probs_bm = torch.stack(probs_bm)
					probs_ba = torch.stack(probs_ba)
					values = torch.stack(values)

					probs_bm_now.append(probs_bm)
					probs_ba_now.append(probs_ba)
					values_now.append(values)

				probs_bm_now = torch.stack(probs_bm_now)
				probs_ba_now = torch.stack(probs_ba_now)
				values_now = torch.stack(values_now)
				# print(values_now.shape)

				if USE_GPU:
					probs_bm_now = probs_bm_now.cuda()
					probs_ba_now = probs_ba_now.cuda()
					values_now = values_now.cuda()
				# print(values_now.shape)

				dist_now_bm = Categorical(logits=probs_bm_now)
				# print(batch['a_bm_n'].shape)
				# print(batch['a_bm_n'])
				dist_entropy_bm = dist_now_bm.entropy()  # dist_entropy_bm.shape=(mini_batch_size, episode_limit, N)
				# batch['a_n'][index].shape=(mini_batch_size, episode_limit, N)
				a_bm_n_batch = batch['a_bm_n'][index].detach()
				# print(probs_bm_now.shape)
				# print(a_bm_n_batch.shape)
				if USE_GPU:
					a_bm_n_batch = a_bm_n_batch.cuda()
				a_bm_logprob_n_now = dist_now_bm.log_prob(
					a_bm_n_batch)  # a_bm_logprob_n_now.shape=(mini_batch_size, episode_limit, N)
				# a/b=exp(log(a)-log(b))
				# print(a_bm_logprob_n_now)

				a_bm_logprob_n_batch = batch['a_bm_logprob_n'][index].detach()
				if USE_GPU:
					a_bm_logprob_n_batch = a_bm_logprob_n_batch.cuda()
				ratios_bm = torch.exp(
					a_bm_logprob_n_now - a_bm_logprob_n_batch)  # ratios_bm.shape=(mini_batch_size, episode_limit, N)
				# ratios_bm = torch.clamp(ratios_bm, max=5.0)

				# print(f'ratios_bm min :{ratios_bm.min().item()}, ratios_bm max {ratios_bm.max().item()}')

				surr1_bm = ratios_bm
				surr2_bm = torch.clamp(ratios_bm, 1 - self.epsilon, 1 + self.epsilon)
				bm_actor_loss = (-torch.min(surr1_bm, surr2_bm) * adv[index]).mean() - self.entropy_coef * dist_entropy_bm.mean()

				print(f'bm loss: {bm_actor_loss} , entropy: {dist_entropy_bm.mean()}')
				# print(surr1_bm)

				dist_now_ba = Categorical(logits=probs_ba_now)
				dist_entropy_ba = dist_now_ba.entropy()  # dist_entropy_bm.shape=(mini_batch_size, episode_limit, N)
				# batch['a_n'][index].shape=(mini_batch_size, episode_limit, N)
				a_ba_n_batch = batch['a_ba_n'][index].detach()
				if USE_GPU:
					a_ba_n_batch = a_ba_n_batch.cuda()
				a_ba_logprob_n_now = dist_now_ba.log_prob(
					a_ba_n_batch)  # a_bm_logprob_n_now.shape=(mini_batch_size, episode_limit, N)
				# a/b=exp(log(a)-log(b))
				a_ba_logprob_n_batch = batch['a_ba_logprob_n'][index].detach()
				if USE_GPU:
					a_ba_logprob_n_batch = a_ba_logprob_n_batch.cuda()

				ratios_ba = torch.exp(
					a_ba_logprob_n_now - a_ba_logprob_n_batch)  # ratios_bm.shape=(mini_batch_size, episode_limit, N)
				# ratios_ba = torch.clamp(ratios_ba, max=5.0)
				# print(f'ratios_ba min :{ratios_bm.min().item()}, ratios_ba max {ratios_bm.max().item()}')

				surr1_ba = ratios_ba
				surr2_ba = torch.clamp(ratios_ba, 1 - self.epsilon, 1 + self.epsilon)
				ba_actor_loss = (-torch.min(surr1_ba, surr2_ba) * adv[index] * ba_masks_batch).sum() / ba_masks_batch.sum() - self.entropy_coef * dist_entropy_ba.mean()
				# ba_actor_loss = ((-torch.min(surr1_ba, surr2_ba) * adv[index] * ba_masks_batch).sum() / ba_masks_batch.sum()
				# 				 - self.entropy_coef * (dist_entropy_ba * ba_masks_batch).sum() / ba_masks_batch.sum())

				print(f'ba loss: {ba_actor_loss} , entropy: {dist_entropy_ba.mean()}')
				# print(f'ba loss: {ba_actor_loss} , entropy: {dist_entropy_ba.mean()}')
				# print(ba_masks_batch.mean().item())
				# print(ba_masks.mean())
				# print(f'ba loss: {ba_actor_loss}')
				# print(ratios_ba, torch.clamp(ratios_ba, 1 - self.epsilon, 1 + self.epsilon))

				if self.use_value_clip:
					values_old = batch["v_n"][index, :-1].detach().unsqueeze(-1)
					if USE_GPU:
						values_old = values_old.cuda()
					values_error_clip = torch.clamp(values_now, values_old - self.epsilon, values_old + self.epsilon) - \
										v_target[index].unsqueeze(-1)
					values_error_original = values_now - v_target[index].unsqueeze(-1)
					critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2).mean() * 0.5
				# print(values_now.shape, values_old.shape, v_target[index].unsqueeze(-1).unsqueeze(-1).shape)
				# print(values_error_clip.shape, values_error_original.shape)
				else:
					# print(v_target[index].unsqueeze(-1).shape)
					critic_loss = ((values_now - v_target[index].unsqueeze(-1)) ** 2).mean()
				# print(f'value now {values_now.shape} v_target {v_target.shape}')
				print(f'critic loss : {critic_loss}')

				self.ac_optimizer.zero_grad()

				if epoch > self.critic_pretrain_epoch:
					ac_loss = (bm_actor_loss + ba_actor_loss)
					ac_loss.backward()

				self.critic_optimizer.zero_grad()

				critic_loss.backward()

				# if self.use_grad_clip:  # Trick 7: Gradient clip
				# 	torch.nn.utils.clip_grad_norm_(list(self.bm_actor.parameters()) + list(self.ba_actor.parameters()),
				# 								   10.)
				#
				# 	torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()), 10.)

				self.ac_optimizer.step()
				self.critic_optimizer.step()

				# print(f'ac loss : {ac_loss} bm actor loss {bm_actor_loss} ba actor loss {ba_actor_loss}')


		if self.use_lr_decay:
			self.lr_decay(epoch)

	def predict(self, input):
		with torch.no_grad():
			pi, value = self.actor_critic.forward(input)
			return pi

	def load_model(self, nn_model_bm, nn_model_ba):
		self.bm_actor.load_state_dict(torch.load(nn_model_bm, map_location=device, weights_only=False))
		self.ba_actor.load_state_dict(torch.load(nn_model_ba, map_location=device, weights_only=False))

	def load_critic(self, nn_model):
		self.critic.load_state_dict(torch.load(nn_model, map_location=device, weights_only=False))

	def save_critic(self, nn_model):
		torch.save(self.critic.state_dict(), nn_model)

	def save_model(self, nn_model_bm, nn_model_ba):
		model_params = self.bm_actor.state_dict()
		torch.save(model_params, nn_model_bm)

		model_params = self.ba_actor.state_dict()
		torch.save(model_params, nn_model_ba)


class Trainer_IL(nn.Module):
	def __init__(self, learning_rate, weight_decay, model):
		super(Trainer_IL, self).__init__()

		self.learning_rate = learning_rate
		self.weight_decay = weight_decay

		self.model = model

		self.loss_function = nn.CrossEntropyLoss(reduction='mean')

		self.optimizer = torch.optim.Adam(
			self.model.parameters(),
			lr=self.learning_rate
		)

	def forward(self, states, bts, labels):
		pi, _ = self.model(states, bts)
		# pi_video = F.softmax(pi, dim=1)
		# print(pi.shape)
		# log_pi = torch.log(pi_video + 1e-8)
		# print(labels.shape)
		loss = self.loss_function(pi, labels)
		# print(log_pi * labels)
		# loss = -(log_pi * labels).mean()
		# print(loss)

		return loss

	def predict(self, input):
		with torch.no_grad():
			pi, _ = self.model.forward(input)
			return pi

	def load_model(self, nn_model):
		self.model.load_state_dict(torch.load(nn_model))

	def save_model(self, nn_model):
		torch.save(self.model.state_dict(), nn_model)
