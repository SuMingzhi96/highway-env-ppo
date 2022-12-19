"""
	The file contains the PPO class to train with.
	NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
			It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

class PPO:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, policy_class, env, **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
		"""
		# Make sure the environment is compatible with our code
		assert(type(env.observation_space) == gym.spaces.Box)
		assert(type(env.action_space) == gym.spaces.Box)

		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)

		# Extract environment information
		self.env = env
		# self.obs_dim = env.observation_space.shape[0]
		self.obs_dim = 30
		self.act_dim = 1

		 # Initialize actor and critic networks
		self.actor = policy_class(self.obs_dim, self.act_dim)                                                   # ALG STEP 1
		self.critic = policy_class(self.obs_dim, 1)

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)

		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}

	def learn(self, total_timesteps):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		while t_so_far < total_timesteps:                                                                       # ALG STEP 2
			# Autobots, roll out (just kidding, we're collecting our batch simulations here)
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()                     # ALG STEP 3

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Logging timesteps so far and iterations so far
			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far

			# Calculate advantage at k-th iteration
			V, _ = self.evaluate(batch_obs, batch_acts)
			A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

			# One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
			# isn't theoretically necessary, but in practice it decreases the variance of 
			# our advantages and makes convergence much more stable and faster. I added this because
			# solving some environments was too unstable without it.
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
				# Calculate V_phi and pi_theta(a_t | s_t)
				V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				# NOTE: we just subtract the logs, which is the same as
				# dividing the values and then canceling the log with e^log.
				# For why we use log probabilities instead of actual probabilities,
				# here's a great explanation: 
				# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
				# TL;DR makes gradient ascent easier behind the scenes.
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor and critic losses.
				# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
				# the performance function, but Adam minimizes the loss. So minimizing the negative
				# performance function maximizes it.
				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())

			# Print a summary of our training so far
			self._log_summary()

			# Save our model if it's time
			if i_so_far % self.save_freq == 0:
				torch.save(self.actor.state_dict(), './ppo_actor.pth')
				torch.save(self.critic.state_dict(), './ppo_critic.pth')

	def rollout(self):
		"""
			Too many transformers references, I'm sorry. This is where we collect the batch of data
			from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.

			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		# Batch data. For more details, check function header.
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_rtgs = []
		batch_lens = []

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		ep_rews = []

		t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			ep_rews = [] # rewards collected per episode

			# Reset the environment. sNote that obs is short for observation.
			obs = self.env.reset()
			# print("obs", obs)
			obs_y = obs[0][1]
			obs = np.array(obs).flatten()[0:30]
			print("obs_y", obs_y)
			# obs_dim = obs.reshape(-1)
			# print("bos-dim", obs_dim)
			# print("obs", obs)
			done = False
			action_bef = 0
			t3 = 0
			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			for ep_t in range(self.max_timesteps_per_episode):
				# If render is specified, render the environment
				if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
					self.env.render()
				self.env.render()
				t += 1  # Increment timesteps ran this batch so far
				print(t)
				# Track observations in this batch
				batch_obs.append(obs)

				# Calculate action and make a step in the env. 
				# Note that rew is short for reward.
				action, log_prob = self.get_action(obs)
				# print("action_ori", action)

				low_bound = self.env.action_space.low
				upper_bound = self.env.action_space.high
				action = np.clip(action, low_bound+0.97, upper_bound-0.97)
				wrong_action = False
				# print("action_in", action)
				t3 = t3
				t2 = t3
				done = False
				rew = 0
				if obs_y <= 0.02 and action >= 0.01:
					print("右转")
					while obs[1] < 0.0217 and not done:
						# print("while1-obs[1]", obs[1])
						t2 = t2 + 1
						obs, reward, done, info = self.env.step(action)
						obs = np.array(obs).flatten()[0:30]
						self.env.render()
						collision = self.env.vehicle.crashed
						if -0.02 > obs[1] > 0.10 or collision:
							done = True
							break
						else:
							done = False
					t2 = t2 + 1
					while t2 != 0 and not done:
						# print("while1-")
						t2 = t2 - 1
						obs, reward, done, info = self.env.step(-action)
						self.env.render()
						collision = self.env.vehicle.crashed
						if -0.02 > obs[0][1] > 0.10 or collision:
							done = True
							break
						else:
							done = False
					action_zheng = [0]
					obs, reward, done, info = self.env.step(action_zheng)
					obs_y = obs[0][1]
					self.env.render()
					rew = 0.1/action[0]
				elif obs_y >= 0.06 and action <= -0.01:
					print("左转")
					while obs[1] > 0.0583 and not done:
						# print("while2-obs[1]", obs[1])
						t2 = t2 + 1
						obs, reward, done, info = self.env.step(action)
						obs = np.array(obs).flatten()[0:30]
						self.env.render()
						collision = self.env.vehicle.crashed
						if -0.02 > obs[1] > 0.10 or collision:
							done = True
							break
						else:
							done = False
					t2 = t2 + 1
					while t2 != 0 and not done:
						# print("while2-")
						# print("t2", t2)
						t2 = t2 - 1
						obs, reward, done, info = self.env.step(-action)
						self.env.render()
						collision = self.env.vehicle.crashed
						if -0.02 > obs[0][1] > 0.10 or collision:
							done = True
							break
						else:
							done = False
					action_zheng = [0]
					obs, reward, done, info = self.env.step(action_zheng)
					# print("done", done)
					obs_y = obs[0][1]
					rew = -0.1/action[0]
					self.env.render()

				elif 0.02 <= obs_y <= 0.06 and action >= 0.01:
					print("右转")
					while obs[1] < 0.0617 and not done:
						# print("while1-obs[1]", obs[1])
						t2 = t2 + 1
						obs, reward, done, info = self.env.step(action)
						obs = np.array(obs).flatten()[0:30]
						self.env.render()
						collision = self.env.vehicle.crashed
						if -0.02 > obs[1] > 0.10 or collision:
							done = True
							break
						else:
							done = False
					t2 = t2 + 1
					while t2 != 0 and not done:
						# print("while1-")
						t2 = t2 - 1
						obs, reward, done, info = self.env.step(-action)
						self.env.render()
						collision = self.env.vehicle.crashed
						if -0.02 > obs[0][1] > 0.10 or collision:
							done = True
							break
						else:
							done = False
					action_zheng = [0]
					obs, reward, done, info = self.env.step(action_zheng)
					obs_y = obs[0][1]
					self.env.render()
					rew = 0.1 / action[0]
				elif 0.02 <= obs_y <= 0.06 and action <= -0.01:
					print("左转")
					while obs[1] > 0.0183 and not done:
						# print("while2-obs[1]", obs[1])
						t2 = t2 + 1
						obs, reward, done, info = self.env.step(action)
						obs = np.array(obs).flatten()[0:30]
						self.env.render()
						collision = self.env.vehicle.crashed
						if -0.02 > obs[1] > 0.10 or collision:
							done = True
							break
						else:
							done = False
					t2 = t2 + 1
					while t2 != 0 and not done:
						# print("while2-")
						# print("t2", t2)
						t2 = t2 - 1
						obs, reward, done, info = self.env.step(-action)
						self.env.render()
						collision = self.env.vehicle.crashed
						if -0.02 > obs[0][1] > 0.10 or collision:
							done = True
							break
						else:
							done = False
					action_zheng = [0]
					obs, reward, done, info = self.env.step(action_zheng)
					# print("done", done)
					obs_y = obs[0][1]
					rew = -0.1 / action[0]
					self.env.render()
				elif -0.01 < action < 0.01:
					print("直行")
					obs_before = obs[1]
					t4 = 10
					while t4 != 0:
						t4 = t4 - 1
						action_zheng = [0]
						obs, reward, done, info = self.env.step(action_zheng)
						if 0.005 < obs_before < 0.02 or 0.045 < obs_before < 0.06 or 0.085 < obs_before < 0.10:
							print("小左转")
							action_zuo = [-0.15] #0.12 有点小
							self.env.step(action_zuo)
							for i in range(10):
								self.env.render()
							action_zuo = [0.15]
							self.env.step(action_zuo)
							for i in range(10):
								self.env.render()
							self.env.step(action_zheng)
							self.env.render()
						elif -0.005 > obs_before > -0.02 or 0.035 > obs_before > 0.02 or 0.075 > obs_before > 0.06:
							print("小右转")
							action_you = [0.15]
							self.env.step(action_you)
							for i in range(10):
								self.env.render()
							action_you = [-0.15]
							self.env.step(action_you)
							for i in range(10):
								self.env.render()
							self.env.step(action_zheng)
							self.env.render()
						self.env.render()
					rew = 15
				else:
					print("错误直行")
					t4 = 4
					obs_before = obs[1]
					while t4 != 0:
						t4 = t4 - 1
						action_zheng = [0]
						obs, reward, done, info = self.env.step(action_zheng)
						if 0.005 < obs_before < 0.02 or 0.045 < obs_before < 0.06 or 0.085 < obs_before < 0.10:
							print("小左转")
							action_zuo = [-0.2]
							self.env.step(action_zuo)
							for i in range(10):
								self.env.render()
							action_zuo = [0.2]
							self.env.step(action_zuo)
							for i in range(10):
								self.env.render()
							self.env.step(action_zheng)
							self.env.render()
						elif -0.005 > obs_before > -0.02 or 0.035 > obs_before > 0.02 or 0.075 > obs_before > 0.06:
							print("小右转")
							action_you = [0.2]
							self.env.step(action_you)
							for i in range(10):
								self.env.render()
							action_you = [-0.2]
							self.env.step(action_you)
							for i in range(10):
								self.env.render()
							self.env.step(action_zheng)
							self.env.render()
						self.env.render()
					rew = -150
					wrong_action = True
				obs = np.array(obs).flatten()[0:30]
				# print("obs平铺", obs)
				# print("obs", np.array(obs).flatten()[0:12])
				# print("扭正")
				# rew = 1
				# print("obs", obs[0:3])
				# print("reward:", rew)
				# print("done:", done)
				# print("action", action)
				# Track recent reward, action, and action log probability
				collision = self.env.vehicle.crashed
				if collision:
					rew = -150
					print("collision")
				print("rew", rew)
				ep_rews.append(rew)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)
				collision = self.env.vehicle.crashed
				if -0.02 > obs[1] > 0.10 or collision or wrong_action:
					break
			# self.env.close()
			# Track episodic lengths and rewards
			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rews)
		self.env.close()
		# Reshape data as tensors in the shape specified in function description, before returning
		# print(batch_obs)
		batch_obs = torch.tensor(batch_obs, dtype=torch.float)
		print("end")
		batch_acts = torch.tensor(batch_acts, dtype=torch.float)
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
		batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4
		# print("batchrew", batch_rews)
		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rews'] = batch_rews
		self.logger['batch_lens'] = batch_lens

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

	def compute_rtgs(self, batch_rews):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
		batch_rtgs = []
		# print("batch_rew", batch_rews)
		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

		return batch_rtgs

	def get_action(self, obs):
		"""
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
		# Query the actor network for a mean action
		mean = self.actor(obs)

		# Create a distribution with the mean action and std from the covariance matrix above.
		# For more information on how this distribution works, check out Andrew Ng's lecture on it:
		# https://www.youtube.com/watch?v=JjB58InuTqM
		dist = MultivariateNormal(mean, self.cov_mat)

		# Sample an action from the distribution
		action = dist.sample()

		# Calculate the log probability for that action
		log_prob = dist.log_prob(action)
		# print(action)
		# Return the sampled action and the log probability of that action in our distribution
		# print(action.detach().numpy())
		return action.detach().numpy(), log_prob.detach()

	def evaluate(self, batch_obs, batch_acts):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)

			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""
		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		V = self.critic(batch_obs).squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		mean = self.actor(batch_obs)
		dist = MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		return V, log_probs

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		# Miscellaneous parameters
		self.render = True                              # If we should render during rollout
		self.render_every_i = 10                        # Only render every n iterations
		self.save_freq = 10                             # How often we save in number of iterations
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time_ns()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))

		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])
		avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rews = str(round(avg_ep_rews, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"Timesteps So Far: {t_so_far}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []
