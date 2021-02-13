import gym
from collections import deque
import numpy as np


class DelayedRewardEnv(gym.Wrapper):
	def __init__(self, env, nstep):
		super(DelayedRewardEnv, self).__init__(env)
		self.env = env
		self.nstep = nstep
		self.reset_buffer()

	def reset_buffer(self):
		self.rewards = []

	def reset(self):
		self.reset_buffer()
		o = self.env.reset()
		return o

	def step(self, action):
		o, r, done, info = self.env.step(action)
		# record
		self.rewards.append(r)
		if len(self.rewards) == self.nstep or done is True:
			r = np.sum(self.rewards)
			self.reset_buffer()
		else:
			r = 0.0
		return o, r, done, info
