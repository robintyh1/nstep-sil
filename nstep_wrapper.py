import gym
from collections import deque
import numpy as np


class NstepWrapper(gym.Wrapper):
	def __init__(self, env, nstep, gamma):
		super(NstepWrapper, self).__init__(env)
		self.env = env
		self.nstep = nstep
		self.gamma = gamma

		# nstep
		self.reset_buffer()

		# multiplier
		self.discount_multiplier = np.array([gamma**i for i in range(nstep)])

	def reset_buffer(self):
		self.obs = deque(maxlen=self.nstep)
		self.obs2 = deque(maxlen=self.nstep)
		self.acts = deque(maxlen=self.nstep)
		self.rews = deque(maxlen=self.nstep)
		self.dones = deque(maxlen=self.nstep)

	def reset(self):
		self.reset_buffer()
		o = self.env.reset()
		self.obs.append(o)
		return o

	def nstep_reward(self, rlist):
		return np.sum(self.discount_multiplier * np.array(rlist))

	def step(self, action):
		o, r, done, info = self.env.step(action)
		# record
		self.obs2.append(o)
		self.rews.append(r)
		self.acts.append(action)
		self.dones.append(done)
		# add to info if necessary
		if len(self.obs2) == self.nstep:
			nstep_r = self.nstep_reward(self.rews)
			nstep_data = [self.obs[0], self.acts[0], nstep_r, self.obs2[-1], self.dones[-1]]
			info.update({'nstep_data_{}'.format(self.nstep): nstep_data})
		# record obs1
		self.obs.append(o)
		return o, r, done, info
