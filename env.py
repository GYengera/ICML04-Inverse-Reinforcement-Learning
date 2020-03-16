import numpy as np
import random
from scipy import sparse
import matplotlib.pyplot as plt

class GridWorld:

	def __init__(self, grid_size, macro_block, gamma):
		self.grid_size = grid_size
		self._macro_block = macro_block
		self._weights_dim = grid_size//macro_block
		self.true_weights = self._create_weights()
		self.D_init = np.ones([self.grid_size**2]) / (self.grid_size**2)
		self.A = np.array([0,1,2,3]) #['left', 'right', 'up', 'down']
		self.gamma = gamma
		self.T, self.T_dense = self._transition()


	def _create_weights(self):
		"""
		@brief: Sample reward weights.
		"""
		sample_weights = np.zeros([self._weights_dim, self._weights_dim])
		for i in range(self._weights_dim):
			for j in range(self._weights_dim):
				sample_weights[i][j] = random.choices([0,1],[0.9,0.1])[0]
				if sample_weights[i][j]==1:
					sample_weights[i][j] = random.uniform(0,1)
		
		sample_weights = sample_weights/sum(sum(sample_weights))
		
		true_weights = np.zeros([self.grid_size,self.grid_size])
		for i in range(self._weights_dim):
			for j in range(self._weights_dim):
				true_weights[i*self._macro_block:(i+1)*self._macro_block,j*self._macro_block:(j+1)*self._macro_block] = \
				sample_weights[i][j]

		true_weights = true_weights.flatten()
		return true_weights

	def initialize_state(self):
		"""
		@brief: Initialize state in GridWorld with uniform probability across the grid.
		"""
		i = random.uniform(0,self.grid_size-1)
		j = random.uniform(0,self.grid_size-1)
		return np.array([i,j])

	def _transition(self):
		"""
		@brief: Defines the transition matrix.

		@return: transition matrix of dimension SxSxA
		"""
		T = np.zeros([np.size(self.A,0), self.grid_size**2, self.grid_size**2])
		for i in range(self.grid_size):
			for j in range(self.grid_size):
				pos = i*self.grid_size + j
				if ((j+1)%self.grid_size==0): #Right wall
					right = pos
					left = pos-1
				elif (j%self.grid_size==0): #Left wall
					right = pos+1
					left = pos
				else:
					right = pos+1
					left = pos-1

				if (i==0): #Top wall
					up = pos
					down = pos + self.grid_size
				elif (i==self.grid_size-1): #Bottom wall
					up = pos - self.grid_size
					down = pos
				else:
					up = pos - self.grid_size
					down = pos + self.grid_size

				T[0,pos,left] += 0.7
				T[0,pos,right] += 0.3/4.
				T[0,pos,up] += 0.3/4.
				T[0,pos,down] += 0.3/4.

				T[1,pos,left] += 0.3/4.
				T[1,pos,right] += 0.7
				T[1,pos,up] += 0.3/4.
				T[1,pos,down] += 0.3/4.

				T[2,pos,left] += 0.3/4.
				T[2,pos,right] += 0.3/4.
				T[2,pos,up] += 0.7
				T[2,pos,down] += 0.3/4.

				T[3,pos,left] += 0.3/4.
				T[3,pos,right] += 0.3/4.
				T[3,pos,up] += 0.3/4.
				T[3,pos,down] += 0.7

				T[:,pos,pos] += 0.3/4.
		
		Trans = []
		Trans.append(sparse.csr_matrix(T[0]))
		Trans.append(sparse.csr_matrix(T[1]))
		Trans.append(sparse.csr_matrix(T[2]))
		Trans.append(sparse.csr_matrix(T[3]))
		return Trans, T

	def get_reward(self, state=None):
		"""
		@brief: Returns reward value at specified state or present state

		@param state: state within gridworld

		@rtype: float
		@return: reward value
		"""
		if state.any()==None:
			state = self.pos

		return self.true_weights[state[0], state[1]]

	def next_states(self, state=None):
		"""
		@brief: Returns possible next states.

		@param state: state within gridworld.

		@return next_states: vector of next states.
		"""
		if state.any()==None:
			state = self.pos

		return np.unique(np.array([state,[max(0,state[0]-1), state[1]],[min(self.grid_size-1,state[0]+1), state[1]],
								[state[0], max(0,state[1]-1)],[state[0], min(self.grid_size-1, state[1]+1)]]),axis=0)

	def policy_transition_matrix(self, policy):
		"""
		@brief: Compute policy specific transition matrix.
		"""
		T_pi = np.zeros([self.grid_size**2, self.grid_size**2])
		for i in range(len(policy)):
			T_pi[i] = self.T_dense[policy[i].astype(int)][i,:]
		return sparse.csr_matrix(np.transpose(T_pi))

	def draw(self, policy, irl_policy, weights):
		"""
		@brief: Visualize comparison between optimal policy and IRL policy.
		"""
		fig, axs = plt.subplots(2)
		w1 = axs[0].pcolor(np.flip(self.true_weights.reshape(self.grid_size, self.grid_size),0))
		w2 = axs[1].pcolor(np.flip(weights.reshape(self.grid_size, self.grid_size),0))
		fig.colorbar(w1, ax = axs[0])
		fig.colorbar(w2, ax = axs[1])
		
		x = np.linspace(0, self.grid_size - 1, self.grid_size) + 0.5
		y = np.linspace(self.grid_size - 1, 0, self.grid_size) + 0.5
		X, Y = np.meshgrid(x, y)
		zeros = np.zeros((self.grid_size, self.grid_size))

		pi_ = np.zeros([len(policy), len(self.A)])
		pi_irl = np.zeros([len(policy), len(self.A)])
		for s in range(len(policy)):
			pi_[s, policy[s]] = 0.45
			pi_irl[s, irl_policy[s]] = 0.45
		pi_ = pi_.reshape(self.grid_size, self.grid_size, 4)
		pi_irl = pi_irl.reshape(self.grid_size, self.grid_size, 4)

		axs[0].quiver(X, Y, zeros, -pi_[:,:,3], scale=1, units='xy')
		axs[0].quiver(X, Y, -pi_[:,:,0], zeros, scale=1, units='xy')
		axs[0].quiver(X, Y, zeros, pi_[:,:,2], scale=1, units='xy')
		axs[0].quiver(X, Y, pi_[:,:,1], zeros, scale=1, units='xy')

		axs[1].quiver(X, Y, zeros, -pi_irl[:,:,3], scale=1, units='xy')
		axs[1].quiver(X, Y, -pi_irl[:,:,0], zeros, scale=1, units='xy')
		axs[1].quiver(X, Y, zeros, pi_irl[:,:,2], scale=1, units='xy')
		axs[1].quiver(X, Y, pi_irl[:,:,1], zeros, scale=1, units='xy')

		axs[0].set_title("Actual reward weights and policy")
		axs[1].set_title("IRL reward weights and policy")
		
		plt.show()