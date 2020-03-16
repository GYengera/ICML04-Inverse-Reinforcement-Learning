"""
Evaluate the optimal policy of an MDP.
"""
import numpy as np

def value_iteration_sparse(env, weights=None, eps=1e-3):
	"""
	@brief: Calculate optimal policy and corresponding optimal value function.

	@param env: environment class

	@param eps: threshold

	@return policy: optimal policy

	@return V: value function 
	"""
	if weights is None:
		weights = env.true_weights

	V = np.zeros([env.grid_size**2])
	policy = np.zeros([env.grid_size**2])

	iteration = 0
	while True:
		v_fn = V
		action_values = np.zeros([len(env.A), env.grid_size**2])

		for a in range(len(action_values)):
			action_values[a] = env.T[a].dot(weights + env.gamma*V)

		V = np.max(action_values, axis=0)

		policy = np.argmax(action_values, axis=0)

		if (max(V-v_fn) < eps):
			break
		iteration += 1

	return policy, V


def compute_feat_exp(env, policy, eps=1e-3):
	"""
	@brief: Compute feature expectation for given policy (equivalent to state visitation frequency). 
	"""
	T_pi = env.policy_transition_matrix(policy)
	mew = np.zeros([env.grid_size**2])
	while True:
		mew_old = mew
		mew = env.D_init + T_pi.dot(env.gamma * mew_old)
		if np.linalg.norm((mew - mew_old), ord=1) < eps:
			break
	return mew