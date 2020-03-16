import numpy as np
from mdp_solver import value_iteration_sparse, compute_feat_exp

def irl(env, expert_policy, eps=1):
	"""
	@brief: Implement projection irl algorithm to obtain reward weights that mimic expert behaviour.

	@param env: environment class

	@param expert_policy: Expert's policy

	@param eps: convergence threshold

	@return weights, policy: solution for reward weights and corresponding policy
	"""
	#sample initial random policy
	policy = np.zeros([env.grid_size**2])
	
	#Compute feature expectations
	mew_init = compute_feat_exp(env, policy)
	mew_exp = compute_feat_exp(env, expert_policy)

	weights = mew_exp - mew_init
	t = np.linalg.norm(weights, ord=2)
	t_old = 0
	iterations = 0
	while (t>eps and (t!=t_old)):
		t_old = t
		policy = value_iteration_sparse(env, weights)[0]
		mew_pred = compute_feat_exp(env, policy)
		x = mew_pred - mew_init
		y = mew_exp - mew_init
		mew_refine = mew_init + ((x.T.dot(y)) / (x.T.dot(x))) * x
		weights = abs(mew_exp - mew_refine)
		t = np.linalg.norm(weights, ord=2)
		mew_init = mew_refine
	return weights, policy