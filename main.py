import argparse
from env import GridWorld
from mdp_solver import value_iteration_sparse
from irl import irl
import numpy as np
import time

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--grid_size', nargs='?', const=1, type=int, default=8)
	parser.add_argument('--macro_block', nargs='?', const=1, type=int, default=2)
	parser.add_argument('--gamma', nargs='?', const=1, type=float, default=0.90)
	args = parser.parse_args()

	start_time = time.time()
	env = GridWorld(args.grid_size, args.macro_block, args.gamma)
	env_time = time.time() - start_time
	print ("Time to create environment: {}".format(env_time))

	start_time = time.time()
	policy, v = value_iteration_sparse(env)
	value_iteration_time = time.time() - start_time
	print ("Time to find optimal policy: {}".format(value_iteration_time))

	
	start_time = time.time()
	weights_inv, policy_inv = irl(env, policy)
	irl_time = time.time() - start_time
	print ("Time to solve IRL problem: {}".format(irl_time))
	

	print ("Displaying comparison between optimal policy and IRL policy:")
	env.draw(policy, policy_inv, weights_inv)
	
if __name__ == '__main__':
	main()