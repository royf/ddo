import numpy as np
import random
import copy
from .AbstractPlanner import *

"""
This module implements Monte Carlo Tree Search
"""


"""
This class stores the main data structure
"""
class MCTSTree(object):

	#initializes with children, parents, and sa tuples
	def __init__(self):
		self.children = []
		self.parent = None
		self.state_action = None
		self.reward = -np.inf

	def backpropagate(self,r):
		self.reward = r
		cur_tree = self.parent
		
		while cur_tree != None and r > cur_tree.reward:
			cur_tree.reward = r
			cur_tree = cur_tree.parent

	def treePrint(self):
		cur_tree = self
		while cur_tree.children != []:
			print(cur_tree.state_action, cur_tree.reward)
			best = np.argmax([c.reward for c in cur_tree.children])
			cur_tree = cur_tree.children[best]

	def argmax(self):
		result = []
		cur_tree = self
		while cur_tree.children != []:
			if cur_tree.state_action != None:
				result.append(cur_tree.state_action)

			best = np.argmax([c.reward for c in cur_tree.children])
			cur_tree = cur_tree.children[best]

		return result


"""
This class performs the tree search
"""
class MCTS(AbstractPlanner):

	#have to pass in an abstractenv descendent
	def __init__(self, domain, 
					   playout_iters=10):

		self.playout_iters = playout_iters

		super(MCTS, self).__init__(domain)

	"""
	Given a max depth and a start state, the 
	planner returns a trajectory
	"""
	def plan(self, max_depth, start=None):
		self.tree = MCTSTree()
		domaincopy = copy.copy(self.domain)
		domaincopy.init(start)
		state, time, reward, term = domaincopy.getState() 

		self.treeSearch(state,
						domaincopy.possibleActions(),
					    reward,
					    max_depth-time, 
					    self.tree)

		return self.tree.argmax()


	#randomly plays out from the current state taking an initial action
	def randomPlayout(self, state, action, reward, remaining_time):

		cur_possible_actions = np.array([action])

		domaincopy = copy.copy(self.domain)
		domaincopy.init(state=state, reward=reward)

		for t in range(0, remaining_time):

			action = np.random.choice(cur_possible_actions)

			domaincopy.play(action)

			cur_possible_actions = domaincopy.possibleActions()

			state, time, reward, term = domaincopy.getState() 

			#break if terminal
			if term:
				break

		return reward


	#Plays a single action, gets an expected state
	def getNextState(self, state, action):

		domaincopy = copy.copy(self.domain)
		domaincopy.init(state=state)
		domaincopy.play(action)
		state, time, reward, term = domaincopy.getState() 
		
		return state, term, domaincopy.possibleActions()



	"""
	This is the recursive method to actually do the treesearch
	"""
	def treeSearch(self,
					init_state, 
					init_actions, 
					init_reward, 
					init_time,
					mctsTree,
					traversed_set=set()):

		#initialize
		cur_state = init_state
		cur_state.flags.writeable = False
		acc_reward = init_reward
		cur_possible_actions = init_actions
		current_time = init_time

		#base case, no more time remaining
		if current_time == 0:
			return

		#expand each node
		for a in cur_possible_actions:

			#if we have already seen it
			if (cur_state.data, a) in traversed_set:
				continue

			#do the random playouts
			playouts = []
			for i in range(0,self.playout_iters):

				playouts.append(self.randomPlayout(cur_state,a,acc_reward,current_time))

			expected_reward = np.mean(playouts)

			#data structure update
			subTree = MCTSTree()
			mctsTree.children.append(subTree)
			subTree.parent = mctsTree
			subTree.state_action = (init_state, a)
			subTree.backpropagate(expected_reward)
			
			new_state, terminal, new_actions = self.getNextState(cur_state, a)

			#recurse
			if not terminal:

				new_set = copy.copy(traversed_set)
				new_set.add((cur_state.data, a))

				self.treeSearch(new_state, 
						   new_actions,
						   expected_reward,
						   current_time-1,
						   subTree,
						   new_set)
			else:
				return




