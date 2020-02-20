'''
certificates.py

Created by 

name: Federico Marcuzzi
e-mail: federico.marcuzzi@unive.it

date 20/02/2020
'''

import numpy as np
from itertools import combinations

'''
 FUNCTION -> : 'decision_path_sets' computes the decision path sets of a given model.
  *INPUT  -> model : tree to inspect
  *INPUT  -> X : set of original instances
  *INPUT  -> check_input : 'decision_path' parameter (from 'scikit-learn: Allow to bypass several input checking. Don’t use this parameter unless you know what you do.')
  *OUTPUT -> : a list containing a decision path for each instance (f+ and f- path)
'''
def decision_path_sets(tr,X,check_input=True):
	n_nodes = tr.tree_.node_count
	feature = tr.tree_.feature
	threshold = tr.tree_.threshold

	dict_node = {}
	for n,f,t in zip(np.arange(n_nodes),feature,threshold):
		dict_node[n] = (f,t)

	node_f = np.zeros((n_nodes,))
	for n,f in zip(np.arange(n_nodes),feature):
		node_f[n] = f

	out_list = []
	# for each instance it calculates the decision path set
	for x in X:
		# compute the path of the instance
		path = tr.decision_path(x.reshape(1,-1),check_input)
		for i in path:
			lista_features = set()
			for node in i.tocoo().col:
				pair = dict_node[node]
				f = pair[0]
				t = pair[1]

				if f >=0 :
					# if the instance goes to the left branch
					if x[f] < t:
						# adds an f- to the set
						lista_features.add(f)
					else:
						# otherwise adds an f+ to the set
						lista_features.add((-1 * f) -1)

			out_list.append(np.asarray(list(lista_features)).astype(int))

	return out_list

'''
 FUNCTION -> : 'get_features_per_ist' it calculates for each tree which instances are involved in the attack of a feature.
  *INPUT  -> model : forest to inspect
  *INPUT  -> X : set of original instances
  *INPUT  -> check_input : 'decision_path' parameter (from 'scikit-learn: Allow to bypass several input checking. Don’t use this parameter unless you know what you do.')
  *OUTPUT -> : for each tree returns a dictionary of features for each instance
'''
def get_features_per_ist(model,X,check_input=True):
	forest = [tr for tr in model]
	n_trees = len(forest)

	n_ist, _ = np.shape(X)
	dict_out = {}
	# for each tree inside the forest.
	for i in np.arange(n_trees):
		# creates a dictionary that maps a feature to a list of instances that contain it.
		dict_ft_insts = {}
		for ist,fts in zip(np.arange(n_ist),decision_path_sets(forest[i],X)):
			for ft in fts:
				if ft not in dict_ft_insts:
					dict_ft_insts[ft] = set()
				dict_ft_insts[ft].add(ist)

		for key,value in dict_ft_insts.items():
			dict_ft_insts[key] = np.asarray(list(value)).astype(int)

		# map the feature-instances dictionary to the tree
		dict_out[i] = dict_ft_insts

	return dict_out

'''
 FUNCTION -> : 'fast_lower_bound' calculate a pessimistic but fast lower-bound of the robustness of the model.
  *INPUT  -> model : forest to be certified
  *INPUT  -> X : set of original instances
  *INPUT  -> y : instances labels
  *INPUT  -> k_start : minimum attacker' budget
  *INPUT  -> k_end : maximum attacker' budget
  *OUTPUT -> : for each attack it returns the indexes of the instances attacked
'''
def fast_lower_bound(model,X,y,k_start,k_end=None):
	forest = [tr for tr in model]
	n_trees = len(forest)

	if k_end == None:
		k_end = k_start

	X = np.asarray(X)
	y = np.asarray(y)
	n_ist, n_feat = np.shape(X)
	trees_idx = np.arange(n_trees)

	# agreement threshold of the forest.
	maj_forest = n_trees // 2 + 1

	# calculate a priori the predictions for each tree
	predictions = np.asarray([forest[i].predict(X) for i in trees_idx])
	# calculates correct predictions: 1 correct, 0 incorrect.
	forest_errors = np.asarray([pred == y for pred in predictions]).T.astype(int)
	# calculate the agreement of the forest in the original instances.
	somma = np.sum(forest_errors,axis=1)

	# 'damage_vectors' is a list of damage vector implemented as a matrix for efficiency.
	damage_vectors = np.zeros((n_ist,n_feat*2))
	list_f_name = np.zeros((n_ist,n_feat*2))
	for i in trees_idx:
		# calculate the decision path for each instance with respect to tree 'i'.
		out = decision_path_sets(forest[i],X)
		# for each instance...
		for j in np.arange(n_ist):
			# ...assigns to each features the prediction of the attacked tree. The tree contributes to the damage vector if the prediction is correct, otherwise it makes no sense to attack it.
			damage_vectors[j,n_feat+out[j]] += forest_errors[j][i]
			list_f_name[j,n_feat+out[j]] = out[j]

	dict_broken = {}
	# calculates possible attackable instances for each attacker's budget.
	for k in np.arange(k_start,k_end+1):
		error_sum = np.zeros((n_ist, ))
		# for each instance...
		for i in np.arange(n_ist):
			# ...extracts its damage vector
			vet_ist = damage_vectors[i]
			vet_ft = list_f_name[i]
			zipped = zip(vet_ist,vet_ft)
			# sorts the features of the damage vector in descending order by number of attacked trees.
			zipped = sorted(zipped,key=lambda x:x[0],reverse=True)
			list_f_prese = []
			counter = 0

			# takes the k features that do the most damage. Keeping the constraints imposed by f+ and f-.
			for z in zipped:
				atk = z[0]
				ft = z[1]

				if ft not in list_f_prese:
					counter += 1
					error_sum[i] += atk
					if ft >= 0:
						list_f_prese.append(ft)
						list_f_prese.append((-1 * ft) -1)
					else:
						list_f_prese.append(ft)
						list_f_prese.append((ft + 1) * -1)

				if counter >= k:
					break

		# find the indexes of instances for which an attack may exist.
		dict_broken[k] = np.where((somma-error_sum)<maj_forest)[0]

	return dict_broken

'''
 FUNCTION -> : 'accurate_lower_bound' calculate a slow but more accurate lower-bound.
  *INPUT  -> model : forest to be certified
  *INPUT  -> X : set of original instances
  *INPUT  -> y : instances labels
  *INPUT  -> k_start : minimum attacker' budget
  *INPUT  -> k_end : maximum attacker' budget
  *OUTPUT -> : for each attack it returns the indexes of the instances attacked
'''
def accurate_lower_bound(model,X,y,k_start,k_end=None):
	if k_end == None:
		k_end = k_start

	forest = [tr for tr in model]
	n_trees = len(forest)

	# calculates for each tree returns a dictionary of features for each instance
	dict_tree_ft_insts = get_features_per_ist(model,X)

	X = np.asarray(X)
	y = np.asarray(y)
	n_ist, n_feat = np.shape(X)
	trees_idx = np.arange(n_trees)
	maj_forest = n_trees // 2 + 1

	# calculates correct predictions: 1 correct, 0 incorrect.
	forest_errors = np.asarray([forest[i].predict(X) == y for i in trees_idx]).T.astype(int)

	# create a dictionary that maps a feature with the trees that use it in the decision path
	dict_ft_tree = {}
	for i in trees_idx:
		for ft in dict_tree_ft_insts[i]:
			if ft not in dict_ft_tree:
				dict_ft_tree[ft] = set()
			dict_ft_tree[ft].add(i)

	dict_broken = {}

	# calculates possible attackable instances for each attacker's budget.
	for k in np.arange(k_start,k_end+1):
		dict_tuple = {}
		atk_ist = np.ones((n_ist, ))
		# computes all combinations of d, k features.
		for idx_f in combinations(dict_ft_tree.keys(),k):
			list_atk_trees_idx = []
			dict_tree_atk_ft = {}
			# given the combination of features calculates the set of possibly attackable trees (Ta).
			for idx in idx_f:
				list_atk_trees_idx += dict_ft_tree[idx]
				for tr in dict_ft_tree[idx]:
					if tr not in dict_tree_atk_ft:
						dict_tree_atk_ft[tr] = set()
					dict_tree_atk_ft[tr].add(idx)

			# set the contribution of trees in Ta to 0 since they are attacked.
			atk_tree_pred = np.zeros((n_ist, ))
			for tr in list_atk_trees_idx:
				pred_tr = np.copy(forest_errors[:,tr])
				for ft in dict_tree_atk_ft[tr]:
					dict_ft_insts = dict_tree_ft_insts[tr]
					if ft in dict_ft_insts:
						idx_inst = dict_ft_insts[ft]
						pred_tr[idx_inst] = 0
				atk_tree_pred += pred_tr

			# calculate indexes of  the non-attackable tree (Ts).
			safe_trees_idx = np.setdiff1d(trees_idx,list_atk_trees_idx)
			# calculate which attacked instances evade the model in this combination.
			predict = np.sum(forest_errors[:,safe_trees_idx],axis=1) + atk_tree_pred
			atk_ist *= (predict>=maj_forest)

		# calculate the  indexes of instances that fool the model
		dict_broken[k] = np.where(atk_ist==0)[0]

	return dict_broken