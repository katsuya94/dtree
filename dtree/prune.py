import numpy as np
import copy

def prune(tree, validate):
	found = True
	best = validate(tree)

	print "%.2f%% before pruning." % (100.0 * best)

	while found:
		frontier = []

		def traverse(node, indices):
			childs_indices = [indices + [idx] for idx, child in enumerate(node.children) if child.children is not None]
			for child_indices in childs_indices:
				frontier.append(child_indices)
			for child_indices in childs_indices:
				traverse(node.children[child_indices[-1]], child_indices)

		if tree.children is not None:
			traverse(tree, [])

		found = False

		def test_prune(node, indices):
			if len(indices) == 0:
				temp = copy.deepcopy(node.children)
				node.children = None
				proposed = validate(tree)
				if proposed > best:
					return proposed
				else:
					node.children = temp
					return None
			else:
				return test_prune(node.children[indices[0]], indices[1:])

		for indices in frontier:
			success = test_prune(tree, indices)
			if success is not None:
				best = success
				found = True
				print "Prune found. (%.2f%%)" % (100.0 * best)
				break

