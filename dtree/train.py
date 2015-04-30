import math

MAXIMUM_NUMERIC_DEPTH = 4

def e(probability):
	if probability == 0.0:
		return -0.0
	else:
		return -probability * math.log(probability, 2)

def divide(numerator, denominator):
	try:
		return numerator / denominator
	except ZeroDivisionError:
		return 0

def train(training, meta, n_uniques, classification_coln):
	class Split(object):
		def __init__(self):
			self.cached_total_entropy = None

		def entropy(self, example_idxs):
			split_counts = [[0 for _ in range(n_uniques[classification_coln])] for _ in range(self.branch())]

			for example_idx in example_idxs:
				example = training[example_idx]
				classifications = range(n_uniques[classification_coln]) if example[classification_coln] is None else [example[classification_coln]]
				for idx in self.split(example):
					for classification in classifications:
						split_counts[idx][classification] += 1

			split_totals = [float(sum(counts)) for counts in split_counts]

			total = sum(split_totals)

			split_proportions = [divide(split_total, total) for split_total in split_totals]

			split_probabilities = [[divide(count, split_total) for count in counts] for counts, split_total in zip(split_counts, split_totals)]

			split_entropies = [sum(e(probability) for probability in probabilities) for probabilities in split_probabilities]

			weighted_entropies = [split_proportion * split_entropy for split_proportion, split_entropy in zip(split_proportions, split_entropies)]

			self.cached_total_entropy = sum(weighted_entropies)

			return self.cached_total_entropy

		def split_examples(self, example_idxs):
			split_idxs = [[] for _ in range(self.branch())]

			for example_idx in example_idxs:
				for idx in self.split(training[example_idx]):
					split_idxs[idx].append(example_idx)

			return split_idxs

		def branch(self):
			raise NotImplementedError()

		def split(self, example):
			raise NotImplementedError()

	class NominalSplit(Split):
		def __init__(self, coln):
			super(NominalSplit, self).__init__()
			self.coln = coln

		def branch(self):
			return n_uniques[self.coln]

		def split(self, example):
			if example[self.coln] is None:
				return range(self.branch())
			else:
				return [example[self.coln]]


	class NumericSplit(Split):
		def __init__(self, coln, w):
			super(NumericSplit, self).__init__()
			self.coln = coln
			self.w = w

		def branch(self):
			return 2

		def split(self, example):
			if example[self.coln] is None:
				return [0, 1]
			if example[self.coln] < self.w:
				return [0]
			else:
				return [1]

	def good_numeric_splits(example_idxs, coln):
		values = [training[example_idx][coln] for example_idx in example_idxs if training[example_idx][coln] is not None]
		if len(values) > 0:
			mean = sum(values) / len(values)
			return [NumericSplit(coln, mean)]
		else:
			return []

	def best(example_idxs, restrictions):
		nominal_splits = [NominalSplit(coln) for coln, datatype in enumerate(meta) if datatype == 'nominal' and not restrictions[coln]]
		numeric_splits = [split for coln, datatype in enumerate(meta) for split in good_numeric_splits(example_idxs, coln) if datatype == 'numeric' and restrictions[coln] <= MAXIMUM_NUMERIC_DEPTH]

		splits = nominal_splits

		best_split = None
		best_entropy = float('inf')

		for split in splits:
			entropy = split.entropy(example_idxs)
			if entropy < best_entropy:
				best_split = split
				best_entropy = entropy

		return best_split

	def homogeneous(example_idxs):
		first_encountered = None
		for example_idx in example_idxs:
			classification = training[example_idx][classification_coln]
			if first_encountered is None:
				first_encountered = classification
			elif classification is not None and classification != first_encountered:
				return None
		return first_encountered

	def majority(example_idxs):
		counts = [0 for _ in range(n_uniques[classification_coln])]
		for example_idx in example_idxs:
			classification = training[example_idx][classification_coln]
			if classification is not None:
				counts[classification] += 1

		max_classification = None
		max_count = 0

		for classification, count in enumerate(counts):
			if count > max_count:
				max_classification = classification
				max_count = count

		return max_classification

	class Node(object):
		def __init__(self, example_idxs, restrictions):
			self.num = len(example_idxs)
			self.homogeneous = homogeneous(example_idxs)
			if self.homogeneous is None:
				self.split = best(example_idxs, restrictions)
				if self.split is None:
					self.homogeneous = majority(example_idxs)
					self.children = []
				else:
					new_restrictions = list(restrictions)
					if type(self.split) == NominalSplit:
						new_restrictions[self.split.coln] = True
					if type(self.split) == NumericSplit:
						new_restrictions[self.split.coln] += 1
					self.children = [Node(example_idxs_subset, new_restrictions) for example_idxs_subset in self.split.split_examples(example_idxs)]

		def classify(self, example):
			if self.homogeneous is not None:
				return self.homogeneous
			else:
				return self.children[self.split.split(example)].classify(example)

	restrictions = [False if datatype == 'nominal' else 0 for datatype in meta]
	restrictions[classification_coln] = True

	tree = Node(range(len(training)), restrictions)

	def traverse(node, level, titles, decodes):
		if node.homogeneous is not None:
			print '  ' * level + '%s: %s (%s)' % (titles[classification_coln], decodes[classification_coln][node.homogeneous], node.num,)
		else:
			if type(node.split) == NominalSplit:
				print '  ' * level + '%s (%s)' % (titles[node.split.coln], node.num,)
			if type(node.split) == NumericSplit:
				print '  ' * level + '%s < %s (%s)' % (titles[node.split.coln], node.split.w, node.num,)
			for child in node.children:
				traverse(child, level + 1, titles, decodes)

	return lambda titles, decodes: traverse(tree, 0, titles, decodes)
