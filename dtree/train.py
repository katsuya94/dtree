import numpy as np

MAXIMUM_NUMERIC_DEPTH = 2

def e(probability):
	if np.isnan(probability) or probability == 0.0:
		return np.float64(-0.0)
	else:
		return -probability * np.log2(probability)

def train(training, meta, n_uniques, classification_coln):
	n_examples = training.shape[0]
	n_classifications = n_uniques[classification_coln]

	def classification_distribution(classification):
		if classification is None:
			distribution = np.full(n_classifications, 1.0 / float(n_classifications), dtype=np.float64)
		else:
			distribution = np.zeros(n_classifications, dtype=np.float64)
			distribution[classification] = 1.0
		return distribution

	class Split(object):
		def __init__(self):
			self.cached_entropy = None

		def entropy(self, examples, nonzero):
			split_counts = np.zeros((self.branch(), n_classifications,), dtype=np.float64)

			for example_idx in nonzero:
				example = training[example_idx, :]
				distribution = classification_distribution(None if np.isnan(example[classification_coln]) else int(example[classification_coln]))
				split_idx = self.split(example)
				if split_idx is None:
					split_counts[:, :] += examples[example_idx] * distribution / float(self.branch())
				else:
					split_counts[split_idx, :] += examples[example_idx] * distribution

			split_totals = np.sum(split_counts, axis=1)
			total = np.sum(split_totals)
			split_proportions = split_totals / total
			split_probabilities = split_counts / split_totals[:, np.newaxis]

			split_entropies = np.empty(self.branch(), dtype=np.float64)
			for split_idx in range(self.branch()):
				split_entropies[split_idx] = np.sum(e(probability) for probability in split_probabilities[split_idx, :])

			weighted_entropies = split_entropies * split_proportions

			self.cached_entropy = np.sum(weighted_entropies)

			return self.cached_entropy

		def examples_subsets(self, examples, nonzero):
			split_examples = np.zeros((self.branch(), n_examples,), dtype=np.float64)

			for example_idx in nonzero:
				split_idx = self.split(training[example_idx, :])
				if split_idx is None:
					split_examples[:, example_idx] += examples[example_idx] / float(self.branch())
				else:
					split_examples[split_idx, example_idx] += examples[example_idx]

			return split_examples

		def branch(self):
			raise NotImplementedError()

		def split(self, example):
			raise NotImplementedError()

		def predicate(self, split_idx, decode, titles):
			raise NotImplementedError()

	class NominalSplit(Split):
		def __init__(self, coln):
			super(NominalSplit, self).__init__()
			self.coln = coln

		def branch(self):
			return n_uniques[self.coln]

		def split(self, example):
			if np.isnan(example[self.coln]):
				return None
			else:
				return int(example[self.coln])

		def predicate(self, split_idx, decode, titles):
			return "%s = %s" % (titles[self.coln], decode(self.coln, split_idx),)

	class NumericSplit(Split):
		def __init__(self, coln, w):
			super(NumericSplit, self).__init__()
			self.coln = coln
			self.w = w

		def branch(self):
			return 2

		def split(self, example):
			if np.isnan(example[self.coln]):
				return None
			elif example[self.coln] < self.w:
				return 0
			else:
				return 1

		def predicate(self, split_idx, decode, titles):
			return "%s %s %s" % (titles[self.coln], '<' if split_idx == 0 else '>=', decode(self.coln, self.w),)

	def good_numeric_splits(nonzero, coln):
		values = training[nonzero, coln]
		if np.count_nonzero(~np.isnan(values)) == 0:
			return []
		median = np.nanmedian(values)
		std = np.nanstd(values)
		ws = [median, median - std, median + std]
		return [NumericSplit(coln, w) for w in ws]

	def best(examples, nonzero, restrictions):
		nominal_splits = [NominalSplit(coln) for coln, datatype in enumerate(meta) if datatype == 'nominal' and not restrictions[coln]]
		numeric_splits = [split for coln, datatype in enumerate(meta) for split in good_numeric_splits(nonzero, coln) if datatype == 'numeric' and restrictions[coln] <= MAXIMUM_NUMERIC_DEPTH]

		splits = nominal_splits + numeric_splits

		best_split = None
		best_entropy = np.inf
		for split in splits:
			entropy = split.entropy(examples, nonzero)
			if entropy < best_entropy:
				best_split = split
				best_entropy = entropy

		return best_split

	def homogeneous(nonzero):
		first_encountered = None
		for example_idx in nonzero:
			classification = training[example_idx, classification_coln]
			if np.isnan(classification):
				continue
			classification = int(classification)
			if first_encountered is None:
				first_encountered = classification
			elif classification != first_encountered:
				return None
		return first_encountered

	def weighted_distribution(nonzero):
		counts = np.zeros(n_classifications, dtype=np.float64)
		for example_idx in nonzero:
			classification = training[example_idx, classification_coln]
			counts += classification_distribution(None if np.isnan(classification) else int(classification))
		return counts / np.sum(counts)

	class Node(object):
		def __init__(self, examples, restrictions, prev_entropy):
			self.children = None
			nonzero = np.nonzero(examples)[0]
			self.num = len(nonzero)
			if self.num == 0:
				self.distribution = classification_distribution(None)
			else:
				homogeneous_classification = homogeneous(nonzero)
				if homogeneous_classification is not None:
					self.distribution = classification_distribution(homogeneous_classification)
				else:
					self.distribution = weighted_distribution(nonzero)
					self.split = best(examples, nonzero, restrictions)
					if self.split is not None and self.split.cached_entropy < prev_entropy:
						new_restrictions = list(restrictions)
						if type(self.split) == NominalSplit:
							new_restrictions[self.split.coln] = True
						if type(self.split) == NumericSplit:
							new_restrictions[self.split.coln] += 1
						examples_subsets = self.split.examples_subsets(examples, nonzero)
						self.children = [Node(examples_subsets[split_idx, :], new_restrictions, self.split.cached_entropy) for split_idx in range(self.split.branch())]

		def classify(self, example):
			if self.children is None:
				return self.distribution
			else:
				split_idx = self.split.split(example)
				if split_idx is None:
					branch = len(self.children)
					split_distributions = np.empty((branch, n_classifications,), dtype=np.float64)
					for split_idx, child in enumerate(self.children):
						split_distributions[split_idx, :] = child.classify(example)
					return np.sum(split_distributions, axis=0) / float(branch)
				else:
					return self.children[self.split.split(example)].classify(example)

	initial_restrictions = [False if datatype == 'nominal' else 0 for datatype in meta]
	initial_restrictions[classification_coln] = True

	tree = Node(np.ones(n_examples, dtype=np.float64), initial_restrictions, np.inf)

	return tree
