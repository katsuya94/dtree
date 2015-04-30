# Parse arguments

import argparse

parser = argparse.ArgumentParser(description='A decision tree utility.')
parser.add_argument('trainfile')
parser.add_argument('metafile')
parser.add_argument('--validate', metavar='validatefile', required=False)
parser.add_argument('--test', metavar='testfile', required=False)

args = parser.parse_args()

# Create internal representation

import csv

csv.register_dialect('dtree', skipinitialspace=True)

meta = None

with open(args.metafile, 'rb') as metafile:
	for row in csv.reader(metafile, dialect='dtree'):
		meta = row

classification_coln = len(meta) - 1

import sets

uniques = [sets.Set() if meta[coln] == 'nominal' else None for coln in range(len(meta))]

count = None

with open(args.trainfile, 'rb') as trainfile:
	reader = csv.reader(trainfile, dialect='dtree')
	next(reader)
	count = 0
	for row in reader:
		count += 1
		for coln, data in enumerate(row):
			if meta[coln] == 'nominal':
				if data != '?':
					uniques[coln].add(data)
				
def invert(lst):
	d = {val: idx for idx, val in enumerate(lst)}
	d['?'] = None
	return d

encodes = [invert(list(unique)) if type(unique) == sets.Set else unique for unique in uniques]
decodes = [{v: k for k, v in encode.iteritems()} if type(encode) == dict else encode for encode in encodes]

def encode(coln, data):
	if meta[coln] == 'nominal':
		return encodes[coln][data]
	else:
		return None if data == '?' else float(data)

titles = None
training = None

with open(args.trainfile, 'rb') as trainfile:
	reader = csv.reader(trainfile, dialect='dtree')
	titles = next(reader)
	training = tuple(tuple(encode(coln, data) for coln, data in enumerate(row)) for row in reader)

print titles

# Learn tree

import math

class Split(object):
	def entropy(self, example_idxs):
		split_counts = [[0 for _ in range(len(uniques[-1]))] for _ in range(self.branch())]

		for example_idx in example_idxs:
			example = training[example_idx]
			classifications = range(len(uniques[-1])) if example[-1] is None else [example[-1]]
			for idx in self.split(example):
				for classification in classifications:
					split_counts[idx][classification] += 1

		split_totals = [float(sum(counts)) for counts in split_counts]

		split_probabilities = [[count / split_total for count in counts] for counts, split_total in zip(split_counts, split_totals)]

		split_entropies = [-sum(probability * math.log(probability, 2) for probability in probabilities) for probabilities in split_probabilities]

		return sum(split_entropies)

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
		self.coln = coln

	def branch(self):
		return len(uniques[self.coln])

	def split(self, example):
		if example[self.coln] is None:
			return range(self.branch())
		else:
			return [example[self.coln]]


class NumericSplit(Split):
	pass

def best(example_idxs, restricted_colns):
	splits = [NominalSplit(coln) for coln, datatype in enumerate(meta) if datatype == 'nominal' and coln not in restricted_colns]

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
		classification = training[example_idx][-1]
		if first_encountered is None:
			first_encountered = classification
		elif classification is not None and classification != first_encountered:
			return None
	return first_encountered

def majority(example_idxs):
	counts = [0 for _ in range(len(uniques[-1]))]
	for example_idx in example_idxs:
		classification = training[example_idx][-1]
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
	def __init__(self, example_idxs, restricted_colns):
		self.homogeneous = homogeneous(example_idxs)
		if self.homogeneous is None:
			self.split = best(example_idxs, restricted_colns)
			if self.split is None:
				self.homogeneous = majority(example_idxs)
			else:
				if type(self.split) == NominalSplit:
					restricted_colns = restricted_colns + [self.split.coln]
				self.children = [Node(example_idxs_subset, restricted_colns) for example_idxs_subset in self.split.split_examples(example_idxs)]

	def classify(self, example):
		if self.homogeneous is not None:
			return self.homogeneous
		else:
			return self.children[self.split.split(example)].classify(example)

tree = Node(range(len(training)), [classification_coln])
