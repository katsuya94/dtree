import argparse
import csv
import sets
import numpy as np
import time

import train
import prune

np.seterr(all='ignore')

# Parse arguments

parser = argparse.ArgumentParser(description='A decision tree utility.')
parser.add_argument('trainfile')
parser.add_argument('metafile')
parser.add_argument('--validate', metavar='validatefile')
parser.add_argument('--learning', metavar='validatefile')
parser.add_argument('--test', metavar='testfile')
parser.add_argument('--prune', metavar='prunefile')
parser.add_argument('--percent', metavar='trainingpercent', type=float, default=100.0)

args = parser.parse_args()

# Create internal representation

csv.register_dialect('dtree', skipinitialspace=True)

meta = None

with open(args.metafile, 'rb') as metafile:
	for row in csv.reader(metafile, dialect='dtree'):
		meta = row

classification_coln = len(meta) - 1

uniques = [sets.Set() if meta[coln] == 'nominal' else None for coln in range(len(meta))]

count = None

with open(args.trainfile, 'rb') as trainfile:
	reader = csv.reader(trainfile, dialect='dtree')
	next(reader)
	count = 0
	print "Pass 1..."
	for row in reader:
		count += 1
		for coln, data in enumerate(row):
			if meta[coln] == 'nominal':
				if data != '?':
					uniques[coln].add(data)
	print "Done."


n_uniques = np.fromiter((len(unique) if type(unique) == sets.Set else -1 for unique in uniques), dtype=np.int64)
				
def invert(lst):
	d = {val: idx for idx, val in enumerate(lst)}
	return d

encodes = [invert(list(unique)) if type(unique) == sets.Set else None for unique in uniques]
decodes = [{v: k for k, v in encode.iteritems()} if type(encode) == dict else encode for encode in encodes]

def encode(coln, data):
	if data == '?':
		return np.nan
	if meta[coln] == 'nominal':
		return encodes[coln].get(data, np.nan)
	if meta[coln] == 'numeric':
		return float(data)

titles = None
training = None

with open(args.trainfile, 'rb') as trainfile:
	reader = csv.reader(trainfile, dialect='dtree')
	titles = next(reader)
	training = np.empty((count, len(meta),), dtype=np.float64)
	print "Pass 2..."
	for example_idx, row in enumerate(reader):
		for coln, data in enumerate(row):
			training[example_idx, coln] = encode(coln, data)
	print "Done."

# Learn tree

def validate(filename, tree):
	correct = 0
	incorrect = 0
	with open(filename, 'rb') as validatefile:
		reader = csv.reader(validatefile, dialect='dtree')
		next(reader)
		for row in reader:
			example = np.fromiter((encode(coln, data) for coln, data in enumerate(row)), dtype=np.float64)
			if not np.isnan(example[classification_coln]):
				actual = int(example[classification_coln])
				distribution = tree.classify(example)
				classification = np.argmax(distribution)
				if actual == classification:
					correct += 1
				else:
					incorrect += 1
	return float(correct) / float(correct + incorrect)

def train_subset(percent):
	percent = max(min(percent, 1.0), 0.0)
	n = min(int(percent * count), count)
	print "Training... %0.2f%% (%d)" % (100.0 * percent, n,)
	subset = training[np.random.choice(count, size=n, replace=False), :]
	start = time.time()
	tree = train.train(subset, meta, n_uniques, classification_coln)
	print "Done. (%.2fs)" % (time.time() - start)
	if args.prune is not None:
		print "Pruning..."
		start = time.time()
		prune.prune(tree, lambda t: validate(args.prune, t))
		print "Done. (%.2fs)" % (time.time() - start)
	return tree

if args.validate is not None:
	tree = train_subset(args.percent / 100.0)
	print "Validating..."
	validity = validate(args.validate, tree)
	print "Validation: %.2f%%" % (100.0 * validity)

if args.learning is not None:
	print "Learning Curve..."
	points = []
	for percent in range(0, 110, 10):
		tree = train_subset(float(percent) / 100.0)
		validity = validate(args.learning, tree)
		points.append((percent, validity,))
	for point in points:
		print point
