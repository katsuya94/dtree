import argparse
import csv
import sets
import numpy as np
import time

import train
import prune

NUM_SAMPLES = 3

np.seterr(all='ignore')

# Parse arguments

parser = argparse.ArgumentParser(description='A decision tree utility.')
parser.add_argument('trainfile', help='training data')
parser.add_argument('metafile', help='meta data (nominal or numeric)')
parser.add_argument('--validate', metavar='validatefile', help='validation on held out data')
parser.add_argument('--learning', metavar='validatefile', help='learning curve based on validations')
parser.add_argument('--test', nargs=2, metavar=('testfile', 'outputfile'), help='generate classifications')
parser.add_argument('--prune', metavar='prunefile', help='prune to improve performance on held out data')
parser.add_argument('--percent', metavar='trainingpercent', type=float, default=100.0, help='percentage of data to train on (random sample)')
parser.add_argument('--dnf', action='store_true', help='print disjunctiive normal form')

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

encodes = [invert(list(unique)) if unique is not None else None for unique in uniques]
decodes = [{v: k for k, v in encode.iteritems()} if encode is not None else None for encode in encodes]

def encode(coln, data):
	if data == '?':
		return np.nan
	elif meta[coln] == 'nominal':
		return encodes[coln].get(data, np.nan)
	elif meta[coln] == 'numeric':
		return float(data)

def decode(coln, data):
	if data == None:
		return '?'
	elif meta[coln] == 'nominal':
		return decodes[coln].get(data, '?')
	elif meta[coln] == 'numeric':
		return '%0.2f' % data

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
	print "\n[ --validate ]"

	tree = train_subset(args.percent / 100.0)

	print "\nValidating..."
	validity = validate(args.validate, tree)
	print "Validation: %.2f%%" % (100.0 * validity)

if args.learning is not None:
	print "\n[ --learning ]"
	points = []
	for percent in range(0, 110, 10):
		validity = 0.0
		for _ in range(NUM_SAMPLES):
			tree = train_subset((args.percent / 100.0) * (float(percent) / 100.0))
			validity += validate(args.learning, tree) / float(NUM_SAMPLES)
			points.append((percent, validity,))
	for point in points:
		print point

if args.dnf:
	print "\n[ --dnf ]"

	tree = train_subset(args.percent / 100.0)

	classification_predicate_lists = [[] for _ in range(n_uniques[classification_coln])]

	def traverse(node, predicates):
		if node.children is None:
			classification_predicate_lists[np.argmax(node.distribution)].append(predicates)
		else:
			for split_idx, child in enumerate(node.children):
				traverse(child, predicates + [node.split.predicate(split_idx, decode, titles)])

	traverse(tree, [])

	for classification, predicate_lists in enumerate(classification_predicate_lists):
		print "\nIF %s THEN %s = %s" % (" OR ".join("(" + " AND ".join(predicate for predicate in predicate_list) + ")" for predicate_list in predicate_lists), titles[classification_coln], decode(classification_coln, classification))

if args.test is not None:
	print "\n[ --test ]"

	tree = train_subset(args.percent / 100.0)

	print "Testing..."

	with open(args.test[0], 'rb') as testfile:
		reader = csv.reader(testfile, dialect='dtree')
		with open(args.test[1], 'wb') as outputfile:
			writer = csv.writer(outputfile, dialect='dtree')
			writer.writerow(next(reader))
			for row in reader:
				example = np.fromiter((encode(coln, data) for coln, data in enumerate(row)), dtype=np.float64)
				classification = np.argmax(tree.classify(example))
				classified = list(row)
				classified[classification_coln] = decode(classification_coln, classification)
				writer.writerow(classified)

	print "Done."

