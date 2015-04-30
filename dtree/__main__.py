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

n_uniques = [len(unique) if type(unique) == sets.Set else None for unique in uniques]
				
def invert(lst):
	d = {val: idx for idx, val in enumerate(lst)}
	d['?'] = None
	return d

encodes = [invert(list(unique)) if type(unique) == sets.Set else None for unique in uniques]
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

# Learn tree

import train

traverse = train.train(training, meta, n_uniques, classification_coln)

traverse(titles, decodes)
