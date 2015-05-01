# dtree - a Python decision tree utility
By Adrien Katsuya Tateno

```
usage: dtree [-h] [--validate validatefile] [--learning validatefile]
             [--test testfile outputfile] [--prune prunefile]
             [--percent trainingpercent] [--dnf]
             trainfile metafile

A decision tree utility.

positional arguments:
  trainfile             training data
  metafile              meta data (nominal or numeric)

optional arguments:
  -h, --help            show this help message and exit
  --validate validatefile
                        validation on held out data
  --learning validatefile
                        learning curve based on validations
  --test testfile outputfile
                        generate classifications
  --prune prunefile     prune to improve performance on held out data
  --percent trainingpercent
                        percentage of data to train on (random sample)
  --dnf                 print disjunctiive normal form
```

## Installation

### Prerequisites
* Python 2.7.8 (2.7.x should work)
  * https://www.python.org/download/releases/2.7.8/
* NumPy 1.9.2
  * Installing with pip is recommended https://pip.pypa.io/en/latest/installing.html

## Usage

Run from the directory containing this file.

1. Read the training data file and generate a decision tree model.
  * `python dtree btrain.csv bmeta.csv`
2. Output the generated decision tree in disjunctive normal form.
  * `python dtree btrain.csv bmeta.csv --dnf`
3. Read the validation data file and report the accuracy of the model on that data.
  * `python dtree btrain.csv bmeta.csv --validate bvalidate.csv`
4. Read a test data file with missing labels in the last column and output a copy of that file with predicted labels in the last column.
  * `python dtree btrain.csv bmeta.csv --test btest.csv bout.csv`
