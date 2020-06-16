# Complete version

This directory contains code / an ipynb report for the abridged Rakuten Challenge. 

### Code requirements:
- `pandas`
- `sklearn`
- `numpy`
- `nltk`
- optional: `networkx` (this is wrapped in `try-except` so is not strictly necessary but only in the `compatibility` folder)

Note that a version of this code that does not require `nltk` is contained in the `compatibility` subdirectory. Please also note that I use a code formatter, `black`, that I've removed from the code because it is non-standard.

### This directory contains the following files:

- `2020_06_15_Kaggle_Challenge.ipynb`: the notebook version of this report
- `util.py`: utility code to shorten this report

### How to read through this:

The `ipynb` version of the report (hopefully) contains a streamlined narrative of this work -- it's a text/code hybrid that serves as a technical report. This report calls some utility functions from `util.py`, and so to get a better sense of what is going on, `util.py` is a good next read.
