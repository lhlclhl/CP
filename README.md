# Crossword Puzzle Resolution via Monte Carlo Tree Search

## Datasets (Puzzles)
### Standard test set
data/puzzles/nyt.new.ra.txt
### Hard test set
data/puzzles/nyt.new.ra.hard.txt
### Validation set
data/puzzles/nyt.valid.txt
### Training set
nyt.shuffle.txt

## Datasets (Clues)
### Clues for test
data/clues_before_2020-09-26
### Clues for validation
data/clues_before_2020-06-18

## Codes
### Algorithm implementation
cps/*
### others
script for testing on standard set: run_standard.py

script for testing on standard set: run_hard.py

aggregating test results: aggres.py

generating data for reward function learning: generate_data.py

training reward function: train_reward.py

## Other data (dictionaries, models, etc.)
[Link](https://placeholder.link)