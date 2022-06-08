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
### Seen clues for test set
data/clues_before_2020-09-26
### Seen clues for validation set
data/clues_before_2020-06-18

## Codes
### Algorithm implementation
cps/*

### Testing on standard set
run_standard.py

### Testing on standard set
run_hard.py

### Aggregating test results
aggres.py

### Generating data for reward function learning
generate_data.py

### Training reward function
train_reward.py

## Instructions

### Installing requirements
python3 -m pip install -r requirements --no-deps

### compiling core modules to accelerate
cythonize -a -i cps/puz_utils1.pyx

## Other data (dictionaries, models, etc.)
[Link](https://u.pcloud.link/publink/show?code=XZvu7RVZJbsfpViTsRhJ0bDNb647lz8mJp57)