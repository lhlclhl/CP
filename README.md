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
- cps/search.py: The main class of the method with several versions of search method
    - class MCTS: the basic implementation of MCTS algorithm for CP
    - class MCTS_NM: MCTS algorithm with neural matching for clue retrieval
    - class Astar: A* algorithm
    - class LDS: Limited Discrepancy Search algorithm 
- cps/candgen.py: candidate generation module
- cps/cdbretr.py: seen clue retrievel module
    - class ClueES: clue retrieval with textual matching
    - class RAM_CTR: clue retrieval with neural matching
- cps/kbretr.py: knowledge base retrievel module
- cps/dictretr_v2.py: dictionary retrievel module
- cps/fillblank.py: blank filling module

### Evaluating on standard test set
> run_standard.py

### Evaluating on hard test set
> run_hard.py

### Aggregating test results
> aggres.py

### Generating data for reward function learning
> generate_data.py

### Training reward function
> train_reward.py

## Instructions

### Installing requirements by create a conda environment
> conda env create -f crossword.yml

### Compiling core modules with Cython to accelerate
> cythonize -a -i cps/puz_utils1.pyx

### Installing and start Elasticsearch service
[Download Elasticsearch](https://www.elastic.co/cn/downloads/elasticsearch)

start ES service (e.g. Windows, go to the ES installation directory): 
> .\bin\elasticsearch-service.bat start

### Download StanfordNLP model
> python -c "import stanfordnlp; stanfordnlp.download('en')"

## Other data (dictionaries, models, etc.)
[Link](https://u.pcloud.link/publink/show?code=XZvu7RVZJbsfpViTsRhJ0bDNb647lz8mJp57)