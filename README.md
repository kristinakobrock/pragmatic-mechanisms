# pragmatic-mechanisms

Implementation of a concept-level reference game in a language emergence paradigm using [EGG](https://github.com/facebookresearch/EGG/tree/main). The implementation builds on the [hierarchical reference game](https://github.com/XeniaOhmer/hierarchical_reference_game/tree/master) by Ohmer et al. (2022) and the [concept game](https://github.com/jayelm/emergent-generalization/tree/master) by Mu & Goodman (2021). A first abstract of this research has been published at the workshop 'Procedural and computational models of semantic and pragmatic processes' held at the 34th European Summer School in Logic, Language and Information (ESSLLI) at University of Ljubljana, Faculty of Computer and Information Science from 31 July – 4 August 2023: Kristina Kobrock, Xenia Ohmer, Elia Bruni and Nicole Gotzner (2023): [Pragmatics in referential communication: An investigation of concept
communication and the role of pragmatics with an emergent communication game](https://prosandcomps.github.io/assets/pdf/abstracts/ProsComps2023_Kobrock.pdf).

## Installing dependencies
We used Python 3.9.15 and PyTorch 1.13.0. Generally, the minimal requirements are Python 3.6 and PyTorch 1.1.0.
`requirements.txt` lists the python packages needed to run this code. Additionally, please make sure you install EGG following the instructions [here](https://github.com/facebookresearch/EGG#installing-egg).
1. (optional) Create a new conda environement:
```
conda create --name emergab python=3.9
conda activate emergab
```
2. EGG can be installed as a package to be used as a library (see [here](https://github.com/facebookresearch/EGG#installing-egg) for more options):
```
pip install git+https://github.com/facebookresearch/EGG.git
```
3. Install packages from the requirements file:
```
pip install -r requirements.txt
```

## Training

Agents can be trained using 'train.py'. The file provides explanations for how to configure agents and training using command line parameters.

For example, to train the agents on data set D(4,8) (4 attributes, 8 values) with vocab size factor 3 (default), using the same hyperparameters as in the paper, you can execute

`python train.py --dimensions 8 8 8 8 --n_epochs 300 --batch_size 32`

Similarly, for data set D(3, 4), the dimensions flag would be

`--dimensions 4 4 4`

Per default, this conducts one run. If you would like to change the number of runs, e.g. to 5, you can specify that using

`--num_of_runs 5`

If you would like to save the results (interaction file, agent checkpoints, a file storing all hyperparameter values, training and validation accuracies over time, plus test accuracy for generalization to novel objects) you can add the flag

`--save True`

## Evaluation

Our results can be found in 'results/'. The subfolders contain the metrics for each run. We stored the final interaction for each run which logs all game-relevant information such as sender input, messages, receiver input, and receiver selections for the training and validation set. Based on these interactions, we evaluated additional metrics after training using the notebook 'evaluate_metrics.ipynb'. We uploaded all metrics but not the interaction files due to their large size.

## Visualization

Visualizations of training and results can be found in the notebooks 'analysis.ipynb' and 'analysis_qualitative.ipynb'. The former reports all results for the 6 different data sets (D(3,4), D(3,8), D(3,16), D(4,4), D(4,8), D(5,4)). The latter contains code to extract message-concept pairs from the interactions and evaluate qualitatively which messages have been produced to refer to which concept in which context condition.

## Grid search results

The folder 'grid_search/' contains the results for the hyperparameter grid search. 'grid_search_length_cost/' contains the results for the length cost parameter search (also given different message lengths).
