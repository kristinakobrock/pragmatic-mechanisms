# emergent-pragmatics

Implementation of an agent-based model that allows to experiment with the role of pragmatic mechanisms in a language emergence paradigm using [EGG](https://github.com/facebookresearch/EGG/tree/main). The implementation builds on the [concept-level reference game](https://github.com/kristinakobrock/context-shapes-language) by Kobrock et al. which builds on the [hierarchical reference game](https://github.com/XeniaOhmer/hierarchical_reference_game/tree/master) by Ohmer et al. (2022) and the [concept game](https://github.com/jayelm/emergent-generalization/tree/master) by Mu & Goodman (2021).
[![DOI](https://zenodo.org/badge/986911991.svg)](https://doi.org/10.5281/zenodo.15497186)

## Installing dependencies
We used Python 3.9.15 and PyTorch 1.13.0. Generally, the minimal requirements are Python 3.6 and PyTorch 1.1.0.
`requirements.txt` lists the python packages needed to run this code. Additionally, please make sure you install EGG following the instructions [here](https://github.com/facebookresearch/EGG#installing-egg).
1. (optional) Create a new conda environement:
```
conda create --name emergprag python=3.9
conda activate emergprag
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

For example, to train the agents on data set D(4,8) (4 attributes, 8 values), using the same hyperparameters as in the paper, you can execute

`python train.py --dimensions 8 8 8 8 --n_epochs 300 --batch_size 16`

Similarly, for data set D(3, 4), the dimensions flag would be

`--dimensions 4 4 4`

Per default, this conducts one run. If you would like to change the number of runs, e.g. to 5, you can specify that using

`--num_of_runs 5`

If you would like to save the results (interaction file, agent checkpoints, a file storing all hyperparameter values, training and validation accuracies over time, plus test accuracy for generalization to novel objects) you can add the flag

`--save True`

By default, agents are trained in the context-aware condition. If you would like to train in the context-unaware (baseline) condition, you can specify the flag

`--context_unaware True`

For reproducibility reasons, the symbolic datasets can be pre-generated with the file `pickle_ds.py` and saved. For training, they can be loaded:

`--load_dataset dim(3,4)_context_sampled_shared_context_sf10.ds`

## Inference

To test the trained agents on the test dataset, we load the trained model (checkpoint), save the test interactions as 'test' and use a larger batch size and again load the respective dataset.

`--load_checkpoint True --save_test_interactions True --save_test_interactions_as test --batch_size 256 --load_dataset dim(3,4)_context_sampled_shared_context_sf10.ds`

To test the trained agents with RSA, we use the same inference procedure as for testing without RSA. Additionally, we load the train interactions (for calculating utilities of the messages) and specify the dataset split on which the inference should be done.

`--load_interaction train --test_rsa test`

To speed up the RSA inference, the test dataset can be limited to a certain number. We have used this setting for large datasets:

`--limit_test_ds 1000`

## Evaluation

Our results can be found in 'results/'. The subfolders contain the metrics for each run. We stored the final interaction for each run which logs all game-relevant information such as sender input, messages, receiver input, and receiver selections for the training and validation set. Based on these interactions, we evaluated additional metrics after training using the notebook 'evaluate_metrics.ipynb'. We uploaded all metrics but not the interaction files due to their large size.

## Visualization

Visualizations of the results can be found in the notebooks 'analysis_context.ipynb' (containing both emergence and inference results) and 'analysis_rsa.ipynb'. 
