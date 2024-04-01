# Monte Carlo Tree Search for Classifier Chain

This repository contains the source code of the implementation of MCTS for Classifier Chains with several [examples](./examples/) of how to use it with Classifier Chains. We only support models which compute probabilities for now. See my Bachelor Thesis report for a detailed explaination of the method. 

The repository is part of my Bachelor Thesis submitted for the degree of Bachelor in Mathenmatics and Computer Science at Ecole Polytechnique. It consists of an 8 to 10 week long full time research internship following a topic linked to one of our double major. I was under supervision of Professor Jesse READ, from LIX. See his [webpage](https://jmread.github.io/index.html) for more details about his works in research and teaching.

#### Monte Carlo Tree Search for Multi-Dimensional Learning with Classifier Chains
*Romain Poggi*, Bachelor of Science at Ecole Polytechnique <br>
*Jesse Read*, Computer Science Laboratory of the École polytechnique <br>
*Bachelor Thesis Report*, [https://drive.google.com/file/d/1-gmiogobxYQINJDOgnwJ1kZrVSOHIX2b/view?usp=sharing](https://drive.google.com/file/d/1-gmiogobxYQINJDOgnwJ1kZrVSOHIX2b/view?usp=sharing)


MCTS for Classifier Chains makes use of the Monte Carlo Tree Search algorithm, a heuristic search algorithm used in decision-making processes. We are see inference as search, where a path is a sequence of labels.

It builds onto the original classifier chains which usses a greedy policy and choses the next node based on its likelihood, which may is often not optimal.

We also try to improve the PCC and MCC methods, which respectively find in a brute force manner the bayesian optimal label combination, while the other samples different paths based on the node's marginal probability. The first method might not always terminate due to the exponential nature of the label space, though it is optimal when it does terminate. The MCC method is the current state-of-the-art for Classifier Chains, which we try to attain in this work.

## Results

Here are the rankings obtained from our tests, which were made in the [data](./data/) directory, precisely in the [evaluation.ipynb](/data/evaluation.ipynb) notebook. For more information on how to reproduce the obtained results, please refer to [data/README.md](/data/README.md).

<div align="center"><strong>Ranking by Exact Match Score</strong></div>

| Dataset   | PCC  | CC  | MCC | MUCB(2) | MEPS(0.2) | MEPS(0.5) | MTMS(1,1) | M1UCB(2) | M1EPS(0.2) |
|-----------|------|-----|-----|---------|-----------|-----------|-----------|----------|------------|
| Music     | 3    | 6   | 3   | 1       | 9         | 5         | 7         | 2        | 8          |
| Scene     | 1    | 8   | 1   | 3       | 6         | 5         | 6         | 4        | 9          |
| Flags     | 3    | 9   | 3   | 5       | 1         | 2         | 6         | 8        | 6          |
| Foodtruck | 2    | 3   | 1   | 5       | 8         | 6         | 4         | 9        | 7          |
| Yeast     | 1    | 3   | 1   | 4       | 7         | 6         | 5         | 9        | 8          |
| Birds     | 9    | 2   | 1   | 3       | 6         | 5         | 4         | 8        | 7          |
| Genbase   | 9    | 2   | 1   | 4       | 6         | 5         | 3         | 8        | 7          |
| avg. rank | 4.0  | 4.71| ***1.57*** | ***3.57***    | 6.14      | 4.85      | 5.0       | 6.85     | 7.43       |

<div align="center"><strong>Ranking by Hamming Score</strong></div>

| Dataset   | PCC  | CC  | MCC | MUCB(2) | MEPS(0.2) | MEPS(0.5) | MTMS(1,1) | M1UCB(2) | M1EPS(0.2) |
|-----------|------|-----|-----|---------|-----------|-----------|-----------|----------|------------|
| Music     | 4    | 3   | 5   | 1       | 8         | 7         | 6         | 2        | 9          |
| Scene     | 3    | 7   | 2   | 1       | 8         | 6         | 5         | 4        | 9          |
| Flags     | 3    | 7   | 3   | 6       | 8         | 1         | 2         | 5        | 9          |
| Foodtruck | 2    | 3   | 1   | 4       | 8         | 6         | 5         | 9        | 7          |
| Yeast     | 2    | 1   | 2   | 4       | 7         | 6         | 5         | 9        | 8          |
| Birds     | 9    | 2   | 1   | 3       | 6         | 5         | 4         | 8        | 7          |
| Genbase   | 9    | 2   | 1   | 4       | 6         | 5         | 3         | 8        | 7          |
| avg. rank | 4.57 | 3.57| ***2***   | ***3.29***    | 7.29      | 5.14      | 4.29      | 6.429    | 8.0        |

Our method therefore achieves 2nd best performance against state-of-the-art methods, all without tuning the hyperparameters. Thus, this repository invites for contributions to be made to further study the method.

## Repository Overview

There are several directories in this repo:

[src/mcts_inference/](src/mcts_inference/): This directory contains the source code for the project. It includes several modules such as `constraints`, `mcts`, `policy`, `utils`, `mcc`, and `pcc`. These modules likely contain the implementation of the Monte Carlo Tree Search (MCTS), Monte Carlo Classifier Chains (MCC), and Probabilistic Classifier Chains (PCC) algorithms, as well as utility functions and constraints used in the project.

[examples/](examples/): This directory contains notebooks that demonstrate how to use the framework built in the project. The notebooks include [mcts.ipynb](/examples/mcts.ipynb), [mcts_vs_mcc.ipynb](/examples/mcts_vs_mcc.ipynb), and [mcts_vs_others.ipynb](/examples/mcts_vs_others.ipynb).

[data/](data/): This directory contains the datasets used to evaluate the methods implemented in the project. It includes raw datasets in `.csv `or `.arff` formats, preprocessed datasets, and results of evaluations. The preprocessing and evaluation are done using the notebooks [data_preprocessing.ipynb](/data/data_preprocessing.ipynb) and [evaluation.ipynb](/data/evaluation.ipynb) respectively.

[tests/](tests/): This directory contains the test files for the project. These tests can be run using pytest.

## Installation

There are different ways to use the package in its current form. One can either install it locally as a package, or simply import it in Python files from the same directory.

1. Clone the repository to your local machine:

```bash
git clone https://github.com/rompoggi/MCTS_ClassifierChain.git
```

2. Navigate to the project directory:

```bash
cd MCTS_ClassifierChain
```

3. Install the dependencies:

You can directly install the package with the following command, which would automatically install the requirements listed in the [```requirements.txt```](/requirements.txt) file.

```bash
pip install -e .
```

Otherwise, one should install the dependencies via the following command:

```bash
pip install -r requirements.txt
```

One can then use the source code via ```import src.mcts_inference.*```, where * can be replaced by the module you want to use.

Of course, you may use `pip3` instead of `pip` depending on your Python installation.


## Data
The project uses datasets located in the data directory. The raw datasets are in the raw_datasets subdirectory and are in `.csv` or `.arff` formats. The [data_preprocessing.ipynb](/data/data_preprocessing.ipynb) notebook is used to preprocess these datasets and store them in the datasets directory. For more detail, refer the [report](https://drive.google.com/file/d/1-gmiogobxYQINJDOgnwJ1kZrVSOHIX2b/view?usp=sharing) or to the [README](/data/README.md) in the directory.

The [evaluation.ipynb](/data/evaluation.ipynb) notebook is used to run evaluations on the preprocessed data. The results of these evaluations are stored as JSON files in the [.results/](/data/.results/) directory.

## Testing
Tests for the project can be run using pytest. The test files are made accessible in the [tests/](tests/) directory. The testing parameters can be found in the [pyproject.toml](pyproject.toml), [setup.cfg](setup.cfg), and [tox.ini](tox.ini) files. Code-cov reports are also used for future maintenance of the project. We invite contributors to be provide tests when proposing new code.

The project uses GitHub Actions for continuous integration. The configuration used can be found in the [/.github/workflows/tests.yml](/.github/workflows/tests.yml) file.


## Contact

Please contact us or post an issue if you have any questions.

For questions related to the package `mcts_inference`:
* *Romain Poggi* ([romain.poggi@polytechnique.edu](romain.poggi@polytechnique.edu) or [romainpoggi323@gmail.com](romainpoggi323@gmail.com))

For questions related to the theoretical aspect of the method:
* *Romain Poggi* ([romain.poggi@polytechnique.edu](romain.poggi@polytechnique.edu) or [romainpoggi323@gmail.com](romainpoggi323@gmail.com))
* *Jesse Read* ([jesse.read@polytechnique.edu](jesse.read@polytechnique.edu))

## Contributing
Contributions to this project are more than welcome. The aim is to further study and improve the method used in this project. Please make sure to follow the coding standards specified in the [`setup.cfg`]("setup.cfg") file under the `[flake8]` section.


## License
[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for more details. 


## Status

[![Tests](https://github.com/rompoggi/MCTS_ClassifierChain/actions/workflows/tests.yml/badge.svg)](https://github.com/rompoggi/MCTS_ClassifierChain/actions)
[![codecov](https://codecov.io/gh/rompoggi/MCTS_ClassifierChain/graph/badge.svg?token=N9FSNH021E)](https://codecov.io/gh/rompoggi/MCTS_ClassifierChain)
[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)