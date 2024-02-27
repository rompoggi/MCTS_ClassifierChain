
This project uses the Monte Carlo Tree Search (MCTS) algorithm for multi-label classification tasks. 

## Monte Carlo Tree Search (MCTS)

MCTS is a heuristic search algorithm used in decision-making processes, particularly in game playing. It involves building a search tree by performing random simulations and using the results of those simulations to make more informed decisions.

![Monte Carlo Tree Search](https://i.stack.imgur.com/GR7qf.png)

The MCTS algorithm consists of four main steps:

- **Selection**: Starting at the root node, the algorithm traverses the tree by selecting the most promising child node until it reaches a node from which not all children have been generated.

- **Expansion**: If the selected node is not a terminal node (i.e., an end point of the game), one or more child nodes are generated.

- **Simulation**: A simulation is run from the newly expanded node according to a default policy (usually random) to a terminal state, resulting in an outcome.

- **Backpropagation**: The outcome of the simulation is backpropagated up the tree, updating the information at each node (like the number of visits and average reward) along the path.

The process is repeated for a certain number of iterations or until a computational budget has been exhausted. The child of the root with the highest average reward is selected as the next action.

## Multi-label Classification

In machine learning, multi-label classification is a type of classification in which an object can belong to more than one class. In other words, classes are not mutually exclusive. For example, a given piece of text might be categorized both as "Science" and "Health".

This is in contrast to multi-class classification, where each object is categorized into one of many classes, and binary classification, where each object is categorized into one of two classes.

In the context of this project, the aim is to improve the performance of multi-label classifiers. This could mean increasing the accuracy of the classifier, reducing the computational cost, or improving some other performance metric. The MCTS algorithm is used to make more informed decisions about which labels should be assigned to each object, potentially improving the performance of the classifier.

## Implementation

The MCTS algorithm is implemented in the [`mcts.py`](command:_github.copilot.openSymbolInFile?%5B%22src%2Fmcts_inference%2Fmcts.py%22%2C%22mcts.py%22%5D "src/mcts_inference/mcts.py") file. The [`MCTSNode`](command:_github.copilot.openSymbolInFile?%5B%22src%2Fmcts_inference%2Fmcts_node.py%22%2C%22MCTSNode%22%5D "src/mcts_inference/mcts_node.py") class represents a node in the MCTS tree. Each node has several attributes, including `state`, `parent`, `children`, `untried_actions`, `visits`, and `reward`.

The [`Policy`](command:_github.copilot.openSymbolInFile?%5B%22src%2Fmcts_inference%2Fpolicy.py%22%2C%22Policy%22%5D "src/mcts_inference/policy.py") class is an abstract base class for different policies that can be used in the MCTS algorithm. It includes several subclasses such as [`Uniform`](command:_github.copilot.openSymbolInFile?%5B%22src%2Fmcts_inference%2Fpolicy.py%22%2C%22Uniform%22%5D "src/mcts_inference/policy.py"), [`Greedy`](command:_github.copilot.openSymbolInFile?%5B%22src%2Fmcts_inference%2Fpolicy.py%22%2C%22Greedy%22%5D "src/mcts_inference/policy.py"), [`EpsGreedy`](command:_github.copilot.openSymbolInFile?%5B%22src%2Fmcts_inference%2Fpolicy.py%22%2C%22EpsGreedy%22%5D "src/mcts_inference/policy.py"), [`UCB`](command:_github.copilot.openSymbolInFile?%5B%22src%2Fmcts_inference%2Fpolicy.py%22%2C%22UCB%22%5D "src/mcts_inference/policy.py"), and [`Thompson_Sampling`](command:_github.copilot.openSymbolInFile?%5B%22src%2Fmcts_inference%2Fpolicy.py%22%2C%22Thompson_Sampling%22%5D "src/mcts_inference/policy.py").

## Installation

To install the MCTS Inference Project, follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/rompoggi/MCTS_ClassifierChain.git
```

2. Navigate to the project directory:

```bash
cd MCTS_ClassifierChain
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains a list of Python packages that the project depends on. You may need to use `pip3` instead of `pip` depending on your Python installation.


## Testing

Tests for the project can be run using pytest. The project uses GitHub Actions for continuous integration, as specified in the [`.circleci/config.yml`](command:_github.copilot.openSymbolInFile?%5B%22.circleci%2Fconfig.yml%22%2C%22.circleci%2Fconfig.yml%22%5D ".circleci/config.yml") file.

## Usage

An example of how to use the MCTS algorithm with a multi-label classifier is provided in the [`mcts.py`](command:_github.copilot.openSymbolInFile?%5B%22src%2Fmcts_inference%2Fmcts.py%22%2C%22mcts.py%22%5D "src/mcts_inference/mcts.py") file under the `if __name__ == "__main__"` clause. This example uses the `ClassifierChain` class from scikit-learn with a logistic regression base classifier.

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for more details.

## Contributing

Contributions to the project are welcome. Please make sure to follow the coding standards specified in the [`setup.cfg`]("setup.cfg") file under the `[flake8]` section.
This repository is that of my bachelor thesis.


## Status

[![Tests](https://github.com/rompoggi/MCTS_ClassifierChain/actions/workflows/tests.yml/badge.svg)](https://github.com/rompoggi/MCTS_ClassifierChain/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/rompoggi/MCTS_ClassifierChain/graph/badge.svg?token=N9FSNH021E)](https://codecov.io/gh/rompoggi/MCTS_ClassifierChain)

- The `Tests` badge shows the status of the automated tests for the project.
- The `License: MIT` badge indicates that the project is licensed under the MIT license.
- The `codecov` badge shows the code coverage percentage of the tests.
