# Data directory

This directory includes the datasets used to evaluate our proposed methods. It includes 3 subdirectories, namely [raw_datasets/](./raw_datasets/), [datasets/](./datasets/), and [.results/](./.results/), as well as 2 notebooks, namely [data_preprocessing.ipynb](./data_preprocessing.ipynb) and [evaluation.ipynb](./evaluation.ipynb).

The [raw_datasets/](./raw_datasets/) directory includes the datasets obtained from their respective sources in either ```.csv``` or ```.arff``` formats. If the raw data is not included on the [GitHub](https://github.com/rompoggi/MCTS_ClassifierChain) page, please contact me (Romain Poggi) at either **romain.poggi@polytechnique.edu** or **ropoggi323@gmail.com**, or eventually Jesse Read at **jesse.read@polytechnique.edu**.

The [data_preprocessing.ipynb](./data_preprocessing.ipynb)notebook provides a way to pre process the data in the [raw_datasets/](./raw_datasets/) directory to make it in a simpler form, and correctly splits the features and labels of each dataset used for evaluation, which are then stored in the [datasets/](./datasets/) directory. Please note that it does so for datasets which were not used for testing, but could be used for improvement in future works on our approach.

Finally, we run our evaluation in the [evaluation.ipynb](./evaluation.ipynb) notebook, assuming that the data was cleaned with the [data_preprocessing.ipynb](./data_preprocessing.ipynb) notebook. The results of different runs are then stored as json files in the [.results/](./.results/) directory. We invite potential contributers to develop a more sophisticaed logging system.
