from mcts_inference.constraints import Constraint
from mcts_inference.mcts import MCTS
from mcts_inference.mcts_config import MCTSConfig
# import numpy as np

if __name__ == "__main__":  # pragma: no cover
    from sklearn.datasets import make_multilabel_classification
    from sklearn.model_selection import train_test_split
    n_samples = 10000
    n_features = 6
    n_classes = 3
    n_labels = 2
    random_state = 0

    X, Y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_labels=n_labels,
        random_state=random_state)

    test_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    from sklearn.multioutput import ClassifierChain
    from sklearn.linear_model import LogisticRegression

    solver = "liblinear"
    base = LogisticRegression(solver=solver)
    chain: ClassifierChain = ClassifierChain(base)

    chain = chain.fit(X_train, Y_train)

    # from tqdm import trange
    # from sklearn.metrics import hamming_loss, zero_one_loss

    secs_lis: list[float] = [0.01, 0.1]  # , 0.5, 1., 2.]

    config = MCTSConfig(n_classes=n_classes, constraint=Constraint(time=True, d_time=0.1), step_once=False, parallel=False, verbose=True)

    M: int = 100
    Y = MCTS(chain, X_test[:M], config=config)

    # M: int = min(100, len(Y_test))

    # hl_mt: list[float] = []
    # hl_ct: list[float] = []
    # hl_mc: list[float] = []

    # zo_mt: list[float] = []
    # zo_ct: list[float] = []
    # zo_mc: list[float] = []

    # run: bool = False
    # run = False

    # y_chain = chain.predict(X_test[:M])
    # for secs in secs_lis:
    #     if not run:
    #         break
    #     _y_mcts = []

    #     for i in trange(M, desc=f"MCTS Inference Constraint={secs}s", unit="it", colour="green"):
    #         _y_mcts.append(MCTS(chain, X_test[i], secs=secs))

    #     y_mcts = np.array(_y_mcts)

    #     hl_mt.append(hamming_loss(y_mcts, Y_test[:M]))
    #     hl_ct.append(hamming_loss(y_chain, Y_test[:M]))
    #     hl_mc.append(hamming_loss(y_chain, y_mcts))

    #     zo_mt.append(zero_one_loss(y_mcts, Y_test[:M]))
    #     zo_ct.append(zero_one_loss(y_chain, Y_test[:M]))
    #     zo_mc.append(zero_one_loss(y_chain, y_mcts))

    # if run:
    #     import matplotlib.pyplot as plt

    #     plt.plot(secs_lis, hl_mt, label="MCTS vs True")
    #     plt.plot(secs_lis, hl_ct, label="Chains vs True")
    #     plt.plot(secs_lis, hl_mc, label="MCTS vs Chains")

    #     plt.title("Hamming Loss Comparison for different times")
    #     plt.xlabel("Seconds")
    #     plt.ylim(0, 1)
    #     plt.ylabel("Hamming Loss")
    #     plt.legend()
    #     plt.show()

    #     plt.plot(secs_lis, zo_mt, label="MCTS vs True")
    #     plt.plot(secs_lis, zo_ct, label="Chains vs True")
    #     plt.plot(secs_lis, zo_mc, label="MCTS vs Chains")

    #     plt.title("Zero One Loss Comparison for time different times")
    #     plt.xlabel("Seconds")
    #     plt.ylim(0, 1)
    #     plt.ylabel("Zero One Loss")
    #     plt.legend()
    #     plt.show()
