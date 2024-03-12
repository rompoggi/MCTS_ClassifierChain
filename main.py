from mcts_inference.constraints import Constraint
from mcts_inference.mcts import MCTS
from mcts_inference.mcts_config import MCTSConfig
from mcts_inference.policy import UCB
from mcts_inference.mc import MC
from mcts_inference.brute_force import brute_force as BF


def main():
    from sklearn.datasets import make_multilabel_classification
    from sklearn.model_selection import train_test_split
    n_samples = 1000
    n_features = 40
    n_classes = 20
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

    
    from sklearn.multioutput import ClassifierChain  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore

    solver = "liblinear"
    base = LogisticRegression(solver=solver)
    chain = ClassifierChain(base)

    M: int = 100
    n_iter = 1000

    chain: ClassifierChain = chain.fit(X_train, Y_train)
    Yc = chain.predict(X_test[:M])

    config = MCTSConfig(n_classes=n_classes, selection_policy=UCB(2), constraint=Constraint(max_iter=True, n_iter=n_iter), verbose=True)
    Ymcts = MCTS(X_test[:M], chain, config=config)

    config = MCTSConfig(n_classes=n_classes, selection_policy=UCB(2), constraint=Constraint(max_iter=True, n_iter=n_iter), verbose=True)
    Ymc = MC(X_test[:M], chain, config=config)

    config = MCTSConfig(n_classes=n_classes, selection_policy=UCB(2) ,constraint=Constraint(max_iter=True, n_iter=n_iter), step_once=False, verbose=True)
    Ym1cts = MCTS(X_test[:M], chain, config=config)

    config = MCTSConfig(n_classes=n_classes, selection_policy=UCB(2) ,constraint=Constraint(time=True, d_time=3.), verbose=True)
    Ybf = BF(X_test[:M], chain, config=config)

    from sklearn.metrics import hamming_loss, zero_one_loss

    loss = hamming_loss
    print(f"{loss(Yc, Y_test[:M])=}")
    print(f"{loss(Ymcts, Y_test[:M])=}")
    print(f"{loss(Ymc, Y_test[:M])=}")
    print(f"{loss(Ym1cts, Y_test[:M])=}")
    print(f"{loss(Ybf, Y_test[:M])=}")

    loss = zero_one_loss
    print(f"{loss(Yc, Y_test[:M])=}")
    print(f"{loss(Ymcts, Y_test[:M])=}")
    print(f"{loss(Ymc, Y_test[:M])=}")
    print(f"{loss(Ym1cts, Y_test[:M])=}")
    print(f"{loss(Ybf, Y_test[:M])=}")

if __name__ == "__main__":  # pragma: no cover
    main()
