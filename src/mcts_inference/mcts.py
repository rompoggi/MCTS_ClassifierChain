import numpy as np
np.set_printoptions(precision=3)
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

import time
class Constraint:
    def __init__(self,time=False,max_iter=False,d_time:float=1.,n_iter:int=100,verbose=False) -> None:
        assert (time or max_iter), f"At least {time=} or {max_iter=} should be True"
        assert(not max_iter or (isinstance(n_iter,int) and n_iter > 0)), f"{n_iter=} should be positive if {max_iter=}"
        assert( (not time) or (d_time > 0)), f"{d_time=} should be positive if {time=}"     
            
        self.time:bool       = time
        self.d_time:float    = d_time
        self.end_time:float = None 
            
        self.max_iter:bool   = max_iter
        self.n_iter:int      = n_iter
        self.curr_iter:int   = None
            
        self.reset()
        
        self.verbose:bool    = verbose
        
    def reset(self) -> None:
        self.end_time = time.time() + self.d_time
        self.curr_iter = -1
        
    def _bool_time(self) -> bool:
        return (not self.time or self.end_time >= time.time())
    
    def _bool_iter(self) -> bool:
        self.curr_iter += 1
        return (not self.max_iter or self.curr_iter < self.n_iter)
    
    def __bool__(self) -> bool:
        if self.verbose: # verbose
            bt = self._bool_time()
            bi = self._bool_iter()
            if not bt:
                print(f"Time Constraint Attained. Current iteration: {self.curr_iter:_}/{self.n_iter:_}")
                return False
            if not bi:
                print(f"Iteration Constraint Attained. Time left: {self.end_time - time.time():.3f}/{self.d_time}s")
                return False
            return True
        return self._bool_time() and self._bool_iter()
    
from enum import Enum
class NormOpt(Enum):
    SOFTMAX = 1
    UNIFORM = 2
    NONE = 3

from typing import Optional, Callable

class MCTSNode:
    pass

class MCTSNode:
    def __init__(self,label:int=0,rank:int=2,n_children:int=2,parent:Optional[MCTSNode]=None,proba:float = 0.,parent_labels:list[int]=[]) -> None:
        self.proba:float               = proba
        self.label:int                 = label
        self.visit_count:int           = 0
        
        self.parent:MCTSNode           = parent
        self.children:list[MCTSNode]   = None
        self.n_children:int            = n_children
        self.rank:int                  = rank
            
        self.parent_labels:list[int]   = parent_labels
        
    def __get__(self,key:int) -> Optional[MCTSNode]:
        assert (key >= 0 and key < self.n_children), f"{key} is not a valid key."
        assert(self.is_expanded()), f"Node not yet expanded. Cannot get the child node at key:{key}."
        return self.children[key]

    def is_terminal(self) -> bool:
        return (self.rank == 0)
    
    def is_expanded(self) -> bool:
        return not (self.children == None)
    
    def expand(self) -> None:
        assert(not self.is_terminal()), f"Cannot expand a terminal node"
        self.children = [MCTSNode(label=i,rank = self.rank-1,n_children = self.n_children,parent=self,parent_labels=self.parent_labels+[i]) for i in range(self.n_children)]
        
    #     def get_parent_labels(self) -> list[int]:
    #         print(f"IN:{self}")
    #         if self.parent is None: # root node
    #             print(f"ROOT OUT:{self}")
    #             return []
    #         if self.parent_labels is None:
    # #             print(f"New parent labels, {self}")
    #             self.parent_labels = self.parent.get_parent_labels()
    #             self.parent_labels.append(self.label)
    # #         print(self.parent_labels)
    #         print(f"OUT:{self}")
    #         return self.parent_labels
    
    def get_parent_labels(self) -> list[int]:
        return self.parent_labels
                
    def __str__(self) -> str:
        out = f"*MCTSNode: L={self.label}, R={self.rank}, P={self.proba:.4f}, PL{self.parent_labels}*"
        return out
    
    def __repr__(self):
        return str(self)
        
    def print_all(self):
        print("\n".join([f"{k}:{v}" for k,v in self.__dict__.items()]))
    
    def delete(self):
        if self.children is None:
            del self
            return
        for child in self.children:
            child.delete()
        del self
        
    def check_correctness(self) -> bool:
        # checks recursively that all the children's visit count sum to that of their parent node.
        if self.children is None:
            return True
        ssum = 0
        for child in self.children:
            if not child.check_correctness():
                return False
            ssum += child.visit_count
        return ssum == self.visit_count
    
    def is_fully_expanded(self) -> bool:
        # checks recursively if the entire tree has been expanded
        if self.is_terminal():
            return True
        for child in self.children:
            if not child.is_fully_expanded():
                return False
        return True
    
    def normalize_proba(self,opt: NormOpt = NormOpt.SOFTMAX) -> None:
        """
        Normalizes the rewards obtained at each node into a distribution
        """
        if self.is_terminal():
            return
        probas = np.array([child.proba for child in self.children])
        
        if opt == NormOpt.SOFTMAX:
            probas = np.exp(probas)
        elif opt == NormOpt.UNIFORM:
            pass
        elif opt == NormOpt.NONE:
            pass  # Do nothing
        
        if opt != NormOpt.NONE:
            probas /= np.sum(probas)
            
        for i,child in enumerate(self.children):
            child.proba = probas[i]

from typing import Any, Dict

def debug(func):
    def wrapper(*args, **kwargs):
        print(f"'{func.__name__}': {args=}, {kwargs=}")
        output = func(*args, **kwargs)
        print(f"'{func.__name__}': {output=}")
        return output
    return wrapper

# @debug
def randmax(A:Any) -> int:
    maxValue=max(A,key=lambda x:x.proba).proba
    index = [i for i in range(len(A)) if A[i].proba==maxValue]
    return np.random.choice(index,)

# @debug
def eps_greedy(node:MCTSNode,eps:float=0.1)->int:
    assert(eps >= 0 and eps <= 1), f"{eps=} should be in the [0,1] range."
    if np.random.rand() < eps: # explore
        return np.random.choice(node.n_children)
    return randmax(node.children)

# @debug
def select(node:MCTSNode,eps:float=0.2)->MCTSNode:
    while(node.is_expanded() and not node.is_terminal()):
        node.visit_count += 1
        ind = eps_greedy(node,eps)
        node = node.children[ind]
    return node

# @debug
def back_prog(node:MCTSNode,reward:float) -> None:
    if node.parent is None: # root node, no need to update
        return
    assert (node.visit_count > 0), f"Node has not yet been visited. A problem appened."
    node.proba = node.proba + (reward - node.proba) / node.visit_count # average proba
    back_prog(node.parent,reward)
    
# @debug
def simulate(node:MCTSNode,model:Any,x:Any,cache) -> float:
    node.visit_count += 1
    while (not node.is_terminal()):
        if not node.is_expanded():
            node.expand()
        node = np.random.choice(node.children) # uniform choice
        node.visit_count += 1
    return get_reward(node,model,x,cache)
    
# @debug
def get_reward(node:MCTSNode,model:Any,x:Any,cache:Dict[tuple,float]={}) -> float:
    labels = node.get_parent_labels()
    if (tuple(labels)) in cache:
        return cache[tuple(labels)]
        
    assert (node.is_terminal()), f"Can only get rewards for a terminal node. Node rank={node.rank}."
    labels = node.get_parent_labels()
    xy = x.reshape(1,-1)
    p = 1
    
    for j in range(len(labels)):
        if j>0:
            # stack the previous y as an additional feature
            xy = np.column_stack([xy, labels[j-1]])

        p *= model.estimators_[j].predict_proba(xy)[0][labels[j]] # (N.B. [0], because it is the first and only row)
        
    cache[tuple(labels)] = p

    return p

def bestChild(root:MCTSNode) -> MCTSNode: # best proba
    return select(root,eps=0).get_parent_labels()

def MCTS(model,x,verbose:bool=False,secs:float=1):
    n_classes = len(model.estimators_)
    root = MCTSNode(label=None,n_children = 2,rank=n_classes,proba=1)
    
    ComputationalConstraint = Constraint(time=True,d_time=secs,max_iter=False,n_iter=0,verbose=verbose)
    
    cache:Dict[tuple,float] = {} # Create a cache to store the reward evaluation to gain inference speed
    while(ComputationalConstraint):
        node = select(root)
        reward = simulate(node,model,x,cache)
        back_prog(node,reward)
    
    return bestChild(root)

def func2():
    n_samples = 1000
    n_features=6
    n_classes=3
    n_labels = 2
    random_state=0

    X, Y = make_multilabel_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_classes=n_classes,
        n_labels=n_labels, 
        random_state=random_state)

    test_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    print(f"{X_train.shape = }\n{X_train}")
    print(f"{Y_train.shape = }\n{Y_train}")

    from sklearn.multioutput import ClassifierChain
    from sklearn.linear_model import LogisticRegression

    solver = "liblinear"
    base = LogisticRegression(solver=solver)
    chain = ClassifierChain(base)

    chain = chain.fit(X_train,Y_train)

    M = 10
    for i in range(M):
        print(f"MCTS Pred:{MCTS(chain,X_test[i])}, ClassifierChain Pred:{chain.predict(X_test[i].reshape(1,-1))[0]} vs True:{Y_test[i]}")

    def func3():
        from tqdm import trange

        secs = 0.1 # 0.5 Second per inference to test
        M = 100
        # M = len(Y_test)

        y_mcts = []
        y_chain = chain.predict(X_test[:M])

        for i in trange(M, desc=f"MCTS Inference Constraint={secs}s", unit="it",colour="green"):
            y_mcts.append(MCTS(chain,X_test[i],secs=secs))
            
        y_mcts = np.array(y_mcts)

        from sklearn.metrics import hamming_loss, zero_one_loss

        print(f"MCTS  vs TRUE : Hamming={hamming_loss(y_mcts,Y_test[:M]):.4f}, ZeroOne={zero_one_loss(y_mcts,Y_test[:M]):.4f}")
        print(f"CHAIN vs TRUE : Hamming={hamming_loss(y_chain,Y_test[:M]):.4f}, ZeroOne={zero_one_loss(y_chain,Y_test[:M]):.4f}")
        print(f"MCTS  vs CHAIN: Hamming={hamming_loss(y_chain,y_mcts):.4f}, ZeroOne={zero_one_loss(y_chain,y_mcts):.4f}")


def main():
    print("Hello World!")
    func2()
    print("Done")

if __name__ == "__main__":
    main()

    