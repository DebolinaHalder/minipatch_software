
import numpy as np
import pandas as pd
from scipy import stats
from .base import Minipatch
from joblib import Parallel, delayed
from .base_learner import BaseLearner
from sklearn.metrics import accuracy_score



class MPForest(Minipatch):
    """
    A minipatch forest classifier.
    A minipatch forest is a meta estimator that fits a number of classifiers on tiny sub-samples of the data points 
    and features and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is 
    controlled with the `minipatch_n_ratio` and `minipatch_m_ratio` parameter.

    Parameters
    ----------
    selector: Classifier
       The selector is a standard sklearn classifier that is fit on each minipatch
    minipatch_m_ratio: float, default=0.5
        Fraction of features selected in each minipatch
    minipatch_n_ratio: float, default=0.5
        Fraction of samples selected in each minipatch
    number_of_patches: int, default=100
        number of minipatches
    oop_score: bool, default=False
        If oop_score is True, out of patch error is calculated


    Attributes
    ----------
    base_selector: BaseLearner
        A classifier that is used to train each minipatch
    n: int
        Number of features in each minipatch
    m: int
        Number of samples in each minipatch
    oop_score_:float
        Out of patch accuracy


    Methods
    -------
    fit(X,y): Build a minipatch forest from the training set (X, y).
    predict(X): Predict class for X

    Examples
    --------
    >>> from minipatch import MPForest
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree
    >>> iris = load_iris()
    >>> X, y = iris.data, iris.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    >>> clf = tree.DecisionTreeClassifier()
    >>> minipatch = MPForest(clf, minipatch_m_ratio= 0.2, minipatch_n_ratio= 0.1, number_of_patches=50, oop_score=True)
    >>> minipatch.fit(X_train, y_train)
    >>> pred = minipatch.predict(X_test)
    >>> print(accuracy_score(Y_test, pred))
    >>> print(minipatch.oop_score_)

    """
    def __init__(self, selector, minipatch_m_ratio=0.05, minipatch_n_ratio=0.5,number_of_patches = 100, oop_score = False):
        self.base_selector = BaseLearner(selector)
        self.minipatch_m_ratio = minipatch_m_ratio
        self.minipatch_n_ratio = minipatch_n_ratio
        self.number_of_patches = number_of_patches
        self.oop_score = oop_score
        self.oop_score_ = None
        self.selectors = []
        

    def _fit_tree(self,X,y,M,N):
        Ik = np.random.choice(N,self.n,replace=False)
        Fk = np.random.choice(M,self.m,replace=False)        
    
        base_selector_on_minipatch = self.base_selector
        base_selector_on_minipatch.fit(X[np.ix_(list(Ik), list(Fk))], y[Ik], Fk, Ik)
        return base_selector_on_minipatch

    def fit(self,X,y):
        m_ratio = self.minipatch_m_ratio  # m/M
        n_ratio = self.minipatch_n_ratio 
        N, M = X.shape

        self.n = np.int(np.round(n_ratio * N))
        self.m = np.int(np.round(m_ratio * M))
        self.X_train = X
        self.y_train = y
        grid = np.arange(self.number_of_patches)
        self.selectors = Parallel(n_jobs=-1,verbose=1)(
        delayed(self._fit_tree)(X,y,M,N) 
        for grid_val in grid
        )
        if self.oop_score == True:
            self.oop_score_ = self._out_of_patch_error()


    def predict(self, X):
        predictions = np.zeros(len(X))
        predX = np.zeros((len(X), len(self.selectors)))
        for i in range(len(self.selectors)):
            pred = (self.selectors[i].predict(X))
            for j in range(len(pred)):
                predX[j][i] = pred[j]
        print(predX.shape)
        for p in range(len(X)):
            predictions[p] = stats.mode(predX[p])[0]
        return predictions

    def predict_proba(self, X):
        pass


    def _out_of_patch_error(self):
        true_value = []
        prediction = []
        for i in range(len(self.X_train)):
            for j in self.selectors:
                if i not in j.Ik:
                    true_value.append(self.y_train[i])
                    prediction.append(j.predict(np.reshape(self.X_train[i],(1,-1))))
        return accuracy_score(true_value, prediction)