from os import F_OK
import numpy as np
from .base import Minipatch

class BaseLearner(Minipatch):
    """
    The BaseLearner class holds necessary informations about the
    base learner of each minipatch

    Parameters
    ----------
    clf: class Classifier
    This is the classifier used to train each minipatch


    Attributes
    ----------
    Fk: array of size(m,)
    This contains the indexes of features used to train the minipatch
    Ik: array of size(n,)
    This contains the indexes of samples used to train the minipatch

    Methods
    -------
    fit(X,y,Fk, Ik): Build a minipatch forest from the training set (X, y).
    predict(X): Predict class for X

    >>

    """






    def __init__(self, clf):
        self.clf = clf


    def fit(self, X, y, Fk, Ik):
        self.Fk = Fk
        self.Ik = Ik
        self.clf.fit(X,y)
        
    def predict(self,X):
        modified_x = X[:,self.Fk]
        return self.clf.predict(modified_x)

    def predict_proba(self, X):
        pass