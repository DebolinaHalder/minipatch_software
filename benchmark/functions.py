#%%
from minipatch import MPForest

def wrapper(m_ratios, n_ratios, clf, number_of_patches, X, y):
    """
    This function is a wrapper function to find different values of oop_score_ for different feature and sample ratios
    of the minipatch.

    Parameters
    ----------
    m_ratios: array of shape(m,)
        Contains the feature ratios to consider
    n_ratios: array of shape(n,)
        Contains the feature ratios to consider
    clf: class Classifier
        The base classifier for the minipatch
    number_of_patches: int
        The number of minipatches to train
    X: array of shape(n,m)
        Data matrix to fit
    y: array of shape(n,)
        Target labels

    Return
    ------
    oop_errors: array of shape (len(m_ratios)*len(n_ratios),)
        1 - oop_score_ for all conbinations

    """
    oop_errors = []
    for feature in m_ratios:
        for sample in n_ratios:
            minipatch = MPForest(clf, feature, sample, number_of_patches, oop_score=True)
            minipatch.fit(X,y)
            oop_errors.append(1-minipatch.oop_score_)
    return oop_errors

