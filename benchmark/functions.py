#%%
from minipatch import MPForest
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

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


def crossValidation(m_ratios, n_ratios, clf, number_of_patches, X, y, cv = 5):
    kf = KFold(n_splits=cv)
    accuracy = 0
    m_ratio_value = None
    n_ratio_value = None
    for feature in m_ratios:
        for sample in n_ratios:
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index].copy(), X[test_index].copy()
                y_train, y_test = y[train_index].copy(), y[test_index].copy()
                minipatch = MPForest(clf, feature, sample, number_of_patches)
                minipatch.fit(X_train,y_train)
                acc = accuracy_score(y_test,minipatch.predict(X_test))
                if acc > accuracy:
                    accuracy = acc
                    m_ratio_value = feature
                    n_ratio_value = sample
    return m_ratio_value, n_ratio_value