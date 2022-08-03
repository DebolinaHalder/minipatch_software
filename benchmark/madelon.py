#%%
from tkinter import N
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from minipatch import MPForest
import functions as f
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn.model_selection import GridSearchCV
import minipatch

#%%
X_train = pd.read_csv('madelon/madelon_train.data',delimiter=' ',header=None)
X_test = pd.read_csv('madelon/madelon_valid.data',delimiter=' ',header=None)
y_train = pd.read_csv('madelon/madelon_train.labels',delimiter=' ',header=None)
y_test = pd.read_csv('madelon/madelon_valid.labels',delimiter=' ',header=None)
X_train = X_train.iloc[: , :-1]
X_test = X_test.iloc[: , :-1]
y_test = y_test.to_numpy()
y_train = y_train.to_numpy()
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
#%%
minipatch_m_ratios = [0.4, 0.45, 0.5]
minipatch_n_ratios = [0.3,0.4,0.5]
clf_mp = tree.DecisionTreeClassifier(max_depth=30)
m_ratio, n_ratio = f.crossValidation(minipatch_m_ratios,minipatch_n_ratios,clf_mp,number_of_patches=250,X=X_train,y=y_train)

iteration = 20
#result_df = pd.DataFrame(columns=['mp', 'rf'])
result_time = pd.DataFrame(columns=['mp', 'rf'])
result_accuracy = pd.DataFrame(columns=['mp', 'rf'])
for i in range(iteration):
    
    minipatch = MPForest(clf_mp,m_ratio,n_ratio,number_of_patches=250)
    start_mp = datetime.now()
    minipatch.fit(X_train, y_train)
    end_mp = datetime.now()
    pred_mp = minipatch.predict(X_test)
    accuracy_mp = accuracy_score(pred_mp, y_test)


    parameters = {'max_features':[0.005, 0.01,0.02]}
    rf = RandomForestClassifier(n_estimators=250, max_depth=30, n_jobs=-2)
    clf_rf = GridSearchCV(rf,parameters)
    start_rf = datetime.now()
    clf_rf.fit(X_train, y_train)
    end_rf = datetime.now()
    time_mp = (end_mp - start_mp).total_seconds()
    time_rf = (end_rf - start_rf).total_seconds()
    pred = clf_rf.predict(X_test)
    accuracy_rf = accuracy_score(pred, y_test)


    result_time = result_time.append({'mp':time_mp,'rf':time_rf}, ignore_index=True)
    result_accuracy = result_accuracy.append({'mp':accuracy_mp,'rf':accuracy_rf}, ignore_index=True)
# %%
result_time.to_csv("madelon_time_2.csv")
result_accuracy.to_csv("madelon_accuracy_2.csv")

#%%
