#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %%
#sns.set(font_scale = 5)
sns.set_context("talk")
fontsize = 35
fig, axs = plt.subplots(1,2, figsize=(40, 20))
mnist = pd.read_csv("mnist_accuracy_2.csv")
madelon = pd.read_csv("madelon_accuracy_2.csv")
religion = pd.read_csv("religion_accuracy_2.csv")
spambase = pd.read_csv("spambase_accuracy_2.csv")
calls = pd.read_csv("calls_accuracy_2.csv")
mnist_time = pd.read_csv("mnist_time_2.csv")
madelon_time = pd.read_csv("madelon_time_2.csv")
religion_time = pd.read_csv("religion_time_2.csv")
spambase_time = pd.read_csv("spambase_time_2.csv")
calls_time = pd.read_csv("calls_time_2.csv")
x_index = ['MNIST','Madelon','Religion','Spam','Calls']



mnist_mp_mean = np.mean(mnist['mp'])
madelon_mp_mean = np.mean(madelon['mp'])
religion_mp_mean = np.mean(religion['mp'])
spambase_mp_mean = np.mean(spambase['mp'])
calls_mp_mean = np.mean(calls['mp'])


mnist_rf_mean = np.mean(mnist['rf'])
madelon_rf_mean = np.mean(madelon['rf'])
religion_rf_mean = np.mean(religion['rf'])
spambase_rf_mean = np.mean(spambase['rf'])
calls_rf_mean = np.mean(calls['rf'])


mnist_mp_err = np.std(mnist['mp'])
madelon_mp_err = np.std(madelon['mp'])
religion_mp_err = np.std(religion['mp'])
spambase_mp_err = np.std(spambase['mp'])
calls_mp_err = np.std(calls['mp'])


mnist_rf_err = np.std(mnist['rf'])
madelon_rf_err = np.std(madelon['rf'])
religion_rf_err = np.std(religion['rf'])
spambase_rf_err = np.std(spambase['rf'])
calls_rf_err = np.std(calls['rf'])

mp_values = [mnist_mp_mean, madelon_mp_mean,religion_mp_mean,spambase_mp_mean,calls_mp_mean]
mp_err = [mnist_mp_err, madelon_mp_err,religion_mp_err,spambase_mp_err,calls_mp_err]
rf_values = [mnist_rf_mean, madelon_rf_mean,religion_rf_mean,spambase_rf_mean,calls_rf_mean]
rf_err = [mnist_rf_err, madelon_rf_err,religion_rf_err,spambase_rf_err,calls_rf_err]

##################
mnist_mp_time_mean = np.mean(mnist_time['mp'])
madelon_mp_time_mean = np.mean(madelon_time['mp'])
religion_mp_time_mean = np.mean(religion_time['mp'])
spambase_mp_time_mean = np.mean(spambase_time['mp'])
calls_mp_time_mean = np.mean(calls_time['mp'])


mnist_rf_time_mean = np.mean(mnist_time['rf'])
madelon_rf_time_mean = np.mean(madelon_time['rf'])
religion_rf_time_mean = np.mean(religion_time['rf'])
spambase_rf_time_mean = np.mean(spambase_time['rf'])
calls_rf_time_mean = np.mean(calls_time['rf'])


mnist_mp_time_err = np.std(mnist_time['mp'])
madelon_mp_time_err = np.std(madelon_time['mp'])
religion_mp_time_err = np.std(religion_time['mp'])
spambase_mp_time_err = np.std(spambase_time['mp'])
calls_mp_time_err = np.std(calls_time['mp'])


mnist_rf_time_err = np.std(mnist_time['rf'])
madelon_rf_time_err = np.std(madelon_time['rf'])
religion_rf_time_err = np.std(religion_time['rf'])
spambase_rf_time_err = np.std(spambase_time['rf'])
calls_rf_time_err = np.std(calls_time['rf'])

mp_time_values = [mnist_mp_time_mean, madelon_mp_time_mean,religion_mp_time_mean,spambase_mp_time_mean,calls_mp_time_mean]
mp_time_err = [mnist_mp_time_err, madelon_mp_time_err,religion_mp_time_err,spambase_mp_time_err,calls_mp_time_err]
rf_time_values = [mnist_rf_time_mean, madelon_rf_time_mean,religion_rf_time_mean,spambase_rf_time_mean,calls_rf_time_mean]
rf_time_err = [mnist_rf_time_err, madelon_rf_time_err,religion_rf_time_err,spambase_rf_time_err,calls_rf_time_err]




#x3 = df['accuracy'].copy()
y = np.arange(1,6)
width = 0.3
axs[0].bar(y-width, mp_values, yerr = mp_err,width=width, color = 'r',label = "MP Forest")

axs[0].bar(y, rf_values,yerr = rf_err, width=width, color = 'b', label = "Random Forest")

axs[0].set_xlabel("Dataset",fontsize = fontsize)
axs[0].set_xticks(list(np.arange(1,6)))
axs[0].set_xticklabels(x_index, rotation=30, ha='right',fontsize = fontsize)

axs[0].set_ylabel("Accuracy", fontsize = fontsize)
axs[0].yaxis.set_tick_params(labelsize=fontsize)

###################
axs[1].bar(y-width, mp_time_values, yerr = mp_time_err,width=width, color = 'r',label = "MP Forest")

axs[1].bar(y, rf_time_values,yerr = rf_time_err, width=width, color = 'b', label = "Random Forest")

axs[1].set_xlabel("Dataset",fontsize = fontsize)
axs[1].set_xticks(list(np.arange(1,6)))
axs[1].set_xticklabels(x_index, rotation=30, ha='right',fontsize = fontsize)

axs[1].set_ylabel("Training Time (sec)", fontsize = fontsize)
axs[1].yaxis.set_tick_params(labelsize=fontsize)
axs[1].legend(loc = 'upper right', fontsize = fontsize)
axs[1].set_yscale('log')
plt.savefig("accuracy_time.pdf")
plt.show()
# %%
