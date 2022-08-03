#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %%
sns.set(font_scale = 3)
sns.set_context("talk")
fig, axs = plt.subplots(1,figsize=(20,20))
pancan = pd.read_csv("pancan_accuracy_1.csv")
madelon = pd.read_csv("madelon_accuracy_1.csv")
religion = pd.read_csv("religion_accuracy_1.csv")
spambase = pd.read_csv("spambase_accuracy_1.csv")
calls = pd.read_csv("calls_accuracy_1.csv")
x_index = ['Pancan','Madelon','Religion','Spambase','Calls']



pancan_mp_mean = np.mean(pancan['mp'])
madelon_mp_mean = np.mean(madelon['mp'])
religion_mp_mean = np.mean(religion['mp'])
spambase_mp_mean = np.mean(spambase['mp'])
calls_mp_mean = np.mean(calls['mp'])


pancan_rf_mean = np.mean(pancan['rf'])
madelon_rf_mean = np.mean(madelon['rf'])
religion_rf_mean = np.mean(religion['rf'])
spambase_rf_mean = np.mean(spambase['rf'])
calls_rf_mean = np.mean(calls['rf'])


pancan_mp_err = np.std(pancan['mp'])
madelon_mp_err = np.std(madelon['mp'])
religion_mp_err = np.std(religion['mp'])
spambase_mp_err = np.std(spambase['mp'])
calls_mp_err = np.std(calls['mp'])


pancan_rf_err = np.std(pancan['rf'])
madelon_rf_err = np.std(madelon['rf'])
religion_rf_err = np.std(religion['rf'])
spambase_rf_err = np.std(spambase['rf'])
calls_rf_err = np.std(calls['rf'])

mp_values = [pancan_mp_mean, madelon_mp_mean,religion_mp_mean,spambase_mp_mean,calls_mp_mean]
mp_err = [pancan_mp_err, madelon_mp_err,religion_mp_err,spambase_mp_err,calls_mp_err]
rf_values = [pancan_rf_mean, madelon_rf_mean,religion_rf_mean,spambase_rf_mean,calls_rf_mean]
rf_err = [pancan_rf_err, madelon_rf_err,religion_rf_err,spambase_rf_err,calls_rf_err]


#x3 = df['accuracy'].copy()
y = np.arange(1,6)
width = 0.3
axs.bar(y-width, mp_values, yerr = mp_err,width=width, color = 'r',label = "MP Forest")
#axs[0].set_ylabel("Fairness Importance Score")
#axs[0].title.set_text("(A) Fairness Feature Importance Score")
#axs[0].set_xlabel("feature")
#axs[0].legend(loc='lower right')
axs.bar(y, rf_values,yerr = rf_err, width=width, color = 'b', label = "Random Forest")
#axs[1].set_ylabel("Occlusion Fairness Importance Score")
axs.set_xlabel("Dataset",fontsize = 35)
axs.set_xticks(list(np.arange(1,6)))
axs.set_xticklabels(x_index, rotation=30, ha='right',fontsize = 35)
#axs[1].title.set_text("(B)Occlusion Fairness Feature Importance Score")
#axs[1].legend(loc='lower right')

#axs.bar(y+width, x3,width=width, color = 'g', label = "Accuracy Score")
axs.set_ylabel("Accuracy", fontsize = 35)
#axs[2].set_xlabel("feature")
#axs.title.set_text()
#axs[2].legend(loc='lower right')
axs.legend(loc = 'lower right')
plt.savefig("accuracy_1.pdf")
plt.show()
# %%
