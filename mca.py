#Libraries for Data Visualization

import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import numpy as np 
import seaborn as sns
import time
import pandas_profiling as pp
import prince

from sklearn.preprocessing import StandardScaler, RobustScaler #Scaling Time and Amount
from mpl_toolkits import mplot3d
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV
from sklearn.model_selection import StratifiedKFold, learning_curve, StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import f1_score, precision_score, precision_recall_fscore_support, fbeta_score, classification_report, roc_curve, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

## Import Data

input=pd.read_csv('StudentsPerformance.csv')
df=input.copy(deep=True)

print('\n\n ============ Printing df head and types ============= \n\n')

print(df.head())
print(df.shape)

print(df.dtypes)

## Types of variables

categorical=['gender','race/ethnicity','parental level of education','lunch','test preparation course']
numeric=['math score','reading score','writing score']

"""

We will use the categorical data as active variables for MCA

"""
print(df[categorical].head())


mca=prince.MCA(n_components=sum([len(df[i].value_counts().index.tolist()) for i in categorical]))

df_mca=mca.fit_transform(df[categorical])

print(df_mca.head())

print(mca.explained_inertia_)
print(mca.col_masses_)
print(mca.eigenvalues_)

cm=df_mca.corr()
eigv=np.linalg.eig(cm)
print(eigv[0])
sns.heatmap(cm,annot=True,cmap='PuBu')
plt.show()


ax=mca.plot_coordinates(X=df[categorical],show_column_labels=True)
plt.show()

sns.scatterplot(data=df_mca,x=0,y=1,hue=df['writing score'],cmap='PuBu')
plt.show()