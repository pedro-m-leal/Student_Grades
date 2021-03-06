#Libraries for Data Visualization

import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import numpy as np 
import seaborn as sns
import time
import pandas_profiling as pp

from sklearn.preprocessing import StandardScaler, RobustScaler #Scaling Time and Amount
from mpl_toolkits import mplot3d
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV
from sklearn.model_selection import StratifiedKFold, learning_curve, StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import f1_score, precision_score, precision_recall_fscore_support, fbeta_score, classification_report, roc_curve, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


## Visualizations functions

def histograms(series):
    if isinstance(series,pd.Series):
        mn=int(min(series))
        mx=int(max(series))
        n_bins=len(range(mn,mx))+1
        ax=sns.distplot(series,n_bins,kde=False)
        ax.set_title('{}'.format(series.name))
        plt.show()
    elif isinstance(series,pd.DataFrame):
        cols=series.columns
        length=len(cols)
        if length%4==0:
            n_rows=4
            n_columns=length//4
        else:
            n_rows=4
            n_columns=length//4+1
        
        fig, axis=plt.subplots(nrows=n_rows,ncols=n_columns)
        for i in range(length):
            mn=int(min(series[cols[i]]))
            mx=int(max(series[cols[i]]))
            n_bins=len(range(mn,mx))+1
            sns.distplot(series[cols[i]],n_bins,kde=False,ax=axis.flatten()[i])
            axis.flatten()[i].set_title(cols[i])
        plt.show()
    else:
        raise ValueError('Please use pd.Series or pd.DataFrame as argument.')
def boxplots(series):
    if isinstance(series,pd.Series):
        ax=sns.boxplot(series)
        ax.set_title('{}'.format(series.name))
        plt.show()
    elif isinstance(series,pd.DataFrame):
        cols=series.columns
        length=len(cols)
        if length%2==0:
            n_rows=2
            n_columns=length//2
        else:
            n_rows=2
            n_columns=length//2+1

        fig, axis=plt.subplots(nrows=n_rows,ncols=n_columns)
        for i in range(length):
            sns.boxplot(series[cols[i]],ax=axis.flatten()[i])
            axis.flatten()[i].set_title(cols[i])
        plt.show()
    else:
        raise ValueError('Please use pd.Series or pd.DataFrame as argument.')
def countplots(series):
    if isinstance(series,pd.Series):
        ax=sns.countplot(series)
        ax.set_title('{}'.format(series.name))
        plt.show()
    elif isinstance(series,pd.DataFrame):
        cols=series.columns
        length=len(cols)
        if length%2==0:
            n_rows=2
            n_columns=length//2
        else:
            n_rows=2
            n_columns=length//2+1

        fig, axis=plt.subplots(nrows=n_rows,ncols=n_columns)
        for i in range(length):
            sns.countplot(series[cols[i]],ax=axis.flatten()[i])
            axis.flatten()[i].set_title(cols[i])
        plt.show()
    else:
        raise ValueError('Please use pd.Series or pd.DataFrame as argument.')
def frequency_plot(df,feature,feature_to_group='LeagueId'):
    freq_prep=train_df[[feature_to_group,feature]].groupby([feature_to_group]).sum()
    counts=freq_prep[feature].values
    leagues=freq_prep.index.values
    length=len(leagues)
    if length%2==0:
        n_rows=2
        n_columns=length//2
    else:
        n_rows=2
        n_columns=length//2+1

    fig, axis=plt.subplots(nrows=n_rows,ncols=n_columns)
    for i in range(len(leagues)):
        feature_counts=train_df[train_df[feature_to_group]==leagues[i]].groupby([feature]).count()[feature_to_group].values
        freqs=np.asarray(feature_counts)/sum(feature_counts)
        feature_values=train_df[train_df[feature_to_group]==leagues[i]].groupby([feature]).count()[feature_to_group].index.values
        sns.barplot(x=feature_values,y=freqs,ax=axis.flatten()[i])
        axis.flatten()[i].set_title('{} : {}'.format(feature_to_group,leagues[i]))
    plt.suptitle('{}'.format(feature))
    plt.show()

## Import Data

input=pd.read_csv('StudentsPerformance.csv')
df=input.copy(deep=True)

print('\n\n ============ Printing df head and create profile ============= \n\n')

print(df.head())
print(df.shape)

print(df.dtypes)

## Pandas Profiling

try:
    fn=open('pandas_profiling_1.html')
    fn.close()
except:
    print('\n---- No profile exists yet. Creating profile. ----\n')
    profile=df.profile_report(title='Pandas Profiling Report')
    profile.to_file(output_file='pandas_profiling_1.html')

## Plots

sns.scatterplot(data=df,x='writing score',y='math score',hue='test preparation course')
plt.show()

sns.heatmap(df.corr(),annot=True,cmap='PuBu')
plt.show()

sns.pairplot(data=df,hue='race/ethnicity',diag_kind='kde',kind='scatter')
plt.show()

sns.catplot(data=df,x='race/ethnicity',kind='count',hue='gender',row='test preparation course',col='lunch')
plt.show()

sns.catplot(data=df,x='race/ethnicity',y='math score',kind='violin',hue='gender',row='test preparation course',col='lunch')
plt.show()