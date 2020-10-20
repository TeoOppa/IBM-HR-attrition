# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 21:20:23 2020

@author: garro
"""
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE

# Load dataset
a = pd.read_csv(r'C:\Users\garro\Desktop\Data Mining\DatasetProject1\Test_HR_Employee_Attrition.csv')
b = pd.read_csv(r'C:\Users\garro\Desktop\Data Mining\DatasetProject1\Train_HR_Employee_Attrition.csv')
df = pd.concat([a,b])

#Create a function for auc roc curve
def ROC_GEN(Title, Labels, Output): 
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr, tpr, _ = roc_curve(Labels, Output)   
    roc_auc = auc(fpr, tpr)    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(Title)
    plt.legend(loc="lower right")
    plt.show()
    
    return;
rows = df.shape[0]
columns = df.shape[1]
print("The dataset contains {0} rows and {1} columns".format(rows, columns))
print(df.head())
# start with semantics and first statistics such as means of the various attributes
print(df.mean())

#Seems like data is composed of different type of values: numerical, strings, floats. This will cause problems for future analysis because for example strings can't be used to fit a KClustering analysis. Deal with the problem later.
#The dataset is composed of 1470 rows and 33 columns. The objective of the analysis will be to determine which attributes are more correlated to the attrition of the employees.


# Display the statistical overview of the employees
print(df.describe())

# Start by comparing some attributes with each other, we expect to find some kind of correlation between some attributes. For example Age, Education, JobLevel
sns.pairplot(df[['Age', 'JobLevel', 'Education']])
plt.show()

# Clustering the dataset by K Means
kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = df._get_numeric_data().dropna(axis=1)
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1],c=labels)
plt.show()


#Then, we make a copy of data to make some changes on the copy so that the original data doesn't affect.Since we want to use output labels of "Attrition" in our numerical process. we have to use the equal numbers. For example "1" and "0" for "Yes" and"No" respectively. We divide the data to two sets of test and train for further processes.
df_copy = df.copy()
df_copy["Attrition"] = df_copy["Attrition"].replace(["Yes","No"],[1,0]);
train = df_copy.sample(frac=0.5, random_state=1)
test = df_copy.loc[~df_copy.index.isin(train.index)]

#### NEED TO SET DTYPE TO FLOAT32 HERE IS THE CODE BUT IT WON'T WORK ### 
pd.set_option('precision', 2)
df.astype(np.float32) 
#### ENDS HERE ###


# Eliminate NaN values from copy
df_copy['BusinessTravel'] = df_copy['BusinessTravel'].fillna("Travel_Rarely")
df_copy['Age'] = df_copy['Age'].fillna("37")
df_copy['Gender'] = df_copy['Gender'].fillna("Male")
df_copy['MonthlyIncome'] = df_copy['MonthlyIncome'].fillna("6549")
df_copy['Over18'] = df_copy['Over18'].fillna("Y")
df_copy['PerformanceRating'] = df_copy['PerformanceRating'].fillna("3")
df_copy['StandardHours'] = df_copy['StandardHours'].fillna("80")
df_copy['TrainingTimesLastYear'] = df_copy['TrainingTimesLastYear'].fillna("3")
df_copy['YearsAtCompany'] = df_copy['YearsAtCompany'].fillna("7")



# It can be seen, Some attributes like StandardHours" don't vary for various records. Thus, we can ignore them for the later processing. On the other hand, since text fields can not be used in numerical processing, we have to ignore them as well. The rest of the attributes are selected for the processing.
Effective_Columns = ["Age", "DailyRate", "DistanceFromHome", "Education", "MonthlyIncome","MonthlyRate" ,"NumCompaniesWorked",
"PercentSalaryHike","PerformanceRating","RelationshipSatisfaction",
"StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany",
"YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager",
"EnvironmentSatisfaction","HourlyRate","JobInvolvement","JobLevel","JobSatisfaction"]




# Random Forest
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
rf.fit(train[Effective_Columns], train["Attrition"])
predictions = rf.predict(test[Effective_Columns])