import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

iris=pd.read_csv('iris.csv',header=0)

#print(iris.shape)
#print(iris.head)
#print(list(iris.columns))
#print(iris.info())
#print(iris.describe())
#print(iris['species'].value_counts())

#%%Data Visualization for better results and understanding
#tmp=iris.drop('species',axis=1)
#g= sns.pairplot(tmp,markers='+',hue='petal_length')
#plt.show()

#%%This can be an effective and attractive way to show multiple distributions 
#of data at once, but keep in mind that the estimations procudure is influenced 
#by the sample size, and violins for relatively small samples might look misleadingly smooth

#g=sns.violinplot(y='species',x='sepal_length',data=iris,inner='quartile')
#plt.show()
#g=sns.violinplot(y='species',x='sepal_width',data=iris,inner='quartile')
#plt.show()
#g=sns.violinplot(y='species',x='petal_length',data=iris,inner='quartile')
#plt.show()
#g=sns.violinplot(y='species',x='petal_width',data=iris,inner='quartile')
#plt.show()

#%%splitting data sets
X=iris.drop(['sepal_length','species'],axis=1)
y=iris['species']

#print(X.head())
print(X.shape)
#print('Y heads\n',y.head())

print(y.shape)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=0)

#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)

ppn=Perceptron(max_iter=40,eta0=0.1,random_state=0)#You can only use max_iter only on fit method not on partial_fit method
ppn.fit(X_train,y_train)
y_pred=ppn.predict(X_test)

#print(y_test,'\n')
#print(y_pred)

print('misclassified',(y_test!=y_pred).sum())
