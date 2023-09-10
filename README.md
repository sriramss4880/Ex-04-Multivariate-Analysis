# Ex-04-Multivariate-Analysis

# AIM :
To perform Multivariate EDA on the given data set.

# EXPLANATION :
Exploratory data analysis is used to understand the messages within a dataset. This technique involves many iterative processes to ensure that the cleaned data is further sorted to better understand the useful meaning.The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.

# ALGORITHM :
# STEP 1:
Import the built libraries required to perform EDA and outlier removal.

# STEP 2:
Read the given csv file.

# STEP 3:
Convert the file into a dataframe and get information of the data.

# STEP 4:
Return the objects containing counts of unique values using (value_counts()).

# STEP 5:
Plot the counts in the form of Histogram or Bar Graph.

# STEP 6:
Use seaborn the bar graph comparison of data can be viewed.

# STEP 7:
Find the pairwise correlation of all columns in the dataframe.corr() .

# STEP 8:
Save the final data set into the file.

# Program and Output:
Name : S.S.SRIRAM

Reg no : 212222230150

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


dt = pd.read_csv("/content/titanic_dataset.csv")
dt
```
<img width="597" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-04-Multivariate-Analysis/assets/121222763/1cbcc48b-9116-40da-bdc5-b3b5b801a6bc">

```
dt.set_index("PassengerId",inplace=True)

dt.describe()
```
<img width="305" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-04-Multivariate-Analysis/assets/121222763/6acbcf40-5d26-4656-9aa2-b66608d2c06f">

```
dt.info()

<img width="172" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-04-Multivariate-Analysis/assets/121222763/79831f68-fcd5-4b1b-86fd-437fdba17a59">
```
```

dt.shape
```
<img width="44" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-04-Multivariate-Analysis/assets/121222763/1cf90407-8433-4f76-a43b-f4806ca0bf70">

```
dt.nunique()
```
<img width="71" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-04-Multivariate-Analysis/assets/121222763/d049e894-f70f-4824-a29e-26d8f1ffa60e">

```
dt["Survived"].value_counts()
```
<img width="120" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-04-Multivariate-Analysis/assets/121222763/aaf43fe2-a046-47e6-8b79-f2e922c81a70">

```
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per
```
<img width="146" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-04-Multivariate-Analysis/assets/121222763/be06eb3e-7e21-4ef1-bf28-ec98cdc2a2d9">

```
sns.catplot(x="Age",col="Survived",kind="count",data=dt)
```
<img width="389" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-04-Multivariate-Analysis/assets/121222763/6db6ed2f-06bc-44ec-982c-57e625dfbc99">

```
fig,ax1=plt.subplots(figsize=(8,5))
graph = sns.countplot (ax=ax1,data=dt,x="Survived",hue="Pclass",palette="rainbow")
graph.set_xticklabels(graph.get_xticklabels())
for p in graph.patches:
  height = p.get_height()
  graph.text(p.get_x()+p.get_width()/2,height+20.8,height ,ha="left")
```
<img width="275" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-04-Multivariate-Analysis/assets/121222763/73e27afe-06f6-47df-9536-aa23a6605f24">

```
dt.boxplot(column="Age",by="Survived")

<img width="224" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-04-Multivariate-Analysis/assets/121222763/dac2af79-60b1-4e24-8bc0-ffebf9fbcdf1">
```
```
sns.scatterplot(x=dt["Age"],y=dt["Fare"])
```
<img width="223" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-04-Multivariate-Analysis/assets/121222763/0e6fd61a-f5e1-4e9f-80a5-023f87d5e5bc">

```
sns.jointplot(x="Age",y="Fare",data=dt)
```
<img width="240" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-04-Multivariate-Analysis/assets/121222763/f399618c-79fb-4567-a314-29d13cdec343">

```
fig,ax1=plt.subplots(figsize=(8,5))
pt=sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Parch',data=dt)
```
<img width="269" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-04-Multivariate-Analysis/assets/121222763/97b94a9a-c25e-476d-92d2-831564edd2f4">

```
sns.catplot(data=dt,col="Survived",x="Parch",hue='Pclass',kind="count")
```
<img width="409" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-04-Multivariate-Analysis/assets/121222763/69ea6a76-50c8-4ca2-be50-5d08955e96ea">

```
g=sns.catplot(data=dt,col="Survived",x="Parch",hue="Pclass",kind ="count",legend=True)
g.fig.set_size_inches(8,5)
g.fig.subplots_adjust(top=0.81,right=0.86)
ax=g.facet_axis(0,0)
for p in ax.patches:
  ax.text(p.get_x()-0.01,p.get_height()*1.02,'{0:.1f}'.
  format(p.get_height()),color='red',rotation="horizontal",size="small")
```
<img width="314" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-04-Multivariate-Analysis/assets/121222763/0c0332a3-7db4-47d6-90e4-4edf7c910539">

```
corr = dt.corr()
sns.heatmap(corr,annot=True)
```
<img width="730" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-04-Multivariate-Analysis/assets/121222763/f51e5d9e-6a2e-4400-939d-57e64151e2b5">

```
sns.pairplot(dt)
```
<img width="658" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-04-Multivariate-Analysis/assets/121222763/013841c3-577f-4756-a16e-a60c37e9aef0">
<img width="659" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-04-Multivariate-Analysis/assets/121222763/6ae0a26e-be68-406e-8327-9c1dcf198ff0">


# RESULT:
Thus the program to perform Multivariate EDA on the given data set is successfully executed.
