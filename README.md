# Ad-Click-Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import plotly.express as px

ad=pd.read_csv("/content/ad_10000records.csv")

print(ad.info())

ad=ad.drop_duplicates()
ad

ad=ad.drop(['City','Timestamp','Ad Topic Line'],axis=1)

ad["Gender"] = ad["Gender"].map({"Male": 1,"Female": 0})

print(ad.shape)
print(ad.info())
print(ad.describe())

le=LabelEncoder()
ad["country"]=le.fit_transform(ad["Country"])
c1_map={index: label for index,label in enumerate(le.classes_)}
print(c1_map)

ad.drop(['Country'],axis=1,inplace=True)
print(ad.info())

counts=ad['country'].value_counts()
Country_m= counts.reset_index()
Country_m.columns= ['country', 'Count']

merged= pd.merge(ad,Country_m,on='country')
print(merged)

merged1 = merged.assign(Group_1=lambda x: x['Count'].apply(lambda val: 1 if val >= 300 else 0))
merged1= merged1.assign(Group_2=lambda x: x['Count'].apply(lambda val: 1 if val >= 200 and val < 300 else 0))
merged1 = merged1.assign(Group_3=lambda x: x['Count'].apply(lambda val: 1 if val >= 100 and val < 200 else 0))
merged1 = merged1.assign(Group_4=lambda x: x['Count'].apply(lambda val: 1 if val >= 0 and val < 100 else 0))

print(merged1)

sns.heatmap(merged1.corr(),annot=True,cmap='mako',fmt="0.2f",annot_kws={"size":10})
plt.title('Correlation matrix')
plt.show()

count=pd.DataFrame(ad['Gender'].value_counts())
count=count.rename_axis(['Sex']).reset_index()
count=count.rename(columns={'Gender':'Count'})
count
px.bar(count, x='Sex', y='Count', color='Sex',template="none")

px.scatter(ad,y='Daily Time Spent on Site',x='Age', color='Clicked on Ad',template="none")

px.scatter(ad,y='Area Income',x='Age', color='Clicked on Ad',template="none")

px.scatter(ad,y='Daily Internet Usage',x='Age', color='Clicked on Ad',template="none",)

fig = px.box(merged1,
             x="Daily Internet Usage",
             color="Clicked on Ad",
             title="Click Through Rate based on Daily Internet Usage",
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

fig = px.box(merged1,
             x="Age",
             color="Clicked on Ad",
             title="Click Through Rate based on Age",
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

fig = px.box(merged1,
             x="Daily Time Spent on Site",
             color="Clicked on Ad",
             title="Click Through Rate based Time Spent on Site",
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

fig = px.box(merged1,
             x="Area Income",
             color="Clicked on Ad",
             title="Click Through Rate based Area Income",
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

outliers_col=["Area Income","Age"]
for column in outliers_col:
  if merged1[column].dtype in ["int64","float64"]:
    Q1=merged1[column].quantile(0.25)
    Q3=merged1[column].quantile(0.75)
    iqr=Q3-Q1
    lower_bound=Q1-1.5*iqr
    upper_bound=Q3+1.5+iqr
    merged1=merged1[(merged1[column]>=lower_bound) & (merged1[column]<=upper_bound)]
    print(merged1)

merged1['Area Income'].plot(kind='box')
plt.xticks(rotation=90)
plt.show()

merged1['Age'].plot(kind='box')
plt.xticks(rotation=90)
plt.show()

X=merged1.drop(['Clicked on Ad','country','Count','Group_1','Group_2','Group_3','Group_4'], axis=1)
y=merged1['Clicked on Ad']

models=[]
models.append(('KNN',KNeighborsClassifier()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('RF',RandomForestClassifier()))
models.append(('SVM',svm.SVC()))

results=[]
names=[]
scoring='accuracy'
kfold=KFold(n_splits=10)
for name,model in models:
    cv_results=cross_val_score(model,X,y,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print('cv results:',cv_results)
    print(f"accuracy of {name} is {cv_results.mean()}")

fig=plt.figure()
fig.suptitle("Algorithm Comparison")
ax=fig.add_subplot(111)
plt.boxplot(results)

ax.set_xticklabels(names)
plt.show()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train1=scaler.fit_transform(X_train)
X_test1=scaler.transform(X_test)

print(X_train1.shape)
print(X_test1.shape)
print(y_train.shape)
print(y_test.shape)

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train1, y_train)

y_pred = random_forest_model.predict(X_test1)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("conf_matrix:\n", confusion_matrix(y_test, y_pred))

plt.subplot(2,2,1)
plt.scatter(x=y_test, y=y_pred, color='b')
plt.title('Actual V/S Predicted points')
plt.xlabel('Actual points')
plt.ylabel('Predicted points')
plt.show()

print("Ads Click Through Rate Prediction : ")
new_data = {
    'Daily Time Spent on Site': float(input("Daily Time Spent on Site: ")),
    'Age': float(input("Age: ")),
    'Area Income': float(input("Area Income: ")),
    'Daily Internet Usage': float(input("Daily Internet Usage: ")),
    'Gender': input("Gender (Male = 1, Female = 0) : ")
}

features = pd.DataFrame([new_data])
new_data_scaled = scaler.transform(features)
predictions = random_forest_model.predict(new_data_scaled)

print("Will the user click on ad =", predictions)
