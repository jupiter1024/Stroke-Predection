import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

data=pd.read_csv("healthcare-dataset-stroke-data.csv")
data

data.shape

data.head(10)

data.isnull().sum()

data.apply(lambda x: x.value_counts().get("Unknown", 0))

data['work_type'].value_counts(normalize=True)

data['smoking_status'].value_counts(normalize=True)

#data.drop(data[data['smoking_status'] == 'Unknown'].index, inplace=True)

data.apply(lambda x: x.value_counts().get("Other", 0))

data.shape

#correlation = data.corr()
#correlation

#sns.histplot(data, x="gender",hue="stroke")

#sns.pairplot(data)

plt.figure(figsize=(25, 5))
sns.displot(data=data,x='age', hue='stroke',kde=False, bins=20)
sns.despine()

#plt.figure(figsize=(15, 5))
#sns.displot(data=data,x='smoking_status', hue='stroke',kde=False, bins=20)
#sns.despine()

#plt.figure(figsize=(15, 5))
#sns.displot(data=data,x='heart_disease', hue='stroke',kde=False, bins=20)
#sns.despine()

#plt.figure(figsize=(15, 5))
#sns.heatmap(correlation, annot=True)
#sns.despine()

#upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(np.bool))


#data=data.drop("work_type",axis=1)
#data=data.drop("Residence_type",axis=1)
#data=data.drop("ever_married",axis=1)
#data=data.drop("smoking_status",axis=1)
#data=data.drop("gender",axis=1)

map_dict = {'Private': 0, 'Self-employed': 1, 'children': 2, 'Govt_job': 3, 'Never_worked': 4}
data['work_type'].replace(map_dict, inplace=True)

data.replace({'smoking_status': {'never smoked': 0, 'Unknown': -1,'smokes':1,'formerly smoked':2}}, inplace=True)

data.replace({'ever_married': {'Yes': 1, 'No': 0}}, inplace=True)


data.replace({'Residence_type': {'Urban': 1, 'Rural': 0}}, inplace=True)


data.replace({'gender': {'Male': 1, 'Female': 0,'Other':0}}, inplace=True)
data



mean_bmi = data['bmi'].mean()
data['bmi'].fillna(value=mean_bmi, inplace=True)


#calculate IQR for each column
#Q1 = data.quantile(0.25)
#Q3 = data.quantile(0.75)
#IQR = Q3 - Q1
#filter out values that are more than 1.5 times the IQR away from the Q1 or Q3
#outliers2 = data[((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
#outliers2

x=data.iloc[:,0:10]
y=data["stroke"]

x

y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

model=KNeighborsClassifier(n_neighbors=10)
model.fit(x_train,y_train)

y_preds=model.predict(x_test)
print(f"Testing score = {accuracy_score(y_test,y_preds)}")

#Testing score = 0.9483568075117371

#classifier = LogisticRegression(random_state = 0)

#classifier.fit(x_train, y_train)

#y_pred = classifier.predict(x_test)

#cm = confusion_matrix(y_test, y_pred)
#cm

#cr=classification_report(y_test, y_pred)
#print(cr)

joblib_file = "stroke prediction"
joblib.dump(model, joblib_file)
loaded_model = joblib.load(open(joblib_file, 'rb'))
y_pred = loaded_model.predict(x_test)
result = np.round(accuracy_score(y_test, y_pred) ,2)
print(result)

