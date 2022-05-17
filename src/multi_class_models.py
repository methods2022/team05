import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv('final.csv')

# toPredict = data[["MARITAL", "RACE", "ETHNICITY", "Body Mass Index", "Diastolic Blood Pressure", "Systolic Blood Pressure", "Heart rate", "Hemoglobin [Mass/volume] in Blood", "Hematocrit [Volume Fraction] of Blood by Automated count", "Tobacco smoking status NHIS", "Total Cholesterol", "Glucose", ]].to_numpy()

#toPredict = data.drop(columns=['PATIENT_ID','BIRTHDATE','DEATHDATE', 'Fetus with unknown complication','Tubal pregnancy', 'Miscarriage in first trimester', 'Preeclampsia','Normal pregnancy']).to_numpy()
toPredictDF = data.drop(columns=['PATIENT_ID','BIRTHDATE','DEATHDATE', 'Fetus with unknown complication','Tubal pregnancy', 'Miscarriage in first trimester', 'Preeclampsia','Normal pregnancy', 'HEALTHCARE_EXPENSES'])
toPredict = toPredictDF.to_numpy()
labels = np.zeros((len(data.index)))
print(toPredict.shape)
for index, row in data.iterrows():
    if row['Fetus with unknown complication'] == 1:
        labels[index] = 0
        continue
    if row['Tubal pregnancy'] == 1:
        labels[index] = 1
        continue
    if row['Miscarriage in first trimester'] == 1:
        labels[index] = 2
        continue
    if row['Preeclampsia'] == 1:
        labels[index] = 3 
        continue
    if row['Normal pregnancy'] == 1:
        labels[index] = 4
        continue

X_train, X_test, y_train, y_test = train_test_split(toPredict, labels, test_size=0.2, random_state=42)

# model = LogisticRegression().fit(X_train, y_train)
model = tree.DecisionTreeClassifier()
model = model.fit(X_train, y_train)
print(model.score(X_test,y_test))

import graphviz

dot_data = tree.export_graphviz(model, out_file=None, 
    feature_names=toPredictDF.columns, 
    class_names=['Fetus with unknown complication','Tubal pregnancy','Miscarriage in first trimester','Preeclampsia','Normal pregnancy' ], 
    filled=True, 
    rounded=True, 
    special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render("smalldata_notemp") 
