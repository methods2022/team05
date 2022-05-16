import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.model_selection import RepeatedKFold
data = pd.read_csv('./final.csv')

def evaluate_model(trained_model, test_inputs, test_y):
    results = trained_model.evaluate(test_inputs, test_y)
    print(str(results[1]*100) + '%')
    return results[1]


targets = ["Fetus with unknown complication", "Tubal pregnancy", "Miscarriage in first trimester", "Preeclampsia", "Normal pregnancy"]
toPredict = data.drop(columns=['PATIENT_ID','BIRTHDATE','DEATHDATE'])
multi_class_target = data[targets].to_numpy()
toPredict = toPredict.drop(columns=targets).to_numpy()


def multi_model(toPredict, multi_class_target, targets,accuracies):
    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
    for train_ix, test_ix in cv.split(toPredict):
        X_train, X_test = toPredict[train_ix], toPredict[test_ix]
        y_train, y_test = multi_class_target[train_ix], multi_class_target[test_ix]
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(100, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(len(targets), activation='sigmoid'))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=.05), loss='binary_crossentropy', metrics=['acc'])
        hist = model.fit(X_train, y_train, batch_size=4, epochs=20)
        accuracies.append(evaluate_model(model, X_test, y_test))
    return accuracies, model

def bi_model(toPredict, data, targets):
    dict_for_storing = {}
    dict_for_storing['name'] = []
    dict_for_storing['acc'] = []
    for name in targets:
        labels = []
        for index, row in data.iterrows():
            if row[name] == 1:
                labels.append(1)
                continue
            else:
                labels.append(0)
                continue

        labels = np.array(labels)
        dict_for_storing['name'].append(name)
        print(name)

        X_train, X_test, y_train, y_test = train_test_split(toPredict, labels, test_size=0.2, random_state=42)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(92, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=.005), loss='binary_crossentropy', metrics=['acc'])
        hist = model.fit(X_train, y_train, batch_size=8, epochs=100)
        print(model.summary())
        acc = evaluate_model(model, X_test, y_test)
        dict_for_storing['acc'].append(acc)
    return dict_for_storing


accuracies = bi_model(toPredict, data, targets)
print(accuracies)

acc, model = multi_model(toPredict,multi_class_target, targets, [])
print(sum(acc)/10)
print(model.summary())