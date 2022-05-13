import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pylab as plt
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("final.csv")
df = df.iloc[:,3:]

feature_names = ["AGE", "QALY", "MARITAL", "HEALTHCARE_EXPENSES", "DALY", "Body Mass Index", "Body Weight", "QOLS", "High Density Lipoprotein Cholesterol", "Odor of Urine", "Glucose [Presence] in Urine by Test strip", "Nitrite [Presence] in Urine by Test strip", "Hemoglobin [Presence] in Urine by Test strip", "Leukocyte esterase [Presence] in Urine by Test strip", "Appearance of Urine"]

X = df[feature_names]

%matplotlib inline
corr = X.corr()

f, ax = plt.subplots(figsize=(10,12))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

_ = sns.heatmap(corr, cmap="YlOrBr", square=True, ax=ax, annot=True, linewidth=0.1)

plt.title("Pearson correlation of Features", y=1.05, size=15)
plt.savefig("figs/correlation.png",dpi=300)
plt.show()

y = df['Miscarriage in first trimester']

onehot_ftrs = ["MARITAL"]

std_ftrs = ["AGE", "QALY", "HEALTHCARE_EXPENSES", "DALY", "Body Mass Index", "Body Weight", "QOLS", "High Density Lipoprotein Cholesterol", "Odor of Urine", "Glucose [Presence] in Urine by Test strip", "Nitrite [Presence] in Urine by Test strip", "Hemoglobin [Presence] in Urine by Test strip", "Leukocyte esterase [Presence] in Urine by Test strip", "Appearance of Urine"]
# One hot encoder 
# Solve the missing values here
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse=False,handle_unknown='ignore'))])
# standard scaler
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])
# collect the transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('std', numeric_transformer, std_ftrs),
        ('onehot', categorical_transformer,onehot_ftrs)])

def MLpipe_KFold_ACC(X, y, preprocessor, ML_algo, param_grid):
    nr_states = 5
    acc_scores = np.zeros(nr_states)
    f1_scores = np.zeros(nr_states)
    auc_scores = np.zeros(nr_states)
    final_models = []

    for i in range(nr_states):
        # first split to separate out the test set
        # we will use kfold on other
        X_other, X_test, y_other, y_test = train_test_split(X,y,test_size = 0.2,random_state=42*i)

        # splitter for other
        kf = KFold(n_splits=4,shuffle=True,random_state=42*i)

        # preprocess the data 
        # get the machine learning algorithms
        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', ML_algo)])

        # use GridSearchCV
        # GridSearchCV loops through all parameter combinations and collects the results 
        grid = GridSearchCV(pipe, param_grid=param_grid,scoring = 'f1',
                            cv=kf, return_train_score = True, n_jobs=-1, verbose=True)

        # fits the model on other
        grid.fit(X_other, y_other)
        # save results into a data frame
        # get the feature names after preprocessor
        feature_names = std_ftrs + \
                list(grid.best_estimator_[0].named_transformers_['onehot'][0].get_feature_names(onehot_ftrs))
        results = pd.DataFrame(grid.cv_results_)
        #print(results)

        print('best model parameters:',grid.best_params_)
        print('validation score:',grid.best_score_) # this is the mean validation score over all iterations
        # save the model
        final_models.append(grid)
        # calculate and save the test score
        y_test_pred = final_models[-1].predict(X_test)
        acc_scores[i] = accuracy_score(y_test,y_test_pred)
        f1_scores[i] = f1_score(y_test,y_test_pred)
        auc_scores[i] = roc_auc_score(y_test,y_test_pred)
        print('test score:',acc_scores[i])
    return final_models, np.array(feature_names), acc_scores, f1_scores, auc_scores

ML_algo = RandomForestClassifier()
param_grid = {
              'classifier__max_depth': [1, 3, 10, 30, 100], # the max_depth should be smaller or equal than the number of features roughly
              'classifier__max_features': [0.5,0.75,1.0] # linearly spaced between 0.5 and 1
              } 
RF_models, ftr_names,acc_scores, f1_scores, auc_scores = MLpipe_KFold_ACC(X,y,preprocessor,ML_algo,param_grid)
print("test scores (acc) for Random Forest Classifier: ")
print(acc_scores)
RF_acc_mean = np.mean(acc_scores)
RF_acc_std = np.std(acc_scores)
RF_f1_mean = np.mean(f1_scores)
RF_auc_mean = np.mean(auc_scores)

print(f"mean: {RF_acc_mean}, std: {RF_acc_std}")

ML_algo = LogisticRegression()
param_grid = {
     'classifier__penalty' : ['elasticnet'],
     'classifier__C' : [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
     'classifier__l1_ratio' :[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
     'classifier__solver' : ['saga'],
     'classifier__max_iter' : [100000]}

EN_models,ftr_names, acc_scores, f1_scores, auc_scores = MLpipe_KFold_ACC(X,y,preprocessor,ML_algo,param_grid)
print("test scores (acc) for Logistic Regression with EN penalty: ")
print(acc_scores)
EN_acc_mean = np.mean(acc_scores)
EN_acc_std = np.std(acc_scores)
EN_f1_mean = np.mean(f1_scores)
EN_auc_mean = np.mean(auc_scores)

print(f"mean: {EN_acc_mean}, std: {EN_acc_std}")

best_model = EN_models[-1]
coefs = best_model.best_estimator_[-1].coef_[0]
coefs = best_model.best_estimator_[-1].coef_[0]
sorted_indcs = np.argsort(np.abs(coefs))

plt.rcParams.update({'font.size': 14})
plt.barh(np.arange(10),coefs[sorted_indcs[-10:]])
plt.yticks(np.arange(10),ftr_names[sorted_indcs[-10:]])
plt.xlabel('coefficient')
plt.title('all scaled feature importance top 10')
#plt.tight_layout()
plt.savefig('figs/EN_coefs_scaled.png',dpi=300)
plt.show()