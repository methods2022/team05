using CSV
using DataFrames
using Statistics
df = CSV.read("final.csv", DataFrame)

#Model training
using ScikitLearn: fit!, predict, @sk_import, fit_transform!
@sk_import model_selection: cross_val_score  
@sk_import metrics: accuracy_score 
@sk_import linear_model: LogisticRegression 
@sk_import ensemble: RandomForestClassifier 
@sk_import tree: DecisionTreeClassifier 

#for Fetus with unknown complication:
feature_names = ["HEALTHCARE_EXPENSES", "AGE", "MARITAL", "Platelets [#/volume] in Blood by Automated count", "Albumin [Mass/volume] in Serum or Plasma", "QALY", "Ketones [Mass/volume] in Urine by Test strip", "Protein [Presence] in Urine by Test strip", "Sodium", "High Density Lipoprotein Cholesterol", "Bilirubin.total [Mass/volume] in Urine by Test strip", "Leukocytes [#/volume] in Blood by Automated count", "Respiratory rate", "DALY", "Body temperature"]

function fetus_model(model, prediction_var="Fetus with unknown complication") 
    #col_names = names(df)[4:130]   
    X = convert(Array, df[feature_names])
    y = convert(Array, df[prediction_var])                    
     
    #Fit the model: 
     fit!(model, X, y) 

     #Make predictions on training set: 
     predictions = predict(model, X) 

     #Print accuracy 
     accuracy = accuracy_score(predictions, y) 
     println("\naccuracy: ",accuracy) 

     #5 fold cross validation 
     cross_score = cross_val_score(model, X, y, cv=5)    
 
     #print cross_val_score 
     println("cross_validation_score: ", mean(cross_score)) 

      
     return mean(cross_score)
end

println("target variable: Fetus with unknown complication")
model = LogisticRegression()
LR_score = fetus_model(model)
println("Score for Logistic Regression: ",LR_score)

model = DecisionTreeClassifier()
DT_score = fetus_model(model)
println("Score for Decision Tree: ",DT_score)

model = RandomForestClassifier(n_estimators=100)
RF_score = fetus_model(model)
println("Score for Random Forest: ",RF_score)


#For miscarriage:
feature_names = ["AGE", "QALY", "MARITAL", "HEALTHCARE_EXPENSES", "DALY", "Body Mass Index", "Body Weight", "QOLS", "High Density Lipoprotein Cholesterol", "Odor of Urine", "Glucose [Presence] in Urine by Test strip", "Nitrite [Presence] in Urine by Test strip", "Hemoglobin [Presence] in Urine by Test strip", "Leukocyte esterase [Presence] in Urine by Test strip", "Appearance of Urine"]

function miscarriage_model(model, prediction_var="Miscarriage in first trimester") 
    #col_names = names(df)[4:130]   
    X = convert(Array, df[feature_names])
    y = convert(Array, df[prediction_var])                    
     
    #Fit the model: 
     fit!(model, X, y) 

     #Make predictions on training set: 
     predictions = predict(model, X) 

     #Print accuracy 
     accuracy = accuracy_score(predictions, y) 
     println("\naccuracy: ",accuracy) 

     #5 fold cross validation 
     cross_score = cross_val_score(model, X, y, cv=5)    
 
     #print cross_val_score 
     println("cross_validation_score: ", mean(cross_score)) 

      
     return mean(cross_score)
end

println("target variable: Miscarriage in first trimester")
model = LogisticRegression()
LR_score = miscarriage_model(model)
println("Score for Logistic Regression: ",LR_score)

model = DecisionTreeClassifier()
DT_score = miscarriage_model(model)
println("Score for Decision Tree: ",DT_score)

model = RandomForestClassifier(n_estimators=100)
RF_score = miscarriage_model(model)
println("Score for Random Forest: ",RF_score)


# For Tubal pregnancy:
feature_names = ["AGE", "HEALTHCARE_EXPENSES", "DALY", "QOLS", "Potassium", "QALY", "Protein [Mass/volume] in Serum or Plasma", "Polyp size greatest dimension by CAP cancer protocols", "Body temperature", "Carbon Dioxide", "Platelet distribution width [Entitic volume] in Blood by Automated count", "Respiratory rate", "Total score [MMSE]", "Hemoglobin.gastrointestinal [Presence] in Stool by Immunologic method"]

function tubal_model(model, prediction_var="Tubal pregnancy") 
    #col_names = names(df)[4:130]   
    X = convert(Array, df[feature_names])
    y = convert(Array, df[prediction_var])                    
     
    #Fit the model: 
     fit!(model, X, y) 

     #Make predictions on training set: 
     predictions = predict(model, X) 

     #Print accuracy 
     accuracy = accuracy_score(predictions, y) 
     println("\naccuracy: ",accuracy) 

     #5 fold cross validation 
     cross_score = cross_val_score(model, X, y, cv=5)    
 
     #print cross_val_score 
     println("cross_validation_score: ", mean(cross_score)) 

      
     return mean(cross_score)
end

println("target variable: Tubal pregnancy")
model = LogisticRegression()
LR_score = tubal_model(model)
println("Score for Logistic Regression: ",LR_score)

model = DecisionTreeClassifier()
DT_score = tubal_model(model)
println("Score for Decision Tree: ",DT_score)

model = RandomForestClassifier(n_estimators=100)
RF_score = tubal_model(model)
println("Score for Random Forest: ",RF_score)


# For Preeclampsia:
feature_names = ["AGE", "QALY", "HEALTHCARE_EXPENSES", "RDW - Erythrocyte distribution width Auto (RBC) [Entitic vol]", "DALY", "Systolic Blood Pressure", "Body Mass Index", "Heart rate", "QOLS", "Body Weight", "Platelet distribution width [Entitic volume] in Blood by Automated count", "RACE", "MCHC [Mass/volume] by Automated count", "RBC Auto (Bld) [#/Vol]", "Triglycerides"]

function preeclampsia_model(model, prediction_var="Preeclampsia") 
    #col_names = names(df)[4:130]   
    X = convert(Array, df[feature_names])
    y = convert(Array, df[prediction_var])                    
     
    #Fit the model: 
     fit!(model, X, y) 

     #Make predictions on training set: 
     predictions = predict(model, X) 

     #Print accuracy 
     accuracy = accuracy_score(predictions, y) 
     println("\naccuracy: ",accuracy) 

     #5 fold cross validation 
     cross_score = cross_val_score(model, X, y, cv=5)    
 
     #print cross_val_score 
     println("cross_validation_score: ", mean(cross_score)) 

      
     return mean(cross_score)
end

println("target variable: preeclampsia")
model = LogisticRegression()
LR_score = preeclampsia_model(model)
println("Score for Logistic Regression: ",LR_score)

model = DecisionTreeClassifier()
DT_score = preeclampsia_model(model)
println("Score for Decision Tree: ",DT_score)

model = RandomForestClassifier(n_estimators=100)
RF_score = preeclampsia_model(model)
println("Score for Random Forest: ",RF_score)


#For Normal Preganancy:
feature_names = ["AGE", "QALY", "HEALTHCARE_EXPENSES", "DALY", "MARITAL", "Body Weight", "Body Mass Index", "Response to cancer treatment", "Heart rate", "Carbon Dioxide", "High Density Lipoprotein Cholesterol", "Regional lymph nodes.clinical [Class] Cancer", "Stage group.clinical Cancer", "Treatment status Cancer", "Potassium", "Distant metastases.clinical [Class] Cancer"]

function normal_model(model, prediction_var="Normal pregnancy") 
    #col_names = names(df)[4:130]   
    X = convert(Array, df[feature_names])
    y = convert(Array, df[prediction_var])                    
     
    #Fit the model: 
     fit!(model, X, y) 

     #Make predictions on training set: 
     predictions = predict(model, X) 

     #Print accuracy 
     accuracy = accuracy_score(predictions, y) 
     println("\naccuracy: ",accuracy) 

     #5 fold cross validation 
     cross_score = cross_val_score(model, X, y, cv=5)    
 
     #print cross_val_score 
     println("cross_validation_score: ", mean(cross_score)) 

      
     return mean(cross_score)
end

println("target variable: Normal pregnancy")
model = LogisticRegression()
LR_score = normal_model(model)
println("Score for Logistic Regression: ",LR_score)

model = DecisionTreeClassifier()
DT_score = normal_model(model)
println("Score for Decision Tree: ",DT_score)

model = RandomForestClassifier(n_estimators=100)
RF_score = normal_model(model)
println("Score for Random Forest: ",RF_score)