using CSV
using DataFrames
using Statistics
df = CSV.read("final.csv", DataFrame)
# rename columns to delete whitespaces
names!(df, Symbol.(replace.(string.(names(df)), Ref(r"\s"=>""))))
col_names = names(df)[4:130]
used_df = df[col_names]

#Model training
using ScikitLearn: fit!, predict, @sk_import, fit_transform!
@sk_import model_selection: cross_val_score  
@sk_import metrics: accuracy_score 
@sk_import linear_model: LogisticRegression 
@sk_import ensemble: RandomForestClassifier 
@sk_import tree: DecisionTreeClassifier 

function classification_model(model, prediction_var) 
    #col_names = names(df)[4:130]
    X_names = filter!(e->e ? prediction_var,col_names)    
    X = convert(Array, df[X_names])
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

# for predictor "Miscarriageinfirsttrimester"
println("Target Variable: Miscarriageinfirsttrimester")
model = LogisticRegression()
predictor_var = "Miscarriageinfirsttrimester"
LR_score = classification_model(model, predictor_var)
println("Logistic Regression score: ", LR_score)

model = DecisionTreeClassifier()
predictor_var = "Miscarriageinfirsttrimester"
DT_score = classification_model(model, predictor_var)
println("Decision Tree score: ", DT_score)

model = RandomForestClassifier(n_estimators=100)
predictor_var = "Miscarriageinfirsttrimester"
RF_score = classification_model(model, predictor_var)
println("Random Forest score: ", RF_score)

# For "Fetuswithunknowncomplication"
println("Target Variable: Fetuswithunknowncomplication")
model = LogisticRegression()
predictor_var = "Fetuswithunknowncomplication"
LR_score = classification_model(model, predictor_var)
println("Logistic Regression score: ", LR_score)

model = DecisionTreeClassifier()
predictor_var = "Fetuswithunknowncomplication"
DT_score = classification_model(model, predictor_var)
println("Decision Tree score: ", DT_score)

model = RandomForestClassifier(n_estimators=100)
predictor_var = "Fetuswithunknowncomplication"
RF_score = classification_model(model, predictor_var)
println("Random Forest score: ", RF_score)

# For "Tubalpregnancy"
println("Target Variable: Tubalpregnancy")
model = LogisticRegression()
predictor_var = "Tubalpregnancy"
LR_score = classification_model(model, predictor_var)
println("Logistic Regression score: ", LR_score)

model = DecisionTreeClassifier()
predictor_var = "Tubalpregnancy"
DT_score = classification_model(model, predictor_var)
println("Decision Tree score: ", DT_score)

model = RandomForestClassifier(n_estimators=100)
predictor_var = "Tubalpregnancy"
RF_score = classification_model(model, predictor_var)
println("Random Forest score: ", RF_score)

# For "Preeclampsia"
println("Target Variable: Preeclampsia")
model = LogisticRegression()
predictor_var = "Preeclampsia"
LR_score = classification_model(model, predictor_var)
println("Logistic Regression score: ", LR_score)

model = DecisionTreeClassifier()
predictor_var = "Preeclampsia"
DT_score = classification_model(model, predictor_var)
println("Decision Tree score: ", DT_score)

model = RandomForestClassifier(n_estimators=100)
predictor_var = "Preeclampsia"
RF_score = classification_model(model, predictor_var)
println("Random Forest score: ", RF_score)

# For "Normalpregnancy"
println("Target Variable: Normalpregnancy")
model = LogisticRegression()
predictor_var = "Normalpregnancy"
LR_score = classification_model(model, predictor_var)
println("Logistic Regression score: ", LR_score)

model = DecisionTreeClassifier()
predictor_var = "Normalpregnancy"
DT_score = classification_model(model, predictor_var)
println("Decision Tree score: ", DT_score)

model = RandomForestClassifier(n_estimators=100)
predictor_var = "Normalpregnancy"
RF_score = classification_model(model, predictor_var)
println("Random Forest score: ", RF_score)