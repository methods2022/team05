using MLJ
using CSV
using DataFrames
using LIBSVM
using ScikitLearn: fit!
using ScikitLearn: predict


function svm_model()
    df = CSV.read("./final.csv", DataFrame)
    targets = ["Fetus with unknown complication", "Tubal pregnancy", "Miscarriage in first trimester", "Preeclampsia", "Normal pregnancy"]

    X = select(df, Not(["PATIENT_ID","BIRTHDATE","DEATHDATE","Fetus with unknown complication", "Tubal pregnancy", "Miscarriage in first trimester", "Preeclampsia", "Normal pregnancy"]))
    for name in targets
        y = Array([])
        for row in eachrow(df)
            if row[name] == 1
                push!(y, 1)
                continue
            else
                push!(y, 0)
            end
        end
        y = categorical(y)
        println(length(y))
        train, test = partition(eachindex(y), 0.8)
        model = fit!(SVC(), Matrix(X[train, :]), y[train])
        ŷ = predict(model, Matrix(X[test,:]))
        acc = mean(ŷ .== y[test]) * 100
        println("Test Set Accuracy on $(name): $acc")
    end


end


svm_model()