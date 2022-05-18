using FeatureSelectors
using CSV
using DataFrames
using DelimitedFiles

df = CSV.read("/Users/noamintz/Desktop/BIOL 1555/FinalProject/final.csv", DataFrame)
df

# from group member julie
smoking_dict = Dict()
smoking_dict["Never smoker"] = "0"
smoking_dict["Former smoker"] = "1"
smoking_dict["Current every day smoker"] = "2"
for row in eachrow(df) 
    row["Tobacco smoking status NHIS"] = smoking_dict[row["Tobacco smoking status NHIS"]]
end 
df[!,"Tobacco smoking status NHIS"] = tryparse.(Int64,df[:,"Tobacco smoking status NHIS"])


# compute pearsons for top 5 values
selector = UnivariateFeatureSelector(method=pearson_correlation, k=5)
targets = ["PATIENT_ID", "BIRTHDATE", "DEATHDATE", "Fetus with unknown complication", 
"Tubal pregnancy", "Miscarriage in first trimester", "Preeclampsia", "Normal pregnancy"]
features = (df[:, filter(x -> !(x in targets), names(df))])
# top 5 for normal pregnancy
normal_preg_5 = select_features(selector,features,df[:,"Normal pregnancy"])
writedlm("normal_preg_top5_features.txt", normal_preg_5)
# top 5 for preeclampsia 
preec_5 = select_features(selector,features,df[:,"Preeclampsia"])
writedlm("preeclampsia_top5_features.txt", preec_5)
# top 5 for first trimester miscarriage
misc_5 = select_features(selector,features,df[:,"Miscarriage in first trimester"])
writedlm("miscarriage_top5_features.txt", misc_5)
# fetus w unknown complication
fetus_unknown_5 = select_features(selector,features,df[:,"Fetus with unknown complication"])
writedlm("fetus_unknown_comp_top5_features.txt", fetus_unknown_5)
# tubal pregnancy
tubal_5 = select_features(selector,features,df[:,"Tubal pregnancy"])
writedlm("tubal_pregnancy_top5_features.txt", tubal_5)
