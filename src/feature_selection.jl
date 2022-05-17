using FeatureSelectors
using CSV
using DataFrames

# Read from the condition.csv file
df = CSV.read("/Users/juliekaram/Downloads/final.csv", DataFrame)
df


# transform smoking info into binary data
smoking_dict = Dict()
smoking_dict["Never smoker"] = "0"
smoking_dict["Former smoker"] = "1"
smoking_dict["Current every day smoker"] = "2"

for row in eachrow(df) 
    row["Tobacco smoking status NHIS"] = smoking_dict[row["Tobacco smoking status NHIS"]]
end 

df[!,"Tobacco smoking status NHIS"] = tryparse.(Int64,df[:,"Tobacco smoking status NHIS"])

# create feature selection for top 15 features
selector = UnivariateFeatureSelector(method=pearson_correlation, k=15)

targets = ["PATIENT_ID", "BIRTHDATE", "DEATHDATE", "Fetus with unknown complication", 
    "Tubal pregnancy", "Miscarriage in first trimester", "Preeclampsia", "Normal pregnancy"]
features = (df[:, filter(x -> !(x in targets), names(df))])

# selection top 15 features for each condition
select_features(
           selector,
           features,
           df[:,"Miscarriage in first trimester"]
       )

select_features(
           selector,
           features,
           df[:,"Fetus with unknown complication"]
       )

select_features(
           selector,
           features,
           df[:,"Tubal pregnancy"]
       )

select_features(
           selector,
           features,
           df[:,"Preeclampsia"]
       )

select_features(
           selector,
           features,
           df[:,"Normal pregnancy"]
       )
