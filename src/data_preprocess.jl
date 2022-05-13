using CSV
using DataFrames
using Statistics
using Dates
using ScikitLearn
# change file paths 
df = CSV.read("bi-classification.csv", DataFrame)

# Create age column
replace!(df.DEATHDATE, missing => today())
age = [Dates.year(t) for t in df.DEATHDATE] - [Dates.year(t) for t in df.BIRTHDATE]
df[!, :AGE] .= age

cols_cat = ["MARITAL","RACE","ETHNICITY","Tobacco smoking status NHIS","Response to cancer treatment","Site of distant metastasis in Breast tumor","Distant metastases.clinical [Class] Cancer", "Primary tumor.clinical [Class] Cancer","Regional lymph nodes.clinical [Class] Cancer","HER2 [Presence] in Breast cancer specimen by Immune stain","HER2 [Presence] in Breast cancer specimen by FISH","Estrogen receptor Ag [Presence] in Breast cancer specimen by Immune stain","Progesterone receptor Ag [Presence] in Breast cancer specimen by Immune stain","Estrogen+Progesterone receptor Ag [Presence] in Tissue by Immune stain","Stage group.clinical Cancer","Treatment status Cancer","Appearance of Urine","Odor of Urine","Sexual orientation", "Clarity of Urine","Color of Urine","Glucose [Presence] in Urine by Test strip","Bilirubin.total [Presence] in Urine by Test strip","Ketones [Presence] in Urine by Test strip","Protein [Presence] in Urine by Test strip","Nitrite [Presence] in Urine by Test strip","Hemoglobin [Presence] in Urine by Test strip","Leukocyte esterase [Presence] in Urine by Test strip","HIV status", "Abuse Status [OMAHA]","Housing status","Are you covered by health insurance or some other kind of health care plan [PhenX]","Cause of Death [US Standard Certificate of Death]"]
println(length(cols_cat))
for col in cols_cat
    replace!(df[col], missing => String("Unknown"));
end

@sk_import preprocessing: LabelEncoder 
labelencoder = LabelEncoder()
# column index for MARITIAL, RACE, ETHNICITY
cols_cat = ["MARITAL","RACE","ETHNICITY","Tobacco smoking status NHIS","Response to cancer treatment","Site of distant metastasis in Breast tumor","Distant metastases.clinical [Class] Cancer", "Primary tumor.clinical [Class] Cancer","Regional lymph nodes.clinical [Class] Cancer","HER2 [Presence] in Breast cancer specimen by Immune stain","HER2 [Presence] in Breast cancer specimen by FISH","Estrogen receptor Ag [Presence] in Breast cancer specimen by Immune stain","Progesterone receptor Ag [Presence] in Breast cancer specimen by Immune stain","Estrogen+Progesterone receptor Ag [Presence] in Tissue by Immune stain","Stage group.clinical Cancer","Treatment status Cancer","Appearance of Urine","Odor of Urine","Sexual orientation", "Clarity of Urine","Color of Urine","Glucose [Presence] in Urine by Test strip","Bilirubin.total [Presence] in Urine by Test strip","Ketones [Presence] in Urine by Test strip","Protein [Presence] in Urine by Test strip","Nitrite [Presence] in Urine by Test strip","Hemoglobin [Presence] in Urine by Test strip","Leukocyte esterase [Presence] in Urine by Test strip","HIV status", "Abuse Status [OMAHA]","Housing status","Are you covered by health insurance or some other kind of health care plan [PhenX]","Cause of Death [US Standard Certificate of Death]"]

for col in cols_cat
     df[col] = fit_transform!(labelencoder, df[col]) 
end

mis_names = names(df, any.(ismissing, eachcol(df)))
for col in mis_names
    df[ismissing.(df[col]),col] = mean(skipmissing(df[col]))
end

CSV.write("final.csv", df)