using CSV
using DataFrames
# change file paths 
df = CSV.read("conditions.csv", DataFrame)
new_df = select(df, [:PATIENT, :DESCRIPTION])
pa_des = new_df[in(["Fetus with unknown complication", "Tubal pregnancy", "Miscarriage in first trimester", "Preeclampsia", "Normal pregnancy"]).(new_df.DESCRIPTION), :]
patient_id = combine(groupby(pa_des, [:PATIENT]), nrow => :count)
patient_ids = Array(patient_id[!, :PATIENT])
df = CSV.read("observations.csv", DataFrame)
des_count = combine(groupby(df, [:DESCRIPTION]), nrow => :count)
meds_df = CSV.read("medications.csv", DataFrame)
obs_df = CSV.read("observations.csv", DataFrame)
#patients_df = CSV.read("patients.csv", DataFrame)
meds_df = meds_df[in.(meds_df[!,:PATIENT], (patient_ids,)), :]
obs_df = obs_df[in.(obs_df[!,:PATIENT], (patient_ids,)), :]
pa_des = new_df[in(["Fetus with unknown complication", "Tubal pregnancy", "Miscarriage in first trimester", "Preeclampsia", "Normal pregnancy"]).(new_df.DESCRIPTION), :]
final_df = DataFrame()
final_df.PATIENT_ID = patient_ids
targets = ["Fetus with unknown complication", "Tubal pregnancy", "Miscarriage in first trimester", "Preeclampsia", "Normal pregnancy"]
for name in targets
    final_df.new_col = zeros(Int8, 332)
    rename!(final_df, :new_col => name)
end
# add target variables categorically to patient_id data
for id in eachrow(final_df)
    for row in eachrow(pa_des)
        if row["PATIENT"]==id["PATIENT_ID"]
            target = row["DESCRIPTION"]
            id[target] = 1
        end 
    end
end 

# add observation data
des_count = combine(groupby(obs_df, [:DESCRIPTION]), nrow => :count)
observations = Array(des_count[!, :DESCRIPTION])
#println(observations)
for obs in observations
    #final_df.new_col = zeros(Int8, 332)
    final_df.new_col = Array{Union{Missing, String}}(missing, 332)
    rename!(final_df, :new_col => obs)
end

for id in eachrow(final_df)
    for row in eachrow(obs_df)
        if row["PATIENT"]==id["PATIENT_ID"]
            target = row["DESCRIPTION"]
            value = row["VALUE"]
            id[target] = value
        end 
    end
end 

# add patients data
patients_df = CSV.read("patients.csv", DataFrame)
pat_df = patients_df[in(patient_ids).(patients_df.Id), :]
pat_df = select(pat_df, [:Id, :BIRTHDATE, :DEATHDATE, :MARITAL,:RACE, :ETHNICITY,:HEALTHCARE_EXPENSES])
rename!(pat_df,:Id => :PATIENT_ID)
# merged df (three csv files)
merged_con_obs_pat_df = leftjoin(pat_df,final_df, on = :PATIENT_ID)

CSV.write("bi-classification.csv", merged_con_obs_pat_df)