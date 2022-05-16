using CSV
using DataFrames
using Plots

function missing_check()
    df = CSV.read("./bi-classification.csv", DataFrame)
    # Get all the non missing columns
    number_of_patients = size(df, 1)
    final_df = describe(df)
    column_titles = Array(final_df[!, :variable])
    column_arr = []
    for word in column_titles
        push!(column_arr, String(word))
    end
    missing_ = Array(final_df[!, :nmissing])
    #zipping the column names and number of missing values
    paired = collect(zip(column_arr,missing_))
    missing_dict = Dict()
    missing_dict["<20"] = Array([])
    missing_dict["<50"] = Array([])
    missing_dict["50-75"] = Array([])
    missing_dict["75-100"] = Array([])
    no_missing_cols = []
    #For loop will store the perecentage of data missing for that feature of interest
    for pair in paired
        if pair[2] == 0
            push!(no_missing_cols, pair[1])
        end
        fraction_missing = pair[2]/number_of_patients
        if fraction_missing < 0.2
            push!(missing_dict["<20"], pair[1])
        elseif fraction_missing < 0.5
            push!(missing_dict["<50"], pair[1])
        elseif fraction_missing < 75
            push!(missing_dict["50-75"], pair[1])
        else
            push!(missing_dict["75-100"], pair[1])
        end
    end

    for key in keys(missing_dict)
        println(key)
        println(length(missing_dict[key]))
        println(missing_dict[key])

    end
    println(length(no_missing_cols))
    println(no_missing_cols)
end
function condition_check()
    #Checking to see which patients have multiple conditions and distribution of conditions
    df = CSV.read("./bi-classification.csv", DataFrame)
    condition_arr = Array([])
    dict_for_repeats = Dict()
    dict_for_condition_dist = Dict()

    targets = ["Fetus with unknown complication", "Tubal pregnancy", "Miscarriage in first trimester", "Preeclampsia", "Normal pregnancy"]
    for id in eachrow(df)
        count = 0
        for name in targets
            if id[name] == 1
                count  = count + 1
                if !haskey(dict_for_condition_dist, name)
                    dict_for_condition_dist[name] = 1
                else
                    dict_for_condition_dist[name] += 1
                end
            end
        end
        if count > 1
            dict_for_repeats[id["PATIENT_ID"]] = count
            push!(condition_arr, count)
        elseif count == 1
            push!(condition_arr, count)
        end
    end
    println("Total number of patients: $(size(df, 1))")
    println("Total number of patients with more than one disease: $(length(dict_for_repeats))")
    println("The percent of patients with multiple diseases $(length(dict_for_repeats)/size(df, 1))")
    println(dict_for_condition_dist)
    column_names = names(df)
    new_df = DataFrame(patient_id = df[!,"PATIENT_ID"], number_of_conds = condition_arr)
    CSV.write("patient_dist.csv", new_df, delim="|")
    histogram(condition_arr, title = "Distribution of Number of Conditions",label = "Frequency")
    savefig("../fig/disease_dist.png")
end

function main()
    missing_check()
    condition_check()
end
main()