using CSV
using DataFrames
using VegaLite
using Statistics
using Dates
using ScikitLearn
using  CategoricalArrays

# Read from the condition.csv file
df = CSV.read("conditions.csv", DataFrame)
new_df = select(df, [:PATIENT, :DESCRIPTION])
pa_des = new_df[in(["Fetus with unknown complication", "Tubal pregnancy", "Miscarriage in first trimester", "Preeclampsia", "Normal pregnancy"]).(new_df.DESCRIPTION), :]
des_count = combine(groupby(pa_des, [:DESCRIPTION]), nrow => :count)

# plot for all complications
des_count |> @vlplot(:bar, x="DESCRIPTION", y="count",width=300,height=300, title="Patients distribution for different complications")


# plots for features vs target variable
final_df = CSV.read("final.csv", DataFrame)
names!(final_df, Symbol.(replace.(string.(names(final_df)), Ref(r"\s"=>""))))
#categorical!(final_df, :MARITAL)

# plot for Marital
categorical!(final_df, :Miscarriageinfirsttrimester)
final_df |>
@vlplot(
    mark={
        :bar,
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    },
    x={"MARITAL", axis={title="Marital status"}},
    y="count()",
    color={
        :Miscarriageinfirsttrimester,
    legend={
            title="Miscarriage in first trimester or not"
        },
    },
    
    width=300,height=300,
    title="Miscarriage in first trimester or not in different Marital status "
)


# plot for RACE
categorical!(final_df, :Miscarriageinfirsttrimester)
final_df |>
@vlplot(
    mark={
        :bar,
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    },
    x={"RACE", axis={title="RACE"}},
    y="count()",
    color={
        :Miscarriageinfirsttrimester,
    legend={
            title="Miscarriage in first trimester or not"
        },
    },
    
    width=300,height=300,
    title="Miscarriage in first trimester or not in different RACEs "
)

# Plot for smoke status
categorical!(final_df, :Miscarriageinfirsttrimester)
final_df |>
@vlplot(
    mark={
        :bar,
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    },
    x={"TobaccosmokingstatusNHIS", axis={title="smoke status"}},
    y="count()",
    color={
        :Miscarriageinfirsttrimester,
    legend={
            title="Miscarriage in first trimester or not"
        },
    },
    
    width=300,height=300,
    title="Miscarriage in first trimester or not in different smoke status "
)


# plot for AGE
categorical!(final_df, :Miscarriageinfirsttrimester)
final_df |>
@vlplot(
    mark={
        :line,
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    },
    x={"AGE", axis={title="Age"}},
    y="count()",
    color={
        :Miscarriageinfirsttrimester,
    legend={
            title="Miscarriage in first trimester"
        },
    },
    
    width=300,height=300,
    title="Miscarriage in first trimester in different ages "
)


# plot for BMI
categorical!(final_df, :Miscarriageinfirsttrimester)
final_df |>
@vlplot(
    mark={
        :line,
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    },
    x={"BodyMassIndex", axis={title="BMI"}},
    y="count()",
    color={
        :Miscarriageinfirsttrimester,
    legend={
            title="Miscarriage in first trimester"
        },
    },
    
    width=300,height=300,
    title="Miscarriage in first trimester in different BMIs "
)