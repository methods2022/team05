# Development of a Multi-Classification Machine Learning Model to Predict Pregnancy Complications from Synthetic EHR Data

### Team05 Hossam Zaki¹, Julie Karam¹, Nasheath Ahmed¹, Qingyan Guo¹, Noa Mintz¹

#### Project Overview

#### Setup and Requirements

#### Data Source

#### Methodologies:
* Data Merge and Preprocess: 
  * The `data_merge.jl` file combines three important csv files based on unique *Patient_ID* to get the final dataset for our project. Then we get 332 recordings with 121 features in all. 
  * In `data_preprocess.jl` file, we use *mean* value to impute missing values for continuous features and use *Unknown* to impute missing values for categorical features.

* Exploratory Data Analysis:
  * The `EDA.jl` helps us to know the relationships between features and target variables. Also we can use *correlation matrix* to learn the dependency between differnt features.

* Feature Selection:

* Bi-classication models:
  * We firstly try to use all features for bi-classification task, the results can be seen as a baseline. The codes are in `bi_all_ftrs.jl`.
  * After the *feature selection* step, we utilize the top 15 important features to do the bi-classification task. The codes are in `bi_top15_ftrs.jl`.
  
* Multi-classification models:

* Supplementary codes: 
  * For interests, we try to pay more attention to *miscarriage*, as the complication appears more often than other complications. The corresponding codes are in `miscarriage_results.py`
