# Development of a Multi-Classification Machine Learning Model to Predict Pregnancy Complications from Synthetic EHR Data

### Team05 Hossam Zaki¹, Julie Karam¹, Nasheath Ahmed¹, Qingyan Guo¹, Noa Mintz¹

#### Project Overview:
Pregnancy complications encompass a wide range of conditions of varying severity and etiology. While the prevalence of common complications have been extensively studied and reported, the number of factors and their interactions along with the time intervals make predictions of pregnancy complications and outcomes very complicated. However, datasets are available that provide comprehensive information on health before and during pregnancy, and subsequent outcomes. Due to the large quantity and complex nature of medical information, ML is recognized as a promising method for supporting diagnosis or predicting clinical outcomes. Therefore, we sought to develop a multi-classification ML-model to predict the occurrence of pregnancy complications based on routinely-collected data found in electronic health records (EHRs). Our study found that the bi-classification models using top 15 selected features performed the best.
* The source codes are in `src` directory
* The result figures are in `fig` directory


#### Setup and Requirements:
* Packages: 
  * CSV
  * DataFrames
  * Statistics
  * ScikitLearn
  * VegaLite
  * Dates

#### Data Source:

The data comes from the 1K sample synthetic dataset from the open-source Synthea tool. You can download the raw data source and place it is in the same folder in the src folder. The scripts should be able to read in these files based on the folder it is in. 
#### Methodologies:
* Data Merge and Preprocess: 
  * The `data_merge.jl` file combines three important csv files based on unique *Patient_ID* to get the final dataset for our project. Then we get 332 recordings with 121 features in all. 
  * In `data_preprocess.jl` file, we use *mean* value to impute missing values for continuous features and use *Unknown* to impute missing values for categorical features.

* Exploratory Data Analysis:
  * The `EDA.jl` helps us to know the relationships between features and target variables. Also we can use *correlation matrix* to learn the dependency between differnt features.

* Feature Selection:
  * The data_analysis.jl allows us to know the percentage of missing features for each of the different features per patient.

* Bi-classication models:
  * We firstly try to use all features for bi-classification task, the results can be seen as a baseline. The codes are in `bi_all_ftrs.jl`.
  * After the *feature selection* step, we utilize the top 15 important features to do the bi-classification task. The codes are in `bi_top15_ftrs.jl`.
  * For the fulluu connected nueral network model, you can run `python neural_net_models.py` for the bi classification results which will be the first result. 
  
* Multi-classification models:
  * For the neural network multi-classification model, we can run `python neural_net_models.py` which will contain the b-classification and multi-class results. 

* Supplementary codes: 
  * For interests, we try to pay more attention to *miscarriage*, as the complication appears more often than other complications. The corresponding codes are in `miscarriage_results.py`





#### References:
1. U.S. Department of Health and Human Services. A Report of the Surgeon General: How Tobacco Smoke Causes Disease: What It Means to You. Atlanta: U.S. Department of Health and Human Services, Centers for Disease Control and Prevention, National Center for Chronic Disease Prevention and Health Promotion, Office on Smoking and Health, 2010..
2. Khatibi, T., Hanifi, E., Sepehri, M.M. et al. Proposing a machine-learning based method to predict stillbirth before and during delivery and ranking the features: nationwide retrospective cross-sectional study. BMC Pregnancy Childbirth 21, 202 (2021).
3. Khatibi, T., Hanifi, E., Sepehri, M.M. et al. Proposing a machine-learning based method to predict stillbirth before and during delivery and ranking the features: nationwide retrospective cross-sectional study. BMC Pregnancy Childbirth 21, 202 (2021). https://doi.org/10.1186/s12884-021-03658-z
4. Davidson L, Boland MR. Towards deep phenotyping pregnancy: a systematic review on artificial intelligence and machine learning methods to improve pregnancy outcomes. Brief Bioinform. 2021 Sep 2;22(5):bbaa369. doi: 10.1093/bib/bbaa369. PMID: 33406530; PMCID: PMC8424395. https://pubmed.ncbi.nlm.nih.gov/33406530/

