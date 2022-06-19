# Final Grade Project
Bachelor’s Degree in Bioinformatics (UPF-UPC-UB-UAB) 
#
In this document, a numbered brief description of the aim of each script in this repository can be found. Document files (.pdf) 0 and 1 contain the Main Manuscript and Supplementary materials. Files from 2 to 15 correspond to pre-processing steps which are in bash (.sh), as Jupiter notebooks (.ipynb) or R scripts (.R). Then, 15 and 16 the disease modeling files in python (.py) and the input and output results in the folder 17,18. Finally, from 19 to 21, the R scripts (.R) for plotting the results for analysis and interpretation can be found. 

0. CAMACHO_MARINA_Main_Manuscript.pdf  

1. CAMACHO_MARINA_Supplementary_Material.pdf

2. matrices.ipynb --> This script takes as input all feature files encoding exposome variables from the UK Biobank cohort including the data of each participant (around half a million) used in this project. As output, it retrieves matrices, with different characteristics explained in detail inside this jupyter notebook, containing all features data cleaned without inconsistencies collected at baseline, which corresponds to the first time visit.

3. bash.pdf --> In this file, there's a summary of the commands used for participant selections at different points of this project with their description.

4. dementia_patients_selection.ipynb --> This script takes as input the eids with dementia and its corresponding date of diagnosis. At the end of this file, we will download 2 data frames containing in each row a record of each diagnosis. One data frame contains the cases after a given year and the other one before that year. For each diagnosis, we will encode the patient's unique identifier (at column eid), the code of the given subtype of the disease (at column diagnosis), and the date of the disorder (at column date). This file was created for future multiclass analysis.

5. patients_selection_after_before.ipynb --> This script takes as input the eids with depression and their corresponding date of diagnosis. It retrieves the same as dementia_patients_selection.ipynb but this file was created to be re-used to find patients with other diseases.

6. stats_dementia_centers.ipynb --> This script was used to explore the number of patients coming from each different assessment center and decide the ones used in internal and external validation. Additionally, it reports the mean age and sex count in internal and external validation to find later an appropriate control group.

7. dementia_matrices.ipynb --> This script separates the internal and external data.

8. healthy_matrices.ipynb --> This script selects and separates the healthy individuals of interest to be used as a control group considering their assessment center, sex, and age.

9. data_quality.ipynb --> This script receives as input a file and retrieves the quality of the data in terms of missing values.

10. crear_matriu_final.ipynb --> This script first creates the final datasets for imputation. Then it reads the imputed datasets using the missForest file and adapts the matrix for running the predictive disease modeling.

11. missForest.r --> This scrip receives .csv files containing missing values an inputs mixed-type data (continuous and categorical) including complex interactions and nonlinear relations.

12. reduced.ipynb --> This script receives the dataset containing all the final exposures used in experiment 1 and creates a reduced dataset with the new ones of interest.

13. comorbidity --> This script was used to create the dataset for analyzing depression comorbidity.

14. feature_scaling.ipynb --> This script normalizes and standardizes the given files.

15. function_SaveDementiaEVAll.py --> function designed to perform the disease modeling with nested cross-validation and external validation in each fold.

16. run_SaveDementiaEVALL.py --> script that calls function_SaveDementiaEVAll and enters the desired parameters.

17. Folder containing final matrices to model predictors. 

18. Folder containing results of the experiments when performing run_SaveDementiaEVALL. 

19. plot_performance_dementia.R --> script that plots the performance of experiments 1-5 in internal and external validation separately.

20. plot_importance_30.R --> script that plots the 30 most important variables of a model.

21. plot_performance_DvsDD.R --> script that plots the performance of experiments 6 in internal and external validation separately.
#
I hope you enjoy the project! Thank you for your time.

Marina Camacho Sanz

