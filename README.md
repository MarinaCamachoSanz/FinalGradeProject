# Final Grade Project
Bachelor’s Degree in Bioinformatics (UPF-UPC-UB-UAB) 




Author: Marina Camacho Sanz
#
In this document a numered brief description of the aim of each main script in this repository can be found. Document files (.pdf) 0 and 1 contain the Main Manuscript and Suplementary materials. Files from 2 to 15 corresponds to pre-processing steps which are in bash (.sh), as jupyter notebooks (.ipynb) or R scripts (.R). Then, 15 and 16 the disease modeling files in python (.py) and the retrieved results in the 17 (a folder). Finally from 18 to 2, the R scripts (.R) for plotting the results for analysis and interpretation can be found. 

0. CAMACHO_MARINA_Main_Manuscript.pdf  

1. CAMACHO_MARINA_Supplementary_Material.pdf

2. matrices.ipynb --> This script takes as input all feature files encoding exposome variables from the UK Biobank cohort including the data of each participant (around half million) used in this project. As output, it retrieves matrices, with different caractheristics explained in detail inside these jupyter notebook, containing all features data cleaned without inconsistencyes collected at baseline, which corresponds to the first time visit.

3. bash.sh --> In this file there's a summary of the commands used for participats selections at different points of this project with their description.

4. dementia_patients_selection.ipynb --> This script takes as input the eids whith dementia and its corresponding date of diagnosis. At the end of these file we will download 2 dataframes contaning in each row a record of each diagnosis. One dataframe contains the cases after a given year, and the other one before that year. For each diagnosis we will encode the patiets unique identifier (at column eid), the code of the given subtype of the disease (at column diagnosis) and the date of the disorder (at column date). This file was created for future multiclass analysis.

5. patients_selection_after_before.ipynb --> This script takes as input the eids whith depression and its corresponding date of diagnosis. It retrieves the same as dementia_patients_selection.ipynb but this file was created to be re-used to find patients with other diseases.

6. stats_dementia_centers.ipynb --> This script was used to explore the amount of patients coming from each different assesment center and decide the ones used in internal and external validation. Additionally, it reports the mean age and sex count in internal and external validation to find later an appropiate control group.

7. dementia_matrices.ipynb --> This script separates the internal and external data.

8. healthy_matrices.ipynb --> This script selects and separates the healthy individuals of interest to be used as control group considering their assesment center, sex and age.

9. data_quality.ipynb --> This script recives as input a file and retrieves the quality of the data in terms of missing values.

10. crear_matriu_final.ipynb --> This script first creates the final datasets for imputation. Then it reads the imputed datasets using missForest file and adapts the matrix for runing the predictive disease modeling.

11. missForest.r --> This scrip receives .csv files containing missing values an inputs mixed-type data (continuous and categorical) including complex interactions and nonlinear relations.

12. reduced.ipynb --> This script receives the dataset containing all the final exposures used in experiment 1 and creates a reduced dataset with the new ones of interest.

13. comorbidity --> This script was used to create the dataset for analysing depression comorbidity.

14. feature_scaling.ipynb --> This script normalizes and standarizes the given files.

15. function_SaveDementiaEVAll.py --> function designed to perform the disease modeling with nested cross-validation and an external validation in each fold.

16. run_SaveDementiaEVALL.py --> script that calls function_SaveDementiaEVAll and enters the desired parameters.

17. Folder containging results of the experiments when performing run_SaveDementiaEVALL. 

18.  papergraph.R -->

19.  DvsDD.R -->

20.  plots30.R --> 
