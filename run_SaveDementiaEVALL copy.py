import function_SaveDementiaEVAll
import pandas as pd
import os

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> input data

#names_files=["input_df_ct_80_age_no.csv","input_df_ct_80_30_age.csv","input_df_ct_80_30.csv","input_df_ct_80.csv","input_df_ct_80_age.csv"]
#external_validation_files=["exval_df_ct_80_age_no.csv", "exval_df_ct_80_30_age.csv","exval_df_ct_80_30.csv","exval_df_ct_80.csv","exval_df_ct_80_age.csv"]

#names_files=["DvsDD_ADASYN.csv"]
#external_validation_files=["DvsDD_ADASYN_ev.csv"]

names_files=["age_30_internal_mm.csv","age_30_internal_zs.csv",
            "age_internal_zs.csv","noage_internal_zs.csv","oage_internal_zs.csv","noage_30_internal_zs.csv"
            "age_internal_mm.csv","noage_internal_mm.csv","oage_internal_mm.csv","noage_30_internal_mm.csv"]

external_validation_files=["age_30_external_mm.csv","age_30_external_zs.csv",
            "age_external_zs.csv","noage_external_zs.csv","oage_external_zs.csv","noage_30_external_zs.csv"
            "age_external_mm.csv","noage_external_mm.csv","oage_external_mm.csv","noage_30_external_mm.csv"]

path_parent="/Users/marinacamacho/Desktop/BCN-AIM/DementiaTFG/input_folder"
output_folder="/Users/marinacamacho/Desktop/BCN-AIM/DementiaTFG/output_folder_EV"

n_repetitions=1
fold_type_out=2
fold_type_inn=2
n_folds_out=7
n_folds_in=5
code_id = 'f.eid'
code_outcome = 'Y'

# process
for k in range(0,len(names_files)):
    print(names_files[k])
    print(external_validation_files[k])
    function_SaveDementiaEVAll.run_prediction(path_parent, names_files[k], output_folder, code_id, code_outcome, n_repetitions,
                                       fold_type_out, fold_type_inn, n_folds_out,n_folds_in, external_validation_files[k])
