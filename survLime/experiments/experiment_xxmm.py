import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import argparse


import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

import seaborn as sns
sns.set()

from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

# Our very own survLime!
from survLime import survlime_explainer
from functools import partial

from derma.general.ingestion.data_loader_csv import SurvivalLoader

from derma.general.preprocessing.transformers import (TransformToNumeric, 
                                                      TransformToDatetime, 
                                                      ComputeAge,
                                                      TransformToObject,
                                                      KeepColumns,
                                                      ComputeAJCC,
                                                      LinkTumourPartToParent,
                                                      TransformCbRegression,
                                                      ConvertCategoriesToNaN,
                                                      ExponentialTransformer,
                                                      RenameLabValues,
                                                      CustomScaler,
                                                      CustomImputer)

from derma.general.preprocessing.encoders import (OrdinalEncoder,
                                                  GenderEncoder,
                                                  AbsentPresentEncoder,
                                                  LABEncoder,
                                                  CategoricalEncoder)

import os
import config_os as settings_file

np.random.seed(123456)
get_target = lambda df: df[['event','duration']]
path = '/home/carlos.hernandez/datasets/csvs/data-surv_20220302.csv'

    
clinical_columns = ['patient_gender', 'patient_eye_color',
       'patient_phototype', 'cutaneous_biopsy_breslow',
       'cutaneous_biopsy_mitotic_index', 'cutaneous_biopsy_ulceration',
       'cutaneous_biopsy_neurotropism', 'cutaneous_biopsy_satellitosis', 'age',
       'primary_tumour_location_coded_acral',
       'primary_tumour_location_coded_head and neck',
       'primary_tumour_location_coded_lower limbs',
       'primary_tumour_location_coded_upper limbs',
       'primary_tumour_location_coded_mucosa',
       'cutaneous_biopsy_histological_subtype_acral_lentiginous',
       'cutaneous_biopsy_histological_subtype_desmoplastic',
       'cutaneous_biopsy_histological_subtype_lentiginous_malignant',
       'cutaneous_biopsy_histological_subtype_mucosal',
       'cutaneous_biopsy_histological_subtype_nevoid',
       'cutaneous_biopsy_histological_subtype_nodular',
       'cutaneous_biopsy_histological_subtype_spitzoid',
       'patient_hair_color_black', 'patient_hair_color_blond',
       'patient_hair_color_red']

lab_columns = ['Recompte de leucòcits', 'Recompte plaquetes', 'Neutròfils absoluts (anal)',
        'Limfòcits absoluts (analitzado', 'Monòcits absoluts (analitzador', 'Eosinòfils absoluts (analitzad',
        'Basòfils absoluts (analitzador', 'Aspartat aminotransferasa (ASA', 'Alanin aminotransferasa (ALAT)',
        'Gamma glutaril transferasa (GG', 'Bilirrubina total', 'Lactat deshidrogenasa (LDH)',
        'Glucosa', 'Creatinina plasmàtica', 'Colesterol total', 'Triglicèrids',
        'Proteïnes totals', 'Beta 2 microglobulina', 'Proteïna S100',
        'Melanoma Inhibitory activity', 'Concentració dhemoglobina']

def obtain_rsf_kwargs(args):
    if args.event_type=='ss':
        kwargs = {'n_estimators' : 300,
                    'min_weight_fraction_leaf': 0,
                    'min_samples_split' : 11,
                    'max_features' : 'auto',
                    'max_depth':10}
    elif args.event_type=='dfs':
        kwargs = {'n_estimators' : 500,
                    'min_weight_fraction_leaf': 0,
                    'min_samples_split' : 15,
                    'max_features' : 'sqrt',
                    'max_depth':11}
    elif args.event_type=='os':
        kwargs = {'n_estimators' : 50,
                    'min_weight_fraction_leaf': 0,
                    'min_samples_split' : 14,
                    'max_features' : 'sqrt',
                    'max_depth':14}
    return kwargs



def main(args):
    X, _, time, event = SurvivalLoader(args.event_type).load_data(path)
    X['duration'] = round(time)

    X['event']    = event
    X.dropna(subset=['duration'],inplace=True)
    X = X[X['duration']>=0]
#   keep_cols = clinical_columns +lab_columns
#   settings_file.keep_cols = {'cols' : keep_cols}

    X_train_previous, X_test, y_train, y_test = train_test_split(X.copy(), np.zeros(len(X)), test_size=0.1, random_state=42)
    for repetition in tqdm(range(args.repetitions)):
        try:
            if args.model == 'cox':
                model_pipe = ('coxph', CoxPHSurvivalAnalysis(alpha=0.0001))
            elif args.model == 'rsf':
                kwargs = obtain_rsf_kwargs(args)
                model_pipe = ('rsf', RandomSurvivalForest(
                                  **kwargs,
                                  random_state=42))
            else:
                AssertionError('Model not implemented')

            pipe = sklearn.pipeline.Pipeline(steps=[
                ('TransformToNumeric', TransformToNumeric(**settings_file.transform_to_numeric)), 
                ('TransformToDatetime', TransformToDatetime(**settings_file.transform_to_datetime)),
                ('TransformToObject', TransformToObject(**settings_file.transform_to_object)),
                ('ComputeAge', ComputeAge(**settings_file.compute_age)),
                ('tr_tm', LinkTumourPartToParent(**settings_file.link_tumour_part_to_parent)),
                ('tr_cb', TransformCbRegression(**settings_file.transform_cb_regression)),
                ('tr0', ConvertCategoriesToNaN(**settings_file.convert_categories_to_nan)),
                ('tr2', GenderEncoder(**settings_file.gender_encoder)),
                ('tr3', AbsentPresentEncoder(**settings_file.absent_present_encoder)),
                ("tr4", CategoricalEncoder(**settings_file.categorical_encoder)),
                ('tr7', LABEncoder(**settings_file.lab_encoder)),
                ('OrdinalEncoder', OrdinalEncoder(**settings_file.ordinal_encoder)),
                ('ComputeAJCC', ComputeAJCC(**settings_file.compute_ajcc)),
                ('tr5', ExponentialTransformer(**settings_file.exponential_transformer)),
                ('KeepColumns', KeepColumns(**settings_file.keep_cols)),
                ('RenameLabValues', RenameLabValues(**settings_file.rename_lab_values)),
                ('CustomImputer', CustomImputer(strategy='mean')),
                ('CustomScaler', CustomScaler()),
                model_pipe
                ])
            

            
            X_train, X_val, y_train, y_val = train_test_split(X_train_previous.copy(), np.zeros(len(X_train_previous)), test_size=0.2, random_state=repetition)
            y_train_list = get_target(X_train)
            
            y_train_list = Surv.from_dataframe(*y_train_list.columns, y_train_list)
            #import ipdb;ipdb.set_trace() 
            fitted_pipe = pipe.fit(X_train.copy(), y_train_list.copy())
            X_test_t = fitted_pipe[:-1].transform(X_test)
            X_train_t = fitted_pipe[:-1].transform(X_train)

            num_neighbors = args.num_neigh
            explainer = survlime_explainer.SurvLimeExplainer(X_train_t, y_train_list,
                                                             model_output_times=pipe[-1].event_times_)

            computation_exp = compute_weights(explainer, X_test_t, pipe[-1], num_neighbors = num_neighbors)
            save_path = f"/home/carlos.hernandez/PhD/survlime-paper/survLime/computed_weights_csv/exp4/{args.event_type}/{args.model}_xxmm_surv_{args.event_type}_weights_rand_seed_{repetition}.csv"
            computation_exp.to_csv(save_path, index=False)

        except:
            print('Error in repetition', repetition)
            # print traceback
            import traceback
            print(traceback.print_exc())
            pass

def compute_weights(
    explainer: survlime_explainer.SurvLimeExplainer,
    x_test:  pd.DataFrame,
    model: CoxPHSurvivalAnalysis,
    num_neighbors: int = 1000
):
    compt_weights = []
    num_pat = num_neighbors
    predict_chf = partial(model.predict_cumulative_hazard_function, return_array=True)
    for test_point in tqdm(x_test.to_numpy()):
        try:
            b, _ = explainer.explain_instance(
                test_point, predict_chf, verbose=False, num_samples=num_pat
            )
                
            b = [x[0] for x in b]
            compt_weights.append(b)
        except:
            b = [None] * len(test_point)
            compt_weights.append(b)

    return pd.DataFrame(compt_weights, columns=model.feature_names_in_)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Obtain SurvLIME results for a given dataset"
    )
    parser.add_argument("--event_type", type=str, default="ss",
                        help="os, dfs or ss"
    )
    parser.add_argument(
        "--model", type=str, default="cox", help="bb model either cox or rsf"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="How many times to repeat the experiment",
    )
    parser.add_argument(
        "--num_neigh",
        type=int,
        default=1000,
        help="Number of neighbours to use for the explanation",
    )
    args = parser.parse_args()
    main(args)

